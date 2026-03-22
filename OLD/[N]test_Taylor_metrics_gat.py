import os
import argparse
import numpy as np
import dask.array as da
import networkx as nx
import torch
import csv
from epynet import Network

# Import utilities from the repository
from utils.graph_utils import get_nx_graph
from utils.DataReader import DataReader
from utils.SensorInstaller import SensorInstaller
from utils.Metrics import Metrics
from utils.MeanPredictor import MeanPredictor
from utils.baselines import interpolated_regularization
from utils.dataloader import build_dataloader

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds',
                    default = 'anytown',
                    type    = str,
                    help    = "Water distribution system."
                    )
parser.add_argument('--tag',
                    default = 'def',
                    type    = str,
                    help    = "Customer tag"
                    )
parser.add_argument('--obsrat',
                    default = 0.05,
                    type    = float,
                    help    = "Observation ratio."
                    )
parser.add_argument('--runs',
                    default = 20,
                    type    = int,
                    help    = "Total experiments."
                    )
parser.add_argument('--gnn',    #TO INCLUDE GATs
                    default = 'cheb1',
                    choices = ['cheb1', 'cheb2', 'gat'],
                    type    = str,
                    help    = "GNN architecture to use.")
args= parser.parse_args()

if args.gnn == 'gat':
   from model.anytown_gat import GATNet as Net
elif args.gnn == 'cheb1':
   from model.anytown import ChebNet as Net
elif args.gnn == 'cheb2':
   from model.anytown_v2 import ChebNet as Net
else:
   from model.anytown_v2 import ChebNet as Net


# --- Configuration ---
WDS_NAME = args.wds #'anytown'
DB_NAME = 'doe_pumpfed_1'
GNN = args.gnn
#BUDGET = 1           # 5% Sensors
TAG = args.tag
OBSRAT = args.obsrat
RUNS = args.runs           # Number of repetitions you trained
BATCH_SIZE = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
base_dir = os.path.dirname(os.path.realpath(__file__))
results_file = os.path.join(base_dir, 'experiments', 'Taylor_metrics.csv')

def compute_metrics(p, p_hat):
    # Calculate Covariance and Standard Deviation for Taylor Diagram
    # p = Real Pressure, p_hat = Predicted Pressure
    msec = da.multiply(p - p.mean(), p_hat - p_hat.mean()).mean()
    sigma = da.sqrt(da.square(p_hat - p_hat.mean()).mean())
    return msec.compute(), sigma.compute()

def run_evaluation():
    print(f" Starting Full Evaluation for {WDS_NAME} ({RUNS} runs)...")
    
    # Load Water Network & Graph ONCE
    inp_path = os.path.join('water_networks', f'{WDS_NAME}.inp')
    wds = Network(inp_path)
    G = get_nx_graph(wds, mode='binary')
    # Laplacian is needed for Interpolation baseline
    L = nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()
    num_nodes = len(wds.junctions)

    for run_id in range(1, RUNS + 1):
        print(f"\n--- Processing Run #{run_id} ---")
        
        # 1. Define Paths for this specific run
        # Filename format: anytown-random-2-binary-def-1
        #run_stamp = f"{WDS_NAME}-random-{OBSRAT}-binary-{GNN}-{TAG}_{run_id}-1" ## FOR 1 SENSOR TESTING
        run_stamp = f"{WDS_NAME}-xrandom-{OBSRAT}-binary-{GNN}-{TAG}-{run_id}"
        model_path = os.path.join(base_dir, 'experiments', 'models', f"{run_stamp}.pt")
        sensor_path = os.path.join(base_dir, 'experiments', 'models', f"{run_stamp}_sensor_nodes.csv")
        
        # 2. Load the EXACT Sensors used in training
        if not os.path.exists(sensor_path):
            print(f" Warning: Sensor file not found for {run_stamp}. Skipping.")
            continue
            
        sensor_shop = SensorInstaller(wds)
        sensor_nodes = np.loadtxt(sensor_path, dtype=np.int32)
        sensor_shop.set_sensor_nodes(sensor_nodes)
        
        # 3. Load Data (Test Set) with this sensor mask
        db_path = os.path.join(base_dir, 'data', f'db_{WDS_NAME}_{DB_NAME}')
        reader = DataReader(
            db_path, 
            n_junc=num_nodes, 
            signal_mask=sensor_shop.signal_mask(),
            node_order=np.array(list(G.nodes))-1
        )
        
        # X is standardized (Inputs), Y is Normalized (Targets)
        tst_x, bias_std, scale_std = reader.read_data('tst', 'junc_heads', 'standardize', cover=True)
        tst_y, bias_nrm, scale_nrm = reader.read_data('tst', 'junc_heads', 'normalize', cover=False)
        
        tst_ldr = build_dataloader(G, tst_x, tst_y, BATCH_SIZE, shuffle=False)
        metrics_nrm = Metrics(bias_nrm, scale_nrm, DEVICE)
        num_graphs = len(tst_x)

        # Get Ground Truth (Unscaled)
        p_real = []
        for batch in tst_ldr:
            batch = batch.to(DEVICE)
            p_real.append(metrics_nrm._rescale(batch.y).reshape(-1, num_nodes).detach().cpu().numpy())
        p_real = da.array(np.concatenate(p_real))

        # --- MODEL 1: GCN (Your Trained Model) ---   TENGO QUE MODIFICAR PARA QUE ME CALCULE LA METRICA DEL GAT !!!!!
        model = Net(tst_x.shape[-1], tst_y.shape[-1]).to(DEVICE) # <-- Changed ChebNet to Net
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        p_gcn = []
        with torch.no_grad():
            for batch in tst_ldr:
                batch = batch.to(DEVICE)
                out = model(batch)
                p_gcn.append(metrics_nrm._rescale(out).reshape(-1, num_nodes).detach().cpu().numpy())
        p_gcn = da.array(np.concatenate(p_gcn))
        
        msec, sigma = compute_metrics(p_real, p_gcn)
        with open(results_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            # CRITICAL: Save it as -v1, -v2, or -gat instead of just -gcn!
            writer.writerow([f"{run_stamp}-{args.gnn}", msec, sigma]) 
        print(f"   {args.gnn} Evaluated")

        # --- MODEL 2: Naive (Mean Predictor) ---
        naive_model = MeanPredictor(DEVICE)
        p_naive = []
        for batch in tst_ldr:
            batch = batch.to(DEVICE)
            # Predicts based on known sensors in the batch
            out = naive_model.pred(batch.y, batch.x[:, -1].type(torch.bool))
            p_naive.append(metrics_nrm._rescale(out).reshape(-1, num_nodes).detach().cpu().numpy())
        p_naive = da.array(np.concatenate(p_naive))

        msec, sigma = compute_metrics(p_real, p_naive)
        with open(results_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{run_stamp}-naive", msec, sigma])
        print(f"   Naive Evaluated")

        # --- MODEL 3: Interpolation (Graph Regularization) ---
        # Note: Interp works on Standardized data, so we rescale differently
        p_interp = interpolated_regularization(L, tst_x)
        p_interp = p_interp * scale_std + bias_std # Rescale standardized -> physical
        p_interp = da.array(p_interp)

        msec, sigma = compute_metrics(p_real, p_interp)
        with open(results_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{run_stamp}-interp", msec, sigma])
        print(f"    Interp Evaluated")

    print(f"\n All evaluations complete! Results saved to {results_file}")

if __name__ == "__main__":
    run_evaluation()