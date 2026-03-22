# -*- coding: utf-8 -*-
import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import ChebConv
from epynet import Network

from utils.graph_utils import get_nx_graph, get_sensitivity_matrix
from utils.DataReader import DataReader
from utils.SensorInstaller import SensorInstaller
from utils.Metrics import Metrics
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds',
                    default = 'anytown',
                    type    = str,
                    help    = "Water distribution system.")
parser.add_argument('--db',
                    default = 'doe_pumpfed_1',
                    type    = str,
                    help    = "DB.")
parser.add_argument('--obsrat',
                    default = 0.05,
                    type    = float,
                    help    = "Observation ratio.")
parser.add_argument('--adj',
                    default = 'binary',
                    choices = ['binary', 'weighted', 'logarithmic', 'pruned'],
                    type    = str,
                    help    = "Type of adjacency matrix.")
parser.add_argument('--deploy',
                    default = 'random',
                    choices = ['master', 'dist', 'hydrodist', 'hds', 'hdvar', 'random', 'xrandom'],
                    type    = str,
                    help    = "Method of sensor deployment.")
parser.add_argument('--epoch',
                    default = 1,
                    type    = int,
                    help    = "Number of epochs.")
parser.add_argument('--idx',
                    default = None,
                    type    = int,
                    help    = "Dev function.")
parser.add_argument('--batch',
                    default = '40',
                    type    = int,
                    help    = "Batch size.")
parser.add_argument('--lr',
                    default = 0.0003,
                    type    = float,
                    help    = "Learning rate.")
parser.add_argument('--decay',
                    default = 0.000006,
                    type    = float,
                    help    = "Weight decay.")
parser.add_argument('--tag',
                    default = 'def',
                    type    = str,
                    help    = "Custom tag.")
parser.add_argument('--deterministic',
                    action  = "store_true",
                    help    = "Setting random seed for sensor placement.")
parser.add_argument('--gnn',    #TO INCLUDE GATs
                    default = 'cheb1',
                    choices = ['cheb1', 'cheb2', 'gat', 'gatres'],
                    type    = str,
                    help    = "GNN architecture to use.")
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# Paths
# ----- ----- ----- ----- ----- -----
wds_name    = args.wds
pathToRoot  = os.path.dirname(os.path.realpath(__file__))
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db) # Data from generatedata.py output
pathToExps  = os.path.join(pathToRoot, 'experiments')
pathToLogs  = os.path.join(pathToExps, 'logs')
run_id  = 1
logs    = [f for f in glob.glob(os.path.join(pathToLogs, '*.csv'))]         # See if there was previous created logs for wds_deploy_budget_weightMatrix_tag
#run_stamp   = wds_name+'-'+args.deploy+'-'+str(args.obsrat)+'-'+args.adj+'-'+args.tag+'-'
run_stamp   = wds_name+'-'+args.deploy+'-'+str(args.obsrat)+'-'+args.adj+'-'+args.gnn+'-'+args.tag+'-'
while os.path.join(pathToLogs, run_stamp + str(run_id)+'.csv') in logs:
    run_id  += 1
run_stamp   = run_stamp + str(run_id)
pathToLog   = os.path.join(pathToLogs, run_stamp+'.csv')
pathToModel = os.path.join(pathToExps, 'models', run_stamp+'.pt')
pathToMeta  = os.path.join(pathToExps, 'models', run_stamp+'_meta.csv') # Parameter in the training call/corresponding log in expriments/logs
pathToSens  = os.path.join(pathToExps, 'models', run_stamp+'_sensor_nodes.csv') # Used sensor locations
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')

# ----- ----- ----- ----- ----- -----
# Saving hyperparams
# ----- ----- ----- ----- ----- -----
hyperparams = {
        'db': args.db,
        'deploy': args.deploy,
        #'budget': args.budget,
        'obsrat': args.obsrat,
        'adj': args.adj,
        'epoch': args.epoch,
        'batch': args.batch,
        'lr': args.lr,
        }
hyperparams = pd.Series(hyperparams)
hyperparams.to_csv(pathToMeta, header=False)

# ----- ----- ----- ----- ----- -----
# Functions
# ----- ----- ----- ----- ----- -----
def train_one_epoch(): # PyTorch training loop
    model.train()
    total_loss  = 0
    for batch in trn_ldr:
        batch   = batch.to(device)
        optimizer.zero_grad()
        out     = model(batch)
        loss    = F.mse_loss(out, batch.y) # MSE error for total loss
        loss.backward()
        optimizer.step()
        total_loss  += loss.item() * batch.num_graphs
    return total_loss / len(trn_ldr.dataset)

def eval_metrics(dataloader): # Measures how the model is doing
    model.eval()
    n   = len(dataloader.dataset)
    tot_loss        = 0
    tot_rel_err     = 0
    tot_rel_err_obs = 0
    tot_rel_err_hid = 0
    for batch in dataloader:
        batch   = batch.to(device)
        out     = model(batch)
        loss    = F.mse_loss(out, batch.y)
        rel_err = metrics.rel_err(out, batch.y)
        rel_err_obs = metrics.rel_err(
            out,
            batch.y,
            batch.x[:, -1].type(torch.bool)
            )
        rel_err_hid = metrics.rel_err(
            out,
            batch.y,
            ~batch.x[:, -1].type(torch.bool)
            )
        tot_loss        += loss.item() * batch.num_graphs
        tot_rel_err     += rel_err.item() * batch.num_graphs
        tot_rel_err_obs += rel_err_obs.item() * batch.num_graphs
        tot_rel_err_hid += rel_err_hid.item() * batch.num_graphs
    loss        = tot_loss / n
    rel_err     = tot_rel_err / n
    rel_err_obs = tot_rel_err_obs / n
    rel_err_hid = tot_rel_err_hid / n
    return loss, rel_err, rel_err_obs, rel_err_hid

# ----- ----- ----- ----- ----- -----
# Loading trn and vld datasets
# ----- ----- ----- ----- ----- -----
wds = Network(pathToWDS)                                                # EPINET use
G   = get_nx_graph(wds, mode=args.adj)                                  # Converts EPANET network .inp file into a graph

if args.deterministic:
    seeds   = [1, 8, 5266, 739, 88867]
    seed    = seeds[run_id % len(seeds)]
else:
    seed    = None

sensor_budget   = int(len(wds.junctions) * args.obsrat)
#sensor_budget   = args.budget
print('Deploying {} sensors...\n'.format(sensor_budget))

sensor_shop = SensorInstaller(wds, include_pumps_as_master=True)        # Where to put sensors based on --deploy arg
if args.deploy == 'master':
    sensor_shop.set_sensor_nodes(sensor_shop.master_nodes)
elif args.deploy == 'dist':
    sensor_shop.deploy_by_shortest_path(
            sensor_budget   = sensor_budget,
            weight_by       = 'length',
            sensor_nodes    = sensor_shop.master_nodes
            )
elif args.deploy == 'hydrodist':
    sensor_shop.deploy_by_shortest_path(
            sensor_budget   = sensor_budget,
            weight_by       = 'iweight',
            sensor_nodes    = sensor_shop.master_nodes
            )
elif args.deploy == 'hds':
    print('Calculating nodal sensitivity to demand change...\n')
    ptb = np.max(wds.junctions.basedemand) / 100
    S   = get_sensitivity_matrix(wds, ptb)
    sensor_shop.deploy_by_shortest_path_with_sensitivity(
            sensor_budget   = sensor_budget,
            node_weights_arr= np.sum(np.abs(S), axis=0),
            weight_by       = 'iweight',
            sensor_nodes    = sensor_shop.master_nodes
            )
elif args.deploy == 'hdvar':
    print('Calculating nodal head variation...\n')
    reader  = DataReader(
                pathToDB,
                n_junc  = len(wds.junctions),
                node_order  = np.array(list(G.nodes))-1
                )
    heads, _, _ = reader.read_data(
        dataset = 'trn',
        varname = 'junc_heads',
        rescale = None,
        cover   = False
        )
    sensor_shop.deploy_by_shortest_path_with_sensitivity(
            sensor_budget   = sensor_budget,
            node_weights_arr= heads.std(axis=0).T[0],
            weight_by       = 'iweight',
            sensor_nodes    = sensor_shop.master_nodes
            )
    del reader, heads
elif args.deploy == 'random':
    sensor_shop.deploy_by_random(
            sensor_budget   = len(sensor_shop.master_nodes)+sensor_budget,
            seed            = seed
            )
elif args.deploy == 'xrandom':
    sensor_shop.deploy_by_xrandom(
            sensor_budget   = sensor_budget,
            seed            = seed,
            sensor_nodes    = sensor_shop.master_nodes
            )
else:
    print('Sensor deployment technique is unknown.\n')
    raise

#if args.idx:
#    sensor_shop.set_sensor_nodes([args.idx])

if args.idx:                                                                            #add these lines to also consider master nodes
    # Grab the default master nodes (tanks/pumps) and add our specific node to them
    combined_nodes = sensor_shop.master_nodes.copy()
    combined_nodes.add(args.idx)
    sensor_shop.set_sensor_nodes(combined_nodes)

np.savetxt(pathToSens, np.array(list(sensor_shop.sensor_nodes)), fmt='%d')

reader  = DataReader(
            pathToDB,
            n_junc  = len(wds.junctions),
            signal_mask = sensor_shop.signal_mask(),
            node_order  = np.array(list(G.nodes))-1
            )
trn_x, _, _ = reader.read_data(                     # Unknown nodes to 0
    dataset = 'trn',
    varname = 'junc_heads',
    rescale = 'standardize',
    cover   = True
    )
trn_y, bias_y, scale_y  = reader.read_data(         # Real data?
    dataset = 'trn',
    varname = 'junc_heads',
    rescale = 'normalize',
    cover   = False
    )
vld_x, _, _ = reader.read_data(
    dataset = 'vld',
    varname = 'junc_heads',
    rescale = 'standardize',
    cover   = True
    )
vld_y, _, _ = reader.read_data(
    dataset = 'vld',
    varname = 'junc_heads',
    rescale = 'normalize',
    cover   = False
    )

    #SELECT Corresponding gnn MODEL (CHEBNET O GAT)
if args.gnn == 'gatres':
    # Since GATRes is generic and doesn't rely on fixed polynomial sizes, 
    # the same model file for all WDS topologies!
    from model.gatres import GATResNet as Net
else:
    if args.wds == 'anytown':
        if args.gnn == 'gat':
            from model.anytown_gat import GATNet as Net
        elif args.gnn == 'cheb1':
            from model.anytown import ChebNet as Net
        else:
            from model.anytown_v2 import ChebNet as Net
    elif args.wds == 'ctown':
        if args.gnn == 'gat':
            from model.ctown_gat import GATNet as Net
        else:
            from model.ctown import ChebNet as Net
    elif args.wds == 'richmond':
        
        from model.richmond import ChebNet as Net
    else:
        print('Water distribution system is unknown.\n')
        raise


model = Net(np.shape(trn_x)[-1], np.shape(trn_y)[-1]).to(device)

if args.gnn == 'gatres':
    # GATRes uses a dynamic number of blocks, so we apply the optimizer to all parameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.decay,
        eps=1e-7
    )
else:
    # Original logic for ChebNet and standard GAT
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=args.decay),
        dict(params=model.conv2.parameters(), weight_decay=args.decay),
        dict(params=model.conv3.parameters(), weight_decay=args.decay),
        dict(params=model.conv4.parameters(), weight_decay=0)
        ],
        lr  = args.lr,
        eps = 1e-7
    )

# ----- ----- ----- ----- ----- -----
# Training
# ----- ----- ----- ----- ----- -----
trn_ldr = build_dataloader(G, trn_x, trn_y, args.batch, shuffle=True)
vld_ldr = build_dataloader(G, vld_x, vld_y, args.batch, shuffle=False)
metrics = Metrics(bias_y, scale_y, device)
estop   = EarlyStopping(min_delta=.00001, patience=30)
results = pd.DataFrame(columns=[
    'trn_loss', 'vld_loss', 'vld_rel_err', 'vld_rel_err_o', 'vld_rel_err_h'
    ])
header  = ''.join(['{:^15}'.format(colname) for colname in results.columns])
header  = '{:^5}'.format('epoch') + header
best_vld_loss   = np.inf
for epoch in range(0, args.epoch):
    trn_loss    = train_one_epoch()
    vld_loss, vld_rel_err, vld_rel_err_obs, vld_rel_err_hid = eval_metrics(vld_ldr)
    new_results = pd.Series({
        'trn_loss'      : trn_loss,
        'vld_loss'      : vld_loss,
        'vld_rel_err'   : vld_rel_err,
        'vld_rel_err_o' : vld_rel_err_obs,
        'vld_rel_err_h' : vld_rel_err_hid
        })
    results = pd.concat([results, pd.DataFrame([new_results])], ignore_index=True) #results = results.append(new_results, ignore_index=True)
    if epoch % 20 == 0:
        print(header)
    values  = ''.join(['{:^15.6f}'.format(value) for value in new_results.values])
    print('{:^5}'.format(epoch) + values)
    if vld_loss < best_vld_loss:
        best_vld_loss   = vld_loss
        torch.save(model.state_dict(), pathToModel)
    if estop.step(torch.tensor(vld_loss)):
        print('Early stopping...')
        break
results.to_csv(pathToLog)

# ----- ----- ----- ----- ----- -----
# Testing
# ----- ----- ----- ----- ----- -----
if best_vld_loss is not np.inf:
    print('Testing...\n')
    del trn_ldr, vld_ldr, trn_x, trn_y, vld_x, vld_y
    tst_x, _, _ = reader.read_data(
        dataset = 'tst',
        varname = 'junc_heads',
        rescale = 'standardize',
        cover   = True
        )
    tst_y, _, _ = reader.read_data(
        dataset = 'tst',
        varname = 'junc_heads',
        rescale = 'normalize',
        cover   = False
        )
    tst_ldr = build_dataloader(G, tst_x, tst_y, args.batch, shuffle=False)
    model.load_state_dict(torch.load(pathToModel))
    model.eval()
    tst_loss, tst_rel_err, tst_rel_err_obs, tst_rel_err_hid = eval_metrics(tst_ldr)
    results = pd.Series({
        'tst_loss'      : tst_loss,                 # 
        'tst_rel_err'   : tst_rel_err,              # Global error
        'tst_rel_err_o' : tst_rel_err_obs,          # Error in observed nodes, should be ~ to 0
        'tst_rel_err_h' : tst_rel_err_hid           # Error in the non-observed nodes, HIGH important
        })
    results.to_csv(pathToLog[:-4]+'_tst.csv')
