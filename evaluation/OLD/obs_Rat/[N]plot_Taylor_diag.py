# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import matplotlib.pyplot as plt

# Ensure we can import the sibling file 'taylorDiagram.py'
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

from taylorDiagram import TaylorDiagram

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds', default='anytown', type=str)
parser.add_argument('--extend', default=None, type=float)
parser.add_argument('--smin', default=0, type=float)
parser.add_argument('--smax', default=1.5, type=float)
parser.add_argument('--legend', action='store_true')
parser.add_argument('--fill', action='store_true')
parser.add_argument('--individual', action='store_true', help="Plot individual runs")
parser.add_argument('--nocenter', action='store_true', help="Do NOT plot the average (center) of the runs")
parser.add_argument('--savepdf', action='store_true')
parser.add_argument('--tag', default='def', type=str)
args = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# DB loading
# ----- ----- ----- ----- ----- -----
#csv_path = os.path.join(current_dir, '..', 'experiments', 'Taylor_metrics_processed.csv')
csv_path = os.path.join(current_dir, '..', 'experiments', 'Taylor_metrics_nodes_processed.csv') # FOR ONLY 1 NODE TESTING FOR ALL NODES --- to comment for other runs

if not os.path.exists(csv_path):
    print(f" Error: File not found at {csv_path}")
    print("    Make sure you ran 'python process_Taylor_metrics.py' first!")
    exit(1)

df = pd.read_csv(csv_path)

wds = args.wds
# Filter for Reference (orig) to get the "1.0" mark
ref_data = df.loc[(df['wds'] == wds) & (df['model'] == 'orig')]

if ref_data.empty:
    print(f" Error: No reference data found for WDS '{wds}'.")
    exit(1)

std_ref = ref_data['sigma_pred'].values[0]

# ----- ----- ----- ----- ----- -----
# Plot assembly
# ----- ----- ----- ----- ----- -----
fig = plt.figure(figsize=(10, 8))
dia = TaylorDiagram(1.0, fig=fig, label='Reference', extend=args.extend, srange=(args.smin, args.smax))
dia.samplePoints[0].set_color('r')
dia.samplePoints[0].set_marker('P')
dia.samplePoints[0].set_markersize(12)
cmap = plt.get_cmap('tab10')

available_ratios = sorted(df['obs_rat'].unique())

# ==========================================
# Plot Individual Points (Semi-transparent)
# ==========================================
if args.individual:
    for i, obs_rat in enumerate(available_ratios):
        # Naive
        naive_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == 'naive')]
        if not naive_df.empty:
            dia.add_sample(
                naive_df['sigma_pred'].values / std_ref, 
                naive_df['corr_coeff'].values,
                marker='s', ms=5, ls='', mfc=cmap(i), mec='none', alpha=0.3
                # marker='s', ms=5, ls='', mfc='green', mec='none', alpha=0.3
            )

        # GCN
        gcn_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == 'gcn')]
        if not gcn_df.empty:
            dia.add_sample(
                gcn_df['sigma_pred'].values / std_ref, 
                gcn_df['corr_coeff'].values,
                marker='o', ms=5, ls='', mfc=cmap(i), mec='none', alpha=0.3
                #marker='o', ms=5, ls='', mfc='blue', mec='none', alpha=0.3
            )

        # Interp
        interp_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == 'interp')]
        if not interp_df.empty:
            dia.add_sample(
                interp_df['sigma_pred'].values / std_ref, 
                interp_df['corr_coeff'].values,
                marker='*', ms=6, ls='', mfc=cmap(i), mec='none', alpha=0.3
                #marker='*', ms=6, ls='', mfc='yellow', mec='none', alpha=0.3
            )

# ==========================================
# Plot Averages / Centers (Large, Hollow)
# ==========================================
if not args.nocenter:
    for i, obs_rat in enumerate(available_ratios):
        # Naive Average
        naive_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == 'naive')]
        if not naive_df.empty:
            dia.add_sample(
                (naive_df['sigma_pred'].values / std_ref).mean(), 
                naive_df['corr_coeff'].values.mean(),
                marker='s', ms=10, ls='', mfc='none', mec=cmap(i), mew=2.5,
                label=f'Naive avg (OR={obs_rat})'
            )

        # GCN Average
        gcn_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == 'gcn')]
        if not gcn_df.empty:
            dia.add_sample(
                (gcn_df['sigma_pred'].values / std_ref).mean(), 
                gcn_df['corr_coeff'].values.mean(),
                marker='o', ms=10, ls='', mfc='none', mec=cmap(i), mew=2.5,
                label=f'GNN avg (OR={obs_rat})'
            )

        # Interp Average
        interp_df = df[(df['wds'] == wds) & (df['obs_rat'] == obs_rat) & (df['model'] == 'interp')]
        if not interp_df.empty:
            dia.add_sample(
                (interp_df['sigma_pred'].values / std_ref).mean(), 
                interp_df['corr_coeff'].values.mean(),
                marker='*', ms=12, ls='', mfc='none', mec=cmap(i), mew=2.5,
                label=f'Interp avg (OR={obs_rat})'
            )

# Add Contours (RMSE)
contours = dia.add_contours(levels=5, colors='0.5', linestyles='dashed', alpha=0.5)
plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

dia.add_grid()
dia._ax.axis[:].major_ticks.set_tick_out(True)

if args.legend:
    # Moved the legend so it doesn't overlap the diagram
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))

plt.title(f"Taylor Diagram - {wds.capitalize()}", y=1.05)
plt.tight_layout()

if args.savepdf:
    plt.savefig(f'taylor-{args.wds}.pdf', format='pdf', bbox_inches='tight')
else:
    plt.show()