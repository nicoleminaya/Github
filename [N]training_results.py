import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# 1. Automatically find all summary files in the current folder
summary_files = glob.glob("*_Training_Summary.csv")

if not summary_files:
    print("No summary CSV files found in the current directory.")
else:
    print(f"Found {len(summary_files)} files. Generating plots...")

    # Load and combine all the CSV files into one Master DataFrame
    dfs = [pd.read_csv(f) for f in summary_files]
    master_df = pd.concat(dfs, ignore_index=True)
    
    # 2. Setup the visual style
    sns.set_theme(style="whitegrid")
    
    # --- YOUR CUSTOM COLOR PALETTE ---
    # This ensures that no matter which plot is generated, the models
    # are always locked to these exact colors.
    custom_colors = {
        'cheb1': '#1f77b4',     # Blue
        'cheb2': '#ff7f0e',     # Orange
        'cheb3': '#2ca02c',     # Green
        'gat': '#d62728',       # Red
        'gat2': '#9467bd',      # Purple
        'gat_hyp': '#8c564b'    # Brown
    }
    
    # =====================================================================
    # Plot 1: Performance vs Observation Ratio (Line Plot)
    # =====================================================================
    g1 = sns.relplot(
        data=master_df,
        x="Ratio", y="vld_rel_err", hue="GNN_Model", col="WDS",
        palette=custom_colors,  # <--- Applied here
        kind="line", marker="o", markersize=8, errorbar="sd",
        linewidth=2, height=5, aspect=1.2, facet_kws={'sharey': False}
    )
    g1.set_axis_labels("Observation Ratio", "Node Rel. Error")
    g1.fig.suptitle("Model Performance vs Observation Ratio (Lower is Better)", y=1.05, fontsize=16, fontweight='bold')
    
    plt.savefig("Plot_1_Performance_vs_Ratio.png", bbox_inches='tight', dpi=300)
    print(" Saved: Plot_1_Performance_vs_Ratio.png")
    plt.close()

    # =====================================================================
    # Plot 2: Model Stability and Variance (Box Plot)
    # =====================================================================
    g2 = sns.catplot(
        data=master_df,
        x="Ratio", y="vld_rel_err", hue="GNN_Model", col="WDS",
        palette=custom_colors,  # <--- Applied here
        kind="box", height=5, aspect=1.2, sharey=False,
        boxprops={'alpha': 0.8}
    )
    g2.set_axis_labels("Observation Ratio", "Node Rel. Error")
    g2.fig.suptitle("Model Stability Across Runs (Spread = Variance)", y=1.05, fontsize=16, fontweight='bold')
    
    plt.savefig("Plot_2_Stability_Boxplots.png", bbox_inches='tight', dpi=300)
    print(" Saved: Plot_2_Stability_Boxplots.png")
    plt.close()

    # =====================================================================
    # Plot 3: Convergence Speed (Bar Plot)
    # =====================================================================
    g3 = sns.catplot(
        data=master_df,
        x="GNN_Model", y="Stopped_At_Epoch", hue="GNN_Model", col="WDS",
        palette=custom_colors,  # <--- Applied here
        kind="bar", errorbar="sd", capsize=.1, height=5, aspect=1.2, sharey=False, legend=False
    )
    g3.set_axis_labels("GNN Model", "Epochs to Converge (Early Stop)")
    g3.fig.suptitle("Convergence Speed: Average Epochs to Stop (Lower = Faster)", y=1.05, fontsize=16, fontweight='bold')
    
    plt.savefig("Plot_3_Convergence_Speed.png", bbox_inches='tight', dpi=300)
    print(" Saved: Plot_3_Convergence_Speed.png")
    plt.close()
    
    print("\nAll plots generated successfully with your custom color palette!")