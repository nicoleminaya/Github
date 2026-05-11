import pandas as pd
import matplotlib.pyplot as plt
import argparse

# ==========================================
# 1. Command-Line Argument Setup
# ==========================================
parser = argparse.ArgumentParser(description='Plot GNN training computation times.')
parser.add_argument('--wds', type=str, required=True, 
                    help='The WDS type (e.g., BWSN, Net3, Richmond, Anytown).')

args = parser.parse_args()
wds_type = args.wds.upper()

# ==========================================
# 2. Configuration & Data Loading
# ==========================================
# Updated colors to match the actual names found in your CSV!
model_colors = {
    'cheb1': '#1f77b4',     # Blue
    'cheb2': '#ff7f0e',     # Orange
    'cheb3': '#2ca02c',     # Green
    'gat': '#d62728',       # Red
    'gat2': '#9467bd',      # Purple
    #'gat_hyp': '#8c564b'    # Brown
}

#csv_filename = "training_computation_times.csv"
#Hanoi 
csv_filename = "[hanoi_noseed_cheb2_v1]training_computation_times_eliminados.csv"
#bwsn csv_filename = "[BWSN_20_RUNS]training_computation_times_eliminados.csv"
#anytown no seed csv_filename = "[ANYTOWN_v2][no_seed]training_computation_times_eliminados.csv"
#anytown seed csv_filename = "[Anytown_SEED]training_computation_times_eliminados.csv"

try:
    # FIX: Explicitly define the 6 column names and skip the broken header row (skiprows=1)
    column_names = ['Experiment_Run', 'Network', 'GNN', 'Ratio', 'tag', 'Time_Seconds']
    df = pd.read_csv(csv_filename, names=column_names, skiprows=1)
except FileNotFoundError:
    print(f"Error: '{csv_filename}' not found. Please ensure the file exists.")
    exit()

# ==========================================
# 3. Filter Data for the Selected WDS ONLY
# ==========================================
# Filter the dataframe safely (lowercase both sides)
df_filtered = df[df['Network'].astype(str).str.lower() == wds_type.lower()]

if df_filtered.empty:
    print(f"Error: No data found for network '{wds_type}'. Exiting.")
    exit()

# ==========================================
# 4. Process Data & Plot
# ==========================================
# Calculate the average time ONLY for the filtered dataset
avg_times = df_filtered.groupby(['Ratio', 'GNN'])['Time_Seconds'].mean().unstack()

plt.figure(figsize=(8, 5))

print(f"--- Plotting Data for {wds_type} ---")
for gnn_model in avg_times.columns:
    clean_name = str(gnn_model).strip().lower()
    line_color = model_colors.get(clean_name, 'black')
    
    plt.plot(avg_times.index, avg_times[gnn_model], 
             marker='o', 
             linewidth=2, 
             label=str(gnn_model).strip().upper(), 
             color=line_color)

# ==========================================
# 5. Format and Save Chart
# ==========================================
plt.title(f'Average Computation Time vs. Observation Ratio ({wds_type})', fontsize=14)
plt.xlabel('Observation Ratio', fontsize=12)
plt.ylabel('Time (Seconds)', fontsize=12)

plt.xticks(df_filtered['Ratio'].unique())
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='GNN Model')
plt.tight_layout()

# Save the graphic
output_image = f'{wds_type}_computation_time_chart.png'
plt.savefig(output_image, dpi=300)
print(f"Success! Chart saved as {output_image}")

plt.show()