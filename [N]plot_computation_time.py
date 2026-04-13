import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data from the CSV file
csv_filename = "[BWSN_20_RUNS]training_computation_times.csv"

try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: {csv_filename} not found. Please run your training script first.")
    exit()

# 2. Calculate the average time for each Ratio and GNN combination
# Since you have multiple runs (e.g., Run 1, 2, 3, 4), this averages them out.
avg_times = df.groupby(['Ratio', 'GNN'])['Time_Seconds'].mean().unstack()

# 3. Create the plot
plt.figure(figsize=(8, 5))

# Plot lines for each GNN model found in the CSV
for gnn_model in avg_times.columns:
    plt.plot(avg_times.index, avg_times[gnn_model], marker='o', linewidth=2, label=gnn_model.upper())

# 4. Format the chart
plt.title('Average Computation Time vs. Observation Ratio', fontsize=14)
plt.xlabel('Observation Ratio', fontsize=12)
plt.ylabel('Time (Seconds)', fontsize=12)

# Set the X-axis ticks to exactly match your ratios
plt.xticks(df['Ratio'].unique())

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='GNN Model')
plt.tight_layout()

# 5. Save the graphic to an image file and show it
output_image = 'computation_time_chart.png'
plt.savefig(output_image, dpi=300)
print(f"Success! Chart saved as {output_image}")

# Optional: Display the plot on your screen
plt.show()