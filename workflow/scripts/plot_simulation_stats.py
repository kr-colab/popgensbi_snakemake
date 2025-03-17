import os
import zarr
import tskit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import ts_simulators


    
# Get information from config
simulator_config = snakemake.params.simulator_config
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)
parameters = simulator.parameters 


# open zarr file; extract targets
z = zarr.open(snakemake.input.zarr, mode="a")
param_values = np.array([arr for arr in z.targets])
df = pd.DataFrame(param_values, columns=parameters)

# use multiprocessing to calculate summary statistics
num_workers = snakemake.threads

def process_tree(i):
    tree_path = os.path.join(snakemake.params.tree_dir, f"{i}.trees")
    ts = tskit.load(tree_path)
    # calculate and return summary statistics for the tree
    return [
        ts.segregating_sites(span_normalise=False), 
        ts.diversity(mode="site", span_normalise=True),
        ts.Tajimas_D(),
    ]
# Create a pool of workers and process the jobs in parallel
with Pool(processes=num_workers) as pool:
    # Using tqdm to display progress; pool.imap preserves the order of the results
    results = list(tqdm(pool.imap(process_tree, range(len(df))),
                        total=len(df),
                        desc="Processing trees"))

stats = np.array(results)    
df["segregating_sites"], df["diversity"], df["Tajimas_D"] = stats.T

for key, stat in zip(["segregating_sites", "diversity", "Tajimas_D"], stats.T):
    if key in z:
        del z[key]
    z.create_dataset(key, data=stat, shape=stat.shape, dtype=stat.dtype)

# plot histograms of stats
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
sns.histplot(df["segregating_sites"], ax=axs[0])
sns.histplot(df["diversity"], ax=axs[1])
sns.histplot(df["Tajimas_D"], ax=axs[2])
plt.savefig(snakemake.output.stats_hist)
plt.clf()


# PairGrid of stats against each parameter
g = sns.PairGrid(df, x_vars=parameters, y_vars=["segregating_sites", "diversity", "Tajimas_D"], height=4, aspect=1.2)
g.map(plt.scatter)

plt.savefig(snakemake.output.stats_vs_params_pairplot)
plt.clf()

# Create heatmaps for each pair of parameters and each statistic
n_params = len(parameters)
# Adjust figure size based on number of parameters
fig_width = n_params * 12  # Scale width by number of parameters
fig_height = n_params * 3  # Scale height by number of parameters
fig = plt.figure(figsize=(fig_width, fig_height))

stats = ["segregating_sites", "diversity", "Tajimas_D"]
titles = ["Segregating Sites", "Diversity", "Tajima's D"]

for i, param_x in enumerate(parameters):
    for j, param_y in enumerate(parameters):
        if i < j:  # Only plot the upper triangle
            for stat_idx, (stat, title) in enumerate(zip(stats, titles)):
                row = i
                col = j + stat_idx * n_params
                plt.subplot(n_params, n_params * 3, row * (n_params * 3) + col + 1)
                
                sns.histplot(
                    data=df,
                    x=param_x,
                    y=param_y,
                    cmap="plasma",
                    cbar=True,
                    weights=stat,
                    bins=30
                )
                plt.title(f"{param_x} vs {param_y}")

# Add super-titles for each set of heatmaps with dynamic positioning
for idx, title in enumerate(titles):
    # Calculate position based on the number of parameters
    x_pos = (1 + 2 * idx) / (2 * 3)  # Divides the width into thirds
    plt.figtext(x_pos, 0.1, title, ha='center', va='center', fontsize=26, fontweight='bold')

plt.tight_layout()
plt.savefig(snakemake.output.stats_heatmaps)
plt.clf()