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
z = zarr.open(snakemake.input.zarr)
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
        ts.diversity(mode="branch", span_normalise=True),
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

# Create heatmaps for each pair of parameters
plt.figure(figsize=(12, 10))
for i, param_x in enumerate(parameters):
    for j, param_y in enumerate(parameters):
        if i < j:  # Only plot the upper triangle
            plt.subplot(len(parameters), len(parameters), i * len(parameters) + j + 1)
            sns.histplot(
                data=df,
                x=param_x,
                y=param_y,
                cmap="YlGnBu",
                cbar=True,
                weights="segregating_sites",
                bins=30
            )
            plt.title(f"{param_x} vs {param_y}")

plt.tight_layout()
plt.savefig(snakemake.output.stats_heatmaps)
plt.clf()