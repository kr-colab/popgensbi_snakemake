import os
import zarr
import tskit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import glob

# get number of windows from glob
# TODO: maybe change this from glob to look at the zarr directly 
tree_files = glob.glob(os.path.join(snakemake.params.tree_dir, "*.trees"))
n_windows = len(tree_files)


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
    results = list(tqdm(pool.imap(process_tree, range(n_windows)),
                        total=n_windows,
                        desc="Processing trees"))

stats = np.array(results)  
df = pd.DataFrame(stats, columns=["segregating_sites", "diversity", "Tajimas_D"])

# open simulation zarr file; get simulation stats
z = zarr.open(snakemake.params.simulation_zarr, mode="r")
sim_stats = np.array([
    z.segregating_sites[:],
    z.diversity[:],
    z.Tajimas_D[:]
]).T  # Transpose to get shape (N, 3)
sim_df = pd.DataFrame(sim_stats, columns=["segregating_sites", "diversity", "Tajimas_D"])

# plot histograms of tree stats vs simulation stats
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
columns = ["segregating_sites", "diversity", "Tajimas_D"]

for i, col in enumerate(columns):
    # primary axis for tree statistics
    ax1 = axs[i]
    # secondary axis for simulation statistics
    ax2 = ax1.twinx()

    # Plot the tree data on the primary y-axis
    sns.histplot(df[col], 
                 ax=ax1, 
                 color='C0', 
                 label='Trees', 
                 alpha=0.5)
    # Plot the simulation data on the secondary y-axis
    sns.histplot(sim_df[col], 
                 ax=ax2, 
                 color='C1', 
                 label='Simulation', 
                 alpha=0.5)

    # Label the y-axes appropriately
    ax1.set_ylabel('Tree counts', color='C0')
    ax2.set_ylabel('Simulation counts', color='C1')
    ax1.set_xlabel(col)

    # Optionally, add legends to identify the plots.
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(snakemake.output.tree_stats_hist)
plt.clf()

