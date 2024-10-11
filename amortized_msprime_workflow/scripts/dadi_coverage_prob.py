import os

import matplotlib.pyplot as plt
import numpy as np
from ts_simulators import *
from scipy import stats 

datadir = snakemake.params.datadir
n_rep = int(snakemake.params.n_rep)
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
bounds = simulator.bounds

num_params = 2
param_names = ["nu", "T"]
alpha_grid = np.arange(0.025, 0.5, 0.025)
z_scores = stats.norm.ppf(1 - alpha_grid)
coverages = np.zeros((num_params, len(alpha_grid)))
# todo - load up MLE, uncerts and theta first, make a list of length n_rep, and loop around alpha_grid just like coverage_prob and do the logical_and & averaging at once.

MLEs = np.zeros((n_rep, num_params))
uncerts = np.zeros((n_rep, num_params))
truths = np.zeros((n_rep, num_params))

for i in range(n_rep):
    MLEs[i] = np.load(os.path.join(datadir, f"test_MLE_{i}.npy"))
    uncerts[i] = np.load(os.path.join(datadir, f"test_uncerts_{i}.npy"))
    truths[i] = np.load(os.path.join(datadir, f"test_theta_{i}.npy"))

cmap = plt.get_cmap("plasma")
fig, axs = plt.subplots(len(bounds), 1, sharey=True, sharex=True)
coverages = np.zeros((num_params, len(z_scores)))

for i in range(num_params):
    ranks = truths[:, i].flatten().argsort().argsort()
    for j, z_score in enumerate(z_scores):
        lower = MLEs[:, i] - uncerts[:, i] * z_score
        upper = MLEs[:, i] + uncerts[:, i] * z_score
        coverage[i, j] = np.sum(np.logical_and(truths[:, i] >= lower, truths[:, i] < upper)) / len(truths)
        axs[i].scatter(ranks, lower - truths[:, i], color=cmap(j / z_scores.size), s=1)
        axs[i].scatter(ranks, upper - truths[:, i], color=cmap(j / z_scores.size), s=1)
    axs[i].axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    axs[i].text(0.98, 0.9, list(bounds.keys())[i], transform=axs[i].transAxes, horizontalalignment='right')
fig.supylabel("(Quantile - true param value)")
fig.supxlabel("Rank order true param value")
plt.savefig(os.path.join(datadir, "ci_rank_param_dadi_godambe.png"))
plt.clf()
       

plt.figure(figsize=(10, 10))
for i in range(len(bounds)):
    plt.scatter(1 - alpha_grid * 2, coverages[i], label=list(bounds.keys())[i])
plt.plot(1 - alpha_grid * 2, 1 - alpha_grid * 2, color="black", linestyle="--")
plt.xlabel("Expected Coverage")
plt.ylabel("Obseved Coverage")
plt.legend()
plt.savefig(os.path.join(datadir, "dadi_godambe_coverage.png"))
np.save(os.path.join(datadir, "dadi_godmabe_coverage.npy"), coverages)
