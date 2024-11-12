# Estimate coverage probability, which is the probability that the true value is within a given expected coverage interval.
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from ts_simulators import *
from ts_processors import *
from sbi import analysis

datadir = snakemake.params.datadir
datasubdir = snakemake.params.datasubdir
posteriordir = snakemake.params.posteriordir
posteriorsubdir = snakemake.params.posteriorsubdir
ts_processor = snakemake.params.ts_processor
n_train = snakemake.params.n_train

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
processor = PROCESSOR_LIST[ts_processor](snakemake)
    
bounds = simulator.bounds

with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "ensemble_posterior.pkl"), "rb") as f:
    posterior = pickle.load(f)   

# Sample posterior with test data as default
n_boot = int(snakemake.params.n_boot) # number of test data
n_sample = 1000
posterior_samples = np.zeros((n_boot, n_sample, len(bounds)))
truths = np.zeros((n_boot, len(bounds)))
for i in range(n_boot):
    x = np.load(os.path.join(datadir, datasubdir, f"test_x_{i}.npy"))
    x = torch.from_numpy(x.reshape(1, *x.shape))
    theta = np.load(os.path.join(datadir, f"test_theta_{i}.npy"))
    truths[i] = theta
    posterior = posterior.set_default_x(x)
    posterior_samples[i] = posterior.sample((n_sample,)).numpy()
np.save(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "posterior_samples_test.npy"), posterior_samples)

cmap = plt.get_cmap("plasma")
fig, axs = plt.subplots(len(bounds), 1, sharey=True, sharex=True)
alpha_grid = np.arange(0.025, 0.5, 0.025)
coverage = np.zeros((len(bounds), len(alpha_grid)))

for i in range(len(bounds)):
    ranks = truths[:, i].flatten().argsort().argsort()
    for j, alpha in enumerate(alpha_grid):
        lower = np.quantile(posterior_samples[:, :, i], alpha, axis=1)
        upper = np.quantile(posterior_samples[:, :, i], 1 - alpha, axis=1)
        prior_width = (1 - 2 * alpha) * (list(bounds.values())[i][1] - list(bounds.values())[i][0])
        inter_width = upper - lower
        coverage[i, j] = np.sum(np.logical_and(truths[:, i] >= lower, truths[:, i] < upper)) / len(truths)
        axs[i].scatter(ranks, lower - truths[:, i], color=cmap(j / alpha_grid.size), s=1)
        axs[i].scatter(ranks, upper - truths[:, i], color=cmap(j / alpha_grid.size), s=1)
    axs[i].axhline(y=0.0, color='black', linestyle='--', linewidth=1)
    axs[i].text(0.98, 0.9, list(bounds.keys())[i], transform=axs[i].transAxes, horizontalalignment='right')
fig.supylabel("(Quantile - true param value)")
fig.supxlabel("Rank order true param value")
plt.savefig(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "ci_rank_param.png"))
plt.clf()

plt.figure(figsize=(10, 10))
for i in range(len(bounds)):
    plt.scatter(1 - alpha_grid * 2, coverage[i], label=list(bounds.keys())[i])
plt.plot(1 - alpha_grid * 2, 1 - alpha_grid * 2, color="black", linestyle="--")
plt.xlabel("Expected Coverage")
plt.ylabel("Obseved Coverage")
plt.legend()
plt.savefig(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "posterior_coverage.png"))

np.save(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "observed_coverage.npy"), coverage)
