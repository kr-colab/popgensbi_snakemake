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
posteriordir = snakemake.params.posteriordir
ts_processor = snakemake.params.ts_processor
n_train = snakemake.params.n_train

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
processor = PROCESSOR_LIST[ts_processor](snakemake)
    
bounds = simulator.bounds

with open(f"{posteriordir}/{ts_processor}/n_train_{n_train}/ensemble_posterior.pkl", "rb") as f:
    posterior = pickle.load(f)   
posterior_samples = np.load(f"{posteriordir}/{ts_processor}/n_train_{n_train}/posterior_samples_test.npy").astype(np.float32)
n_boot = posterior_samples.shape[0]
n_sample = posterior_samples.shape[1]
alpha_hpd = np.zeros(n_boot)

for i in range(n_boot):
    x = np.load(os.path.join(datadir, ts_processor, f"test_x_{i}.npy"))
    x = torch.from_numpy(x.reshape(1, *x.shape))
    # true param
    theta = np.load(os.path.join(datadir, f"test_theta_{i}.npy")).astype(np.float32)
    posterior = posterior.set_default_x(x)
    alpha_hpd[i] = np.sum((posterior.log_prob(posterior_samples[i]) > posterior.log_prob(theta)).numpy()) / n_sample

alpha_grid = np.arange(0.05, 1, 0.05)
coverage = np.zeros(len(alpha_grid))
for j, alpha in enumerate(alpha_grid):
    coverage[j] = np.sum(np.array(alpha_hpd >= alpha)) / n_boot

plt.figure(figsize=(10,10))
plt.scatter(1 - alpha_grid, coverage)
plt.plot(1 - alpha_grid, 1 - alpha_grid, color='black', linestyle="--")
plt.xlabel("Expected Coverage")
plt.ylabel("Observed Coverage (Highest Posterior Density)")
plt.savefig(f"{posteriordir}/{ts_processor}/n_train_{n_train}/posterior_coverage_hpd.png")
np.save(f"{posteriordir}/{ts_processor}/n_train_{n_train}/observed_coverage_hpd.npy", coverage)
