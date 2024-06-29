import numpy as np
import matplotlib.pyplot as plt
import os
from ts_simulators import *

# track confidence interval of each parameter at each round and plot it againts number of rounds
datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
ts_processor = snakemake.params.ts_processor
sim_rounds = snakemake.params.sim_rounds
n_sims_per_round = snakemake.params.n_sims_per_round

posterior_samples = np.load(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/default_obs_samples.npy")
widths = []
for i, param_samples in enumerate(posterior_samples.T):
    lower_bound = np.quantile(param_samples, 0.025)
    upper_bound = np.quantile(param_samples, 0.975)
    width = upper_bound - lower_bound
    widths.append(width)
widths = np.array(widths)

if int(sim_rounds) > 0:
    confidence_intervals = np.load(f"{posteriordir}/{ts_processor}/sim_round_{int(sim_rounds)-1}/confidence_intervals.npy")
    confidence_intervals = np.vstack((confidence_intervals, widths))
else:
    confidence_intervals = widths
np.save(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/confidence_intervals.npy", confidence_intervals)


simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
bounds = simulator.bounds
labels = list(bounds.keys())
max_min_list = list(list(v) for v in bounds.values())
plt.figure(figsize=(10, 10))
for i, param_samples in enumerate(posterior_samples.T):
    plt.plot(np.arange(int(sim_rounds)+1) * n_sims_per_round, confidence_intervals.T[i] / (max_min_list[i][1] - max_min_list[i][0]), label=labels[i])
plt.legend()
plt.xlabel("Number of simulations")
plt.ylabel("Confidence Interval Width / Parameter Range")
plt.savefig(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/confidence_intervals.png")