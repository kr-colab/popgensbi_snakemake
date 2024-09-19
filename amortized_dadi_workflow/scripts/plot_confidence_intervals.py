import numpy as np
import matplotlib.pyplot as plt
import os
from dadi_simulators import *

# track confidence interval of each parameter at each round and plot it againts number of rounds
datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
n_trains = snakemake.params.n_trains
max_n_train = snakemake.params.max_n_train
for j, n_train in enumerate(n_trains):
    posterior_samples = np.load(os.path.join(posteriordir, f"n_train_{int(float(n_train))}", "default_obs_samples.npy"))
    widths = []
    for i, param_samples in enumerate(posterior_samples.T):
        lower_bound = np.quantile(param_samples, 0.025)
        upper_bound = np.quantile(param_samples, 0.975)
        width = upper_bound - lower_bound
        widths.append(width)
    widths = np.array(widths)
    if j == 0:
        confidence_intervals = widths
    else:
        confidence_intervals = np.vstack((confidence_intervals, widths))
np.save(f"{posteriordir}confidence_intervals.npy", confidence_intervals)


simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
bounds = simulator.bounds
labels = list(bounds.keys())
max_min_list = list(list(v) for v in bounds.values())
plt.figure(figsize=(10, 10))
for i, param_samples in enumerate(posterior_samples.T):
    plt.plot(n_trains, confidence_intervals.T[i] / (max_min_list[i][1] - max_min_list[i][0]), label=labels[i])
plt.legend()
plt.xlabel("# training data")
plt.ylabel("Confidence Interval Width / Parameter Range")
plt.savefig(f"{posteriordir}confidence_intervals.png")