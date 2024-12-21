# run simulation-based calibration
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from ts_simulators import *
from ts_processors import *
from sbi.analysis import check_sbc, run_sbc, get_nltp, sbc_rank_plot


datadir = snakemake.params.datadir
datasubdir = snakemake.params.datasubdir
posteriordir = snakemake.params.posteriordir
posteriorsubdir = snakemake.params.posteriorsubdir
ts_processor = snakemake.params.ts_processor
n_train = snakemake.params.n_train

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
processor = PROCESSOR_LIST[ts_processor](snakemake)

# Set up device
device = torch.device(snakemake.params.device if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "ensemble_posterior.pkl"), "rb") as f:
    posterior = pickle.load(f)   

num_sbc_runs = int(snakemake.params.n_boot)  

xs = []
thetas = []
for i in range(num_sbc_runs):
    x = torch.from_numpy(np.load(os.path.join(datadir, datasubdir, f"test_x_{i}.npy")))
    xs.append(x)
    theta = torch.from_numpy(np.load(os.path.join(datadir, f"test_theta_{i}.npy")))
    thetas.append(theta)

xs = torch.stack(xs)
print(xs.shape)
thetas = torch.stack(thetas)
print(thetas.shape)

# move tensors to device
xs = xs.to(device)
thetas = thetas.to(device)
# print to make sure they are on the device
print(f"xs device: {xs.device}")
print(f"thetas device: {thetas.device}")

num_posterior_samples = 1_000
ranks, dap_samples = run_sbc(
    thetas, xs, posterior, num_posterior_samples=num_posterior_samples
)
check_stats = check_sbc(
    ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples
)


with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "sbc_stats.pkl"), 'wb') as file:
    pickle.dump(check_stats, file)

from sbi.analysis import sbc_rank_plot

f, ax = sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=num_posterior_samples,
    plot_type="hist",
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)

plt.savefig(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "sbc_rank_hist.png"))

f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf")
plt.savefig(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "sbc_rank_cdf.png"))