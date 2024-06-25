import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from dadi_simulators import *
from sbi import analysis

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
sim_rounds = snakemake.params.sim_rounds

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)

bounds = simulator.bounds
theta_star = simulator.true_values

with open(f"{posteriordir}/sim_round_{sim_rounds}/ensemble_posterior.pkl", "rb") as f:
    posterior = pickle.load(f)

fs_star = np.load(os.path.join(datadir, "fs_star.npy"))
fs_star = torch.from_numpy(fs_star.reshape(1, *fs_star.shape))
samples = posterior.sample((100_000,), x=fs_star.cuda(), show_progress_bars=True).cpu().numpy()
np.save(f"{posteriordir}/sim_round_{sim_rounds}/default_obs_samples.npy", samples)

# find MAP parameters
posterior = posterior.set_default_x(fs_star)
map_thetas = posterior.map(show_progress_bars=True)

# simulate with MAP parameters
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
model_fs = simulator(map_thetas)

map_thetas = map_thetas.cpu().numpy()
model_fs = model_fs.squeeze().cpu().numpy()

# save model fs and map_thetas
np.save(os.path.join(posteriordir, f"sim_round_{sim_rounds}/model_fs.npy"), model_fs)
np.save(os.path.join(posteriordir, f"sim_round_{sim_rounds}/map_thetas.npy"), map_thetas)

# plot posterior with MAP parameter values (red) and true parameter values (yellow)
_ = analysis.pairplot(samples,
        labels=list(bounds.keys()),
        limits=list(list(v) for v in bounds.values()),
        ticks=list(list(v) for v in bounds.values()),
        figsize=(10, 10),
        points=[list(theta_star.values()), list(map_thetas)],
        points_colors=["y", "r"])
plt.savefig(f"{posteriordir}/sim_round_{sim_rounds}/default_obs_corner.png")