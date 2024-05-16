import os
import pickle

import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from ts_simulators import *
from ts_processors import *

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
rounds = snakemake.params.rounds

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
    
bounds = simulator.bounds
theta_star = simulator.true_values

with open(f"{posteriordir}/round_{rounds}/posterior.pkl", "rb") as f:
    posterior = pickle.load(f)   

try:
    x_obs = np.load(os.path.join(datadir, "x_obs.npy"))
    # Reshape x_obs to (1, *x_obs.shape) so that it knows that batch size is 1
    x_obs = torch.from_numpy(x_obs.reshape(1, *x_obs.shape))
    samples = posterior.sample((1_000_000,), x=x_obs.cuda(), show_progress_bars=True).cpu().numpy()

    np.save(f"{posteriordir}/round_{rounds}/default_obs_samples.npy", samples)
    _ = corner.corner(
        samples,
        labels=list(bounds.keys()),
        truths=list(theta_star.values()),
        range=list(bounds.values()),
    )
    plt.savefig(f"{posteriordir}/round_{rounds}/default_obs_corner.png")
except KeyError:
    print(
        "No Default parameter set found -- Not sampling and plotting posterior distribution..."
    )
