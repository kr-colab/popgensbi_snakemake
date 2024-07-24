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
sim_rounds = snakemake.params.sim_rounds

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
    
bounds = simulator.bounds
theta_star = simulator.true_values

with open(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/ensemble_posterior.pkl", "rb") as f:
    posterior = pickle.load(f)   

device = posteriors[-1]._device
try:
    x_obs = np.load(os.path.join(datadir, ts_processor, "x_obs.npy"))
    # Reshape x_obs to (1, *x_obs.shape) so that it knows that batch size is 1
    x_obs = torch.from_numpy(x_obs.reshape(1, *x_obs.shape))
    samples = posterior.sample((100_000,), x=x_obs.to(device), show_progress_bars=True).cpu().numpy()
    np.save(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/default_obs_samples.npy", samples)
    _ = analysis.pairplot(samples, 
            labels=list(bounds.keys()), 
            limits=list(list(v) for v in bounds.values()),
            ticks=list(list(v) for v in bounds.values()), 
            figsize=(10, 10), 
            points=np.array(list(theta_star.values())))
    plt.savefig(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/default_obs_corner.png")
except KeyError:
    print(
        "No Default parameter set found -- Not sampling and plotting posterior distribution..."
    )
