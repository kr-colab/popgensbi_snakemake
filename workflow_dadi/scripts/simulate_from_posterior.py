import dadi
import numpy as np
from dadi_simulators import *
import os
import torch

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
sim_rounds = snakemake.params.sim_rounds
sample_idx = snakemake.params.sample_idx

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)

samples = np.load(os.path.join(posteriordir, f"sim_round_{sim_rounds}/default_obs_samples.npy"))
sample = samples[int(sample_idx), :]
theta = torch.from_numpy(sample)

fs = simulator(theta)
fs = fs.squeeze().cpu().numpy()
if not os.path.isdir(os.path.join(posteriordir, f"sim_round_{sim_rounds}/sample_fs/")):
    os.mkdir(os.path.join(posteriordir, f"sim_round_{sim_rounds}/sample_fs/"))
np.save(os.path.join(posteriordir, f"sim_round_{sim_rounds}/sample_fs/fs_sample_{sample_idx}.npy"), fs)