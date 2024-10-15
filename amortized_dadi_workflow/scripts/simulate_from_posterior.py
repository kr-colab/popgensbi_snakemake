import dadi
import numpy as np
from dadi_simulators import *
import os
import torch

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
n_train = snakemake.params.n_train
sample_idx = snakemake.params.sample_idx

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)

samples = np.load(os.path.join(posteriordir, f"n_train_{n_train}/default_obs_samples.npy"))
sample = samples[int(sample_idx), :]
theta = torch.from_numpy(sample)

fs = simulator(theta)
fs = fs.squeeze().cpu().numpy()
if not os.path.isdir(os.path.join(posteriordir, f"n_train_{n_train}/sample_fs/")):
    os.mkdir(os.path.join(posteriordir, f"n_train_{n_train}/sample_fs/"))
np.save(os.path.join(posteriordir, f"n_train_{n_train}/sample_fs/fs_sample_{sample_idx}.npy"), fs)