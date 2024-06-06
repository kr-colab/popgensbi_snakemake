import torch
import dadi
import stdpopsim
import os
import numpy as np
from dadi_simulators import *

datadir = snakemake.params.datadir
if not os.path.isdir(f"{datadir}/"):
    os.mkdir(f"{datadir}/")

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)

theta_star = simulator.true_values
theta_star = torch.tensor(list(theta_star))
fs_star = simulator(theta_star)
np.save(os.path.join(datadir, "fs_star.npy"), fs_star)
