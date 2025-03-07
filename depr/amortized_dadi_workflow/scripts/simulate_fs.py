
import torch

import msprime
import stdpopsim
import os
import numpy as np
import pickle
from dadi_simulators import *

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
num_simulations = snakemake.params.num_simulations

if not os.path.isdir(f"{datadir}"):
    os.mkdir(f"{datadir}")

demog_model = snakemake.params.demog_model

simulator = MODEL_LIST[demog_model](snakemake)

theta = simulator.prior.sample((1,))

fs = simulator(theta)
fs = fs.squeeze().cpu().numpy()
np.save(os.path.join(datadir, f"fs_{num_simulations}.npy"), fs)
theta = theta.squeeze().cpu().numpy()
np.save(os.path.join(datadir, f"theta_{num_simulations}.npy"), theta)
