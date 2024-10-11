import torch

import msprime
import stdpopsim
import os
import numpy as np
import pickle
from ts_simulators import *

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
num_simulations = snakemake.params.num_simulations

if not os.path.isdir(f"{datadir}"):
    os.mkdir(f"{datadir}")

demog_model = snakemake.params.demog_model

simulator = MODEL_LIST[demog_model](snakemake)

theta = simulator.prior.sample((1,))

ts = simulator(theta)
with open(os.path.join(datadir, f"test_{num_simulations}.trees"), "wb") as ts_file:
    ts.dump(ts_file)

theta = theta.squeeze().cpu().numpy()
np.save(os.path.join(datadir, f"test_theta_{num_simulations}.npy"), theta)
