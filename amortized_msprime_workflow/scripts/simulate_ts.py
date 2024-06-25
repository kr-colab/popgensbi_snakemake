## Reproducing AraTha/Africa2Epoch_1h18 of popgensbi

import torch

import msprime
import stdpopsim
import os
import numpy as np
import pickle
from ts_simulators import *
from ts_processors import *

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
num_simulations = snakemake.params.num_simulations
sim_rounds = snakemake.params.sim_rounds

if not os.path.isdir(f"{datadir}/sim_round_{sim_rounds}/"):
    os.mkdir(f"{datadir}/sim_round_{sim_rounds}/")

demog_model = snakemake.params.demog_model

simulator = MODEL_LIST[demog_model](snakemake)

theta = simulator.prior.sample((1,))

ts = simulator(theta)
with open(os.path.join(datadir, f"sim_round_{sim_rounds}/", f"{num_simulations}.trees"), "wb") as ts_file:
    ts.dump(ts_file)
theta = theta.squeeze().cpu().numpy()
np.save(os.path.join(datadir, f"sim_round_{sim_rounds}/", f"theta_{num_simulations}.npy"), theta)
