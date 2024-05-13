import torch

import msprime
import stdpopsim
import os
import numpy as np
from ts_simulators import *
from ts_processors import *

datadir = snakemake.params.datadir
if not os.path.isdir(f"{datadir}/"):
    os.mkdir(f"{datadir}/")

demog_model = snakemake.params.demog_model
if demog_model == "AraTha_2epoch":
    simulator = AraTha_2epoch_simulator(snakemake)
if snakemake.params.ts_processor == "dinf":
    processor = dinf_extract(snakemake)

theta_star = simulator.true_values
ts_star = simulator(list(theta_star.values()))
with open(os.path.join(datadir, "ts_star.trees"), "wb") as ts_file:
    ts_star.dump(ts_file)
x_obs = processor(ts_star).squeeze().cpu().numpy()
np.save(os.path.join(datadir, "x_obs.npy"), x_obs)
