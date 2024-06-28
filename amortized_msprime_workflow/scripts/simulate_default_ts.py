import torch

import os
import numpy as np
from ts_simulators import *

datadir = snakemake.params.datadir
if not os.path.isdir(f"{datadir}/"):
    os.mkdir(f"{datadir}/")

simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)

theta_star = simulator.true_values
ts_star = simulator(list(theta_star.values()))
with open(os.path.join(datadir, "ts_star.trees"), "wb") as ts_file:
    ts_star.dump(ts_file)
