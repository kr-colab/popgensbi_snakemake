# load all ld stat from the same simulation and save the average

import tskit
import os
import torch
from ts_simulators import *
from ts_processors import *
import numpy as np
import moments
import pickle

datadir = snakemake.params.datadir
num_simulations = snakemake.params.num_simulations
ts_processor = snakemake.params.ts_processor
n_segs = snakemake.params.n_segs # total number of segments

ld_stats = {}
for i in range(n_segs):
    with open(f"{datadir}/{ts_processor}/ld_stat_{num_simulations}_{i}.pkl", "rb") as f:
        ld_stat = pickle.load(f)
        ld_stats[i] = ld_stat

stats = ld_stats[0]["stats"]
means = moments.LD.Parsing.means_from_region_data(ld_stats, stats)
output = []
# will stack means of D2, Dz and pi2 and append replicate of H
for i in range(len(means)-1):
    output.append(means[i])

rep = len(output[0])
for i in range(len(means[-1])):
    output.append(np.repeat(means[-1][i], rep))
output = np.stack(output)

np.save(f"{datadir}/{ts_processor}/x_{num_simulations}.npy", output)
