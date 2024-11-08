import tskit
import os
from ts_processors import *
import torch
from ts_simulators import *
import numpy as np

datadir = snakemake.params.datadir
datasubdir = snakemake.params.datasubdir
# num_simulations = snakemake.params.num_simulations
ts_processor = snakemake.params.ts_processor
tsname = snakemake.params.tsname
xname = snakemake.params.xname

if not os.path.isdir(f"{datadir}/{datasubdir}"):
    os.mkdir(f"{datadir}/{datasubdir}")

with open(os.path.join(datadir, tsname), "rb") as ts_file:
    ts = tskit.load(ts_file)

processor = PROCESSOR_LIST[ts_processor](snakemake)

x = processor(ts)
# x is tensor, so change it to numpy first and save it as .npy
x = x.squeeze().cpu().numpy()
np.save(os.path.join(datadir, datasubdir, xname), x)
