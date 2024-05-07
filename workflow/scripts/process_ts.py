import tskit
import os
from ts_processors import *
import torch
from ts_simulators import *
import numpy as np

outdir = snakemake.params.outdir
num_simulations = snakemake.params.num_simulations
n_snps = snakemake.params.n_snps

with open(f"{outdir}/{num_simulations}.trees", "rb") as ts_file:
    ts = tskit.load(ts_file)


simulator = AraTha_2epoch_simulator()
n_sample = simulator.n_sample
# Todo : make the processor customizable
processor = dinf_extract(n_snps=n_snps)
x = processor(ts)
# x is tensor, so change it to numpy first and save it as .npy
x = x.squeeze().cpu().numpy()
np.save(f"{outdir}/x_{num_simulations}.npy",x)
