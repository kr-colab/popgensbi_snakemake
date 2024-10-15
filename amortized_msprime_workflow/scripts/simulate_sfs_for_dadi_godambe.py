# simulate with test param values and save sfs for bootstrap data needed for dadi godambe
import torch

import msprime
import stdpopsim
import os
import numpy as np
import pickle
from ts_simulators import *

datadir = snakemake.params.datadir
num_simulations = snakemake.params.num_simulations
num_rep_dadi = snakemake.params.num_rep_dadi

if not os.path.isdir(f"{datadir}"):
    os.mkdir(f"{datadir}")

demog_model = snakemake.params.demog_model

simulator = MODEL_LIST[demog_model](snakemake)

theta = np.load(os.path.join(datadir, f"test_theta_{num_simulations}.npy"))

ts = simulator(list(theta))
sfs = ts.allele_frequency_spectrum(
            sample_sets = [ts.samples(population=i) for i in range(ts.num_populations) if len(ts.samples(population=i))>0], 
            windows = None, 
            mode = 'site', 
            span_normalise = False, 
            polarised = True)
sfs = sfs / sum(sfs.flatten())

np.save(os.path.join(datadir, f"test_{num_simulations}_sfs_rep_{num_rep_dadi}.npy"), sfs)