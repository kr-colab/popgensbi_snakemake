## Reproducing AraTha/Africa2Epoch_1h18 of popgensbi

import torch

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import msprime
import stdpopsim
import os
import numpy as np
import pickle
from ts_simulators import *

outdir = snakemake.params.outdir
num_simulations = snakemake.params.num_simulations
if not os.path.isdir(outdir):
    os.mkdir(outdir)

simulator = AraTha_2epoch_simulator()
prior = simulator.prior        
# Simulate only once (parallelized by snakemake)
# sample one theta
theta = prior.sample((1,))
ts = simulator(theta)
with open(f"{outdir}/{num_simulations}.trees", "wb") as ts_file:
    ts.dump(ts_file)
theta = theta.squeeze().cpu().numpy()
np.save(f"{outdir}/theta_{num_simulations}.npy", theta)
