import dadi
import numpy as np
import os
from dadi_simulators import *

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
sim_rounds = snakemake.params.sim_rounds

# compare observed sfs (data) against sfs simulated with MAP parameters

# load data
data = np.load(os.path.join(datadir, "fs_star.npy"))

# get model fs
model_fs = np.load(os.path.join(posteriordir, f"sim_round_{sim_rounds}/model_fs.npy"))

import pylab
pylab.figure(figsize=(8,6))
if len(data.shape) == 1:
    dadi.Plotting.plot_1d_comp_multinom(dadi.Spectrum(model_fs), dadi.Spectrum(data))
elif len(data.shape) == 2:
    dadi.Plotting.plot_2d_comp_multinom(dadi.Spectrum(model_fs), dadi.Spectrum(data), vmin=1e-5)
pylab.savefig(os.path.join(posteriordir, f"sim_round_{sim_rounds}/2d_comp_multinom.png"))

