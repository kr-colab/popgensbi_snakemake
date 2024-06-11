import dadi
import numpy as np
import os
from dadi_simulators import *

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
sim_rounds = snakemake.params.sim_rounds

# compare observed sfs (data) against average sfs from posterior samples

# load data
data = np.load(os.path.join(datadir, "fs_star.npy"))

# load posterior samples
model_fs = np.load(os.path.join(posteriordir, f"sim_round_{sim_rounds}/sample_fs/fs_sample_0.npy"))
for i in range(1, 1000):
    model_fs += np.load(os.path.join(posteriordir, f"sim_round_{sim_rounds}/sample_fs/fs_sample_{i}.npy"))
model_fs /= 1000

import pylab
pylab.figure(figsize=(8,6))
dadi.Plotting.plot_2d_comp_multinom(dadi.Spectrum(model_fs), dadi.Spectrum(data))
pylab.savefig(os.path.join(posteriordir, f"sim_round_{sim_rounds}/2d_comp_multinom.png"))