import torch

import os
import numpy as np
from ts_processors import *

datadir = snakemake.params.datadir
# use ts processor name to name the subdirectory
ts_processor = snakemake.params.ts_processor 
processor = PROCESSOR_LIST[snakemake.params.ts_processor](snakemake)

ts_star = tskit.load(os.path.join(datadir, "ts_star.trees"))
x_obs = processor(ts_star).squeeze().cpu().numpy()
np.save(os.path.join(datadir, ts_processor, "x_obs.npy"), x_obs)