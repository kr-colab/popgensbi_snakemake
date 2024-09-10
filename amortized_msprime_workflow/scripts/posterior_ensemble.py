import glob
import os
import pickle
from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior
import torch
import numpy as np

posteriordir = snakemake.params.posteriordir
ts_processor = snakemake.params.ts_processor
sim_rounds = snakemake.params.sim_rounds

posterior_files = glob.glob(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/posterior_rep_*.pkl")
posteriors = []
for pf in posterior_files:
    with open(pf, "rb") as f:
        posteriors.append(pickle.load(f))

device = posteriors[-1]._device
ensemble = EnsemblePosterior(posteriors)

with open(f"{posteriordir}/{ts_processor}/sim_round_{sim_rounds}/ensemble_posterior.pkl", "wb") as f:
    pickle.dump(ensemble, f)

