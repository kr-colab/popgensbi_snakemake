import glob
import os
import pickle
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
import torch
import numpy as np

posteriordir = snakemake.params.posteriordir
n_train = snakemake.params.n_train

posterior_files = glob.glob(f"{posteriordir}/n_train_{n_train}/posterior_rep_*.pkl")
posteriors = []
for pf in posterior_files:
    with open(pf, "rb") as f:
        posteriors.append(pickle.load(f))
device = posteriors[-1]._device
weights = 1.0 / len(posteriors) * torch.ones(len(posteriors)).to(device)
ensemble = NeuralPosteriorEnsemble(posteriors, weights=weights)

with open(f"{posteriordir}/n_train_{n_train}/ensemble_posterior.pkl", "wb") as f:
    pickle.dump(ensemble, f)

