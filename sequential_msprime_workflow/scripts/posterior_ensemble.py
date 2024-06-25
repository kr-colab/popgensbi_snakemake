import glob
import os
import pickle
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
import torch
import numpy as np

posteriordir = snakemake.params.posteriordir
rounds = snakemake.params.rounds

posterior_files = glob.glob(f"{posteriordir}/round_{rounds}/posterior_*.pkl")
posteriors = []
for pf in posterior_files:
    with open(pf, "rb") as f:
        posteriors.append(pickle.load(f))
device = posteriors[-1]._device
weights = 1.0 / len(posteriors) * torch.ones(len(posteriors)).to(device)
ensemble = NeuralPosteriorEnsemble(posteriors, weights=weights)

with open(f"{posteriordir}/round_{rounds}/ensemble_posterior.pkl", "wb") as f:
    pickle.dump(ensemble, f)

datadir = snakemake.params.datadir

# Draw set of thetas from the posterior for the next round of simulations
x_obs = np.load(os.path.join(datadir, "x_obs.npy"))
x_obs = x_obs.reshape(1, *x_obs.shape)
proposal = ensemble.set_default_x(torch.from_numpy(x_obs))
thetas = proposal.sample((snakemake.params.n_train_sims,))
np.save(os.path.join(datadir, f"round_{rounds}/", "thetas.npy"), thetas.cpu().numpy())