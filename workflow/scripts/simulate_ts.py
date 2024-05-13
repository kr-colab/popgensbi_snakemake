## Reproducing AraTha/Africa2Epoch_1h18 of popgensbi

import torch

import msprime
import stdpopsim
import os
import numpy as np
import pickle
from ts_simulators import *
from ts_processors import *

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
num_simulations = snakemake.params.num_simulations
rounds = snakemake.params.rounds

if not os.path.isdir(f"{datadir}/round_{rounds}/"):
    os.mkdir(f"{datadir}/round_{rounds}/")

demog_model = snakemake.params.demog_model

if demog_model == "AraTha_2epoch":
    simulator = AraTha_2epoch_simulator(snakemake)
if snakemake.params.ts_processor == "dinf":
    processor = dinf_extract(snakemake)


if int(rounds) == 0:
    proposal = simulator.prior
else:
    # Use previous round's posterior as proposal
    previous_round = int(rounds) - 1
    with open(os.path.join(posteriordir, f"round_{previous_round}/", "posterior.pkl"), "rb") as f:
        proposal = pickle.load(f)
    x_obs = np.load(os.path.join(datadir, "x_obs.npy"))
    # Reshape x_obs to (1, *x_obs.shape) so that it knows that batch size is 1
    x_obs = x_obs.reshape(1, *x_obs.shape)
    proposal = proposal.set_default_x(torch.from_numpy(x_obs))
# Sample one theta from the proposal (parallelization handled by snakemake)
theta = proposal.sample((1,))
ts = simulator(theta)
with open(os.path.join(datadir, f"round_{rounds}/", f"{num_simulations}.trees"), "wb") as ts_file:
    ts.dump(ts_file)
theta = theta.squeeze().cpu().numpy()
np.save(os.path.join(datadir, f"round_{rounds}/", f"theta_{num_simulations}.npy"), theta)
