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
    theta = proposal.sample((1,))
else:
    # In rounds > 0, we have sampled thetas from the previous round posterior
    previous_round = int(rounds) - 1
    thetas = np.load(os.path.join(datadir, f"round_{previous_round}/", "thetas.npy"))
    # use one of the thetas (num_simulations-th) as the theta for the current simulation
    theta = torch.from_numpy(thetas[int(num_simulations), :])

ts = simulator(theta)
with open(os.path.join(datadir, f"round_{rounds}/", f"{num_simulations}.trees"), "wb") as ts_file:
    ts.dump(ts_file)
theta = theta.squeeze().cpu().numpy()
np.save(os.path.join(datadir, f"round_{rounds}/", f"theta_{num_simulations}.npy"), theta)
