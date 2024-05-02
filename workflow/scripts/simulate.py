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
import defopt
import numpy as np
import pickle
from simulators import AraTha_2epoch_simulator
def main(num_simulations: int, outdir: str, num_workers: int, prefix: str):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    simulator = AraTha_2epoch_simulator()
    prior = simulator.prior
    # check if prior and simulator meet sbi's requirements
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)
    
    # Simulate and create a list of theta and x
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations, num_workers=num_workers, )

    # save theta and x to an npy file
    theta = theta.squeeze().cpu().numpy()
    x = x.squeeze().cpu().numpy()
    if prefix != "":
        prefix += "_"
    np.save(
        f"{outdir}/{prefix}{num_simulations}_simulations.npy",
        {"theta": theta, "x": x},
    )

if __name__ == "__main__":
    defopt.run(main)