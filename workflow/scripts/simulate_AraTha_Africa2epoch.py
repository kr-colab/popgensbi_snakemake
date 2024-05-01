## Reproducing AraTha/Africa2Epoch_1h18 of popgensbi

import torch

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.utils import BoxUniform
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import msprime
import stdpopsim
from process_ts import dinf_extract
import os
import defopt
import numpy as np

def main(num_simulations: int, outdir: str, num_workers: int, prefix: str):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    true_values = {"N_A": 746148, "N_0": 100218, "t_1": 568344, "mutation_rate": 7e-9}

    # Set up prior
    BOUNDS = {
    "N_A": (10_000, 1_000_000),
    "N_0": (10_000, 1_000_000),
    "t_1": (10_000, 1_000_000),
    "mutation_rate": (0, 1.0e-8),
    }
    low = [BOUNDS[p][0] for p in BOUNDS.keys()]
    high = [BOUNDS[p][1] for p in BOUNDS.keys()]
    prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")

    # save true values and bounds for plotting at the end of NPE inference
    true_values_file = f"{outdir}/AraTha_2epoch_true.npy"
    bounds_file = f"{outdir}/AraTha_2epoch_bounds.npy"
    if not os.path.exists(true_values_file):
        np.save(true_values_file, true_values)
    if not os.path.exists(bounds_file):
        np.save(bounds_file, BOUNDS)

    
    # Set up simulator
    def simulator(theta):
        N_A, N_0, t_1, mutation_rate = theta.squeeze().cpu().tolist()
        species = stdpopsim.get_species("AraTha")
        contig = species.get_contig(length=1e6, mutation_rate=mutation_rate)
        model = stdpopsim.PiecewiseConstantSize(N_A, (t_1, N_0))
        engine = stdpopsim.get_engine("msprime")

        n_sample = 10

        ts = engine.simulate(model, contig, samples = {"pop_0":n_sample})

        n_snps = 2000
        ploidy = 2
        maf_thresh = 0.05
        phased = False

        result = dinf_extract(ts, n_sample, n_snps, ploidy, phased, maf_thresh)
        return result

    # check if prior and simulator meet sbi's requirements
    prior, num_parameters, prior_returns_numpy = process_prior(prior)
    simulator = process_simulator(simulator, prior, prior_returns_numpy)
    check_sbi_inputs(simulator, prior)

    # finally simulate and create a list of theta and x
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