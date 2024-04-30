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
# Set up prior
BOUNDS = {
    "N_A": (10_000, 1_000_000),
    "N_0": (10_000, 1_000_000),
    "t_1": (10_000, 1_000_000),
    "mutation_rate": (0, 1.0e-8),
}
low = [BOUNDS[p][0] for p in BOUNDS.keys()]
high = [BOUNDS[p][1] for p in BOUNDS.keys()]
prior = BoxUniform(low=tensor(low), high=tensor(high), device="cuda" if cuda else "cpu")

# set up simulator (in progress)
def simulator(theta):
    N_A, N_0, t_1, mutation_rate = theta.squeeze().cpu().tolist()
    population_configurations=[
            msprime.PopulationConfiguration(initial_size=N_0)
        ],
    demographic_events=[
            msprime.PopulationParametersChange(time=t_1, initial_size=N_A)
        ]
    ts = msprime.simulate(length=1e6, 
                population_configurations=population_configurations,
                demographic_events=demographic_events,
                recombination_rate=8.06452e-10,
                mutation_rate=7e-09)
    # Todo : make process_ts.py with classes of various summary stats of ts
    return result