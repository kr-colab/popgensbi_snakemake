## ts_simulators outputs tree sequence. This cannot be used as a simulator for simulate_for_sbi!

import tskit
import torch
import numpy as np

from sbi.utils import BoxUniform


class BaseSimulator:
    def __init__(self, config: dict, default: dict):
        for key, default in default.items():
            setattr(self, key, config.get(key, default))


class YRI_CEU(BaseSimulator):
    """
    Simulate a model defined in dadi's manual
    (https://dadi.readthedocs.io/en/latest/examples/YRI_CEU/YRI_CEU/).
    Ancestral population changes in size; splits into YRI/CEU; CEU population
    undergoes a bottleneck upon splitting and subsequently grows exponentially.
    There is continuous symmetric migration between YRI and CEU.
    """

    default_config = {
        # FIXED PARAMETERS
        "samples": {"YRI": 10, "CEU": 10},
        "sequence_length": 10e6,
        "recombination_rate": 1.5e-8,
        "mutation_rate": 1.5e-8,
        # RANDOM PARAMETERS (UNIFORM)
        "N_A": [1e2, 1e5],
        "N_YRI": [1e2, 1e5],
        "N_CEU_initial": [1e2, 1e5],
        "N_CEU_final": [1e2, 1e5],
        "M": [0, 5e-4],
        "Tp": [0, 6e4],
        "T": [0, 6e4],
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        self.parameters = ["N_A", "N_YRI", "N_CEU_initial", "N_CEU_final", "M", "Tp", "T"]
        self.prior = BoxUniform(
            low=torch.tensor([getattr(self, p)[0] for p in self.parameters]), 
            high=torch.tensor([getattr(self, p)[1] for p in self.parameters]), 
        )


    def __call__(self, seed: int = None) -> (tskit.TreeSequence, np.ndarray):

        torch.manual_seed(seed)
        theta = self.prior.sample().numpy()
        N_A, N_YRI, N_CEU_initial, N_CEU_final, M, Tp, T = theta

        import demes
        import msprime

        graph = demes.Builder()
        graph.add_deme(
            "ancestral", 
            epochs=[dict(start_size=N_A, end_time=Tp + T)]
        )
        graph.add_deme(
            "AMH", 
            ancestors=["ancestral"], 
            epochs=[dict(start_size=N_YRI, end_time=T)],
        )
        graph.add_deme(
            "CEU", 
            ancestors=["AMH"], 
            epochs=[dict(start_size=N_CEU_initial, end_size=N_CEU_final)],
        )
        graph.add_deme(
            "YRI", 
            ancestors=["AMH"], 
            epochs=[dict(start_size=N_YRI)],
        )
        graph.add_migration(demes=["CEU", "YRI"], rate=M)

        demog = msprime.Demography.from_demes(graph.resolve())
        ts = msprime.sim_ancestry(
            self.samples,
            demography=demog,
            sequence_length=self.sequence_length,
            recombination_rate=self.recombination_rate,
            random_seed=seed,
        )
        ts = msprime.sim_mutations(ts, rate=self.mutation_rate, random_seed=seed)

        return ts, theta


