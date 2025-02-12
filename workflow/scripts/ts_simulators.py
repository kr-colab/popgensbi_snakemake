## ts_simulators outputs tree sequence. This cannot be used as a simulator for simulate_for_sbi!

import tskit
import msprime
import demes
import torch
import numpy as np
import stdpopsim
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

class AraTha_2epoch(BaseSimulator):
    """
    Simulate the African2Epoch_1H18 model from stdpopsim for Arabidopsis thaliana.
    The model consists of a single population that undergoes a size change.
    """
    species = stdpopsim.get_species("AraTha")
    model = species.get_demographic_model("African2Epoch_1H18")

    default_config = {
        # FIXED PARAMETERS
        "samples": {"SouthMiddleAtlas": 10},
        "sequence_length": 10e6,
        "recombination_rate": 1.5e-8,
        "mutation_rate": 1.5e-8,
        # RANDOM PARAMETERS (UNIFORM)
        "nu": [0.01, 1],      # Ratio of current to ancestral population size
        "T": [0.01, 1.5],     # Time of size change (scaled)
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        self.parameters = ["nu", "T"]
        self.prior = BoxUniform(
            low=torch.tensor([getattr(self, p)[0] for p in self.parameters]),
            high=torch.tensor([getattr(self, p)[1] for p in self.parameters]),
        )

    def __call__(self, seed: int = None) -> (tskit.TreeSequence, np.ndarray):
        torch.manual_seed(seed)
        theta = self.prior.sample().numpy()
        nu, T = theta

        species = self.species
        contig = species.get_contig(
            length=self.sequence_length,
        )
        model = self.model
        
        # Scale the population size and time parameters
        N_A = model.model.events[0].initial_size  # ancestral population size
        model.populations[0].initial_size = nu * N_A  # current population size
        model.model.events[0].time = T * 2 * N_A  # time of size change

        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(
            model, 
            contig, 
            samples=self.samples,
            random_seed=seed
        )

        return ts, theta

class VariablePopulationSize(BaseSimulator):
    """
    Simulate a model with varying population size across multiple time windows.
    The model consists of a single population that undergoes multiple size changes.
    """
    default_config = {
        # FIXED PARAMETERS
        "samples": {"pop": 10},
        "sequence_length": 10e6,
        "mutation_rate": 1.5e-8,
        # TIME WINDOWS AND SIZE RANGES
        "num_time_windows": 3,
        "min_pop_size": 1e2,
        "max_pop_size": 1e5,
        "min_recomb_rate": 1e-9,
        "max_recomb_rate": 1e-7,
        # TIME PARAMETERS
        "max_time": 100000,  # Maximum time for population events
        "time_rate": 0.1,    # Rate at which time changes across windows
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        # Parameters will be population sizes for each time window plus recombination rate
        self.parameters = [f"N_{i}" for i in range(self.num_time_windows)] + ["recomb_rate"]
        
        # Set up the prior distributions
        pop_size_lows = np.log10(self.min_pop_size) * torch.ones(self.num_time_windows)
        pop_size_highs = np.log10(self.max_pop_size) * torch.ones(self.num_time_windows)
        recomb_lows = self.min_recomb_rate * torch.ones(1)
        recomb_highs = self.max_recomb_rate * torch.ones(1)
        
        # Combine into single prior
        self.prior = BoxUniform(
            low=torch.cat([pop_size_lows, recomb_lows]),
            high=torch.cat([pop_size_highs, recomb_highs])
        )
        
        # Calculate fixed time points for population size changes
        self.change_times = self._calculate_change_times()

    def _calculate_change_times(self) -> np.ndarray:
        """
        Calculate the times at which population size changes occur using an exponential spacing.
        
        Returns
        -------
        np.ndarray
            Array of time points for population size changes
        """
        times = [(np.exp(np.log(1 + self.time_rate * self.max_time) * i / 
                        (self.num_time_windows - 1)) - 1) / self.time_rate 
                for i in range(self.num_time_windows)]
        return np.around(times).astype(int)

    def sample_parameters(self) -> np.ndarray:
        """
        Sample parameters from the prior distribution.
        
        Returns
        -------
        np.ndarray
            Array containing population sizes and recombination rate
        """
        theta = self.prior.sample()
        # Convert population sizes from log10 space
        pop_sizes = 10 ** theta[:-1]
        # Combine with recombination rate
        return torch.cat([pop_sizes, theta[-1:]])

    def __call__(self, seed: int = None) -> (tskit.TreeSequence, np.ndarray):
        if seed is not None:
            torch.manual_seed(seed)
        
        min_snps = 400
        max_attempts = 20  # Prevent infinite loops
        attempt = 0
        
        while attempt < max_attempts:
            theta = self.sample_parameters()
            
            # Create demography
            demography = msprime.Demography()
            # Convert tensors to numpy arrays for msprime
            pop_sizes = theta[:-1].numpy()  # All but last element are population sizes
            recomb_rate = float(theta[-1])  # Convert single value to Python float
            
            # Add initial population with name matching the samples config
            demography.add_population(name="pop0", initial_size=float(pop_sizes[0]))
            
            # Add population size changes at calculated time intervals
            for i in range(1, len(pop_sizes)):
                demography.add_population_parameters_change(
                    time=self.change_times[i], 
                    initial_size=float(pop_sizes[i]),
                    growth_rate=0,
                    population="pop0"  # Specify which population to change
                )

            # Simulate ancestry
            ts = msprime.sim_ancestry(
                samples={"pop0": self.samples["pop"]},
                demography=demography,
                sequence_length=self.sequence_length,
                recombination_rate=recomb_rate,
                random_seed=seed
            )
            
            # Add mutations
            ts = msprime.sim_mutations(ts, rate=self.mutation_rate, random_seed=seed)
            
            # Check if we have enough SNPs after MAF filtering
            geno = ts.genotype_matrix().T
            num_sample = geno.shape[0]
            if (geno==2).any():
                num_sample *= 2
            
            row_sum = np.sum(geno, axis=0)
            keep = np.logical_and.reduce([
                row_sum != 0,
                row_sum != num_sample,
                row_sum > num_sample * 0.05,  # MAF threshold
                num_sample - row_sum > num_sample * 0.05
            ])
            
            if np.sum(keep) >= min_snps:
                return ts, theta.numpy()
                
            attempt += 1
            if seed is not None:
                seed += 1  # Increment seed for next attempt
        
        raise RuntimeError(f"Failed to generate tree sequence with at least {min_snps} SNPs after MAF filtering and {max_attempts} attempts")

