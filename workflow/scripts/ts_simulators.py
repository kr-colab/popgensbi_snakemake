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
        "num_time_windows": 3,
        # RANDOM PARAMETERS (UNIFORM)
        "pop_sizes": [1e2, 1e5],      # Range for population sizes (log10 space)
        "recomb_rate": [1e-9, 1e-7],  # Range for recombination rate
        # TIME PARAMETERS
        "max_time": 100000,  # Maximum time for population events
        "time_rate": 0.1,    # Rate at which time changes across windows
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        # Set up parameters list
        self.parameters = [f"N_{i}" for i in range(self.num_time_windows)] + ["recomb_rate"]
        
        # Create parameter ranges in the same format as AraTha_2epoch
        # Population sizes (in log10 space)
        pop_size_ranges = [[np.log10(self.pop_sizes[0]), np.log10(self.pop_sizes[1])] 
                          for _ in range(self.num_time_windows)]
        # Add recombination rate range
        param_ranges = pop_size_ranges + [self.recomb_rate]
        
        # Set up prior using BoxUniform
        self.prior = BoxUniform(
            low=torch.tensor([r[0] for r in param_ranges]),
            high=torch.tensor([r[1] for r in param_ranges])
        )
        
        # Calculate fixed time points for population size changes
        self.change_times = self._calculate_change_times()

    def _calculate_change_times(self) -> np.ndarray:
        """Calculate the times at which population size changes occur using an exponential spacing."""
        times = [(np.exp(np.log(1 + self.time_rate * self.max_time) * i / 
                        (self.num_time_windows - 1)) - 1) / self.time_rate 
                for i in range(self.num_time_windows)]
        return np.around(times).astype(int)

    def __call__(self, seed: int = None) -> (tskit.TreeSequence, np.ndarray):
        if seed is not None:
            torch.manual_seed(seed)
        
        min_snps = 400
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            # Sample parameters directly from prior (like AraTha_2epoch)
            theta = self.prior.sample().numpy()
            
            # Convert population sizes from log10 space
            pop_sizes = 10 ** theta[:-1]  # All but last element are population sizes
            recomb_rate = theta[-1]  # Last element is recombination rate
            
            # Create demography
            demography = msprime.Demography()
            demography.add_population(name="pop0", initial_size=float(pop_sizes[0]))
            
            # Add population size changes at calculated time intervals
            for i in range(1, len(pop_sizes)):
                demography.add_population_parameters_change(
                    time=self.change_times[i],
                    initial_size=float(pop_sizes[i]),
                    growth_rate=0,
                    population="pop0"
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
                row_sum > num_sample * 0.05,
                num_sample - row_sum > num_sample * 0.05
            ])
            
            if np.sum(keep) >= min_snps:
                return ts, theta
                
            attempt += 1
            if seed is not None:
                seed += 1
        
        raise RuntimeError(f"Failed to generate tree sequence with at least {min_snps} SNPs after {max_attempts} attempts")


class recombination_rate(BaseSimulator):
    """
    Simulate a one population model where recombination rate varies
    among replicates 
    """

    default_config = {
        # FIXED PARAMETERS
        "samples": {0: 10},
        "sequence_length": 1e6,
        "mutation_rate": 1.5e-8,
        "pop_size": 1e4,
        # RANDOM PARAMETERS (UNIFORM)
        "recombination_rate": [0, 1e-8],
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        self.parameters = ["recombination_rate"]
        self.prior = BoxUniform(
            low=torch.tensor([getattr(self, p)[0] for p in self.parameters]), 
            high=torch.tensor([getattr(self, p)[1] for p in self.parameters]), 
        )


    def __call__(self, seed: int = None) -> (tskit.TreeSequence, np.ndarray):

        torch.manual_seed(seed)
        theta = self.prior.sample().numpy()
        recombination_rate = theta.item()

        ts = msprime.sim_ancestry(
            self.samples,
            population_size=self.pop_size,
            sequence_length=self.sequence_length,
            recombination_rate=recombination_rate,
            random_seed=seed,
            discrete_genome=False, # don't want overlapping mutations
        )
        ts = msprime.sim_mutations(ts, rate=self.mutation_rate, random_seed=seed)

        return ts, theta


class Cattle_21Gen(BaseSimulator):
    """
    Simulate a cattle population with population size across 21 time windows.
    The model consists of a single population that undergoes multiple size changes.
    """
    default_config = {
        # FIXED PARAMETERS
        "samples": {"pop0": 25},
        "sequence_length": 2e6,
        "mutation_rate": 1e-8,
        "num_time_windows": 21,
        "maf": 0.05,
        # RANDOM PARAMETERS (UNIFORM)
        "pop_sizes": [10, 1e5],      # Range for population sizes (log10 space)
        "pop_changes": [-1, 1],      # Range for population size changes (* 10 ** beta)
        "recomb_rate": [1e-9, 1e-8],  # Range for recombination rate
        # TIME PARAMETERS
        "max_time": 130000,  # Maximum time for population events
        "time_rate": 0.1,    # Rate at which time changes across windows
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        # Set up parameters list
        self.parameters = ["N_0"] + [f"beta_{i}" for i in range(1,self.num_time_windows)] + ["recomb_rate"]
        
        # Create parameter ranges in the same format as AraTha_2epoch
        # Population sizes (in log10 space)
        pop_size_range = [[np.log10(self.pop_sizes[0]), np.log10(self.pop_sizes[1])]]
        pop_change_ranges = [[self.pop_changes[0], self.pop_changes[1]]
                             for _ in range(1,self.num_time_windows)]
        # Add recombination rate range
        param_ranges = pop_size_range + pop_change_ranges + [self.recomb_rate]
        
        # Set up prior using BoxUniform
        self.prior = BoxUniform(
            low=torch.tensor([r[0] for r in param_ranges]),
            high=torch.tensor([r[1] for r in param_ranges])
        )
        
        # Calculate fixed time points for population size changes
        self.change_times = self._calculate_change_times()

    def _calculate_change_times(self) -> np.ndarray:
        """Calculate the times at which population size changes occur using an exponential spacing."""
        times = [(np.exp(np.log(1 + self.time_rate * self.max_time) * i / 
                        (self.num_time_windows - 1)) - 1) / self.time_rate 
                for i in range(self.num_time_windows)]
        return np.around(times).astype(int)
    
    def _generate_dependent_pop_sizes(self) -> np.ndarray:
        """
        Generate a sequence of population sizes where the first population size
        is sampled from a uniform distribution corresponding to the most recent
        time window. The following population sizes are generated following
        N_i = N_{i-1} * 10 ^ β for i in [1,...,num_time_windows], unless N_i
        is outside of pop_ranges. If so N_i is set to the max/min population size
        """
        prior_sample = self.prior.sample().numpy()
        # The first value is uniformly sampled within the log10 bounds
        # Transform into population size N_0
        prior_sample[0] = prior_sample[0] * 10 ** prior_sample[0]
        # For subsequent time windows, calculate the new value based on the previous one and beta
        for i in range(1, self.num_time_windows):
            new_value = prior_sample[i - 1] * 10 ** prior_sample[i]
            if new_value > self.pop_sizes[1]:
                new_value = self.pop_sizes[1]
            if new_value < self.pop_sizes[0]:
                new_value = self.pop_sizes[0]
            prior_sample[i] = new_value

        # Return N_i and recombination rate
        return prior_sample

    def __call__(self, seed: int = None) -> (tskit.TreeSequence, np.ndarray):
        if seed is not None:
            torch.manual_seed(seed)
        
        min_snps = 400
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            # Sample parameters directly from prior (like AraTha_2epoch)
            theta = self._generate_dependent_pop_sizes()
            
            pop_sizes = theta[:-1]  # All but last element are population sizes
            recomb_rate = theta[-1]  # Last element is recombination rate
            
            # Create demography
            demography = msprime.Demography()
            demography.add_population(name="pop0", initial_size=float(pop_sizes[0]))
            
            # Add population size changes at calculated time intervals
            for i in range(1, len(pop_sizes)):
                demography.add_population_parameters_change(
                    time=self.change_times[i],
                    initial_size=float(pop_sizes[i]),
                    growth_rate=0,
                    population="pop0"
                )

            # Simulate ancestry
            ts = msprime.sim_ancestry(
                samples=self.samples,
                demography=demography,
                sequence_length=self.sequence_length,
                recombination_rate=recomb_rate,
                random_seed=seed
            )
            
            # Add mutations
            ts = msprime.sim_mutations(ts, rate=self.mutation_rate, random_seed=seed)
            
            # Check if we have enough SNPs after MAF filtering
            geno = ts.genotype_matrix().T
            num_sample = ts.num_samples
            
            row_sum = np.sum(geno, axis=0)
            keep = np.logical_and.reduce([
                row_sum > num_sample * self.maf,
                num_sample - row_sum > num_sample * self.maf
            ])
            
            if np.sum(keep) >= min_snps:
                return ts, theta
                
            attempt += 1
            if seed is not None:
                seed += 1
        
        raise RuntimeError(f"Failed to generate tree sequence with at least {min_snps} SNPs after {max_attempts} attempts")
