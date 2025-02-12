## ts_simulators outputs tree sequence. This cannot be used as a simulator for simulate_for_sbi!

import tskit
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

class VaryingPopulationSize(BaseSimulator):
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
        self.pop_size_dist = utils.BoxUniform(
            low=np.log10(self.min_pop_size) * torch.ones(self.num_time_windows),
            high=np.log10(self.max_pop_size) * torch.ones(self.num_time_windows)
        )
        self.recomb_rate_dist = utils.BoxUniform(
            low=self.min_recomb_rate * torch.ones(1),
            high=self.max_recomb_rate * torch.ones(1)
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
        Sample population sizes and recombination rate from their prior distributions.
        
        Returns
        -------
        np.ndarray
            Array containing population sizes and recombination rate
        """
        # Sample initial population sizes from log-uniform distribution
        pop_sizes = 10 ** self.pop_size_dist.sample()
        
        # Sample recombination rate
        recomb_rate = self.recomb_rate_dist.sample()
        
        # Combine parameters
        theta = torch.cat([pop_sizes, recomb_rate])
        return theta.numpy()

    def log_prob(self, values):
        """
        Calculate log probability of parameters.
        
        Parameters
        ----------
        values : torch.Tensor or np.ndarray
            Array containing population sizes and recombination rate
            
        Returns
        -------
        torch.Tensor
            Log probability of the parameters
        """
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
            
        pop_sizes = values[:-1]
        recomb_rate = values[-1:]
        
        # Check if values are within bounds
        if ((pop_sizes >= self.min_pop_size).all() and 
            (pop_sizes <= self.max_pop_size).all() and
            (recomb_rate >= self.min_recomb_rate).all() and 
            (recomb_rate <= self.max_recomb_rate).all()):
            
            # Calculate log probabilities
            pop_log_prob = self.pop_size_dist.log_prob(torch.log10(pop_sizes))
            recomb_log_prob = self.recomb_rate_dist.log_prob(recomb_rate)
            
            return pop_log_prob + recomb_log_prob
        else:
            return torch.tensor(float('-inf'))

    def __call__(self, seed: int = None) -> (tskit.TreeSequence, np.ndarray):
        torch.manual_seed(seed)
        theta = self.sample_parameters()
        
        # Create demography
        demography = msprime.Demography()
        pop_sizes = theta[:-1]  # All but last element are population sizes
        recomb_rate = theta[-1]  # Last element is recombination rate
        
        # Add initial population
        demography.add_population(initial_size=pop_sizes[0])
        
        # Add population size changes at calculated time intervals
        for i in range(1, len(pop_sizes)):
            demography.add_population_parameters_change(
                time=self.change_times[i], 
                initial_size=pop_sizes[i],
                growth_rate=0
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

        return ts, theta


def SNP_afs(ts, SNP_min, data_path, scenario):
    """
    Analyze Single Nucleotide Polymorphisms (SNPs) from a set of tree sequences and filter them based on a minimum
    count threshold.

    Parameters
    ----------
    ts (iterator of msprime.Tree): An iterator over tree sequences generated by msprime or similar tool.
    SNP_min (int): Minimum number of SNPs required to consider the data valid.
    data_path (str): Path to the directory where SNP data should be stored.
    scenario (int): Identifier for the scenario under which the simulation is run, used for file naming.

    Returns
    -------
    tuple of lists: Returns a tuple containing two lists, one with SNP arrays and one with positions, if the minimum
    SNP count is met; otherwise, returns None.
    """

    SNPs_of_one_pop = []
    POSs_of_one_pop = []
    for rep, tree in enumerate(ts):
        mts = msprime.sim_mutations(tree, rate=1e-8)
        snp, pos = get_SNPs(mts)
        if len(pos) < SNP_min:
            return None
        else:
            SNPs_of_one_pop.append(snp)
            POSs_of_one_pop.append(pos)

    for idx, (snp, pos) in enumerate(zip(SNPs_of_one_pop, POSs_of_one_pop)):
        np.savez_compressed(os.path.join(data_path, f'checked_SNPs/scenario_{scenario}_rep_{idx}'), SNP=snp, POS=pos)
    return SNPs_of_one_pop, POSs_of_one_pop


def simulate_scenario(p, data_path, num_sample, SNP_min, num_rep, population_time, segment_length):
    """
    Simulate population for a single set of parameters and generate SNP data, saving results only for scenarios that
    meet SNP minimums.

    Parameters
    ----------
    p (list): A list containing population size and recombination rate data for a particular scenario.
    data_path (str): Base directory to store output data.
    num_sample (int): Number of samples to simulate.
    SNP_min (int): Minimum number of SNPs required to save data.
    num_rep (int): Number of replicates to simulate.
    population_time (list): Times at which population parameters change.
    segment_length (float): Length of the genomic segment to simulate.
    """
    demography = msprime.Demography()

    idx = int(p[0])
    pop_sizes = p[1:-1]
    recomb_rate = p[-1]
    demography.add_population(initial_size=pop_sizes[0])

    for i in range(1, len(pop_sizes)):
        demography.add_population_parameters_change(time=population_time[i], initial_size=pop_sizes[i], growth_rate=0)

    ts = msprime.sim_ancestry(
        num_sample,
        sequence_length=segment_length,
        ploidy=1,
        num_replicates=num_rep,
        demography=demography,
        recombination_rate=recomb_rate)

    # Check SNP data and save to files if it meets the SNP_min requirement.
    passed_SNP = SNP_afs(ts, SNP_min, data_path, idx)
    if passed_SNP is not None:
        SNP_400 = np.stack([s[:, :SNP_min] for s in passed_SNP[0]])
        POS_400 = np.stack([s[:SNP_min] for s in passed_SNP[1]])
        np.savez_compressed(os.path.join(data_path, f'SNP400/scenario_{idx}'), SNP=SNP_400, POS=POS_400)


def simulate_population(param, data_path, num_sample, SNP_min, num_rep, population_time, segment_length, max_cores):
    """
    Simulate population based on a set of parameters and generate SNP data, saving results only for scenarios that
    meet SNP minimums.

    Parameters
    ----------
    param (list): List of parameters where each element is a list containing population size and recombination rate
        data for a particular scenario.
    data_path (str): Base directory to store output data.
    num_sample (int): Number of samples to simulate.
    SNP_min (int): Minimum number of SNPs required to save data.
    num_rep (int): Number of replicates to simulate.
    population_time (list): Times at which population parameters change.
    segment_length (float): Length of the genomic segment to simulate.
    max_cores (int): Maximum number of threads to use.
    """
    os.makedirs(os.path.join(data_path, 'checked_SNPs'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'SNP400'), exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_cores) as executor:
        futures = [
            executor.submit(simulate_scenario, p, data_path, num_sample, SNP_min, num_rep, population_time,
                            segment_length)
            for p in param
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")


def get_SNPs(mts):
    """
    Extract SNP information and their positions from a given msprime/mutated tree sequence.

    Parameters
    ----------
    mts(msprime.MutationTreeSequence): The mutated tree sequence object from which to extract SNPs.

    Returns
    -------

    """
    positions = [variant.site.position for variant in mts.variants()]
    positions = np.array(positions) - np.array([0] + positions[:-1])
    positions = positions.astype(int)

    SNPs = mts.genotype_matrix().T.astype(np.uint8)
    return SNPs, positions
