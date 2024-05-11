import stdpopsim
import torch
from sbi.utils import BoxUniform

## ts_simulators outputs tree sequence. This cannot be used as a simulator for simulate_for_sbi!
class AraTha_2epoch_simulator:
    def __init__(self, snakemake):
        species = stdpopsim.get_species("AraTha")
        model = species.get_demographic_model("African2Epoch_1H18")
        try:
            self.n_sample = snakemake.params.n_sample
        except AttributeError:
            self.n_sample = 10
        try:
            self.N_A_true = snakemake.params.N_A_true
        except AttributeError:
            self.N_A_true = model.model.events[0].initial_size
        try:
            self.N_0_true = snakemake.params.N_0_true
        except AttributeError:
            self.N_0_true = model.populations[0].initial_size
        try:
            self.t_1_true = snakemake.params.t_1_true
        except AttributeError:
            self.t_1_true = model.model.events[0].time
        try:
            self.mutation_rate_true = snakemake.params.mutation_rate_true
        except AttributeError:
            self.mutation_rate_true = model.mutation_rate
        try:
            self.N_A_low = snakemake.params.N_A_low
        except AttributeError:
            self.N_A_low = 10_000
        try:
            self.N_A_high = snakemake.params.N_A_high
        except AttributeError:
            self.N_A_high = 1_000_000
        try:
            self.N_0_low = snakemake.params.N_0_low
        except AttributeError:
            self.N_0_low = 10_000
        try:
            self.N_0_high = snakemake.params.N_0_high
        except AttributeError:
            self.N_0_high = 1_000_000
        try:
            self.t_1_low = snakemake.params.t_1_low
        except AttributeError:
            self.t_1_low = 1_000
        try:
            self.t_1_high = snakemake.params.t_1_high
        except AttributeError:
            self.t_1_high = 1_000_000
        try:
            self.mutation_rate_low = snakemake.params.mutataion_rate_low
        except AttributeError:
            self.mutation_rate_low = 0
        try:
            self.mutation_rate_high = snakemake.params.mutation_rate_high
        except AttributeError:
            self.mutation_rate_high = 1.0e-8

        self.true_values = {"N_A": self.N_A_true, "N_0": self.N_0_true, "t_1": self.t_1_true, "mutation_rate": self.mutation_rate_true}
        self.bounds = {"N_A": (self.N_A_low, self.N_A_high),
                        "N_0": (self.N_0_low, self.N_0_high),
                        "t_1": (self.t_1_low, self.t_1_high),
                        "mutation_rate": (self.mutation_rate_low, self.mutation_rate_high),
                        }
        low = [self.bounds[p][0] for p in self.bounds.keys()]
        high = [self.bounds[p][1] for p in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, theta):
        if type(theta) is torch.Tensor:
            N_A, N_0, t_1, mutation_rate = theta.squeeze().cpu().tolist()
        elif type(theta) is list:
            N_A, N_0, t_1, mutation_rate = theta
        
        species = stdpopsim.get_species("AraTha")
        contig = species.get_contig(length=10e6, mutation_rate=mutation_rate)
        model = stdpopsim.PiecewiseConstantSize(N_A, (t_1, N_0))
        engine = stdpopsim.get_engine("msprime")

        ts = engine.simulate(model, contig, samples={"pop_0": self.n_sample})

        return ts