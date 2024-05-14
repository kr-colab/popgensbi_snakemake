import stdpopsim
import torch
from sbi.utils import BoxUniform
import numpy as np

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

class HomSap_Africa_1b08_simulator:
    def __init__(self, snakemake):
        species = stdpopsim.get_species("HomSap")
        model = species.get_demographic_model("Africa_1B08")
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
            self.N_A_low = 100
        try:
            self.N_A_high = snakemake.params.N_A_high
        except AttributeError:
            self.N_A_high = 30_000
        try:
            self.N_0_low = snakemake.params.N_0_low
        except AttributeError:
            self.N_0_low = 100
        try:
            self.N_0_high = snakemake.params.N_0_high
        except AttributeError:
            self.N_0_high = 30_000
        try:
            self.t_1_low = snakemake.params.t_1_low
        except AttributeError:
            self.t_1_low = 10
        try:
            self.t_1_high = snakemake.params.t_1_high
        except AttributeError:
            self.t_1_high = 10_000
        try:
            self.mutation_rate_low = snakemake.params.mutataion_rate_low
        except AttributeError:
            self.mutation_rate_low = 0
        try:
            self.mutation_rate_high = snakemake.params.mutation_rate_high
        except AttributeError:
            self.mutation_rate_high = 4.0e-8

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
        
        species = stdpopsim.get_species("HomSap")
        contig = species.get_contig(length=10e6, mutation_rate=mutation_rate)
        model = stdpopsim.PiecewiseConstantSize(N_A, (t_1, N_0))
        engine = stdpopsim.get_engine("msprime")

        ts = engine.simulate(model, contig, samples={"pop_0": self.n_sample})

        return ts

class gammaDFE_cnst_N_simulator:
    def __init__(self, snakemake):
        species = stdpopsim.get_species("HomSap")
        contig = species.get_contig('chr1', left=0, right=1e6)
        dfe = species.get_dfe("Gamma_K17")
        try:
            self.n_sample = snakemake.params.n_sample
        except AttributeError:
            self.n_sample = 100
        try:
            self.mutation_rate_true = snakemake.params.mutation_rate_true
        except AttributeError:
            self.mutation_rate_true = contig.mutation_rate
        try:
            self.shape_true = snakemake.params.shape_true
        except AttributeError:
            self.shape_true = dfe.mutation_types[1].distribution_args[1]
        try:
            self.mean_true = snakemake.params.mean_true
        except AttributeError:
            self.mean_true = dfe.mutation_types[1].distribution_args[0]
        try:
            self.p_true = snakemake.params.p_true
        except AttributeError:
            self.p_true = dfe.proportions[1] / sum(dfe.proportions)
        try:
            self.mutation_rate_low = snakemake.params.mutation_rate_low
        except AttributeError:
            self.mutation_rate_low = 1e-8
        try:
            self.mutation_rate_high = snakemake.params.mutation_rate_high
        except AttributeError:
            self.mutation_rate_high = 3e-8
        try:
            self.shape_low = snakemake.params.shape_low
        except AttributeError:
            self.shape_low = 0.01
        try:
            self.shape_high = snakemake.params.shape_high
        except AttributeError:
            self.shape_high = 1.
        try:
            self.mean_low = snakemake.params.mean_low
        except AttributeError:
            self.mean_low = -0.05
        try:
            self.mean_high = snakemake.params.mean_high
        except AttributeError:
            self.mean_high = 0.
        try:
            self.p_low = snakemake.params.p_low
        except AttributeError:
            self.p_low = 0.6
        try:
            self.p_high = snakemake.params.p_high
        except AttributeError:
            self.p_high = 0.8
        try:
            self.N = snakemake.params.N
        except AttributeError:
            self.N = 10000
        self.true_values = {"mutation_rate": self.mutation_rate_true, "mean": self.mean_true, "shape": self.shape_true, "p": self.p_true}
        self.bounds = {"mutation_rate": (self.mutation_rate_low, self.mutation_rate_high),
                        "mean": (self.mean_low, self.mean_high),
                        "shape": (self.shape_low, self.shape_high),
                        "p": (self.p_low, self.p_high),
                        }
        low = [self.bounds[p][0] for p in self.bounds.keys()]
        high = [self.bounds[p][1] for p in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, theta):
        if type(theta) is torch.Tensor:
            mutation_rate, mean, shape, p = theta.squeeze().cpu().tolist()
        elif type(theta) is list:
            mutation_rate, mean, shape, p = theta
        
        species = stdpopsim.get_species("HomSap")
        contig = species.get_contig('chr1', left=0, right=1e6)
        contig.mutation_rate = mutation_rate

        dfe = species.get_dfe("Gamma_K17")
        dfe.mutation_types[1].distribution_args = [mean, shape]
        dfe.proportions = [1-p, p]
        contig.add_dfe(intervals=np.array([[0, int(contig.length)]]), DFE=dfe)
        model = stdpopsim.PiecewiseConstantSize(self.N)
        engine = stdpopsim.get_engine("slim")

        ts = engine.simulate(model, contig, samples={"pop_0": self.n_sample})

        return ts
