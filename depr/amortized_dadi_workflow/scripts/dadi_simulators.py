import stdpopsim
import dadi
import numpy as np
from sbi.utils import BoxUniform
import torch

class BaseSimulator:
    def __init__(self, snakemake, params_default):
        for key, default in params_default.items():
            if key in snakemake.params.keys():
                setattr(self, key, snakemake.params[key])
            else:
                setattr(self, key, default)

class PonAbe_IM_simulator(BaseSimulator):
    '''
    simulate split and migration model of Orangutan
    Default values are set to the values from stdpopsim
    It uses dadi to simulate jsfs and Poisson sample from it
    '''
    species = stdpopsim.get_species("PonAbe")
    model = species.get_demographic_model("TwoSpecies_2L11")
    params_default = {
        "ns" : (100, 100), # sample size of two populations
        "pts" : 300, # number of grid points
        # ratio of population split = s : 1-s
        "s_true": (model.populations[0].initial_size 
                * np.exp(-model.populations[0].growth_rate 
                    * model.model.events[0].time) 
                    / model.model.events[-1].initial_size),
        # size of population 1 relative to ancestral population
        "nu1_true": (model.populations[0].initial_size 
            / model.model.events[-1].initial_size),
        # size of population 2 relative to ancestral population
        "nu2_true": (model.populations[1].initial_size
            / model.model.events[-1].initial_size),
        # time of population split scaled by 2 * ancestral population size
        "T_true": (model.model.events[0].time 
            / (2 * model.model.events[-1].initial_size)),
        # migration rate from population 1 to population 2 * 2 * ancestral population size
        "m12_true": (model.model.migration_matrix[0, 1] 
            * 2 * model.model.events[-1].initial_size),
        # migration rate from population 2 to population 1 * 2 * ancestral population size
        "m21_true": (model.model.migration_matrix[1, 0]
            * 2 * model.model.events[-1].initial_size),
        "s_low": 0.05,
        "s_high": 0.95,
        "nu1_low": 0.01,
        "nu1_high": 5,
        "nu2_low": 0.01,
        "nu2_high": 5,
        "T_low": 0.01,
        "T_high": 5,
        "m12_low": 0.0,
        "m12_high": 2,
        "m21_low": 0.0,
        "m21_high": 2,
        "n_seg_sites": 1e4
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, PonAbe_simulator.params_default)
        self.true_values = {"s": self.s_true, 
            "nu1": self.nu1_true, 
            "nu2": self.nu2_true, 
            "T": self.T_true, 
            "m12": self.m12_true, 
            "m21": self.m21_true}
        self.bounds = {"s": (self.s_low, self.s_high),
            "nu1": (self.nu1_low, self.nu1_high),
            "nu2": (self.nu2_low, self.nu2_high),
            "T": (self.T_low, self.T_high),
            "m12": (self.m12_low, self.m12_high),
            "m21": (self.m21_low, self.m21_high)}
        low = [self.bounds[key][0] for key in self.bounds.keys()]
        high = [self.bounds[key][1] for key in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")
    def __call__(self, theta):
        s, nu1, nu2, T, m12, m21 = theta.squeeze().cpu().tolist()
        params = (s, nu1, nu2, T, m12, m21)
        fs = dadi.Demographics2D.IM(params, self.ns, self.pts)
        fs_sample = (fs * self.n_seg_sites).sample()
        # normalize fs_sample
        fs_sample = fs_sample / fs_sample.sum()
        return torch.tensor(fs_sample)

class AraTha_2epoch_simulator(BaseSimulator):
    '''
    simulates 2 epoch model of Arabidopsis thaliana
    using stdpopsim's catalog for default parameter values
    Outputs sfs
    '''
    species = stdpopsim.get_species("AraTha")
    model = species.get_demographic_model("African2Epoch_1H18")
    params_default = {
        "ns": (20,), # sample size
        "pts": 100, # number of grid points
        # ratio of current population size to ancestral population size
        "nu_true": model.populations[0].initial_size / model.model.events[0].initial_size,
        # time of size change scaled by 2 * ancestral population size
        "T_true": model.model.events[0].time / (2 * model.model.events[0].initial_size),
        "nu_low": 0.01,
        "nu_high": 1,
        "T_low": 0.01,
        "T_high": 1.5, 
        "contig_length": 10e6 # synonymous sequence length 
        }
    def __init__(self, snakemake):
        super().__init__(snakemake, AraTha_2epoch_simulator.params_default)
        self.true_values = {"nu": self.nu_true, "T": self.T_true}
        self.bounds = {"nu": (self.nu_low, self.nu_high), "T": (self.T_low, self.T_high)}
        low = [self.bounds[key][0] for key in self.bounds.keys()]
        high = [self.bounds[key][1] for key in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")
    def __call__(self, theta):
        nu, T = theta.squeeze().cpu().tolist()
        params = (nu, T)
        fs = dadi.Demographics1D.two_epoch(params, self.ns, self.pts)
        # in Huber et al. it is assumed that number of segregating synonymous sites = 4 N_e mu L
        # this will be used to scale the sfs to Poisson sample from it
        # (Ne is ancestral population size)
        n_seg_sites = 4 * self.model.model.events[0].initial_size * self.model.mutation_rate * self.contig_length
        fs_sample = (fs * n_seg_sites).sample()
        # normalize fs_sample
        fs_sample = fs_sample / fs_sample.sum()
        return torch.tensor(fs_sample)

class YRI_CEU_simulator(BaseSimulator):
    '''
    simulate a custom model as defined in dadi's manual.
    Downsample simulated fs using the number of segregating sites
    '''
    params_default = {
        "ns": (20, 20), # sample size
        "pts": 100,
        "nu1F_true": 1.880,
        "nu2B_true": 0.0724,
        "nu2F_true": 1.764,
        "m_true": 0.930,
        "Tp_true": 0.363,
        "T_true": 0.112,
        "nu1F_low": 1e-2,
        "nu1F_high": 10,
        "nu2B_low": 1e-2,
        "nu2B_high": 10,
        "nu2F_low": 1e-2,
        "nu2F_high": 10,
        "m_low": 0,
        "m_high": 10,
        "Tp_low": 0,
        "Tp_high": 3,
        "T_low": 0,
        "T_high": 3,
        "contig_length": 10e6
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, YRI_CEU_simulator.params_default)
        self.true_values = {"nu1F": self.nu1F_true, "nu2B": self.nu2B_true, "nu2F": self.nu2F_true, "m": self.m_true, "Tp": self.Tp_true, "T": self.T_true}
        self.bounds = {"nu1F": (self.nu1F_low, self.nu1F_high), "nu2B": (self.nu2B_low, self.nu2B_high), "nu2F": (self.nu2F_low, self.nu2F_high), "m": (self.m_low, self.m_high), "Tp": (self.Tp_low, self.Tp_high), "T": (self.T_low, self.T_high)}
        low = [self.bounds[key][0] for key in self.bounds.keys()]
        high = [self.bounds[key][1] for key in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")
    def __call__(self, theta):
        nu1F, nu2B, nu2F, m, Tp, T = theta.squeeze().cpu().tolist()
        n1, n2 = self.ns
        xx = yy = dadi.Numerics.default_grid(self.pts)
        phi = dadi.PhiManip.phi_1D(xx)
        phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F)
        phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
        nu2_func = lambda t: nu2B * (nu2F / nu2B) ** (t / T)
        phi = dadi.Integration.two_pops(phi, xx, T, nu1=nu1F, nu2=nu2_func, m12=m, m21=m)
        fs = dadi.Spectrum.from_phi(phi, (n1, n2), (xx, yy))
        n_seg_sites = 4 * self.model.model.events[0].initial_size * self.model.mutation_rate * self.contig_length
        fs_sample = (fs * n_seg_sites).sample()
        # normalize fs_sample
        fs_sample = fs_sample / fs_sample.sum()
        return torch.tensor(fs_sample)



MODEL_LIST = {
    "PonAbe_IM": PonAbe_IM_simulator, 
    "AraTha_2epoch": AraTha_2epoch_simulator,
    "YRI_CEU": YRI_CEU_simulator
}