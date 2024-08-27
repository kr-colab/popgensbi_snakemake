## ts_simulators outputs tree sequence. This cannot be used as a simulator for simulate_for_sbi!

import stdpopsim
import torch
from sbi.utils import BoxUniform
import numpy as np
from zuko.distributions import Uniform, Joint, Sort
from zuko.distributions import BoxUniform as BoxUniformZuko
class BaseSimulator:
    def __init__(self, snakemake, params_default):
        for key, default in params_default.items():
            if key in snakemake.params.keys():
                setattr(self, key, snakemake.params[key])
            else:
                setattr(self, key, default)

# Modified to match dadi simulator            
class AraTha_2epoch_simulator(BaseSimulator):
    species = stdpopsim.get_species("AraTha")
    model = species.get_demographic_model("African2Epoch_1H18")
    params_default = {
        "n_sample": 10,
        "nu_true": model.populations[0].initial_size / model.model.events[0].initial_size,
        "T_true": model.model.events[0].time / (2 * model.model.events[0].initial_size),
        "nu_low": 0.01,
        "nu_high": 1,
        "T_low": 0.01,
        "T_high": 1.5,
        "contig_length": 10e6,
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, AraTha_2epoch_simulator.params_default)
        self.true_values = {"nu": self.nu_true, "T": self.T_true}
        self.bounds = {"nu": (self.nu_low, self.nu_high), "T": (self.T_low, self.T_high)}
        low = [self.bounds[p][0] for p in self.bounds.keys()]
        high = [self.bounds[p][1] for p in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, theta):
        if type(theta) is torch.Tensor:
            nu, T = theta.squeeze().cpu().tolist()
        elif type(theta) is list:
            nu, T = theta
        
        species = AraTha_2epoch_simulator.species
        contig = species.get_contig(length=self.contig_length)
        model = AraTha_2epoch_simulator.model
        model.populations[0].initial_size = nu * model.model.events[0].initial_size
        model.model.events[0].time = T * 2 * model.model.events[0].initial_size
        engine = stdpopsim.get_engine("msprime")

        ts = engine.simulate(model, contig, samples={"SouthMiddleAtlas": self.n_sample})

        return ts

class AraTha_2epoch_genetic_map_simulator(BaseSimulator):
    '''
    Simulator for AraTha 2 epoch model with genetic map (SalomeAveraged_TAIR10)
    Use a subset of chromosome 1 (from the left, clipped on the right) for simulation
    '''
    species = stdpopsim.get_species("AraTha")
    model = species.get_demographic_model("African2Epoch_1H18")
    N_anc = model.model.events[0].initial_size
    params_default = {
        "n_sample": 10,
        "nu_true": model.populations[0].initial_size / N_anc,
        "T_true": model.model.events[0].time / (2 * N_anc),
        "nu_low": 0.01,
        "nu_high": 1,
        "T_low": 0.01,
        "T_high": 1.5,
        "contig_length": 1e6,
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, AraTha_2epoch_genetic_map_simulator.params_default)
        self.true_values = {"nu": self.nu_true, "T": self.T_true}
        self.bounds = {"nu": (self.nu_low, self.nu_high), "T": (self.T_low, self.T_high)}
        low = [self.bounds[p][0] for p in self.bounds.keys()]
        high = [self.bounds[p][1] for p in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, theta):
        if type(theta) is torch.Tensor:
            nu, T = theta.squeeze().cpu().tolist()
        elif type(theta) is list:
            nu, T = theta
        
        species = AraTha_2epoch_genetic_map_simulator.species
        contig = species.get_contig("chr1", genetic_map="SalomeAveraged_TAIR10", left=0, right=self.contig_length)
        model = AraTha_2epoch_genetic_map_simulator.model
        model.populations[0].initial_size = nu * model.model.events[0].initial_size
        model.model.events[0].time = T * 2 * model.model.events[0].initial_size
        engine = stdpopsim.get_engine("msprime")

        ts = engine.simulate(model, contig, samples={"SouthMiddleAtlas": self.n_sample})

        return ts

class HomSap_ooa_archaic_simulator(BaseSimulator):
    '''
    simulate based on the OutOfAfricaArchaicAdmixture_5R19 model, perturbing inferred parameter values.
    Note that the paramters are not normalized based on ancestral pop size.
    '''
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfricaArchaicAdmixture_5R19")
    # extract true parameter values from the stdpopsim catalog model
    N_A_true = model.populations[-1].initial_size
    N_YRI_true = model.populations[0].initial_size
    N_CEU_final = model.populations[1].initial_size
    N_CHB_final = model.populations[2].initial_size
    r_CEU_true = model.populations[1].growth_rate
    r_CHB_true = model.populations[2].growth_rate
    # OOA pop size is found from model.model.events
    N_OOA_true = model.model.events[14].initial_size
    # timing of each event, convert to yr by multiplying by generation time
    T_arch_adm_end_true = model.model.events[0].time * model.generation_time
    T_EU_AS_true = model.model.events[6].time * model.generation_time
    T_B_true = model.model.events[15].time * model.generation_time
    T_arch_afr_mig_true = model.model.events[19].time * model.generation_time
    T_AF_true = model.model.events[20].time * model.generation_time
    T_arch_afr_split_true = model.model.events[21].time * model.generation_time
    T_nean_split_true = model.model.events[22].time * model.generation_time
    # with T_EU_AS_true, we can get initial size of CEU and CHB
    N_CEU_true = N_CEU_final * np.exp(-r_CEU_true * T_EU_AS_true / model.generation_time)
    N_CHB_true = N_CHB_final * np.exp(-r_CHB_true * T_EU_AS_true / model.generation_time)
    # migration rate of current populations can be found from migration matrix
    m_YRI_CEU_true = model.model.migration_matrix[0, 1]
    m_CEU_CHB_true = model.model.migration_matrix[1, 2]
    # migration rates of archaic populations should be found from model.model.events (migration rate change events)
    m_AF_arch_af_true = model.model.events[0].rate
    m_OOA_nean_true = model.model.events[2].rate
    m_AF_B_true = model.model.events[8].rate

    params_default = {
        "samples": {"YRI": 10, "CEU": 10, "CHB": 10},
        "N_A_true": N_A_true,
        "N_YRI_true": N_YRI_true,
        "N_CEU_true": N_CEU_true,
        "N_CHB_true": N_CHB_true,
        "N_OOA_true": N_OOA_true,
        "r_CEU_true": r_CEU_true,
        "r_CHB_true": r_CHB_true,
        "T_arch_adm_end_true": T_arch_adm_end_true,
        "T_EU_AS_true": T_EU_AS_true,
        "T_B_true": T_B_true,
        "T_arch_afr_mig_true": T_arch_afr_mig_true,
        "T_AF_true": T_AF_true,
        "T_arch_afr_split_true": T_arch_afr_split_true,
        "T_nean_split_true": T_nean_split_true,
        "m_YRI_CEU_true": m_YRI_CEU_true,
        "m_CEU_CHB_true": m_CEU_CHB_true,
        "m_AF_arch_af_true": m_AF_arch_af_true,
        "m_OOA_nean_true": m_OOA_nean_true,
        "m_AF_B_true": m_AF_B_true,
        "N_A_low": 100,
        "N_A_high": 50_000,
        "N_YRI_low": 100,
        "N_YRI_high": 50_000,
        "N_CEU_low": 100,
        "N_CEU_high": 50_000,
        "N_CHB_low": 100,
        "N_CHB_high": 50_000,
        "N_OOA_low": 100,
        "N_OOA_high": 50_000,
        "r_CEU_low": 0,
        "r_CEU_high": 0.005,
        "r_CHB_low": 0,
        "r_CHB_high": 0.005,
        "T_arch_adm_end_low": 10_000,
        "T_arch_adm_end_high": 1_000_000,
        "T_EU_AS_low": 10_000,
        "T_EU_AS_high": 1_000_000,
        "T_B_low": 10_000,
        "T_B_high": 1_000_000,
        "T_arch_afr_mig_low": 10_000,
        "T_arch_afr_mig_high": 1_000_000,
        "T_AF_low": 10_000,
        "T_AF_high": 1_000_000,
        "T_arch_afr_split_low": 10_000,
        "T_arch_afr_split_high": 1_000_000,
        "T_nean_split_low": 10_000,
        "T_nean_split_high": 1_000_000,
        "m_YRI_CEU_low": 0.0,
        "m_YRI_CEU_high": 0.001,
        "m_CEU_CHB_low": 0.0,
        "m_CEU_CHB_high": 0.001,
        "m_AF_arch_af_low": 0.0,
        "m_AF_arch_af_high": 0.001,
        "m_OOA_nean_low": 0.0,
        "m_OOA_nean_high": 0.001,
        "m_AF_B_low": 0.0,
        "m_AF_B_high": 0.001,
        "contig_length": 1e6,
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, HomSap_ooa_archaic_simulator.params_default)
        self.true_values = {
            "N_A": self.N_A_true,
            "N_YRI": self.N_YRI_true,
            "N_CEU": self.N_CEU_true,
            "N_CHB": self.N_CHB_true,
            "N_OOA": self.N_OOA_true,
            "r_CEU": self.r_CEU_true,
            "r_CHB": self.r_CHB_true,
            "T_arch_adm_end": self.T_arch_adm_end_true,
            "T_EU_AS": self.T_EU_AS_true,
            "T_B": self.T_B_true,
            "T_arch_afr_mig": self.T_arch_afr_mig_true,
            "T_AF": self.T_AF_true,
            "T_arch_afr_split": self.T_arch_afr_split_true,
            "T_nean_split": self.T_nean_split_true,
            "m_YRI_CEU": self.m_YRI_CEU_true,
            "m_CEU_CHB": self.m_CEU_CHB_true,
            "m_AF_arch_af": self.m_AF_arch_af_true,
            "m_OOA_nean": self.m_OOA_nean_true,
            "m_AF_B": self.m_AF_B_true,
        }
        self.bounds = {
            "N_A": (self.N_A_low, self.N_A_high),
            "N_YRI": (self.N_YRI_low, self.N_YRI_high),
            "N_CEU": (self.N_CEU_low, self.N_CEU_high),
            "N_CHB": (self.N_CHB_low, self.N_CHB_high),
            "N_OOA": (self.N_OOA_low, self.N_OOA_high),
            "r_CEU": (self.r_CEU_low, self.r_CEU_high),
            "r_CHB": (self.r_CHB_low, self.r_CHB_high),
            "T_arch_adm_end": (self.T_arch_adm_end_low, self.T_arch_adm_end_high),
            "T_EU_AS": (self.T_EU_AS_low, self.T_EU_AS_high),
            "T_B": (self.T_B_low, self.T_B_high),
            "T_arch_afr_mig": (self.T_arch_afr_mig_low, self.T_arch_afr_mig_high),
            "T_AF": (self.T_AF_low, self.T_AF_high),
            "T_arch_afr_split": (self.T_arch_afr_split_low, self.T_arch_afr_split_high),
            "T_nean_split": (self.T_nean_split_low, self.T_nean_split_high),
            "m_YRI_CEU": (self.m_YRI_CEU_low, self.m_YRI_CEU_high),
            "m_CEU_CHB": (self.m_CEU_CHB_low, self.m_CEU_CHB_high),
            "m_AF_arch_af": (self.m_AF_arch_af_low, self.m_AF_arch_af_high),
            "m_OOA_nean": (self.m_OOA_nean_low, self.m_OOA_nean_high),
            "m_AF_B": (self.m_AF_B_low, self.m_AF_B_high),
        }
        low = [self.bounds[p][0] for p in self.bounds.keys()]
        high = [self.bounds[p][1] for p in self.bounds.keys()]
        if hasattr(snakemake.params, "device") and snakemake.params.device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        N_distribution = BoxUniformZuko(torch.tensor(low[:5], dtype=torch.float32).to(device), torch.tensor(high[:5], dtype=torch.float32).to(device))
        r_distribution = BoxUniformZuko(torch.tensor(low[5:7], dtype=torch.float32).to(device), torch.tensor(high[5:7], dtype=torch.float32).to(device))
        T_distribution = Sort(Uniform(torch.tensor(low[7], dtype=torch.float32).to(device), torch.tensor(high[7], dtype=torch.float32).to(device)), 7)
        m_distribution = BoxUniformZuko(torch.tensor(low[14:], dtype=torch.float32).to(device), torch.tensor(high[14:], dtype=torch.float32).to(device))
        self.prior = Joint(N_distribution, r_distribution, T_distribution, m_distribution)

    def __call__(self,theta):
        if type(theta) == torch.Tensor:
            theta = theta.squeeze().cpu().tolist()
            
        N_A,N_YRI, N_CEU, N_CHB, N_OOA = theta[:5]
        r_CEU,r_CHB = theta[5:7]
        # Note that these are in years not in generations
        T_arch_adm_end,T_EU_AS,T_B,T_arch_afr_mig,T_AF,T_arch_afr_split,T_nean_split= theta[7:14]
        m_YRI_CEU,m_CEU_CHB,m_AF_arch_af,m_OOA_nean,m_AF_B = theta[14:]
        # Now we need to replace parameter values in model with the values from theta
        species = stdpopsim.get_species("HomSap")
        model = species.get_demographic_model("OutOfAfricaArchaicAdmixture_5R19")
        model.model.populations[0].initial_size = N_YRI
        model.model.populations[1].initial_size = N_CEU / np.exp(-r_CEU * T_EU_AS / model.generation_time)
        model.model.populations[1].growth_rate = r_CEU
        model.model.populations[2].initial_size = N_CHB / np.exp(-r_CHB * T_EU_AS / model.generation_time)
        model.model.populations[2].growth_rate = r_CHB
        model.model.populations[3].initial_size = N_A
        model.model.populations[4].initial_size = N_A
        model.model.migration_matrix[0,1] = m_YRI_CEU
        model.model.migration_matrix[1,0] = m_YRI_CEU
        model.model.migration_matrix[1,2] = m_CEU_CHB
        model.model.migration_matrix[2,1] = m_CEU_CHB
        # change demographic events based on theta
        # first event is migration turned on between modern and archaic humans
        model.model.events[0].time = T_arch_adm_end / model.generation_time
        model.model.events[0].rate = m_AF_arch_af
        model.model.events[1].time = T_arch_adm_end / model.generation_time
        model.model.events[1].rate = m_AF_arch_af
        model.model.events[2].time = T_arch_adm_end / model.generation_time
        model.model.events[2].rate = m_OOA_nean
        model.model.events[3].time = T_arch_adm_end / model.generation_time
        model.model.events[3].rate = m_OOA_nean
        model.model.events[4].time = T_arch_adm_end / model.generation_time
        model.model.events[4].rate = m_OOA_nean
        model.model.events[5].time = T_arch_adm_end / model.generation_time
        model.model.events[5].rate = m_OOA_nean
            # CEU and CHB merge into B with rate changes at T_EU_AS
        model.model.events[6].time = T_EU_AS / model.generation_time
        model.model.events[7].time = T_EU_AS / model.generation_time
        model.model.events[8].time = T_EU_AS / model.generation_time
        model.model.events[8].rate = m_AF_B
        model.model.events[9].time = T_EU_AS / model.generation_time
        model.model.events[9].rate = m_AF_B
        model.model.events[10].time = T_EU_AS / model.generation_time
        model.model.events[10].rate = m_AF_arch_af
        model.model.events[11].time = T_EU_AS / model.generation_time
        model.model.events[11].rate = m_AF_arch_af
        model.model.events[12].time = T_EU_AS / model.generation_time
        model.model.events[12].rate = m_OOA_nean
        model.model.events[13].time = T_EU_AS / model.generation_time
        model.model.events[13].rate = m_OOA_nean
        model.model.events[14].time = T_EU_AS / model.generation_time
        model.model.events[14].initial_size = N_OOA
        # Population B merges into YRI at T_B
        model.model.events[15].time = T_B / model.generation_time
        model.model.events[16].time = T_B / model.generation_time
        model.model.events[17].time = T_B / model.generation_time
        model.model.events[17].rate = m_AF_arch_af
        model.model.events[18].time = T_B / model.generation_time
        model.model.events[18].rate = m_AF_arch_af
        # Beginning of migration between African and archaic African populations
        model.model.events[19].time = T_arch_afr_mig / model.generation_time
        # Size changes to N_A (N_0 in stdpopsim source code) at T_AF
        model.model.events[20].time = T_AF / model.generation_time
        model.model.events[20].initial_size = N_A
        # Archaic African merges with moderns
        model.model.events[21].time = T_arch_afr_split / model.generation_time
        # Neanderthal merges with moderns
        model.model.events[22].time = T_nean_split / model.generation_time
        engine = stdpopsim.get_engine("msprime")
        contig = species.get_contig(length=self.contig_length)
        ts = engine.simulate(model, contig, samples=self.samples)

        return ts

class HomSap_Africa_1b08_simulator(BaseSimulator):
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("Africa_1B08")
    params_default = {
        "n_sample": 10,
        "N_A_true": model.model.events[0].initial_size,
        "N_0_true": model.populations[0].initial_size,
        "t_1_true": model.model.events[0].time,
        "mutation_rate_true": model.mutation_rate,
        "N_A_low": 100,
        "N_A_high": 30_000,
        "N_0_low": 100,
        "N_0_high": 30_000,
        "t_1_low": 10,
        "t_1_high": 10_000,
        "mutation_rate_low": 0,
        "mutation_rate_high": 4.0e-8,
        "contig_length": 10e6,
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, HomSap_Africa_1b08_simulator.params_default)
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
        
        species = HomSap_Africa_1b08_simulator.species
        contig = species.get_contig(length=self.contig_length, mutation_rate=mutation_rate)
        model = HomSap_Africa_1b08_simulator.model
        model.populations[0].initial_size = N_0
        model.model.events[0].initial_size = N_A
        model.model.events[0].time = t_1
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(model, contig, samples={"African_Americans": self.n_sample})

        return ts

class gammaDFE_cnst_N_simulator(BaseSimulator):
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig('chr1', left=0, right=1e6)
    dfe = species.get_dfe("Gamma_K17")
    params_default = {
        "n_sample": 100,
        "mutation_rate_true": contig.mutation_rate,
        "shape_true": dfe.mutation_types[1].distribution_args[1],
        "mean_true": dfe.mutation_types[1].distribution_args[0],
        "p_true": dfe.proportions[1] / sum(dfe.proportions),
        "mutation_rate_low": 1e-8,
        "mutation_rate_high": 3e-8,
        "shape_low": 0.01,
        "shape_high": 1.,
        "mean_low": -0.05,
        "mean_high": 0.,
        "p_low": 0.6,
        "p_high": 0.8,
        "N": 10000
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, gammaDFE_cnst_N_simulator.params_default)
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
        
        species = gammaDFE_cnst_N_simulator.species
        contig = gammaDFE_cnst_N_simulator.contig
        contig.mutation_rate = mutation_rate

        dfe = gammaDFE_cnst_N_simulator.dfe
        dfe.mutation_types[1].distribution_args = [mean, shape]
        dfe.proportions = [1-p, p]
        contig.add_dfe(intervals=np.array([[0, int(contig.length)]]), DFE=dfe)
        model = stdpopsim.PiecewiseConstantSize(self.N)
        engine = stdpopsim.get_engine("slim")

        ts = engine.simulate(model, contig, samples={"pop_0": self.n_sample})

        return ts

class AnaPla_split_migration_simulator(BaseSimulator):
    species = stdpopsim.get_species("AnaPla")
    model = species.get_demographic_model("MallardBlackDuck_2L19")
    params_default = {
        "samples": {"Mallard": 10, "Black_duck": 10}, 
        "N_A_true": model.populations[-1].initial_size, # ancestral population size
        "N_B_true": model.populations[0].initial_size, # Black duck population size
        "N_M_true": model.populations[1].initial_size, # Mallard population size
        "T_true": model.model.events[0].time, # split time
        "m_true": model.model.migration_matrix[0, 1], # migration rate
        "N_A_low": 1e3,
        "N_A_high": 1e7,
        "N_B_low": 1e3,
        "N_B_high": 1e7,
        "N_M_low": 1e3,
        "N_M_high": 1e7,
        "T_low": 1e3,
        "T_high": 1e7,
        "m_low": 0,
        "m_high": 1e-1,
        "contig_length": 1e6
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, AnaPla_split_migration_simulator.params_default)
        self.true_values = {"N_A": self.N_A_true, "N_B": self.N_B_true, "N_M": self.N_M_true, "T": self.T_true, "m": self.m_true}
        self.bounds = {"N_A": (self.N_A_low, self.N_A_high),
                        "N_B": (self.N_B_low, self.N_B_high),
                        "N_M": (self.N_M_low, self.N_M_high),
                        "T": (self.T_low, self.T_high),
                        "m": (self.m_low, self.m_high),
                        }
        low = [self.bounds[p][0] for p in self.bounds.keys()]
        high = [self.bounds[p][1] for p in self.bounds.keys()]
        self.prior = BoxUniform(low=torch.tensor(low), high=torch.tensor(high), device="cuda" if torch.cuda.is_available() else "cpu")
    def __call__(self, theta):
        if type(theta) is torch.Tensor:
            N_A, N_B, N_M, T, m = theta.squeeze().cpu().tolist()
        elif type(theta) is list:
            N_A, N_B, N_M, T, m = theta
        
        model = AnaPla_split_migration_simulator.model
        model.populations[-1].initial_size = N_A
        model.populations[0].initial_size = N_B
        model.populations[1].initial_size = N_M
        model.model.events[0].time = T
        model.model.migration_matrix[0, 1] = m
        model.model.migration_matrix[1, 0] = m
        contig = self.species.get_contig(length=self.contig_length, mutation_rate=model.mutation_rate)
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(model, contig, samples=self.samples)
        return ts

class PonAbe_IM_msprime_simulator(BaseSimulator):
    '''
    simulate split and migration moel of Orangutan
    edited to match dadi simulator (PonAbe_IM_sample and PonAb_IM)
    '''
    species = stdpopsim.get_species("PonAbe")
    model = species.get_demographic_model("TwoSpecies_2L11")
    params_default = {
        "samples": {"Bornean":50,"Sumatran":50}, 
        # choose contig length so that 4 N_anc mu L = 1e4
        "contig_length": int(1e4 / (4 * model.mutation_rate * model.model.events[-1].initial_size)),
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
        "m21_high": 2
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, PonAbe_IM_msprime_simulator.params_default)
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
        if type(theta) is torch.Tensor:
            s, nu1, nu2, T, m12, m21 = theta.squeeze().cpu().tolist()
        elif type(theta) is list:
            s, nu1, nu2, T, m12, m21 = theta

        model = PonAbe_IM_msprime_simulator.model
        model.populations[0].initial_size = nu1 * model.model.events[-1].initial_size
        model.populations[1].initial_size = nu2 * model.model.events[-1].initial_size
        for i in range(len(model.model.events)):
            model.model.events[i].time = T * 2 * model.model.events[-1].initial_size
        r_B = -1 * np.log(s / nu1) / (T * 2 * model.model.events[-1].initial_size)
        r_S = -1 * np.log((1-s) / nu2) / (T * 2 * model.model.events[-1].initial_size)
        model.populations[0].growth_rate = r_B
        model.populations[1].growth_rate = r_S
        model.model.migration_matrix[0, 1] = m12 / (2 * model.model.events[-1].initial_size)
        model.model.migration_matrix[1, 0] = m21 / (2 * model.model.events[-1].initial_size)
        contig = self.species.get_contig(length=self.contig_length, mutation_rate=model.mutation_rate)
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(model, contig, samples=self.samples)
        return ts


MODEL_LIST = {
    "AraTha_2epoch": AraTha_2epoch_simulator,
    "AraTha_2epoch_genetic_map": AraTha_2epoch_genetic_map_simulator,
    "HomSap_ooa_archaic_simulator": HomSap_ooa_archaic_simulator,
    "HomSap_2epoch": HomSap_Africa_1b08_simulator,
    "gammaDFE_cnst_N": gammaDFE_cnst_N_simulator,
    "AnaPla_split_migration": AnaPla_split_migration_simulator,
    "PonAbe_IM_msprime": PonAbe_IM_msprime_simulator
}