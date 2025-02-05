import dinf
import tskit
import torch
import numpy as np

from dinf.misc import ts_individuals


class BaseProcessor:
    def __init__(self, config: dict, default: dict):
        for key, default in default.items():
            setattr(self, key, config.get(key, default))


# TODO: something seems to be going wrong here if pops have different sizes
# Not convinced the padding is working as intended
class dinf_multiple_pops(BaseProcessor):

    default_config = {
        "n_snps": 500,
        "ploidy": 2,
        "phased": False,
        "maf_thresh": 0.05
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:        
        # Recover population names and corresponding sampled individuals
        pop_names = [pop.metadata['name'] for pop in ts.populations()]

        # Make a dictionary with sampled individuals for each population except for unsampled ones
        sampled_pop_names = [pop for pop in pop_names if len(ts_individuals(ts, pop)) > 0]
        individuals = {name: ts_individuals(ts, name) for name in sampled_pop_names}
        num_individuals = {name: len(individuals[name]) for name in sampled_pop_names}

        extractor = dinf.feature_extractor.MultipleHaplotypeMatrices(
            num_individuals=num_individuals, 
            num_loci={pop: self.n_snps for pop in sampled_pop_names},
            ploidy={pop: self.ploidy for pop in sampled_pop_names},
            global_phased=self.phased,
            global_maf_thresh=self.maf_thresh
        )

        # Get a dictionary of feature matrices, one for each population
        feature_matrices = extractor.from_ts(ts, individuals=individuals)

        # Each feature matrix has dimension of (num_individual, num_loci, 2). 
        # because num_individual can be different for each population, pad them with -1
        max_num_individuals = max([num_individuals[pop] for pop in sampled_pop_names])

        for pop in individuals.keys():
            feature_matrices[pop] = torch.from_numpy(feature_matrices[pop])
            num_individuals = feature_matrices[pop].shape[0]
            if num_individuals < max_num_individuals:
                feature_matrices[pop] = torch.nn.functional.pad(
                    feature_matrices[pop], 
                    (0, 0, 0, 0, 0, max_num_individuals - num_individuals), 
                    "constant", 
                    -1,
                )

        output_mat = torch.stack([v for v in feature_matrices.values()]).permute(0, 3, 1, 2)
        # the output tensor is (# populations, # channels, # individuals, # snps)
        # where the number of channels is 2 (positions and genotypes)
        # and -1 is padded on the bottom and right if # snps or # individuals differ
        return output_mat.numpy()


class genotypes_and_distances(BaseProcessor):

    default_config = {
        "max_snps": 2000,
        "phased": False,
        "min_freq": 0.0,
        "max_freq": 1.0,
        "position_scaling": 1e3,
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        geno = ts.genotype_matrix()
        freq = geno.sum(axis=1) / geno.shape[1] 
        keep = np.logical_and(
            freq >= self.min_freq,
            freq <= self.max_freq,
        )
        if not self.phased:
            diploid_map = np.zeros((ts.num_samples, ts.num_individuals))
            for i, ind in enumerate(ts.individuals()):
                diploid_map[ind.nodes, i] = 1.0
            geno = geno @ diploid_map
        pos = np.append(ts.sites_position[0], np.diff(ts.sites_position)) / self.position_scaling
        geno = np.concatenate([geno, pos.reshape(ts.num_sites, -1)], axis=-1)
        # filter SNPs
        geno = geno[keep]
        geno = geno[:self.max_snps]
        # the output tensor is (snps, individuals) with no padding,
        # and will be ragged across simulations, with the SNP dimension not
        # exceeding `max_snps`
        return geno

class dinf_extract(BaseProcessor):
    """
    Extract a genotype matrix from a tree sequence using dinf's feature extractor
    """
    params_default = {
        "n_snps": 500,
        "ploidy": 2,
        "phased": False,
        "maf_thresh": 0.05
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, dinf_extract.params_default)

    def __call__(self, ts):        
        '''
        input : tree sequence of one population
        output : genotype matrix + positional encoding (dinf's format)
        '''
        if self.ploidy == 2:
            if self.phased == False:
                n_sample = int(ts.num_samples / self.ploidy)
            else:
                n_sample = ts.num_samples
        elif self.ploidy == 1:
            n_sample = ts.num_samples
            
        extractor = dinf.feature_extractor.HaplotypeMatrix(
            num_individuals=n_sample, 
            num_loci=self.n_snps,
            ploidy=self.ploidy,
            phased=self.phased,
            maf_thresh=self.maf_thresh
        )
        feature_matrix = extractor.from_ts(ts)
        return np.transpose(feature_matrix, (2, 0, 1))