import dinf
import torch

class dinf_extract:
    def __init__(self, snakemake):
        try:
            self.n_snps = snakemake.params.n_snps
        except AttributeError:
            self.n_snps = 500
        try:
            self.ploidy = snakemake.params.ploidy
        except AttributeError:
            self.ploidy = 2
        try:
            self.phased = snakemake.params.phased
        except AttributeError:
            self.phased = False
        try:
            self.maf_thresh = snakemake.params.maf_thresh
        except AttributeError:
            self.maf_thresh = 0.05

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
        return torch.from_numpy(feature_matrix).float().permute(2, 0, 1)