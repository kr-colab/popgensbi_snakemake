import dinf
import torch

# Change this to class
# get n_sample from ts.num_sample, ploidy =2, phased = False
# n_snps and maf_thresh are from config. (snakemake.params.)
class dinf_extract:
    def __init__(self, n_snps=500,
                ploidy=2,
                phased=False,
                maf_thresh=0.05):
        self.n_snps = n_snps
        self.ploidy = ploidy
        self.phased = phased
        self.maf_thresh = maf_thresh
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