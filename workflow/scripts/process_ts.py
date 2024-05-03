import dinf
import torch

## Todo - Restructure simulate.py and process_ts.py so that ts gets processed after all of the siimulations?
## (That is what happens if we pre-train an encoder to learn summary statistics)
def dinf_extract(ts, n_sample, n_snps, ploidy, phased, maf_thresh):
    '''
    input : tree sequence of one population
    output : genotype matrix + positional encoding (dinf's format)
    '''
    extractor = dinf.feature_extractor.HaplotypeMatrix(
        num_individuals=n_sample, 
        num_loci=n_snps,
        ploidy=ploidy,
        phased=phased,
        maf_thresh=maf_thresh
    )
    feature_matrix = extractor.from_ts(ts)
    return torch.from_numpy(feature_matrix).float().permute(2, 0, 1)