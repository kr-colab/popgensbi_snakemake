import dinf
import torch
import numpy as np

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

class three_channel_feature_matrices:
    def __init__(self, snakemake):
        try:
            self.n_snps = snakemake.params.n_snps
        except AttributeError:
            self.n_snps = 500
        try:
            self.maf_thresh = snakemake.params.maf_thresh
        except AttributeError:
            self.maf_thresh = 0.05

    def __call__(self, ts):

        # genotype matrix (gm) shape = (total number of variants, number of individuals * 2 (for dipoid))
            gm = ts.genotype_matrix()
            positions = np.array(ts.tables.sites.position)
            # threshold by MAF
            ac0 = np.sum(gm == 0, axis=1)
            ac1 = np.sum(gm == 1, axis=1)
            keep = np.minimum(ac0, ac1) >= self.maf_thresh * gm.shape[1]
            gm = gm[keep]
            positions = positions[keep]
            # mut_type = 0 for neutral site, 1 for non-neutral site
            # If there are multiple mutations at a site, the last mutation is considered
            mut_types = np.array(
            [int(ts.site(i).mutations[0].metadata['mutation_list'][-1]['mutation_type'] > 1) for i in np.where(keep)[0]]
            )
            # trim or pad with zeros to resize gm to be (n_snps, num_individuals * 2)
            delta = gm.shape[0] - self.n_snps
            first_position = ts.site(0).position

            if delta >= 0:
                left = delta // 2
                right = left + self.n_snps
                gm = gm[left:right]
                positions = positions[left:right]
                delta_positions = np.diff(positions, prepend = first_position)
                mut_types = mut_types[left:right]
            else:
                pad_left = -delta // 2
                pad_right = self.n_snps - gm.shape[0] - pad_left
                gm_left = np.zeros((pad_left, gm.shape[1]), dtype = gm.dtype)
                gm_right = np.zeros((pad_right, gm.shape[1]), dtype = gm.dtype)
                gm = np.concatenate((gm_left, gm, gm_right))

                positions_left = np.zeros(pad_left, dtype = positions.dtype)
                right_pad_value = 0 if len(positions) == 0 else positions[-1]
                positions_right = np.full(pad_right, right_pad_value, dtype = positions.dtype)
                positions = np.concatenate((positions_left, positions, positions_right))
                delta_positions = np.diff(positions, prepend = positions[0])

                # pad mut_types with zeros
                mut_types_left = np.zeros(pad_left)
                mut_types_right = np.zeros(pad_right)
                mut_types = np.concatenate((mut_types_left, mut_types, mut_types_right))
                
            delta_positions = delta_positions / ts.sequence_length 
            # tile positional information vector to match the shape of gm
            # shape of delta_position_matrix = (num_individuals * 2, n_snps)
            delta_positions_matrix = np.tile(delta_positions, [gm.shape[1], 1])

            feature_matrix = np.zeros(shape = (gm.shape[1], self.n_snps, 3))
            feature_matrix[..., 0] = gm.T
            # tile positional information vector to match the shape of gm.T
            feature_matrix[..., 1] = np.tile(delta_positions, [gm.shape[1], 1])
            # same thing for mut_types
            feature_matrix[..., 2] = np.tile(mut_types, [gm.shape[1], 1])

            return torch.from_numpy(feature_matrix).float().permute(2, 0, 1)
