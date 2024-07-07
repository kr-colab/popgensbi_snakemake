import dinf
import torch
import numpy as np

class BaseProcessor:
    def __init__(self, snakemake, params_default):
        for key, default in params_default.items():
            if key in snakemake.params.keys():
                setattr(self, key, snakemake.params[key])
            else:
                setattr(self, key, default) 
        

class dinf_extract(BaseProcessor):
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
        return torch.from_numpy(feature_matrix).float().permute(2, 0, 1)

class dinf_extract_multiple_pops(BaseProcessor):
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
        from dinf.misc import ts_individuals
        # recover population names and corresponding sampled individuals
        pop_names = [pop.metadata['name'] for pop in ts.populations()]
        # make a dictionary with sampled individuals for each population except for unsampled ones
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
        # we get a dictionary of feature matrices, one for each population
        feature_matrices = extractor.from_ts(ts, individuals=individuals)

        # Each feature matrix has dimension of (num_individual, num_loci, 2). 
        # because num_individual can be different for each population, we need to pad them with -1
        max_num_individuals = max([num_individuals[pop] for pop in sampled_pop_names])

        for pop in individuals.keys():
            feature_matrices[pop] = torch.from_numpy(feature_matrices[pop])
            num_individuals = feature_matrices[pop].shape[0]
            if num_individuals < max_num_individuals:
                feature_matrices[pop] = torch.nn.functional.pad(feature_matrices[pop], (0, 0, 0, 0, 0, max_num_individuals - num_individuals), "constant", -1)

        output_mat = torch.stack([v for v in feature_matrices.values()]).permute(0, 3, 1, 2)
        return output_mat


class three_channel_feature_matrices(BaseProcessor):
    params_default = {
        "n_snps": 500,
        "maf_thresh": 0.05
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, three_channel_feature_matrices.params_default)

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


class tskit_sfs(BaseProcessor):
    '''
    Normalized sfs (sum to 1)
    '''
    params_default = {
        "sample_sets": None,
        "windows": None,
        "mode": "site",
        "span_normalise": False,
        "polarised": False
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, tskit_sfs.params_default)
    def __call__(self, ts):
        sfs = ts.allele_frequency_spectrum(
            sample_sets = self.sample_sets, 
            windows = self.windows, 
            mode = self.mode, 
            span_normalise = self.span_normalise, 
            polarised = self.polarised)
        sfs = sfs / sum(sfs)
        return torch.from_numpy(sfs).float()

class tskit_jsfs(BaseProcessor):
    '''
    Joint site frequency spectrum for two-population model
    Normalized sfs (sum to 1)
    '''
    params_default = {
        "windows": None,
        "mode": "site",
        "span_normalise": False,
        "polarised": False
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, tskit_sfs.params_default)
    def __call__(self, ts):
        sfs = ts.allele_frequency_spectrum(
            sample_sets = [ts.samples(population=0), ts.samples(population=1)], 
            windows = self.windows, 
            mode = self.mode, 
            span_normalise = self.span_normalise, 
            polarised = self.polarised)
        sfs = sfs / sum(sum(sfs))
        return torch.from_numpy(sfs).float()




class tskit_sfs_selection(BaseProcessor):
    ## get SFS with synnonymous and non-synonymous mutations separately and append the two arrays to get a single array
    params_default = {
        "span_normalise": True,
        "polarised": False
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, tskit_sfs_selection.params_default)
    def __call__(self, ts):
        nonsyn_counts = []
        syn_counts = []
        for var in ts.variants():
            count = sum(var.genotypes)
            # If there are multiple mutations at a site, the last mutation is considered
            if var.site.mutations[-1].metadata['mutation_list'][-1]['mutation_type']==2:
                if self.polarised:
                    nonsyn_counts.append(min(count, ts.num_samples - count))
                else:
                    nonsyn_counts.append(count)
            else:
                if self.polarised:
                    syn_counts.append(min(count, ts.num_samples - count))
                else:
                    syn_counts.append(count)
        
        # compute span normalized SFS (length = ts.num_samples + 1 like in tskit)
        nonsyn_sfs = np.histogram(nonsyn_counts, bins = np.arange(-0.5, ts.num_samples + 1.5))[0]
        syn_sfs = np.histogram(syn_counts, bins = np.arange(-0.5, ts.num_samples + 1.5))[0]

        if self.span_normalise:
            nonsyn_sfs = nonsyn_sfs / ts.sequence_length
            syn_sfs = syn_sfs / ts.sequence_length
        
        sfs_combined = np.append(nonsyn_sfs, syn_sfs)
        return torch.from_numpy(sfs_combined).float()


class moments_LD_stats(BaseProcessor):
    '''
    create a matrix of recombination bin edges, LD statistics from moments from ts
    '''
    params_default = {
        "n_bins": 10,
    }
    def __init__(self, snakemake):
        super().__init__(snakemake, moments_LD_stats.params_default)
    def __call__(self, ts):
        import moments
        import gzip
        import os 
        datadir = self.datadir
        rounds = self.rounds
        num_simulations = self.num_simulations
        vcf_name = f"{datadir}/round_{rounds}/{num_simulations}.vcf"
        with open(vcf_name, "w+") as fout:
            ts.write_vcf(fout)
        os.system(f"gzip {vcf_name}")
        vcf_file = f"{vcf_name}.gz"
        map_file = f"{datadir}/round_{rounds}/{num_simulations}.map.txt"
        Map_cm = ts.sequence_length * contig.recombination_map.rate[0] * 100
        with open(map_file, "w+") as fout:
            fout.write("pos\tMap(cM)\n")
            fout.write("0\t0\n")
            fout.write(f"{int(ts.sequence_length)}\t{Map_cm}\n")
        r_bins = np.logspace(round(np.log10(contig.recombination_map.rate[0]))+1, 
                        round(np.log10(contig.recombination_map.rate[0] * contig.length))-1,
                        self.n_bins)
        if ts.num_populations > 1:
            pops_rep = []
            for n in range(ts.num_populations):
                for i in range(int(len(ts.samples(n))/2)):
                    pops_rep.append(ts.population(n).metadata['name'])
            pop_file = f"{datadir}/round_{rounds}/{num_simulations}.map.txt"                    
            with open(pop_file, "w+") as fout:
                fout.write("sample\tpop\n")
                for j, pop in enumerate(pops_rep):
                    fout.write(f"tsk_{j}\t{pop}\n")
            pops = [ts.population(n).metadata['name'] for n in range(ts.num_populations)]
            ld_stats = moments.LD.Parsing.comput_ld_statistics(
                vcf_file,
                rec_map_file = map_file,
                pop_file = pop_file,
                pops=pops,
                r_bins=r_bins,
                report=False
            )
        else:
            ld_stats = moments.LD.Parsing.compute_ld_statistics(
                vcf_file,
                rec_map_file = map_file,
                r_bins=r_bins,
                report=False
            )
        
        means = moments.LD.Parsing.means_from_region_data({0:ld_stats}, ld_stats['stats'], norm_idx=0)
        ld_stats_mat = np.zeros((len(means[0])+len(means[-1])+1, len(means)-1))
        # first row is edges of recombination bins
        ld_stats_mat[0] = 0.5 * (r_bins[1:] + r_bins[:-1])
        # next rows are LD statistics
        for i in range(len(means[0])):
            ld_stats_mat[i+1] = [means[j][i] for j in range(len(means)-1)]
        # final rows are copies of H
        for i in range(len(means[-1])):
            ld_stats_mat[-i-1] = np.ones(len(means)-1) * means[-1][-i-1]
        # flatten ld_stats_mat
        ld_stats_mat = ld_stats_mat.flatten()
        return torch.from_numpy(ld_stats_mat).float()

PROCESSOR_LIST = {
    "dinf": dinf_extract,
    "dinf_multiple_pops": dinf_extract_multiple_pops,
    "three_channel_feature_matrices": three_channel_feature_matrices,
    "tskit_sfs": tskit_sfs,
    "tskit_jsfs": tskit_jsfs,
    "tskit_sfs_selection": tskit_sfs_selection, 
    "moments_LD_stats": moments_LD_stats
}