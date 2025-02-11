import dinf
import tskit
import torch
import numpy as np
import moments
from moments.LD import Parsing
from moments.LD.Parsing import compute_ld_statistics
from dinf.misc import ts_individuals
import os
from functools import partial
import ray


class BaseProcessor:
    def __init__(self, config: dict, default: dict):
        for key, value in default.items():
            setattr(self, key, config.get(key, value))

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

# TODO: something seems to be going wrong here if pops have different sizes
# Not convinced the padding is working as intended
class cnn_extract(BaseProcessor):
    """
    Extract genotype matrices from tree sequences using dinf's feature extractor.
    Handles both single and multiple population cases automatically.
    """
    default_config = {
        "n_snps": 500,
        "ploidy": 2,
        "phased": False,
        "maf_thresh": 0.05
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        # Get population names and corresponding sampled individuals
        pop_names = [pop.metadata['name'] for pop in ts.populations()]
        sampled_pop_names = [pop for pop in pop_names if len(ts_individuals(ts, pop)) > 0]
        
        if len(sampled_pop_names) == 1:
            # Single population case
            if self.ploidy == 2:
                if not self.phased:
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
            
        else:
            # Multiple populations case
            individuals = {name: ts_individuals(ts, name) for name in sampled_pop_names}
            num_individuals = {name: len(individuals[name]) for name in sampled_pop_names}

            extractor = dinf.feature_extractor.MultipleHaplotypeMatrices(
                num_individuals=num_individuals,
                num_loci={pop: self.n_snps for pop in sampled_pop_names},
                ploidy={pop: self.ploidy for pop in sampled_pop_names},
                global_phased=self.phased,
                global_maf_thresh=self.maf_thresh
            )

            # Get feature matrices for each population
            feature_matrices = extractor.from_ts(ts, individuals=individuals)

            # Pad matrices if populations have different sizes
            max_num_individuals = max(num_individuals.values())
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
            # Output tensor shape:
            # - Multiple pops: (# populations, # channels, # individuals, # snps)
            # - Channels are positions and genotypes
            # - Padded with -1 if # individuals differ
            return output_mat.numpy()
        
class tskit_sfs(BaseProcessor):
    '''
    Site frequency spectrum processor that handles both single and multiple populations.
    For single population: returns normalized SFS
    For multiple populations: returns normalized joint SFS
    '''
    default_config = {
        "sample_sets": None,
        "windows": None,
        "mode": "site",
        "span_normalise": False,
        "polarised": False
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)

    def __call__(self, ts: tskit.TreeSequence) -> torch.Tensor:
        # Get number of populations with samples
        sampled_pops = [i for i in range(ts.num_populations) if len(ts.samples(population=i)) > 0]

        if len(sampled_pops) == 1:
            # Single population case
            sfs = ts.allele_frequency_spectrum(
                sample_sets=[ts.samples(population=sampled_pops[0])],
                windows=self.windows,
                mode=self.mode,
                span_normalise=self.span_normalise,
                polarised=self.polarised
            )
            sfs = sfs / np.sum(sfs)
        else:
            # Multiple populations case
            sample_sets = [ts.samples(population=i) for i in sampled_pops]
            sfs = ts.allele_frequency_spectrum(
                sample_sets=sample_sets,
                windows=self.windows,
                mode=self.mode,
                span_normalise=self.span_normalise,
                polarised=self.polarised
            )
            sfs = sfs / np.sum(sfs)
        
        return sfs

class moments_LD_stats(BaseProcessor):
    '''
    Processor that computes LD statistics using moments.LD.Parsing.compute_ld_statistics.
    This processor extracts segments from the tree sequence, writes temporary VCF files,
    computes LD statistics using the provided b_bins, and returns a flattened vector
    of stacked LD statistics.
    '''
    default_config = {
        "pop_file": "pop_file.txt",
        "pops": None,
        "bp_bins": [1e2, 1e3],
        "n_segs": 1,
        "use_genotypes": True,  # False for phased data
        "ts_processor": os.path.join("output_directory", "moments_ld"),  # Specify a subdirectory
    }
    
    def __init__(self, config: dict):
        try:
            import snakemake
            self.datadir = snakemake.config.get('project_dir', '.')
        except (ImportError, AttributeError):
            self.datadir = '.'
            
        config['datadir'] = self.datadir
        super().__init__(config, self.default_config)
        self.randn = np.random.randint(0, 9999999)
        
        # Initialize ray once during initialization
        if not ray.is_initialized():
            num_cpus = 10
            ray.init(num_cpus=num_cpus)
    
    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        # Create output directory for VCF files if it doesn't exist
        vcf_dir = os.path.join(self.datadir, self.ts_processor)
        os.makedirs(vcf_dir, exist_ok=True)

        # Define remote function for processing segments
        @ray.remote
        def process_segment(seg_idx, datadir, ts_processor, ts, seg_len, randn, 
                           pops, bp_bins):
            # Include the function definition here instead of importing
            def get_ld_stats(seg_idx, datadir, ts_processor, ts, seg_len, randn, 
                            pops, bp_bins):
                # Create output directory in the worker
                vcf_dir = os.path.join(datadir, ts_processor)
                os.makedirs(vcf_dir, exist_ok=True)
                
                vcf_name = os.path.join(vcf_dir, f"{randn}_seg_{seg_idx}.vcf")
                site_mask = np.ones(ts.num_sites, dtype=bool)
                # Unmask only those sites within the segment boundaries
                for site in ts.sites():
                    if seg_idx * seg_len < site.position < (seg_idx + 1) * seg_len:
                        site_mask[site.id] = 0
                
                # Write VCF file
                with open(vcf_name, "w+") as fout:
                    ts.write_vcf(fout, site_mask=site_mask)
                
                # Read VCF header to get sample names
                with open(vcf_name, 'r') as f:
                    for line in f:
                        if line.startswith('#CHROM'):
                            header = line.strip().split('\t')
                            sample_names = header[9:]  # Get sample columns (after FORMAT)
                            break
                
                # Create the population file in the same directory as the VCF
                pop_file_path = os.path.join(vcf_dir, "pop_file.txt")
                if not os.path.exists(pop_file_path):
                    with open(pop_file_path, "w") as f:
                        f.write("sample pop\n")  # Write header
                        for sample in sample_names:
                            # Get population ID from the sample index
                            sample_idx = int(sample.replace('tsk_', ''))
                            pop_id = ts.node(sample_idx).population
                            pop_obj = ts.population(pop_id)
                            if pop_obj.metadata is not None and "name" in pop_obj.metadata:
                                pop_name = pop_obj.metadata["name"]
                            else:
                                pop_name = f"pop{pop_id}"
                            f.write(f"{sample} {pop_name}\n")

                os.system(f"gzip {vcf_name}")
                ld_stat = moments.LD.Parsing.compute_ld_statistics(
                    f"{vcf_name}.gz",
                    pop_file=pop_file_path,
                    pops=pops,
                    bp_bins=bp_bins,
                    report=True
                )
                os.system(f"rm {vcf_name}.gz")
                os.system(f"rm {vcf_name[:-3]}h5")
                return ld_stat

            return get_ld_stats(seg_idx, datadir, ts_processor, ts, seg_len, randn,
                              pops, bp_bins)

        # Get the length of each segment
        seg_len = int(ts.sequence_length / self.n_segs)
        
        # Create remote tasks
        futures = []
        for seg_idx in range(self.n_segs):
            # output logging as to which segment is being processed
            print(f"Processing segment {seg_idx} of {self.n_segs}")
            futures.append(process_segment.remote(
                seg_idx,
                self.datadir,
                self.ts_processor,
                ts,
                seg_len,
                self.randn,
                self.pops,
                self.bp_bins
            ))

        # Get results
        ld_stats = ray.get(futures)
        
        # Convert to dictionary format
        ld_stats_dict = {i: stat for i, stat in enumerate(ld_stats)}
        
        # Extract stats and compute means from the collected segment data.
        stats = ld_stats_dict[0]["stats"]
        means = moments.LD.Parsing.means_from_region_data(ld_stats_dict, stats)
        
        output = []
        # Stack means of D2, Dz and pi2.
        for i in range(len(means) - 1):
            output.append(means[i])
        
        rep = len(output[0])
        # Append a replicate of H.
        for i in range(len(means[-1])):
            output.append(np.repeat(means[-1][i], rep))
        
        output = np.stack(output)
        # Return flattened output.
        return output.flatten()

    def __del__(self):
        if ray.is_initialized():
            ray.shutdown()
