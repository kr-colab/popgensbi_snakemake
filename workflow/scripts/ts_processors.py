import dinf
import tskit
import torch
import numpy as np
import allel
import pandas as pd
from dinf.misc import ts_individuals


class BaseProcessor:
    def __init__(self, config: dict, default: dict):
        for key in config:
            if key == "class_name": continue
            assert key in default, f"Option {key} not available for processor"
        for key, default in default.items():
            setattr(self, key, config.get(key, default))


class genotypes_and_distances(BaseProcessor):
    """
    Genotype matrix and distance to next SNP
    """

    default_config = {
        "max_snps": 2000,
        "min_freq": 0.0,
        "max_freq": 1.0,
        "polarised": True,
        "phased": False,
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
        if not self.polarised: 
            geno[freq > 0.5] = 1 - geno[freq > 0.5]
        if not self.phased:
            individual_map = np.zeros((ts.num_samples, ts.num_individuals))
            for i, ind in enumerate(ts.individuals()):
                individual_map[ind.nodes, i] = 1.0
            geno = geno @ individual_map
        pos = (
            np.append(ts.sites_position[0], np.diff(ts.sites_position))
            / self.position_scaling
        )
        geno = np.concatenate([geno, pos.reshape(ts.num_sites, -1)], axis=-1)
        # filter SNPs
        geno = geno[keep]
        geno = geno[: self.max_snps]
        # the output tensor is (snps, individuals) with no padding,
        # and will be ragged across simulations, with the SNP dimension not
        # exceeding `max_snps`
        return geno


# FIXME: bug if pops have different sizes?
class cnn_extract(BaseProcessor):
    """
    Extract genotype matrices from tree sequences using dinf's feature extractor.
    Handles both single and multiple population cases automatically.
    """

    default_config = {
        "n_snps": 500,  # extract at most this many SNPs
        "phased": False,  # if False use diploid genotypes
        "maf_thresh": 0.05,  # use SNPs with at least this minor allele frequecny
        "polarised": False,  # if False then polarise so that the minor allele is derived
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        if self.polarised: 
            raise ValueError("Polarised features not implemented")

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        # Get population names and corresponding sampled individuals
        pop_names = [pop.metadata["name"] for pop in ts.populations()]
        sampled_pop_names = [
            pop for pop in pop_names if len(ts_individuals(ts, pop)) > 0
        ]
        ploidy = ts.individual(0).nodes.size

        if len(sampled_pop_names) == 1:
            # Single population case
            extractor = dinf.feature_extractor.HaplotypeMatrix(
                num_individuals=ts.num_individuals,
                num_loci=self.n_snps,
                ploidy=ploidy,
                phased=self.phased,
                maf_thresh=self.maf_thresh,
            )
            feature_matrix = extractor.from_ts(ts)
            return np.transpose(feature_matrix, (2, 0, 1))

        else:
            # Multiple populations case
            individuals = {name: ts_individuals(ts, name) for name in sampled_pop_names}
            num_individuals = {
                name: len(individuals[name]) for name in sampled_pop_names
            }

            # Get feature matrices for each population
            extractor = dinf.feature_extractor.MultipleHaplotypeMatrices(
                num_individuals=num_individuals,
                num_loci={pop: self.n_snps for pop in sampled_pop_names},
                ploidy={pop: ploidy for pop in sampled_pop_names},
                global_phased=self.phased,
                global_maf_thresh=self.maf_thresh,
            )
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

            output_mat = torch.stack(
                [v for v in feature_matrices.values()]
            ).permute(0, 3, 1, 2)
            # Output tensor shape:
            # - Multiple pops: (# populations, # channels, # individuals, # snps)
            # - Channels are positions and genotypes
            # - Padded with -1 if # individuals differ
            return output_mat.numpy()


class tskit_sfs(BaseProcessor):
    """
    Site frequency spectrum processor that handles both single and multiple populations.
    For single population: returns normalized SFS
    For multiple populations: returns normalized joint SFS
    """

    default_config = {
        "sample_sets": None,
        "windows": None,
        "mode": "site",
        "span_normalise": False,
        "polarised": False,
        "normalised": True,
        "log1p": False,
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)

    def __call__(self, ts: tskit.TreeSequence) -> torch.Tensor:
        sampled_pops = [
            i for i in range(ts.num_populations) if len(ts.samples(population=i)) > 0
        ]
        sample_sets = [ts.samples(population=i) for i in sampled_pops]
        sfs = ts.allele_frequency_spectrum(
            sample_sets=sample_sets,
            windows=self.windows,
            mode=self.mode,
            span_normalise=self.span_normalise,
            polarised=self.polarised,
        )
        if self.normalised:
            sfs /= np.sum(sfs)
        if self.log1p:
            sfs = np.log1p(sfs)
        return sfs


class SPIDNA_processor(BaseProcessor):

    default_config = {
        "maf": 0.05, 
        "relative_position": True, 
        "n_snps": 400,
        "polarised": True,
        "phased": True,
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        # Extract genotype matrix and positions
        snp = ts.genotype_matrix()  # Shape: (n_variants, n_samples)
        pos = ts.sites_position

        # Handle relative positions
        if self.relative_position:
            abs_pos = np.array(pos)
            pos = abs_pos - np.concatenate(([0], abs_pos[:-1]))
        pos = pos / ts.sequence_length  # Normalize positions to [0, 1] range

        # MAF filtering
        if self.maf != 0:
            num_sample = ts.num_samples
            row_sum = np.sum(
                snp, axis=1
            )  # Sum along rows since matrix isn't transposed
            keep = np.logical_and.reduce(
                [
                    row_sum != 0,
                    row_sum != num_sample,
                    row_sum > num_sample * self.maf,
                    num_sample - row_sum > num_sample * self.maf,
                ]
            )
            snp = snp[keep]
            pos = pos[keep]

        if not self.polarised: 
            freq = snp.sum(axis=1) / snp.shape[1]
            snp[freq > 0.5] = 1 - snp[freq > 0.5]

        if not self.phased:
            individual_map = np.zeros((ts.num_samples, ts.num_individuals))
            for i, ind in enumerate(ts.individuals()):
                individual_map[ind.nodes, i] = 1.0
            snp = snp @ individual_map

        # Handle case where we have fewer than n_snps SNPs
        n_samples = snp.shape[1]
        n_snps = snp.shape[0]
        if n_snps < self.n_snps:
            # Pad with -1 to reach n_snps SNPs (consistent with cnn_extract padding)
            snp_padded = np.full((self.n_snps, n_samples), -1, dtype=snp.dtype)
            snp_padded[:n_snps, :] = snp[:, :n_samples]
            snp = snp_padded

            # Pad positions with -1 to indicate padding
            pos_padded = np.full(self.n_snps, -1, dtype=pos.dtype)
            pos_padded[:n_snps] = pos
            pos = pos_padded
        else:
            # We have enough SNPs, just take the first n_snps
            snp = snp[:self.n_snps, :n_samples]
            pos = pos[:self.n_snps]

        # Create output tensor matching legacy format
        # First create position channel (1, n_snps)
        pos_channel = pos.reshape(1, -1)

        # Stack channels
        output_val = np.concatenate(
            [
                pos_channel,  # Shape: (1, n_snps)
                snp.T,  # Now transpose only at the end to match expected output format
            ]
        )

        return output_val.astype(np.float32)


class ReLERNN_processor(BaseProcessor):

    default_config = {
        "n_snps": 2000,
        "min_freq": 0.0,
        "max_freq": 1.0,
        "phased": True,
        "polarised": True,
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)
        if not self.phased: 
            raise ValueError("Unphased features not implemented")

    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        geno = ts.genotype_matrix()
        freq = geno.sum(axis=1) / geno.shape[1]
        keep = np.logical_and(
            freq >= self.min_freq,
            freq <= self.max_freq,
        )
        if not self.polarised:
            geno[freq > 0.5] = 1 - geno[freq > 0.5]
        geno = geno * 2 - 1  # recode ancestral to -1, derived to 1
        pos = ts.sites_position / ts.sequence_length
        geno = np.concatenate([pos.reshape(ts.num_sites, -1), geno], axis=-1)
        # filter SNPs
        geno = geno[keep]
        geno = geno[:self.n_snps]
        # Pad with zeros if the number of rows is less than max_snps
        if geno.shape[0] < self.n_snps:
            pad_rows = self.n_snps - geno.shape[0]
            geno = np.pad(
                geno, ((0, pad_rows), (0, 0)), mode="constant", constant_values=0
            )

        return geno


class tskit_windowed_sfs_plus_ld(BaseProcessor):
    """
    Summary statistics processor that returns a vector of the mean r2 across distances and the mean afs
    where the mean is taken over windows.

    Mean currently only for the single population case.
    """

    default_config = {
        "sample_sets": None,
        "mode": "site",
        "span_normalise": False,
        "polarised": True,
        "window_size": 1_000_000,
    }

    def __init__(self, config: dict):
        super().__init__(config, self.default_config)

    # TODO: this was copied from Yuxin's code, need to clean it up.
    # not sure if we need the circular option at all?
    def _LD(self, haplotype, pos_vec, size_chr, circular=True, distance_bins=None):
        """
        Compute LD on a subsampled set of SNP pairs and return a DataFrame containing the mean r^2
        per distance bin.

        Gap sizes follow powers of 2. For each gap, SNP pairs are sampled with a random phase shift, 
        then LD is computed and squared. Results are binned by physical distance and averaged.

        Parameters
        ----------
        haplotype : array(n_SNP, n_samples)
        pos_vec : array(n_SNP,)
            Absolute genomic positions in [0, size_chr].
        size_chr : int
            Chromosome length.
        circular : bool
            Whether to consider the chromosome circular. If circular, the maximum distance between
            two SNPs is half the chromosome. (Currently not used in distance computation.)
        distance_bins : int or sequence of numbers, optional
            If an int k is given, LD is averaged in k-1 logarithmic bins between 10^2 and size_chr,
            with 0 inserted as the first edge. If a sequence is given, those values are used as
            the bin edges directly. If None, 19 log-spaced bins between 10^2 and size_chr are used,
            with 0 inserted as the first edge.

        Returns
        -------
        pandas.DataFrame
            Index: pandas.IntervalIndex of distance bins. Columns: 'mean_r2' containing the
            mean of r^2 within each distance bin.

        Notes
        -----
        - Subsampling is stochastic due to random shifts when forming SNP pairs.
        - LD per pair is computed as allel.rogers_huff_r(...)**2.
        """

        # Set up distance bins if not provided (kept here for potential grouping)
        if distance_bins is None or isinstance(distance_bins, int):
            if isinstance(distance_bins, int):
                n_bins = distance_bins - 1
            else:
                n_bins = 19
            distance_bins = np.logspace(2, np.log10(size_chr), n_bins)
            distance_bins = np.insert(distance_bins, 0, [0])

        n_SNP, n_samples = haplotype.shape
        gaps = (2 ** np.arange(0, np.log2(n_SNP), 1)).astype(int)

        selected_snps = []
        for gap in gaps:
            snps = np.arange(0, n_SNP, gap) + np.random.randint(0, (n_SNP - 1) % gap + 1)
            snp_pairs = np.unique([((snps[i] + i) % n_SNP, (snps[i + 1] + i) % n_SNP) for i in range(len(snps) - 1)], axis=0)

            snp_pairs = snp_pairs[snp_pairs[:, 0] < snp_pairs[:, 1]]
            last_pair = snp_pairs[-1]
            max_value = n_SNP - gap - 1

            while len(snp_pairs) <= min(300, max_value):
                random_shift = np.random.randint(1, n_SNP) % n_SNP
                new_pair = (last_pair + random_shift) % n_SNP
                snp_pairs = np.unique(
                    np.concatenate([snp_pairs, new_pair.reshape(1, 2)]), axis=0
                )
                last_pair = new_pair
                snp_pairs = snp_pairs[snp_pairs[:, 0] < snp_pairs[:, 1]]
            selected_snps.append(snp_pairs)

        # Collect r2 values into a DataFrame.
        agg_bins = {"snp_dist": ["mean"], "r2": ["mean"]}

        ld = pd.DataFrame()
        for i, snps_pos in enumerate(selected_snps):
            sd = pd.DataFrame((np.diff(pos_vec[snps_pos])), columns=["snp_dist"])
            sd["dist_group"] = pd.cut(sd.snp_dist, bins=distance_bins)
            sr = [allel.rogers_huff_r(snps) ** 2 for snps in haplotype[snps_pos]]
            sd["r2"] = sr
            sd["gap_id"] = i
            ld = pd.concat([ld, sd])

        ld2 = ld.dropna().groupby("dist_group",observed=False).agg(agg_bins)

        # Flatten the MultiIndex columns and rename explicitly
        ld2.columns = ['_'.join(col).strip() for col in ld2.columns.values]
        ld2 = ld2.rename(columns={
            'snp_dist_mean': 'mean_dist',
            'r2_mean': 'mean_r2'
        })
        return ld2[['mean_r2']]


    def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
        # Get number of populations with samples
        sampled_pops = [
            i for i in range(ts.num_populations) if len(ts.samples(population=i)) > 0
        ]
        assert len(sampled_pops) == 1, "Only single population case currently supported"
        sequence_length = ts.sequence_length
        windows = np.arange(0, sequence_length, self.window_size)
        # iterate over windows such that the last window is the remainder
        # start at 0, end at sequence_length, step by 1_000_000
        ld_stats = []
        afs_stats = []
        for i in range(len(windows)):
            if i == len(windows) - 1:
                window_end = sequence_length
            else:
                window_end = windows[i + 1]
            window_start = windows[i]
            ts_win = ts.keep_intervals([(window_start, window_end)]).trim()
            ts_win_positions = ts_win.tables.sites.position
            # calculate LD stats for each window
            ld_stats.append(
                self._LD(
                    ts_win.genotype_matrix(),
                    ts_win_positions,
                    window_end - window_start,
                )
            )

            # get the AFS for each window
            afs = ts_win.allele_frequency_spectrum(
                mode="site",
                polarised=self.polarised,
                span_normalise=self.span_normalise,
            )
            # normalize the AFS
            # afs = afs / np.sum(afs)
            afs_stats.append(afs)

        # calculate the mean r2 for each window at each distance
        mean_r2_values = (
            pd.concat(ld_stats)["mean_r2"]
            .groupby(level=0, observed=False)
            .mean()  # Automatically skips NaNs by default
            .fillna(0)  # Replace any remaining NaNs with 0
        )

        # calculate the mean afs for each window
        mean_afs_values = np.stack(afs_stats).mean(axis=0)[1:-1]
        sum_stats = np.concatenate((mean_r2_values, mean_afs_values))
        return sum_stats
