Processors
==========

Processors transform tree sequences from simulators into tensor representations suitable for neural network training. They serve as the bridge between population genetic simulations and machine learning models.

Overview
--------

The processor pipeline:

1. **Tree Sequence Input**: Receives tree sequences from simulators
2. **Feature Extraction**: Converts genetic data into numerical tensors
3. **Preprocessing**: Applies filtering, normalization, and formatting
4. **Output**: Returns tensors ready for embedding networks

Each processor is designed to work with specific embedding network architectures, ensuring compatible tensor shapes and data representations.

Processor Types
---------------

Genotype-based Processors
~~~~~~~~~~~~~~~~~~~~~~~~~

These processors extract genotype matrices and related features directly from tree sequences.

**genotypes_and_distances**
   Simple processor that extracts genotype matrices with inter-SNP distances.
   
   - Filters SNPs by allele frequency
   - Optionally phases/unphases genotypes
   - Adds scaled positional information
   - Output shape: (n_snps, n_individuals + 1)

**cnn_extract**
   Sophisticated processor using dinf's HaplotypeMatrix for CNN-compatible features.
   
   - Handles single and multiple populations
   - Creates position and genotype channels
   - Pads populations to equal sizes when needed
   - Output shape varies by population structure:
     - Single pop: (2, n_individuals, n_snps)
     - Multiple pops: (n_pops, 2, n_individuals, n_snps)

Summary Statistics Processors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These processors compute population genetic summary statistics.

**tskit_sfs**
   Computes site frequency spectra (SFS) for single or joint populations.
   
   - Supports normalized and unnormalized SFS
   - Optional log transformation
   - Handles both folded and unfolded spectra
   - Output: 1D array (single pop) or multi-dimensional (joint SFS)

**tskit_windowed_sfs_plus_ld**
   Advanced processor combining SFS with linkage disequilibrium (LD) statistics.
   
   - Computes mean r² across distance bins
   - Calculates windowed SFS
   - Aggregates statistics across genomic windows
   - Currently supports single population only

Network-specific Processors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These processors format data for specific embedding network architectures.

**SPIDNA_processor**
   Formats data specifically for SPIDNA embedding networks.
   
   - Creates position channel and SNP channels
   - Applies MAF filtering
   - Fixed output: 400 SNPs × 20 samples
   - Output shape: (n_samples + 1, 400)

**ReLERNN_processor**
   Formats data for ReLERNN architecture.
   
   - Requires phased genotypes
   - Recodes alleles to -1/1
   - Normalizes positions to [0,1]
   - Pads to fixed SNP count
   - Output shape: (n_snps, n_samples + 1)

Configuration
-------------

Processors are configured in the workflow YAML files:

.. code-block:: yaml

   processor:
     class_name: cnn_extract
     parameters:
       n_snps: 500
       ploidy: 2
       phased: False
       maf_thresh: 0.05

Common Parameters
~~~~~~~~~~~~~~~~~

**Genotype processors:**

- ``n_snps`` or ``max_snps``: Maximum number of SNPs to retain
- ``phased``: Whether to maintain phase information
- ``min_freq`` / ``max_freq``: Allele frequency filters
- ``maf_thresh``: Minor allele frequency threshold

**Summary statistics processors:**

- ``normalised``: Whether to normalize statistics
- ``polarised``: Use ancestral state information
- ``window_size``: Size of genomic windows
- ``mode``: "site" or "branch" mode for statistics

Processor-Network Compatibility
-------------------------------

Each processor is designed to work with specific embedding networks:

.. list-table:: Processor-Network Compatibility
   :header-rows: 1
   :widths: 40 60

   * - Processor
     - Compatible Networks
   * - genotypes_and_distances
     - RNN, generic MLPs
   * - cnn_extract
     - ExchangeableCNN
   * - tskit_sfs
     - SummaryStatisticsEmbedding
   * - tskit_windowed_sfs_plus_ld
     - SummaryStatisticsEmbedding
   * - SPIDNA_processor
     - SPIDNA_embedding_network
   * - ReLERNN_processor
     - ReLERNN

Usage Examples
--------------

Single Population CNN
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   processor:
     class_name: cnn_extract
     parameters:
       n_snps: 1000
       ploidy: 2
       phased: True
       maf_thresh: 0.01

Multiple Population CNN
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   processor:
     class_name: cnn_extract
     parameters:
       n_snps: 500  # Per population
       ploidy: 2
       phased: False
       maf_thresh: 0.05

Summary Statistics
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   processor:
     class_name: tskit_sfs
     parameters:
       normalised: True
       polarised: False
       log1p: True  # Log transform

Custom Processors
-----------------

To create a custom processor:

1. Inherit from ``BaseProcessor``
2. Define ``default_config`` with parameters
3. Implement ``__call__(self, ts)`` to return a tensor

Example:

.. code-block:: python

   class MyProcessor(BaseProcessor):
       default_config = {
           "my_param": 42,
           "filter_singletons": True
       }
       
       def __init__(self, config: dict):
           super().__init__(config, self.default_config)
       
       def __call__(self, ts: tskit.TreeSequence) -> np.ndarray:
           # Extract features from tree sequence
           features = self.extract_features(ts)
           
           # Apply preprocessing
           if self.filter_singletons:
               features = self.filter(features)
           
           return features

Technical Notes
---------------

- All processors return numpy arrays or torch tensors
- Output shapes must be consistent for batching
- Variable-length outputs are padded with -1 or 0
- Processors handle both haploid and diploid data
- Population structure is preserved in multi-population processors
- MAF filtering is applied before size limits