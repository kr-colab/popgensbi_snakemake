API Reference
=================

This section provides detailed API documentation for the popgensbi package modules.

Tree Sequence Processors
------------------------

The ``ts_processors`` module transforms tree sequences into tensor representations for neural networks.

.. currentmodule:: workflow.scripts.ts_processors

BaseProcessor
~~~~~~~~~~~~~

.. autoclass:: BaseProcessor
   :members:
   :undoc-members:
   :show-inheritance:

   Base class for all processors. Handles configuration and default parameters.

genotypes_and_distances
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: genotypes_and_distances
   :members:
   :undoc-members:
   :show-inheritance:

   Extracts genotype matrix with inter-SNP distances.

cnn_extract
~~~~~~~~~~~

.. autoclass:: cnn_extract
   :members:
   :undoc-members:
   :show-inheritance:

   Feature extractor for CNN architectures using dinf's HaplotypeMatrix.

tskit_sfs
~~~~~~~~~

.. autoclass:: tskit_sfs
   :members:
   :undoc-members:
   :show-inheritance:

   Computes site frequency spectra for single or multiple populations.

tskit_windowed_sfs_plus_ld
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tskit_windowed_sfs_plus_ld
   :members:
   :undoc-members:
   :show-inheritance:

   Combines windowed SFS with linkage disequilibrium statistics.

SPIDNA_processor
~~~~~~~~~~~~~~~~

.. autoclass:: SPIDNA_processor
   :members:
   :undoc-members:
   :show-inheritance:

   Processor specifically designed for SPIDNA embedding networks.

ReLERNN_processor
~~~~~~~~~~~~~~~~~

.. autoclass:: ReLERNN_processor
   :members:
   :undoc-members:
   :show-inheritance:

   Processor for ReLERNN architecture with phased genotype requirements.

Embedding Networks
------------------

The ``embedding_networks`` module provides neural network architectures that process tensor outputs from processors.

.. currentmodule:: workflow.scripts.embedding_networks

RNN
~~~

.. autoclass:: RNN
   :members:
   :undoc-members:
   :show-inheritance:

   A recurrent neural network using bidirectional GRU layers for processing sequential genetic data.

   **Parameters:**
   
   - **input_size** (*int*) -- The input size of the GRU layer (e.g., num_individuals * ploidy)
   - **output_size** (*int*) -- The dimension of the output feature vector
   - **num_layers** (*int, optional*) -- Number of GRU layers (default: 2)
   - **dropout** (*float, optional*) -- Dropout probability (default: 0.0)

   **Architecture:**
   
   - Bidirectional GRU with configurable layers
   - MLP head with dropout for final embedding

ExchangeableCNN
~~~~~~~~~~~~~~~

.. autoclass:: ExchangeableCNN
   :members:
   :undoc-members:
   :show-inheritance:

   Implements the Exchangeable CNN (permutation-invariant CNN) from Chan et al. 2018.
   This architecture builds in invariance to permutations of individuals in haplotype matrices.

   **Parameters:**
   
   - **output_dim** (*int, optional*) -- Dimension of the final output vector (default: 64)
   - **input_rows** (*list of int, optional*) -- Number of rows (samples) per population
   - **input_cols** (*list of int, optional*) -- Number of cols (SNPs) per population
   - **channels** (*int, optional*) -- Number of input channels (default: 2)
   - **symmetric_func** (*str, optional*) -- Symmetric pooling function: "max", "mean", or "sum" (default: "max")

   **Architecture:**
   
   - Two CNN layers with 2D convolutions (kernel heights = 1)
   - ELU activation and batch normalization
   - Symmetric pooling layer for permutation invariance
   - Global average pooling
   - Feature extractor MLP

   **Notes:**
   
   - Supports multiple populations with different dimensions
   - Automatically masks padded values (-1) when processing multiple populations
   - First CNN layer uses wider kernel and stride for long-range LD capture

SummaryStatisticsEmbedding
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SummaryStatisticsEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

   Identity embedding layer for pre-computed summary statistics.

   **Parameters:**
   
   - **output_dim** (*int, optional*) -- Not used, maintained for API consistency

   **Input Formats:**
   
   - Single population SFS: shape (num_samples + 1,)
   - Joint SFS: shape (num_samples_pop1 + 1, num_samples_pop2 + 1)

   **Notes:**
   
   - Simply passes through pre-computed summary statistics
   - Automatically flattens multi-dimensional statistics
   - Converts numpy arrays to torch tensors if needed

SPIDNA_embedding_network
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SPIDNA_embedding_network
   :members:
   :undoc-members:
   :show-inheritance:

   SPIDNA (Spatially-aware Population genomics with Deep neural Networks) architecture
   for processing genetic data with positional information.

   **Parameters:**
   
   - **output_dim** (*int, optional*) -- Dimension of output features (default: 64)
   - **num_block** (*int, optional*) -- Number of SPIDNA blocks (default: 3)
   - **num_feature** (*int, optional*) -- Number of convolutional features (default: 32)

   **Architecture:**
   
   - Separate convolutional processing for position and SNP data
   - Sequential SPIDNA blocks with residual connections
   - Progressive feature aggregation across blocks

   **Input Format:**
   
   - Shape: (batch, channels, samples, snps)
   - Channel 0: positional information
   - Channels 1+: SNP/haplotype data

ReLERNN
~~~~~~~

.. autoclass:: ReLERNN
   :members:
   :undoc-members:
   :show-inheritance:

   ReLERNN architecture following the design from https://github.com/kr-colab/ReLERNN/.
   Combines recurrent processing of haplotypes with positional information.

   **Parameters:**
   
   - **input_size** (*int*) -- Input size for GRU (num_individuals * ploidy)
   - **n_snps** (*int*) -- Number of SNPs in the input data
   - **output_size** (*int, optional*) -- Output embedding dimension (default: 64)
   - **shuffle_genotypes** (*bool, optional*) -- Shuffle genotypes during training (default: False)

   **Architecture:**
   
   - Bidirectional GRU for haplotype processing
   - Separate linear layer for positional encoding
   - Concatenated features passed through MLP
   - Dropout for regularization

   **Input Format:**
   
   - Shape: (batch, sequence_length, 1 + input_size)
   - First feature: positional data
   - Remaining features: haplotype representation

Supporting Classes
~~~~~~~~~~~~~~~~~~

.. autoclass:: SymmetricLayer
   :members:
   :undoc-members:
   :show-inheritance:

   Permutation-invariant pooling layer.

   **Parameters:**
   
   - **axis** (*int*) -- Dimension along which to apply the symmetric function
   - **func** (*str, optional*) -- Function type: "max", "mean", or "sum" (default: "max")

.. autoclass:: SPIDNABlock
   :members:
   :undoc-members:
   :show-inheritance:

   Basic building block for SPIDNA architecture.

   **Parameters:**
   
   - **num_feature** (*int*) -- Number of feature channels
   - **output_dim** (*int*) -- Output dimension for feature aggregation

   **Architecture:**
   
   - Convolutional layer with batch normalization
   - Sample-wise averaging for feature extraction
   - Residual connection to output
   - Max pooling for spatial dimension reduction 