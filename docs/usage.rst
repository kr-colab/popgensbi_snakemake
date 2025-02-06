Usage
=====

This package is a Snakemake workflow for simulation-based inference in population genetics.
The goal of this package is to provide a flexible and modular framework for running
neural posterior estimation for population genetic inference.

The workflow is designed to be executed from the command line, and it can run either on a local machine
or a high performance computing cluster. The entire pipeline is managed by the Snakemake workflow manager.

Important files in this package include:

- ``workflow/Snakefile``: The main workflow file. This file contains the rules and
  functions that define the workflow, executing a complete neural posterior
  estimation process based on a given configuration.
- ``workflow/environment.yaml``: The Conda environment file for the workflow that lists all necessary dependencies.

In addition, the workflow relies on a configuration file (in YAML format) that contains the parameters for both
simulation and inference. For example, the file ``workflow/AraTha_2epoch_cnn.yaml`` is used to run neural
posterior estimation for a two-epoch demographic model with a CNN embedding network.

Basic Usage
-----------

To run the workflow, use the following command:

.. code-block:: bash

    snakemake --configfile workflow/AraTha_2epoch_cnn.yaml --snakefile workflow/Snakefile

This command will execute the entire pipeline—from simulation to neural posterior estimation.

Configuration Files
-------------------

The configuration file is organized into several sections controlling various aspects of the workflow:

1. **Project Directory**:

   - ``project_dir``: Specifies the directory containing the project files. Adjust this path to point to your
     actual project directory.

2. **Resource Allocation**:

   - ``cpu_resources``: Defines resources for CPU-only tasks, including:
     
     - ``runtime``: Maximum time allocated for the task.
     - ``mem_mb``: Memory (in MB) allocated for the task.
     
   - ``gpu_resources``: Defines resources for GPU tasks, including:
     
     - ``runtime``: Maximum time allocated for GPU tasks.
     - ``mem_mb``: Memory (in MB) allocated for GPU tasks.
     - ``gpus``: Number of GPUs to be used.
     - ``slurm_partition``: SLURM partition to use for job scheduling.
     - ``slurm_extra``: Additional SLURM options for GPU allocation.

3. **Simulation Parameters**:

   - ``random_seed``: A seed value for random number generation to ensure reproducibility.
   - ``chunk_size``: The number of instances to simulate per worker.
   - ``n_train``, ``n_val``, ``n_test``: The number of training, validation, and test instances, respectively.

4. **Model Training Configuration**:

   - ``train_embedding_net_separately``: Boolean flag indicating whether to train the embedding network separately
     from the normalizing flow.
   - ``use_cache``: Boolean flag indicating whether to load features into CPU memory.
   - ``optimizer``: The optimization algorithm to be used (e.g., "Adam").
   - ``batch_size``: The size of the batches used during training.
   - ``learning_rate``: The learning rate for the optimizer.
   - ``max_num_epochs``: Maximum number of training epochs.
   - ``stop_after_epochs``: Number of epochs with no improvement after which training should stop.
   - ``clip_max_norm``: Maximum norm for gradient clipping.
   - ``packed_sequence``: Boolean flag indicating whether to use packed sequences during training.

5. **Simulator Configuration**:

   - ``simulator``: Contains the simulator class and its parameters, including:
     
     - ``class_name``: Name of the simulator class to be used.
     - ``sequence_length``: Length of the simulated sequences.
     - ``mutation_rate``: Mutation rate for the simulation.
     - ``recombination_rate``: Recombination rate for the simulation.
     - ``samples``: Number of samples to simulate for each population (e.g., YRI and CEU).

6. **Processor Configuration**:

   - ``processor``: Contains the processor class and its parameters, including:
     
     - ``class_name``: Name of the processor class to be used.
     - ``n_snps``: Number of SNPs to be processed.
     - ``maf_thresh``: Minor allele frequency threshold.

7. **Embedding Network Configuration**:

   - ``embedding_network``: Contains the embedding network class and its parameters, including:
     
     - ``class_name``: Name of the embedding network class to be used.
     - ``output_dim``: Output dimension of the embedding network.
     - ``input_rows``: Number of input rows.
     - ``input_cols``: Number of input columns.



