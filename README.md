# Snakemake workflow: `popgensbi`

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)
[![GitHub actions status](https://github.com/<owner>/<repo>/workflows/Tests/badge.svg?branch=main)](https://github.com/<owner>/<repo>/actions?query=branch%3Amain+workflow%3ATests)


A Snakemake workflow for simulation-based inference in population genetics.


## Usage

Unlike a standard Snakemake workflow in [Snakemake Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/?usage=<owner>%2F<repo>), there are three different workflows - amortized msprime workflow, amortized dadi workflow, and sequential msprime workflow.

### Amortized msprime workflow
To run amoritzed msprime workflow on talapas (HPC at UOregon), run `snakemake --executor slurm --workflow-profile ~/.config/snakemake/yourprofile/ --snakefile amortized_msprime_workflow/main_workflow.smk` in terminal, replacing workflow profile with your own profile. If you are using a different cluster, you probably have to change `slurm_partition1 and `slurm_extra` under `rule train_npe`, `rule posterior_ensemble`, and `rule plot_posterior` in `amortized_msprime_workflow/main_workflow.smk` - check your cluster documentation for the name of the gpu partition and how to get up `--gres`. 

note:
The profile used in the workflow assumes snakemake version 8.X. If you are using a different version, you may have to change the profile.
An example profile is provided in `example_profile/config.yaml`.

This workflow uses msprime to simulate a demographic model, saves tree sequences, process them so that it can be used as input data to train neural posterior estimator, and visualize joint posterior. The config file called in line 5 of `main_workflow.smk` defines, number of simulations per round, number of rounds, number of repeated NPE training per round for ensemble learning, data directory to save tree sequences, posterior directory to save posteriors and plots. In addition, a config file should specify the name of a demographic model, ts (tree sequence) processor, and embedding network. Check out `ts_simulators.py` for a list of demographic models, `ts_processors.py` for ts processors, and `embedding_networks.py` for networks. Depending on the object, there can be more paramters that you can specify in config and override default values. (e.g. number of SNPs to use for feature matrix, if you are using dinf's feature matrix as a ts processor). You can change line 5 to use any yaml file in `config/amortized_msprime/`.

After each round, you will get joint posterior plots. We also plot the width of confidence interval of each parameter as a function of the number of rounds. As you add more rounds, the width will shrink and (theoretically) converge to true uncertainty range. 

Importantly, you can reuse the trained NPE to get a new joint posterior with a different observed data in an amortized workflow. 

Training histories of each round are saved in in `yourposteriordir/sim_round_*/sbi_logs`. You can see the loss trajectory using Tensorboard.

### Amortized dadi workflow
To run amortized dadi workflow, run `snakemake --executor slurm --workflow-profile ~/.config/snakemake/yourprofile/ --snakefile amortized_msprime_workflow/main_workflow.smk`.
Simlar to the previous workflow, check if `resources` in the snakemake file is compatible with the cluster you are using.
This workflow uses dadi to simulate joint frequency spectra, which is directly used as input data for training NPE. It uses CNN embedding network to further summarize jsfs.
Same as before, you can change the observed fs and get a new posterior without training NPE again. 

### Sequential msprime workflow
To run amortized dadi workflow, run `snakemake --executor slurm --workflow-profile ~/.config/snakemake/yourprofile/ --snakefile sequential_msprime_workflow/Snakefile`.
Simlar to the previous workflow, check if `resources` in the snakemake file is compatible with the cluster you are using.
Use any yaml file in `config/sequential_msprime` for config.
The training result of NPE depends on observed data when `total_rounds` is greater than 1 because posterior after each round is used to sample new set of parameters for the next round of NPE learning. In other words, you will have to retrain NPE with every new observed data. However, you will need less training data in general, and confidence intervals will converge faster. 





