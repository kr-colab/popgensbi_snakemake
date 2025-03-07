# Snakemake workflow: `popgensbi`

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)
[![Read the Docs](https://img.shields.io/readthedocs/pip/stable.svg)](https://popgensbi-snakemake.readthedocs.io/en/latest/)



A Snakemake workflow for simulation-based inference in population genetics.


## Contents

The main workflow for this package is contained in `workflow/training_workflow.smk`. 
This will run a complete neural posterior estimation workflow based on a given config file. For more details, see the [docs](https://popgensbi-snakemake.readthedocs.io/en/latest/).

## Environment setup

Current environment is set up with conda. To install the environment, run

```bash
conda env create -f environment.yaml
```

## Basic usage

To run the workflow, you first need to create a config file containing the details of 
the simulation and inference you want to perform. An example of such a config file is
contained in `workflow/AraTha_2epoch_cnn.yaml`. This config file is used to run the
neural posterior estimation of a two-epoch demographic model with a CNN embedding network.

To run the workflow, you can use the following command:

```bash
snakemake --configfile config/AraTha_2epoch_cnn.yaml --snakefile workflow/training_workflow.smk
```

## Simulate data for VCF prediction

Example data for the VCF prediction pipeline may be created via

```bash
python resources/util/simulate-vcf.py \
  --outpath "example_data/AraTha_2epoch" \
  --configfile workflow/config/AraTha_2epoch_cnn.yaml
```

## VCF prediction pipeline

To generate predictions along windows in the VCF,

```bash
snakemake --configfile config/AraTha_2epoch_cnn.yaml --snakefile workflow/prediction_workflow.smk
```

TODO: more details on input files, when this is finalized

## Cluster usage quickstart

To run the workflow on a cluster, you should first create a profile config file for
you cluster. We've provided an example profile in `example_profile/config.yaml`,
which is meant to be used with the SLURM cluster at University of Oregon.

To use this profile, you can run the following command:

```bash
snakemake --executor slurm --configfile config/AraTha_2epoch_cnn.yaml --workflow-profile ~/.config/snakemake/yourprofile/ --snakefile workflow/Snakefile
```
note:
The profile used in the workflow assumes snakemake version 8.X. If you are using a different version, you may have to change the profile.







