# second part of the main workflow. Train NPE and visualize the posterior and confidence interval

import os

# Set up config
configfile: "config/amortized_msprime/YRI_CEU_moments.yaml"

n_ensemble = config["n_ensemble"] # number of times repeat SNPE training for ensemble learning
n_trains = config["n_trains"] # training set sizes (list of integers)
n_trains = [int(float(n)) for n in n_trains]
max_n_train = n_trains[-1]
datadir = config["datadir"] # directory for training data
posteriordir = config["posteriordir"] # output directory for posterior
ts_processor = config["ts_processor"] # name of the ts processor used

rule all:
    input:
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_estimator_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "inference_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "default_obs_samples.npy"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "default_obs_corner.png"), k=n_trains),
        os.path.join(posteriordir, ts_processor, "confidence_intervals.png"),
        os.path.join(posteriordir, ts_processor, "confidence_intervals.npy")


rule train_npe:
    message:
        "training neural posterior estimators with {wildcards.k} data points rep {wildcards.e}..."
    input:
        lambda wildcards: expand(os.path.join(datadir, ts_processor, "x_{l}.npy"), l=range(int(wildcards.k))),
        lambda wildcards: expand(os.path.join(datadir, "theta_{l}.npy"), l=range(int(wildcards.k)))
    output:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_rep_{e}.pkl"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_estimator_rep_{e}.pkl"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "inference_rep_{e}.pkl")
    log:
        "logs/train_npe_n_train_{k}_rep_{e}.log"
    resources:
        mem_mb="20000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        n_train="{k}",
        ensemble="{e}",
        **{k: v for k, v in config.items()}
    script: "scripts/train_npe.py"

rule posterior_ensemble:
    message:
        "creating an ensemble posterior for training data size {wildcards.k}"
    input:
        lambda wildcards: expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_rep_{e}.pkl"), e=range(n_ensemble), k=[wildcards.k])
    output:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl")
    log:
        "logs/posterior_ensemble_n_train_{k}.log"
    resources:
        mem_mb="32000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        n_train="{k}",
        **{k: v for k, v in config.items()}
    script: "scripts/posterior_ensemble.py"
    
rule plot_posterior:
    message: "visualizing learned posterior for training dataset size {wildcards.k}..."
    input: 
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl"),
        os.path.join(datadir, ts_processor, "x_obs.npy"),
    output:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "default_obs_samples.npy"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "default_obs_corner.png")
    log: "logs/plot_posterior_n_train_{k}.log"
    resources:
        mem_mb="5000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        n_train=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plotting.py"

rule plot_ci:
    message: "plotting confidence intervals for training data set sizes {n_trains}..."
    input:
        expand(os.path.join(posteriordir, ts_processor, "n_train_{n}", "default_obs_samples.npy"), n=n_trains)
    output:
        os.path.join(posteriordir, ts_processor, "confidence_intervals.npy"),
        os.path.join(posteriordir, ts_processor, "confidence_intervals.png")
    params:
        **{k: v for k, v in config.items()}
    script: "scripts/plot_confidence_intervals.py"


