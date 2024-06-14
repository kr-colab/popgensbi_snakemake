# run this snakefile after finish running workflow_dadi.smk and want to add more rounds of simulations

import os

# Set up config
configfile: "config/PonAbe_IM_sample.yaml"

n_sims_per_round = config["n_sims_per_round"] # number of simulations per round
n_rounds = config["n_rounds"] # number of rounds set by config. If the maximum number of rounds run so far is larger, replace it with that number.
n_ensemble = config["n_ensemble"] # number of times repeat SNPE training for ensemble learning
datadir = config["datadir"] # directory for training data
posteriordir = config["posteriordir"] # output directory for posterior
n_extra_rounds = 10

rule all:
    input:
        expand(os.path.join(datadir, "sim_round_{k}/", "fs_{i}.npy"), k=list(range(n_rounds, n_rounds + n_extra_rounds)), i=range(n_sims_per_round)),
        expand(os.path.join(datadir, "sim_round_{k}/", "theta_{i}.npy"), k=list(range(n_rounds, n_rounds + n_extra_rounds)), i=range(n_sims_per_round)),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "posterior_rep_{e}.pkl"), k=list(range(n_rounds, n_rounds + n_extra_rounds)), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "posterior_estimator_rep_{e}.pkl"), k=list(range(n_rounds, n_rounds + n_extra_rounds)), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "inference_rep_{e}.pkl"), k=list(range(n_rounds, n_rounds + n_extra_rounds)), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "ensemble_posterior.pkl"), k=list(range(n_rounds, n_rounds + n_extra_rounds))),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "default_obs_samples.npy"), k=list(range(n_rounds, n_rounds + n_extra_rounds))),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "default_obs_corner.png"), k=list(range(n_rounds, n_rounds + n_extra_rounds))),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "confidence_intervals.png"), k=list(range(n_rounds, n_rounds + n_extra_rounds))),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "confidence_intervals.npy"), k=list(range(n_rounds, n_rounds + n_extra_rounds))),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "2d_comp_multinom.png"), k=list(range(n_rounds, n_rounds+ n_extra_rounds))),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "map_thetas.npy"), k=list(range(n_rounds, n_rounds + n_extra_rounds))),
        expand(os.path.join(posteriordir, "sim_round_{k}/", "model_fs.npy"), k=list(range(n_rounds, n_rounds + n_extra_rounds)))

rule simulate_ts:
    message:
        "simulating tree sequences for round {wildcards.k}..."
    output:
        os.path.join(datadir, "sim_round_{k}/", "fs_{i}.npy"),
        os.path.join(datadir, "sim_round_{k}/", "theta_{i}.npy"),
    log:
        "logs/simulate_ts_round_{k}_{i}.log"
    params:
        num_simulations=lambda wildcards: wildcards.i,
        sim_rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_fs.py"


rule train_npe:
    message:
        "training neural posterior estimators for round {wildcards.k} rep {wildcards.e}..."
    input:
        lambda wildcards: expand(os.path.join(datadir, "sim_round_{k}/", "fs_{i}.npy"), i=range(n_sims_per_round), k=[wildcards.k]),
        lambda wildcards: expand(os.path.join(datadir, "sim_round_{k}/", "theta_{i}.npy"), i=range(n_sims_per_round), k=[wildcards.k])
    output:
        os.path.join(posteriordir, "sim_round_{k}/", "posterior_rep_{e}.pkl"),
        os.path.join(posteriordir, "sim_round_{k}/", "posterior_estimator_rep_{e}.pkl"),
        os.path.join(posteriordir, "sim_round_{k}/", "inference_rep_{e}.pkl")
    log:
        "logs/train_npe_round_{k}_rep_{e}.log"
    resources:
        mem_mb="20000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        sim_rounds="{k}",
        ensemble="{e}",
        **{k: v for k, v in config.items()}
    script: "scripts/train_npe.py"

rule posterior_ensemble:
    message:
        "creating an ensemble posterior for round {wildcards.k}"
    input:
        lambda wildcards: expand(os.path.join(posteriordir, "sim_round_{k}/", "posterior_rep_{e}.pkl"), e=range(n_ensemble), k=[wildcards.k])
    output:
        os.path.join(posteriordir, "sim_round_{k}/", "ensemble_posterior.pkl")
    log:
        "logs/posterior_ensemble_round_{k}.log"
    resources:
        mem_mb="32000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        sim_rounds="{k}",
        **{k: v for k, v in config.items()}
    script: "scripts/posterior_ensemble.py"
  

rule plot_posterior:
    message: "visualizing learned posterior for round {wildcards.k}..."
    input: 
        os.path.join(posteriordir, "sim_round_{k}/", "posterior.pkl"),
        os.path.join(datadir, "fs_star.npy"),
    output:
        os.path.join(posteriordir, "sim_round_{k}/", "default_obs_samples.npy"),
        os.path.join(posteriordir, "sim_round_{k}/", "default_obs_corner.png"),
        os.path.join(posteriordir, "sim_round_{k}/model_fs.npy"),
        os.path.join(posteriordir, "sim_round_{k}/map_thetas.npy")

    log: "logs/plot_posterior_round_{k}.log"
    resources:
        mem_mb="5000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        sim_rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plotting.py"

rule plot_ci:
    message: "plotting confidence intervals for round {wildcards.k}..."
    input:
        lambda wildcards: os.path.join(posteriordir, "sim_round_{}/confidence_intervals.npy".format(int(wildcards.k) -1)) if int(wildcards.k) >= 1 else [],
        os.path.join(posteriordir, "sim_round_{k}/", "default_obs_samples.npy")
    output:
        os.path.join(posteriordir, "sim_round_{k}/", "confidence_intervals.npy"),
        os.path.join(posteriordir, "sim_round_{k}/", "confidence_intervals.png")
    params:
        sim_rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plot_confidence_intervals.py"

rule plot_2d_comp_multinom:
    message: "compare default fs to simulated fs from MAP parameters"
    input:
        os.path.join(posteriordir, "sim_round_{k}/model_fs.npy"),
        os.path.join(datadir, "fs_star.npy")
    output:
        os.path.join(posteriordir, "sim_round_{k}/2d_comp_multinom.png"), 
    params:
        sim_rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plot_2d_comp_multinom.py"
