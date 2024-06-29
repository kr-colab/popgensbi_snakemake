# use after running main_workflow
# make sure all *.trees are available for this workflow 
# Specifically, run main_workflow with the same
# datadir, demog_model, n_sims_per_round,n_rounds, n_ensemble 
# as the config used in this Snakefile.

# this is identical to main_workflow.smk except it doesn't have rule simulate_default_ts or rule simulate_ts.
# We use existing tree sequences, but make new summary stats(x.npy's) and train NPE on the new summary stats.
import os

# Set up config
configfile: "config/amortized_msprime/AraTha_2epoch.yaml"

n_sims_per_round = config["n_sims_per_round"] # number of simulations per round
n_rounds = config["n_rounds"] # number of rounds
n_ensemble = config["n_ensemble"] # number of times repeat SNPE training for ensemble learning
datadir = config["datadir"] # directory for training data
posteriordir = config["posteriordir"] # output directory for posterior
ts_processor = config["ts_processor"] # name of the ts processor used

rule all:
    input:
        os.path.join(datadir, ts_processor, "x_obs.npy"),
        expand(os.path.join(datadir, ts_processor, "sim_round_{k}/x_{i}.npy"), k=list(range(n_rounds)), i=range(n_sims_per_round)),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/posterior_rep_{e}.pkl"), k=list(range(n_rounds)), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/posterior_estimator_rep_{e}.pkl"), k=list(range(n_rounds)), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/inference_rep_{e}.pkl"), k=list(range(n_rounds)), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/ensemble_posterior.pkl"), k=list(range(n_rounds))),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/default_obs_samples.npy"), k=list(range(n_rounds))),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/default_obs_corner.png"), k=list(range(n_rounds))),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/confidence_intervals.png"), k=list(range(n_rounds))),
        expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/confidence_intervals.npy"), k=list(range(n_rounds)))


rule process_default_ts:
    message:
        "getting summary stats from default tree sequences..."
    input:
        os.path.join(datadir, "ts_star.trees")
    output:
        os.path.join(datadir, ts_processor, "x_obs.npy")
    log:
        "logs/process_default_ts.log"
    params:
        **{k: v for k, v in config.items()}
    script:
        "scripts/process_default_ts.py"


rule process_ts:
    message:
        "process ts for round {wildcards.k}..."
    input:
        os.path.join(datadir, "sim_round_{k}/", "{i}.trees")
    output:
        os.path.join(datadir, ts_processor, "sim_round_{k}/x_{i}.npy")
    log:
        "logs/process_ts_round_{k}_{i}.log"
    params:
        num_simulations=lambda wildcards: wildcards.i,
        sim_rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script:
        "scripts/process_ts.py"

rule train_npe:
    message:
        "training neural posterior estimators for round {wildcards.k} rep {wildcards.e}..."
    input:
        lambda wildcards: expand(os.path.join(datadir, ts_processor, "sim_round_{k}/x_{i}.npy"), i=range(n_sims_per_round), k=[wildcards.k]),
        lambda wildcards: expand(os.path.join(datadir, "sim_round_{k}/theta_{i}.npy"), i=range(n_sims_per_round), k=[wildcards.k])
    output:
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/posterior_rep_{e}.pkl"),
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/posterior_estimator_rep_{e}.pkl"),
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/inference_rep_{e}.pkl")
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
        lambda wildcards: expand(os.path.join(posteriordir, ts_processor, "sim_round_{k}/", "posterior_rep_{e}.pkl"), e=range(n_ensemble), k=[wildcards.k])
    output:
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/", "ensemble_posterior.pkl")
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
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/ensemble_posterior.pkl"),
        os.path.join(datadir, ts_processor, "x_obs.npy"),
    output:
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/default_obs_samples.npy"),
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/default_obs_corner.png")
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
        lambda wildcards: os.path.join(posteriordir, ts_processor, "sim_round_{}/confidence_intervals.npy".format(int(wildcards.k) -1)) if int(wildcards.k) >= 1 else [],
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/", "default_obs_samples.npy")
    output:
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/", "confidence_intervals.npy"),
        os.path.join(posteriordir, ts_processor, "sim_round_{k}/", "confidence_intervals.png")
    params:
        sim_rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plot_confidence_intervals.py"


