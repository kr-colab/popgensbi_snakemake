
import os

# Set up config
configfile: "config/amortized_dadi/AraTha_2epoch.yaml"

n_sims = config["n_sims"] # number of simulations
n_ensemble = config["n_ensemble"] # number of times repeat SNPE training for ensemble learning
datadir = config["datadir"] # directory for training data
posteriordir = config["posteriordir"] # output directory for posterior
n_trains = config["n_trains"] # training set sizes
n_trains = [int(float(n)) for n in n_trains]
max_n_train = n_trains[-1]

rule all:
    input:
        os.path.join(datadir, "fs_star.npy"),
        expand(os.path.join(datadir, "fs_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(datadir, "theta_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(posteriordir, "n_train_{k}", "posterior_rep_{e}.pkl"), k=list(n_trains), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, "n_train_{k}", "posterior_estimator_rep_{e}.pkl"), k=list(n_trains), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, "n_train_{k}", "inference_rep_{e}.pkl"), k=list(n_trains), e=range(n_ensemble)),
        expand(os.path.join(posteriordir, "n_train_{k}", "ensemble_posterior.pkl"), k=list(n_trains)),
        expand(os.path.join(posteriordir, "n_train_{k}", "default_obs_samples.npy"), k=list(n_trains)),
        expand(os.path.join(posteriordir, "n_train_{k}", "default_obs_corner.png"), k=list(n_trains)),
        os.path.join(posteriordir, "confidence_intervals.png"),
        os.path.join(posteriordir, "confidence_intervals.npy"),
        expand(os.path.join(posteriordir, "n_train_{k}/", "2d_comp_multinom.png"), k=list(n_trains)),
        expand(os.path.join(posteriordir, "n_train_{k}/", "map_thetas.npy"), k=list(n_trains)),
        expand(os.path.join(posteriordir, "n_train_{k}/", "model_fs.npy"), k=list(n_trains))


rule simulate_default:
    message:
        "simulating with default thetas..."
    output:
        os.path.join(datadir, "fs_star.npy")
    log:
        "logs/simulate_default.log"
    params:
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_default.py"

rule simulate:
    message:
        "simulating {wildcards.i}-th sfs..."
    output:
        os.path.join(datadir, "fs_{i}.npy"),
        os.path.join(datadir, "theta_{i}.npy"),
    log:
        "logs/simulate_sfs_{i}.log"
    params:
        num_simulations=lambda wildcards: wildcards.i,
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_fs.py"


rule train_npe:
    message:
        "training neural posterior estimators with {wildcards.k} data points rep {wildcards.e}..."
    input:
        lambda wildcards: expand(os.path.join(datadir, "fs_{l}.npy"), l=range(int(wildcards.k))),
        lambda wildcards: expand(os.path.join(datadir, "theta_{l}.npy"), l=range(int(wildcards.k)))
    output:
        os.path.join(posteriordir, "n_train_{k}/", "posterior_rep_{e}.pkl"),
        os.path.join(posteriordir, "n_train_{k}/", "posterior_estimator_rep_{e}.pkl"),
        os.path.join(posteriordir, "n_train_{k}/", "inference_rep_{e}.pkl")
    log:
        "logs/train_npe_n_{k}_rep_{e}.log"
    resources:
        mem_mb="20000"
    params:
        n_train="{k}",
        ensemble="{e}",
        **{k: v for k, v in config.items()}
    script: "scripts/train_npe.py"

rule posterior_ensemble:
    message:
        "creating an ensemble posterior for training data size {wildcards.k}"
    input:
        lambda wildcards: expand(os.path.join(posteriordir, "n_train_{k}/", "posterior_rep_{e}.pkl"), e=range(n_ensemble), k=[wildcards.k])
    output:
        os.path.join(posteriordir, "n_train_{k}/", "ensemble_posterior.pkl")
    log:
        "logs/posterior_ensemble_n_{k}.log"
    resources:
        mem_mb="32000"
    params:
        n_train="{k}",
        **{k: v for k, v in config.items()}
    script: "scripts/posterior_ensemble.py"
    
rule plot_posterior:
    message: "visualizing learned posterior for training data size {wildcards.k}..."
    input: 
        os.path.join(posteriordir, "n_train_{k}/", "ensemble_posterior.pkl"),
        os.path.join(datadir, "fs_star.npy"),
    output:
        os.path.join(posteriordir, "n_train_{k}/", "default_obs_samples.npy"),
        os.path.join(posteriordir, "n_train_{k}/", "default_obs_corner.png"),
        os.path.join(posteriordir, "n_train_{k}/model_fs.npy"),
        os.path.join(posteriordir, "n_train_{k}/map_thetas.npy")

    log: "logs/plot_posterior_round_{k}.log"
    resources:
        mem_mb="5000"
    params:
        n_train=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plotting.py"

rule plot_ci:
    message: "plotting confidence intervals with training dataset sizes {n_trains}..."
    input:
        expand(os.path.join(posteriordir, "n_train_{n}/", "default_obs_samples.npy"), n=n_trains)
    output:
        ci_npy = os.path.join(posteriordir, "confidence_intervals.npy"),
        ci_png = os.path.join(posteriordir, "confidence_intervals.png")
    params:
        max_n_train="{max_n_train}",
        **{k: v for k, v in config.items()}
    script: "scripts/plot_confidence_intervals.py"

rule plot_2d_comp_multinom:
    message: "compare default fs to simulated fs from MAP parameters for training data size {wildcards.k}..."
    input:
        os.path.join(posteriordir, "n_train_{k}/model_fs.npy"),
        os.path.join(datadir, "fs_star.npy")
    output:
        os.path.join(posteriordir, "n_train_{k}/2d_comp_multinom.png"), 
    params:
        n_train=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plot_2d_comp_multinom.py"
