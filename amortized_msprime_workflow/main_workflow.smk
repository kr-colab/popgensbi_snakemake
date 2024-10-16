
import os

# Set up config
configfile: "config/amortized_msprime/AraTha_2epoch_genetic_map_sfs.yaml"

n_sims = config["n_sims"] # number of simulations
n_ensemble = config["n_ensemble"] # number of times repeat SNPE training for ensemble learning
n_trains = config["n_trains"] # training set sizes (list of integers)
n_trains = [int(float(n)) for n in n_trains]
max_n_train = n_trains[-1]
datadir = config["datadir"] # directory for training data
posteriordir = config["posteriordir"] # output directory for posterior
ts_processor = config["ts_processor"] # name of the ts processor used
n_rep = 1000 # number of replicate with true params value for coverage probability estimation

rule all:
    input:
        os.path.join(datadir, "ts_star.trees"),
        os.path.join(datadir, ts_processor, "x_obs.npy"),
        expand(os.path.join(datadir, "test_{r}.trees"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, ts_processor, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "{i}.trees"), i=range(n_sims)),
        expand(os.path.join(datadir, "theta_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(datadir, ts_processor, "x_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_estimator_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "inference_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "default_obs_samples.npy"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "default_obs_corner.png"), k=n_trains),
        os.path.join(posteriordir, ts_processor, "confidence_intervals.png"),
        os.path.join(posteriordir, ts_processor, "confidence_intervals.npy"),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_coverage.png"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "observed_coverage.npy"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_samples_test.npy"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "ci_rank_param.png"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "sbc_stats.pkl"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "sbc_rank_hist.png"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "sbc_rank_cdf.png"), k=n_trains),



rule simulate_default_ts:
    message:
        "simulating with default thetas..."
    output:
        os.path.join(datadir, "ts_star.trees")
    log:
        "logs/simulate_default_ts.log"
    params:
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_default_ts.py"

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

rule simulate_test_ts:
    message:
        "simulate {wildcards.r}-th ts for coverage test..."
    output:
        os.path.join(datadir, "test_{r}.trees"),
        os.path.join(datadir, "test_theta_{r}.npy")
    params:
        num_simulations=lambda wildcards: wildcards.r,
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_test_ts.py"

rule process_test_ts:
    message:
        "process {wildcards.r}-th ts for coverage test..."
    input:
        os.path.join(datadir, "test_{r}.trees")
    output:
        os.path.join(datadir, ts_processor, "test_x_{r}.npy")
    params:
        num_simulations=lambda wildcards: wildcards.r,
        **{k: v for k, v in config.items()}
    script:
        "scripts/process_test_ts.py"

rule simulate_ts:
    message:
        "simulating {wildcards.i}-th ts..."
    output:
        os.path.join(datadir, "{i}.trees"),
        os.path.join(datadir, "theta_{i}.npy"),
    log:
        "logs/simulate_ts_{i}.log"
    params:
        num_simulations=lambda wildcards: wildcards.i,
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_ts.py"

rule process_ts:
    message:
        "process {wildcards.i}-th ts..."
    input:
        os.path.join(datadir, "{i}.trees")
    output:
        os.path.join(datadir, ts_processor, "x_{i}.npy")
    log:
        "logs/process_ts_{i}.log"
    params:
        num_simulations=lambda wildcards: wildcards.i,
        **{k: v for k, v in config.items()}
    script:
        "scripts/process_ts.py"

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

rule plot_coverage_prob:
    message: "estimate coverage probability for posterior learned from {wildcards.k} sims"
    input:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl"),
        expand(os.path.join(datadir, ts_processor, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep))
    output:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_samples_test.npy"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "ci_rank_param.png"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_coverage.png"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "observed_coverage.npy")
    resources:
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        n_train=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/coverage_prob.py"

rule plot_coverage_prob_hpd:
    message: "estimate coverage probability for posterior learned from {wildcards.k} sims"
    input:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl"),
        expand(os.path.join(datadir, ts_processor, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep))
    output:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "sbc_stats.pkl"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "sbc_rank_hist.png"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "sbc_rank_cdf.png")
    resources:
        mem_mb="10000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        n_train=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/run_sbc.py"



