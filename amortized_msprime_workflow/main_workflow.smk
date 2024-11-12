# Todo - consolidate simulate_ts and process_ts scripts
# define outputfilename as a param so that the same simulate or process ts script can be used across all rules.
import os

# Set up config
configfile: "config/amortized_msprime/testing.yaml"

n_sims = config["n_sims"] # number of simulations
n_ensemble = config["n_ensemble"] # number of times repeat SNPE training for ensemble learning
n_trains = config["n_trains"] # training set sizes (list of integers)
n_trains = [int(float(n)) for n in n_trains]
max_n_train = n_trains[-1]
datadir = config["datadir"] # directory for training data
posteriordir = config["posteriordir"] # output directory for posterior
posteriorsubdir = config.get("posteriorsubdir", config["ts_processor"]) # name of the subdirectory of posterior (default - name of the ts_processsor)
ts_processor = config["ts_processor"] # name of the ts processor used

if "datasubdir" in config:
    datasubdir = config["datasubdir"] # name of the subdirectory for training data (default - name of the ts_processor used)
else:
    datasubdir = ts_processor
if "posteriorsubdir" in config:
    posteriorsubdir = config["posteriorsubdir"]
else:
    posteriorsubdir = ts_processor
if "n_rep_coverage" in config:
    n_rep = config["n_rep_coverage"]
else:
    n_rep = 1000

rule all:
    input:
        os.path.join(datadir, "ts_star.trees"),
        os.path.join(datadir, "theta_star.trees"),
        os.path.join(datadir, datasubdir, "x_obs.npy"),
        expand(os.path.join(datadir, "test_{r}.trees"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, datasubdir, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "{i}.trees"), i=range(n_sims)),
        expand(os.path.join(datadir, "theta_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(datadir, datasubdir, "x_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_estimator_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "inference_rep_{e}.pkl"), k=n_trains, e=range(n_ensemble)),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "default_obs_samples.npy"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "default_obs_corner.png"), k=n_trains),
        os.path.join(posteriordir, posteriorsubdir, "confidence_intervals.png"),
        os.path.join(posteriordir, posteriorsubdir, "confidence_intervals.npy"),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_coverage.png"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "observed_coverage.npy"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_samples_test.npy"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ci_rank_param.png"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_stats.pkl"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_rank_hist.png"), k=n_trains),
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_rank_cdf.png"), k=n_trains),

wildcard_constraints:
    i="\d+"

rule simulate_default_ts:
    message:
        "simulating with default thetas..."
    output:
        os.path.join(datadir, "ts_star.trees"),
        os.path.join(datadir, "theta_star.npy")
    log:
        "logs/simulate_default_ts.log"
    params:
        tsname="ts_star.trees",
        thetaname="theta_star.npy",
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_ts.py"

rule process_default_ts:
    message:
        "getting summary stats from default tree sequences..."
    input:
        os.path.join(datadir, "ts_star.trees")
    output:
        os.path.join(datadir, datasubdir, "x_obs.npy")
    log:
        "logs/process_default_ts.log"
    params:
        tsname="ts_star.trees",
        xname="x_obs.npy",
        **{k: v for k, v in config.items()}
    script:
        "scripts/process_ts.py"

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
        tsname="{i}.trees",
        thetaname="theta_{i}.npy",
        **{k: v for k, v in config.items()}
    group:
        "sim"
    script:
        "scripts/simulate_ts.py"

rule process_ts:
    message:
        "process {wildcards.i}-th ts..."
    input:
        os.path.join(datadir, "{i}.trees")
    output:
        os.path.join(datadir, datasubdir, "x_{i}.npy")
    log:
        "logs/process_ts_{i}.log"
    params:
        num_simulations=lambda wildcards: wildcards.i,
        tsname="{i}.trees",
        xname="x_{i}.npy",
        **{k: v for k, v in config.items()}
    group:
        "process"
    script:
        "scripts/process_ts.py"

rule simulate_test_ts:
    message:
        "simulate {wildcards.r}-th ts for coverage test..."
    output:
        os.path.join(datadir, "test_{r}.trees"),
        os.path.join(datadir, "test_theta_{r}.npy")
    params:
        num_simulations=lambda wildcards: wildcards.r,
        tsname="test_{r}.trees",
        thetaname="test_theta_{r}.npy",
        **{k: v for k, v in config.items()}
    group:
        "sim_test"
    script:
        "scripts/simulate_ts.py"

rule process_test_ts:
    message:
        "process {wildcards.r}-th ts for coverage test..."
    input:
        os.path.join(datadir, "test_{r}.trees")
    output:
        os.path.join(datadir, datasubdir, "test_x_{r}.npy")
    params:
        num_simulations=lambda wildcards: wildcards.r,
        tsname="test_{r}.trees",
        xname="test_x_{r}.npy",
        **{k: v for k, v in config.items()}
    group:
        "process_test"
    script:
        "scripts/process_ts.py"

rule train_npe:
    message:
        "training neural posterior estimators with {wildcards.k} data points rep {wildcards.e}..."
    input:
        lambda wildcards: expand(os.path.join(datadir, datasubdir, "x_{l}.npy"), l=range(int(wildcards.k))),
        lambda wildcards: expand(os.path.join(datadir, "theta_{l}.npy"), l=range(int(wildcards.k)))
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_rep_{e}.pkl"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_estimator_rep_{e}.pkl"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "inference_rep_{e}.pkl")
    log:
        "logs/train_npe_n_train_{k}_rep_{e}.log"
    resources:
        mem_mb="80000",
        slurm_partition="gpu",
        gpus=1,
        #slurm_extra="--gres=gpu:nvidia_h100_80gb_hbm3:1"
    params:
        n_train="{k}",
        ensemble="{e}",
        **{k: v for k, v in config.items()}
    script: "scripts/train_npe.py"

rule posterior_ensemble:
    message:
        "creating an ensemble posterior for training data size {wildcards.k}"
    input:
        lambda wildcards: expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_rep_{e}.pkl"), e=range(n_ensemble), k=[wildcards.k])
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl")
    log:
        "logs/posterior_ensemble_n_train_{k}.log"
    resources:
        mem_mb="20000",
        gpus=1,
        slurm_partition="gpu",
        #slurm_extra="--gres=gpu:nvidia_h100_80gb_hbm3:1"
    params:
        n_train="{k}",
        **{k: v for k, v in config.items()}
    script: "scripts/posterior_ensemble.py"
    
rule plot_posterior:
    message: "visualizing learned posterior for training dataset size {wildcards.k}..."
    input: 
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"),
        os.path.join(datadir, posteriorsubdir, "x_obs.npy"),
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "default_obs_samples.npy"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "default_obs_corner.png")
    log: "logs/plot_posterior_n_train_{k}.log"
    resources:
        mem_mb="5000",
        gpus=1,
        slurm_partition="gpu",
        #slurm_extra="--gres=gpu:nvidia_h100_80gb_hbm3b:1"
    params:
        n_train=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/plotting.py"

rule plot_ci:
    message: "plotting confidence intervals for training data set sizes {n_trains}..."
    input:
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{n}", "default_obs_samples.npy"), n=n_trains)
    output:
        os.path.join(posteriordir, posteriorsubdir, "confidence_intervals.npy"),
        os.path.join(posteriordir, posteriorsubdir, "confidence_intervals.png")
    params:
        **{k: v for k, v in config.items()}
    script: "scripts/plot_confidence_intervals.py"

rule plot_coverage_prob:
    message: "estimate coverage probability for posterior learned from {wildcards.k} sims"
    input:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"),
        expand(os.path.join(datadir, posteriorsubdir, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep))
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_samples_test.npy"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ci_rank_param.png"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_coverage.png"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "observed_coverage.npy")
    resources:
        slurm_partition="gpu",
        gpus=1,
        #slurm_extra="--gres=gpu:nvidia_h100_80gb_hbm3:1"
    params:
        n_train=lambda wildcards: wildcards.k,
        n_boot=n_rep,
        **{k: v for k, v in config.items()}
    script: "scripts/coverage_prob.py"

rule run_sbc:
    message: "estimate coverage probability for posterior learned from {wildcards.k} sims"
    input:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"),
        expand(os.path.join(datadir, posteriorsubdir, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep))
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_stats.pkl"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_rank_hist.png"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_rank_cdf.png")
    resources:
        mem_mb="10000",
        gpus=1,
        slurm_partition="gpu",
        #slurm_extra="--gres=gpu:nvidia_h100_80gb_hbm3:1"
    params:
        n_train=lambda wildcards: wildcards.k,
        n_boot=n_rep,
        **{k: v for k, v in config.items()}
    script: "scripts/run_sbc.py"



