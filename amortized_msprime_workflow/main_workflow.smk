import os

# Set up config
configfile: "config/amortized_msprime/YRI_CEU_dinf.yaml"

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
    datasubdir = config["datasubdir"]
else:
    datasubdir = config["ts_processor"]
if "posteriorsubdir" in config:
    posteriorsubdir = config["posteriorsubdir"]
else:
    posteriorsubdir = ts_processor
if "n_rep_coverage" in config:
    n_rep = config["n_rep_coverage"]
else:
    n_rep = 1000

localrules: 
    all,
    simulate_all,
    simulate_test_all,
    process_all,
    train_all,
    analyze_all,
    process_default_ts,
    plot_ci,
    #process_ts_batch,
    #process_test_ts_batch

BATCH_SIZE = 10  # Adjust based on your needs and memory constraints
TEST_BATCH_SIZE = 10  # Adjust based on your needs

rule simulate_all:
    input:
        expand(os.path.join(datadir, "batch_{batch}", "sims.done"), 
               batch=range((n_sims + BATCH_SIZE - 1) // BATCH_SIZE))

rule simulate_test_all:
    input:
        os.path.join(datadir, "ts_star.trees"),
        os.path.join(datadir, "theta_star.npy"),
        expand(os.path.join(datadir, "test_batch_{batch}", "test_sims.done"), 
               batch=range((n_rep + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE))

rule process_all:
    input:
        os.path.join(datadir, datasubdir, "x_obs.npy"),
        expand(os.path.join(datadir, datasubdir, "batch_{batch}/features.done"), 
               batch=range((n_sims + BATCH_SIZE - 1) // BATCH_SIZE)),
        expand(os.path.join(datadir, datasubdir, "test_batch_{batch}/test_features.done"),
               batch=range((n_rep + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE))

rule train_all:
    input:
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"), k=n_trains)

rule analyze_all:
    input:
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
        expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_rank_cdf.png"), k=n_trains)

rule all:
    input:
        rules.simulate_all.input,
        rules.simulate_test_all.input,
        rules.process_all.input,
        rules.train_all.input,
        rules.analyze_all.input

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
        datasubdir=datasubdir,
        **{k: v for k, v in config.items()}
    script:
        "scripts/process_ts.py"

rule simulate_ts_batch:
    message:
        "simulating batch {wildcards.batch} of tree sequences..."
    output:
        trees = os.path.join(datadir, "batch_{batch}", "sims.done")
    params:
        batch_id = lambda wildcards: int(wildcards.batch),
        batch_size = BATCH_SIZE,
        **{k: v for k, v in config.items()}
    group:
        "sim"
    resources:
        mem_mb = 16000,
        time = "2:00:00"
    script:
        "scripts/simulate_ts_batch.py"

rule process_ts_batch:
    message:
        "processing batch {wildcards.batch} of tree sequences..."
    input:
        os.path.join(datadir, "batch_{batch}", "sims.done")
    output:
        done = os.path.join(datadir, datasubdir, "batch_{batch}", "features.done")
    log:
        os.path.join("logs", "process_ts_batch_{batch}.log")
    params:
        batch_id = lambda wildcards: int(wildcards.batch),
        batch_size = BATCH_SIZE,
        datasubdir = datasubdir,
        **{k: v for k, v in config.items()}
    group: "process"
    resources:
        mem_mb = 4000,
        time = "1:00:00"
    script: "scripts/process_ts_batch.py"

rule train_npe:
    message:
        "training neural posterior estimators with {wildcards.k} data points rep {wildcards.e}..."
    input:
        # Calculate how many batches we need for k training samples
        sim_markers = lambda w: [os.path.join(datadir, f"batch_{b}", "sims.done") 
                               for b in range((int(w.k) + BATCH_SIZE - 1) // BATCH_SIZE)],
        feature_markers = lambda w: [os.path.join(datadir, datasubdir, f"batch_{b}", "features.done") 
                                   for b in range((int(w.k) + BATCH_SIZE - 1) // BATCH_SIZE)]
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_rep_{e}.pkl"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_estimator_rep_{e}.pkl"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "inference_rep_{e}.pkl")
    log:
        "logs/train_npe_n_train_{k}_rep_{e}.log"
    resources:
        runtime = "3h",
        mem_mb = 25000,
        threads=10,
        slurm_partition="kerngpu,gpu",
        gpus=1,
        slurm_extra="--gres=gpu:1 --constraint=gpu-80gb"
    params:
        n_train="{k}",
        ensemble="{e}",
        datasubdir=datasubdir,
        posteriorsubdir=posteriorsubdir,
        **{k: v for k, v in config.items()}
    script: "scripts/train_npe.py"

rule posterior_ensemble:
    message:
        "creating an ensemble posterior for training data size {wildcards.k}"
    input:
        lambda wildcards: expand(os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_rep_{e}.pkl"), 
                               e=range(n_ensemble), k=[wildcards.k])
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl")
    log:
        "logs/posterior_ensemble_n_train_{k}.log"
    resources:
        mem_mb="20000",
        gpus=1,
        slurm_partition="gpu,kerngpu",
        slurm_extra="--gres=gpu:1 --constraint=a100"
    params:
        n_train="{k}",
        posteriorsubdir=posteriorsubdir,
        **{k: v for k, v in config.items()}
    script: "scripts/posterior_ensemble.py"
    
rule plot_posterior:
    message: "visualizing learned posterior for training dataset size {wildcards.k}..."
    input: 
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"),
        os.path.join(datadir, datasubdir, "x_obs.npy"),
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "default_obs_samples.npy"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "default_obs_corner.png")
    log: "logs/plot_posterior_n_train_{k}.log"
    resources:
        mem_mb="5000",
        gpus=1,
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=a100"
    params:
        n_train=lambda wildcards: wildcards.k,
        datasubdir=datasubdir,
        posteriorsubdir=posteriorsubdir,
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
        posteriorsubdir=posteriorsubdir,
        **{k: v for k, v in config.items()}
    script: "scripts/plot_confidence_intervals.py"

rule plot_coverage_prob:
    message: "estimate coverage probability for posterior learned from {wildcards.k} sims"
    input:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"),
        expand(os.path.join(datadir, datasubdir, "test_batch_{batch}/test_features.done"),
               batch=range((n_rep + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE)),
        expand(os.path.join(datadir, "test_batch_{batch}/test_sims.done"),
               batch=range((n_rep + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE))
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_samples_test.npy"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ci_rank_param.png"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "posterior_coverage.png"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "observed_coverage.npy")
    resources:
        slurm_partition="kerngpu,gpu",
        gpus=1,
        slurm_extra="--gres=gpu:1 --constraint=a100"
    params:
        n_train=lambda wildcards: wildcards.k,
        n_boot=n_rep,
        datasubdir=datasubdir,
        posteriorsubdir=posteriorsubdir,
        **{k: v for k, v in config.items()}
    script: "scripts/coverage_prob.py"

rule run_sbc:
    message: "estimate coverage probability for posterior learned from {wildcards.k} sims"
    input:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "ensemble_posterior.pkl"),
        expand(os.path.join(datadir, datasubdir, "test_batch_{batch}/test_features.done"),
               batch=range((n_rep + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE)),
        expand(os.path.join(datadir, "test_batch_{batch}/test_sims.done"),
               batch=range((n_rep + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE))
    output:
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_stats.pkl"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_rank_hist.png"),
        os.path.join(posteriordir, posteriorsubdir, "n_train_{k}", "sbc_rank_cdf.png")
    resources:
        mem_mb="10000",
        gpus=1,
        slurm_partition="gpu,kerngpu",
        slurm_extra="--gres=gpu:nvidia_a100_80gb_pcie:1"
    params:
        n_train=lambda wildcards: wildcards.k,
        n_boot=n_rep,
        datasubdir=datasubdir,
        posteriorsubdir=posteriorsubdir,
        device="cuda:0",
        **{k: v for k, v in config.items()}
    script: "scripts/run_sbc.py"

rule simulate_test_ts_batch:
    message:
        "simulate batch {wildcards.batch} of test tree sequences..."
    output:
        trees = os.path.join(datadir, "test_batch_{batch}", "test_sims.done")
    params:
        batch_id = lambda wildcards: int(wildcards.batch),
        batch_size = TEST_BATCH_SIZE,
        **{k: v for k, v in config.items()}
    group:
        "sim_test"
    resources:
        mem_mb = 16000,
        time = "2:00:00"
    script:
        "scripts/simulate_test_ts_batch.py"

rule process_test_ts_batch:
    message:
        "processing batch {wildcards.batch} of test tree sequences..."
    input:
        os.path.join(datadir, "test_batch_{batch}", "test_sims.done")
    output:
        done = os.path.join(datadir, datasubdir, "test_batch_{batch}", "test_features.done")
    log:
        "logs/process_test_ts_batch_{batch}.log"
    params:
        batch_id = lambda wildcards: int(wildcards.batch),
        batch_size = TEST_BATCH_SIZE,
        datasubdir = datasubdir,
        **{k: v for k, v in config.items()}
    group: "process_test"
    resources:
        mem_mb = 4000,
        time = "1:00:00"
    script: "scripts/process_ts_batch.py"



