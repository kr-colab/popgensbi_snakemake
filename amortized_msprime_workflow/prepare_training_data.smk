# First part of the main workflow - simulate tree sequences and process them for training NPE
# Also create default ts and x (is not repeated in the 'add_training_data.smk')

import os

# Set up config
configfile: "config/amortized_msprime/YRI_CEU_moments.yaml"

n_sims = config["n_sims"] # total number of simulations (i.e. n_sims - 1 is the highest index)
datadir = config["datadir"] # directory for training data
ts_processor = config["ts_processor"] # name of the ts processor used

rule all:
    input:
        os.path.join(datadir, "ts_star.trees"),
        os.path.join(datadir, ts_processor, "x_obs.npy"),
        expand(os.path.join(datadir, "{i}.trees"), i=range(n_sims)),
        expand(os.path.join(datadir, "theta_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(datadir, ts_processor, "x_{i}.npy"), i=range(n_sims)),


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
