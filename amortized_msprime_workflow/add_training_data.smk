# First part of the main workflow, but when you already have some training data to re-use.
# - simulate tree sequences and process them for training NPE


import os

# Set up config
configfile: "config/amortized_msprime/YRI_CEU_sfs.yaml"

n_sims_start = 30000 # which index to start with (in case some training data still exists.)
n_sims = 50000 # total number of simulations (i.e. n_sims - 1 is the highest index)
datadir = config["datadir"] # directory for training data
ts_processor = config["ts_processor"] # name of the ts processor used

rule all:
    input:
        expand(os.path.join(datadir, "{i}.trees"), i=range(n_sims_start, n_sims)),
        expand(os.path.join(datadir, "theta_{i}.npy"), i=range(n_sims_start, n_sims)),
        expand(os.path.join(datadir, ts_processor, "x_{i}.npy"), i=range(n_sims_start, n_sims)),


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
