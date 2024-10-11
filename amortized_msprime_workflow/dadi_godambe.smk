import os

# Set up config
configfile: "config/amortized_msprime/AraTha_2epoch_genetic_map_sfs.yaml"


datadir = config["datadir"] # directory for training data

n_rep = 1000 # number of replicate with true params value for coverage probability estimation 
n_rep_dadi = 100 # number of bootstrap sfs for godambe for each rep

rule all:
    input:
        expand(os.path.join(datadir, "test_{r}_sfs_rep_{i}.npy"), r=range(n_rep), i=range(n_rep_dadi)),
        expand(os.path.join(datadir, "test_MLE_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_uncerts_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_GIM_{r}.npy"), r=range(n_rep)),
        os.path.join(datadir, "ci_rank_param_dadi_godambe.png"),
        os.path.join(datadir, "dadi_godambe_coverage.png"),
        os.path.join(datadir, "dadi_godambe_coverage.npy")

rule simulate_sfs:
    message:
        "simulate sfs for dadi godambe with test theta {wildcards.r}, rep {wildcards.i}"
    input:
        os.path.join(datadir, "test_theta_{r}.npy")
    output:
        os.path.join(datadir, "test_{r}_sfs_rep_{i}.npy")
    params:
        num_simulations=lambda wildcards: wildcards.r,
        num_rep_dadi=lambda wildcards: wildcards.i,
        **{k: v for k, v in config.items()}
    script:
        "scripts/simulate_sfs_for_dadi_godambe.py"

rule find_MLE_uncerts:
    message:
        "find MLE and uncertainties by running dadi Godambe with theta {wildcards.r}"
    input:
        expand(os.path.join(datadir, "test_{{r}}_sfs_rep_{i}.npy"), i=range(n_rep_dadi)),
        os.path.join(datadir, "test_{r}.trees")
    output:
        os.path.join(datadir, "test_MLE_{r}.npy"),
        os.path.join(datadir, "test_uncerts_{r}.npy"),
        os.path.join(datadir, "test_GIM_{r}.npy")
    params:
        n_rep_dadi=n_rep_dadi,
        num_simulations=lambda wildcards: wildcards.r,
        **{k: v for k, v in config.items()}
    script:
        "scripts/find_MLE_uncerts.py"

rule dadi_coverage_prob:
    message:
        "get coverage prob and make plots"
    input:
        expand(os.path.join(datadir, "test_MLE_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_uncerts_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep))
    output:
        os.path.join(datadir, "ci_rank_param_dadi_godambe.png"),
        os.path.join(datadir, "dadi_godambe_coverage.png"),
        os.path.join(datadir, "dadi_godambe_coverage.npy")
    params:
        n_rep=n_rep,
        **{k: v for k, v in config.items()}
    script:
        "scripts/dadi_coverage_prob.py"

