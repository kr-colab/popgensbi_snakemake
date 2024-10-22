import os

# Set up config
configfile: "config/amortized_msprime/YRI_CEU_moments2.yaml"

n_sims = config["n_sims"] # number of simulations
datadir = config["datadir"] # directory for training data
posteriordir = config["posteriordir"] # output directory for posterior
ts_processor = config["ts_processor"] # name of the ts processor used
n_segs = config["n_segs"] # number of segments 

rule all:
    input:
        expand(os.path.join(datadir, ts_processor, "x_{i}.npy"), i=range(n_sims)),
        expand(os.path.join(datadir, ts_processor, "ld_stat_{i}_{j}.pkl"), i=range(n_sims), j=range(n_segs)),

rule get_ld_stats:
    message:
        "getting ld stats of {wildcards.j}-th segment of {wildcards.i}-th ts..."
    input:
        os.path.join(datadir, "{i}.trees")
    output:
        os.path.join(datadir, ts_processor, "ld_stat_{i}_{j}.pkl")
    params:
        num_simulations=lambda wildcards: wildcards.i,
        n_seg_idx=lambda wildcards: wildcards.j,
        **{k: v for k, v in config.items()}
    group:
        "ld_stat"
    script:
        "scripts/get_ld_stat.py"

rule get_avg_ld_stats:
    message:
        "get average of ld stats of {wildcards.i}-th ts..."
    input:
        lambda wildcards: expand(os.path.join(datadir, ts_processor, "ld_stat_{i}_{j}.pkl"), i=[wildcards.i], j=range(n_segs))
    output:
        os.path.join(datadir, ts_processor, "x_{i}.npy")
    params:
        num_simulations=lambda wildcards: wildcards.i,
        **{k: v for k, v in config.items()}
    group:
        "ld_stat_avg"
    script:
        "scripts/get_mean_of_ld_stats.py"
