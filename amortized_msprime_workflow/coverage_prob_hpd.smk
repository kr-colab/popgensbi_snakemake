import os

# Set up config
configfile: "config/amortized_msprime/AraTha_2epoch_genetic_map_dinf.yaml"

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
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, ts_processor, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_coverage_hpd.png"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "observed_coverage_hpd.npy"), k=n_trains),
        expand(os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_samples_test.npy"), k=n_trains),

rule plot_coverage_prob_hpd:
    message: "estimate coverage probability for posterior learned from {wildcards.k} sims"
    input:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "ensemble_posterior.pkl"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_samples_test.npy"),
        expand(os.path.join(datadir, ts_processor, "test_x_{r}.npy"), r=range(n_rep)),
        expand(os.path.join(datadir, "test_theta_{r}.npy"), r=range(n_rep))
    output:
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "posterior_coverage_hpd.png"),
        os.path.join(posteriordir, ts_processor, "n_train_{k}", "observed_coverage_hpd.npy")
    resources:
        mem_mb="100000",
    params:
        n_train=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "scripts/coverage_prob_hpd.py"



