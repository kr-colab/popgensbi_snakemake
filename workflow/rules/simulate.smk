n_train_sims_per_job = config["n_train_sims_per_job"] # number of simulations per training data simulating job
n_train_reps = config["n_train_reps"] # number of training data simulation jobs to run

# Total number of training simulations is n_train_sims_per_job*n_train_reps

rule simulate_train:
    message: "simulating training sets..."
    output: "results/AraTha_2epoch/rep_{i}_{Ntrain}_simulations.npy"
    log: "log/simulate_train_rep_{i}_{Ntrain}.log"
    conda: "envs/demog.yaml"
    shell: "python scripts/simulate.py {wildcards.Ntrain} results/AraTha_2epoch/ 1 rep_{wildcards.i}"
