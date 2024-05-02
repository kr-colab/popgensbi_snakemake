
n_test_sims = 2000 # number of simulations to use for testing
n_train_sims_per_job = 250 # number of simulations per training data simulating job
n_train_reps = 300 # number of training data simulation jobs to run

# Total number of training simulations is n_train_sims_per_job*n_train_reps

rule all:
    input:
        expand("results/AraTha_2epoch/rep_{i}_{Ntrain}_simulations.npy", i=range(n_train_reps), Ntrain=n_train_sims_per_job),
        expand("results/AraTha_2epoch/test_{Ntest}_simulations.npy", Ntest=n_test_sims),

rule simulate_test:
    message: "simulating test data..."
    output: expand("results/AraTha_2epoch/test_{Ntest}_simulations.npy", Ntest=n_test_sims)
    shell: "python scripts/simulate.py {n_test_sims} results/AraTha_2epoch/ 1 test"


rule simulate_train:
    message: "simulating training sets..."
    output: "results/AraTha_2epoch/rep_{i}_{Ntrain}_simulations.npy"
    shell: "python scripts/simulate.py {wildcards.Ntrain} results/AraTha_2epoch/ 1 rep_{wildcards.i}"
