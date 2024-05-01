# Todo - make a rule to simulate AraTha 2 epoch model, prepare data for SNPE inference.


n_test_sims = 2000 # number of simulations to use for testing
n_train_sims_per_job = 250 # number of simulations per training data simulating job
n_train_reps = 300 # number of training data simulation jobs to run

# Total number of training simulations is n_train_sims_per_job*n_train_reps

n_ensemble = 10 # number of posterior estimators to train which then get combined into an ensemble

rule all:
    input:
        expand("results/rep_{i}_{Ntrain}_simulations.npy", i=range(n_train_reps), Ntrain=n_train_sims_per_job),
        expand("results/test_{Ntest}_simulations.npy", Ntest=n_test_sims),

rule simulate_test:
    message: "simulating test data..."
    output: expand("results/test_{Ntest}_simulations.npy", Ntest=n_test_sims)
    shell: "python scripts/simulate_AraTha_Africa2epoch.py {n_test_sims} ../results/ -n 1 -p test"


rule simulate_train:
    message: "simulating training sets..."
    output: "results/rep_{i}_{Ntrain}_simulations.npy"
    shell: "python scripts/simulate_AraTha_Africa2epoch.py {wildcards.Ntrain} ../results/ -n 1 -p rep_{wildcards.i}"
