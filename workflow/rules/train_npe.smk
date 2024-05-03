n_train_sims_per_job = config["n_train_sims_per_job"] # number of simulations per training data simulating job
n_train_reps = config["n_train_reps"] # number of training data simulation jobs to run

# Total number of training simulations is n_train_sims_per_job*n_train_reps


rule train_npe:
    message: "training neural posterior estimators..."
    input:
        expand("results/AraTha_2epoch/rep_{i}_{Ntrain}_simulations.npy", i=range(n_train_reps), Ntrain=n_train_sims_per_job),
    output:
        "results/AraTha_2epoch/posteriors/posterior.pkl",
    resources:
        mem_mb="32000",
        slurm_partition="kerngpu",
        slurm_extra="--gres=gpu:1 --constraint=a100"
    shell: "python scripts/train_npe.py results/AraTha_2epoch results/AraTha_2epoch/posteriors/"