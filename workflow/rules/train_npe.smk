import os
n_train_sims = config["n_train_sims"] # number of simulations for training
datadir = config["datadir"] # directory for training data
outdir = config["outdir"] # output directory for posterior

rule train_npe:
    message: "training neural posterior estimators..."
    input:
        expand(os.path.join(datadir, "{i}.trees"), i=range(n_train_sims)),
        expand(os.path.join(datadir, "theta_{i}.npy"), i=range(n_train_sims)),
        expand(os.path.join(datadir, "x_{j}.npy"), j=range(n_train_sims)),
    output:
        os.path.join(outdir, "posterior.pkl"),
    resources:
        mem_mb="32000",
        slurm_partition="kerngpu",
        slurm_extra="--gres=gpu:1 --constraint=a100"
    log: "logs/train_npe.log"
    params:
        datadir=datadir,
        outdir=outdir
    script: "../scripts/train_npe.py"