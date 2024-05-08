import os
datadir = config["datadir"] # where to find input data
outdir = config["outdir"] # where to save posterior

rule plot_posterior:
    message: "visualizing learned posterior..."
    input: os.path.join(outdir, "posterior.pkl"),
    output:
        os.path.join(outdir, "default_obs_samples.npy"),
        os.path.join(outdir, "default_obs_corner.png")
    log: "logs/plot_posterior.log"
    resources:
        mem_mb="5000",
        slurm_partition="kerngpu",
        slurm_extra="--gres=gpu:1 --constraint=a100"
    params:
        outdir=outdir,
        n_snps=config["n_snps"]
    # scripts is outside rules
    script: "../scripts/plotting.py"