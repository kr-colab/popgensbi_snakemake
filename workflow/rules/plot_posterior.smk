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
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        **{k: v for k, v in config.items()}
    # scripts is outside rules
    script: "../scripts/plotting.py"
