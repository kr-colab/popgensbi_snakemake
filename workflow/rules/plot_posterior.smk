rule plot_posterior:
    message: "visualizing learned posterior for round {wildcards.k}..."
    input: 
        os.path.join(posteriordir, "round_{k}/", "ensemble_posterior.pkl"),
        os.path.join(datadir, "x_obs.npy")
    output:
        os.path.join(posteriordir, "round_{k}/", "default_obs_samples.npy"),
        os.path.join(posteriordir, "round_{k}/", "default_obs_corner.png")
    log: "logs/plot_posterior_round_{k}.log"
    resources:
        mem_mb="5000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    # scripts is outside rules
    script: "../scripts/plotting.py"
