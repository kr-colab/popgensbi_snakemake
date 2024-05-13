rule plot_posterior:
    message: "visualizing learned posterior for round {wildcards.k}..."
    input: 
        os.path.join(posteriordir, "round_{k}/", "posterior.pkl"),
        os.path.join(datadir, "x_obs.npy")
    output:
        os.path.join(posteriordir, "round_{k}/", "default_obs_samples.npy"),
        os.path.join(posteriordir, "round_{k}/", "default_obs_corner.png")
    log: "logs/plot_posterior_round_{k}.log"
    params:
        rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    # scripts is outside rules
    script: "../scripts/plotting.py"
