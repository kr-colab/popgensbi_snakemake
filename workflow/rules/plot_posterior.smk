rule plot_posterior:
    message: "visualizing learned posterior..."
    input: "results/AraTha_2epoch/posteriors/posterior.pkl",
    output:
        "results/AraTha_2epoch/posteriors/default_obs_samples.npy",
        "results/AraTha_2epoch/posteriors/default_obs_corner.png"
    log: "log/plot_posterior.log"
    conda: "envs/demog.yaml"
    resources:
        mem_mb="5000",
        slurm_partition="kerngpu",
        slurm_extra="--gres=gpu:1 --constraint=a100"
    shell: "python scripts/plotting.py results/AraTha_2epoch results/AraTha_2epoch/posteriors/"