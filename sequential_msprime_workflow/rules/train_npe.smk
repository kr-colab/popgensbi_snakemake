rule train_npe:
    message:
        "training neural posterior estimators for round {wildcards.k}..."
    input:
        trees=lambda wildcards: expand(os.path.join(datadir, "round_{k}/", "{i}.trees"), i=range(n_train_sims), k=[wildcards.k]),
        theta=lambda wildcards: expand(os.path.join(datadir, "round_{k}/", "theta_{i}.npy"), i=range(n_train_sims), k=[wildcards.k]),
        x=lambda wildcards: expand(os.path.join(datadir, "round_{k}/", "x_{j}.npy"), j=range(n_train_sims), k=[wildcards.k]),
        x_obs=lambda wildcards: os.path.join(datadir, "x_obs.npy")
    output:
        posterior=os.path.join(posteriordir, "round_{k}/", "posterior_{e}.pkl")
    log:
        "logs/train_npe_round_{k}_{e}.log"
    resources:
        mem_mb="32000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        rounds="{k}",
        ensemble="{e}",
        **{k: v for k, v in config.items()}
    script: "../scripts/train_npe.py"

rule posterior_ensemble:
    message:
        "creating an ensemble posterior for round {wildcards.k} and sampling thetas for the next round..."
    input:
        posteriors=lambda wildcards: expand(os.path.join(posteriordir, "round_{k}/", "posterior_{e}.pkl"), e=range(n_ensemble), k=[wildcards.k])
    output:
        ensemble_posterior=os.path.join(posteriordir, "round_{k}/", "ensemble_posterior.pkl"),
        thetas=os.path.join(datadir, "round_{k}/", "thetas.npy")
    log:
        "logs/posterior_ensemble_round_{k}.log"
    resources:
        mem_mb="32000",
        slurm_partition="gpu",
        slurm_extra="--gres=gpu:1 --constraint=gpu-10gb"
    params:
        rounds="{k}",
        **{k: v for k, v in config.items()}
    script: "../scripts/posterior_ensemble.py"