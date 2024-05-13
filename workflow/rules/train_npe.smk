rule train_npe:
    message:
        "training neural posterior estimators for round {wildcards.k}..."
    input:
        trees=lambda wildcards: expand(os.path.join(datadir, "round_{k}/", "{i}.trees"), i=range(n_train_sims), k=[wildcards.k]),
        theta=lambda wildcards: expand(os.path.join(datadir, "round_{k}/", "theta_{i}.npy"), i=range(n_train_sims), k=[wildcards.k]),
        x=lambda wildcards: expand(os.path.join(datadir, "round_{k}/", "x_{j}.npy"), j=range(n_train_sims), k=[wildcards.k]),
    output:
        posterior=os.path.join(posteriordir, "round_{k}/", "posterior.pkl"),
    log:
        "logs/train_npe_round_{k}.log"
    params:
        rounds="{k}",
        **{k: v for k, v in config.items()}
    script: "../scripts/train_npe.py"