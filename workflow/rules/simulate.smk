rule simulate_default:
    message:
        "simulating with default thetas..."
    output:
        os.path.join(datadir, "x_obs.npy"),
        os.path.join(datadir, "ts_star.trees")
    log:
        "logs/simulate_default.log"
    params:
        **{k: v for k, v in config.items()}
    script:
        "../scripts/simulate_default.py"

rule simulate_ts:
    message:
        "simulating tree sequences for round {wildcards.k}..."
    input:
        [
        lambda wildcards: os.path.join(posteriordir, "round_{}/posterior.pkl".format(int(wildcards.k) - 1)) if int(wildcards.k) >= 1 else [], 
        lambda wildcards: os.path.join(datadir, "x_obs.npy"),
        ]
    output:
        os.path.join(datadir, "round_{k}/", "{i}.trees"),
        os.path.join(datadir, "round_{k}/", "theta_{i}.npy"),
    log:
        "logs/simulate_ts_round_{k}_{i}.log"
    params:
        num_simulations=lambda wildcards: wildcards.i,
        rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script:
        "../scripts/simulate_ts.py"

rule process_ts:
    message: "process tree sequences for round {wildcards.k}..."
    input: os.path.join(datadir, "round_{k}/", "{j}.trees")
    output: os.path.join(datadir, "round_{k}/", "x_{j}.npy")
    log: "logs/process_ts_round_{k}_{j}.log"
    params:
        num_simulations=lambda wildcards: wildcards.j,
        rounds=lambda wildcards: wildcards.k,
        **{k: v for k, v in config.items()}
    script: "../scripts/process_ts.py"
        
