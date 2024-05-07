datadir = config["datadir"]

rule all:
    input:
        expand(os.path.join(datadir, "{i}.trees"), i=range(n_train_sims)),
        expand(os.path.join(datadir, "theta_{i}.npy"), i=range(n_train_sims)),
        expand(os.path.join(datadir, "x_{j}.npy"), j=range(n_train_sims)),

rule simulate_ts:
    message: "simulating tree sequences..."
    output: 
        os.path.join(datadir, "{i}.trees"),
        os.path.join(datadir, "theta_{i}.npy"),
    log: "logs/simulate_ts_{i}.log"
    params: 
        num_simulations=lambda wildcards: wildcards.i,
        outdir=config["datadir"]
    script: "../scripts/simulate_ts.py"

rule process_ts:
    message: "process tree sequences..."
    input: os.path.join(datadir, "{j}.trees")
    output: os.path.join(datadir, "x_{j}.npy")
    log: "logs/process_ts_{j}.log"
    params:
        n_snps=config["n_snps"],
        num_simulations=lambda wildcards: wildcards.j,
        outdir=config["datadir"]
    script: "../scripts/process_ts.py"
        
