"""
Train embedding network and normalising flow on simulations
"""

import os
import numpy as np

CPU_RESOURCES = config.get("cpu_resources", {})
GPU_RESOURCES = config.get("gpu_resources", {})

module common:
    snakefile: "common.smk"
    config: config

# very annoyingly: imported rules are first in the rule order and will get
# executed by default, yet we have to do the following to access variables, 
# so we should never actually define any rules in `common`
use rule * from common

PROJECT_DIR = common.PROJECT_DIR
EMBEDDING_NET = common.EMBEDDING_NET
NORMALIZING_FLOW = common.NORMALIZING_FLOW
RANDOM_SEED = common.RANDOM_SEED
TRAIN_SEPARATELY = common.TRAIN_SEPARATELY

# Simulation design
N_TRAIN = int(config["n_train"])
N_VAL = int(config["n_val"])
N_TEST = int(config["n_test"])

# Directory structure
PLOT_DIR = os.path.join(PROJECT_DIR, "plots")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
TREE_DIR = os.path.join(PROJECT_DIR, "trees")
TENSOR_DIR = os.path.join(PROJECT_DIR, "tensors")

# Divvy up training instances into chunks
SPLIT_SIZES = [N_TRAIN, N_VAL, N_TEST]
SPLIT_NAMES = ["sbi_train", "sbi_val", "sbi_test"]
ZARR_FIELDS = ["features", "targets"]
if TRAIN_SEPARATELY:
    SPLIT_SIZES += [N_TRAIN, N_VAL]
    SPLIT_NAMES += ["pre_train", "pre_val"]
N_CHUNK = int(config["n_chunk"])


scattergather:
    split = N_CHUNK,


localrules: 
    simulate_all,
    process_all,
    train_all,
    analyze_all,
    clean_simulation,
    clean_processing,
    clean_all,


rule analyze_all:
    input:
        expectation = os.path.join(PLOT_DIR, "posterior_expectation.png"),
        calibration = os.path.join(PLOT_DIR, "posterior_calibration.png"),
        concentration = os.path.join(PLOT_DIR, "posterior_concentration.png"),
        at_prior_mean = os.path.join(PLOT_DIR, "posterior_at_prior_mean.png"),
        at_prior_low = os.path.join(PLOT_DIR, "posterior_at_prior_low.png"),
        at_prior_high = os.path.join(PLOT_DIR, "posterior_at_prior_high.png"),


rule train_all:
    input:
        embedding_net = EMBEDDING_NET,
        normalizing_flow = NORMALIZING_FLOW,


rule process_all:
    input:
        done = gather.split(os.path.join(TREE_DIR, "{scatteritem}-process.done")), 


rule simulate_all:
    input:
        done = gather.split(os.path.join(TREE_DIR, "{scatteritem}-simulate.done")), 


rule setup_training:
    message: "Setting up simulations and zarr storage..."
    output:
        zarr = directory(os.path.join(TENSOR_DIR, "zarr")),
        yaml = scatter.split(os.path.join(TREE_DIR, "{scatteritem}.yaml")),
    params:
        split_sizes = SPLIT_SIZES,
        split_names = SPLIT_NAMES,
        random_seed = RANDOM_SEED,
    threads: 1
    resources: **CPU_RESOURCES
    script:
        "scripts/setup_training.py"


rule simulate_batch:
    message:
        "Simulating batch {wildcards.scatteritem} of tree sequences..."
    input:
        zarr = rules.setup_training.output.zarr,
        yaml = os.path.join(TREE_DIR, "{scatteritem}.yaml"),
    output:
        done = touch(os.path.join(TREE_DIR, "{scatteritem}-simulate.done")),
    params:
        simulator_config = config["simulator"],
    threads: 1
    resources: **CPU_RESOURCES
    script:
        "scripts/simulate_batch.py"


rule process_batch:
    message:
        "Processing batch {wildcards.scatteritem} of tree sequences..."
    input:
        zarr = rules.setup_training.output.zarr,
        done = rules.simulate_batch.output.done,
        yaml = os.path.join(TREE_DIR, "{scatteritem}.yaml"),
    output:
        done = touch(os.path.join(TREE_DIR, "{scatteritem}-process.done")),
    threads: 1
    resources: **CPU_RESOURCES
    params:
        processor_config = config["processor"],
    script: 
        "scripts/process_batch.py"


rule train_embedding_net:
    message:
        "Training embedding network separately..."
    input:
        zarr = rules.setup_training.output.zarr,
        done = rules.process_all.input,
    output:
        network = os.path.join(PROJECT_DIR, "pretrain_embedding_network"),
    log:
        tensorboard = os.path.join(LOG_DIR, "pretrain_embedding_network"),
    threads: 4
    resources: **GPU_RESOURCES
    params:
        optimizer = config["optimizer"],
        batch_size = config["batch_size"],
        use_cache = config["use_cache"],
        learning_rate = config["learning_rate"],
        max_num_epochs = config["max_num_epochs"],
        stop_after_epochs = config["stop_after_epochs"],
        clip_max_norm = config["clip_max_norm"],
        packed_sequence = config["packed_sequence"],
        embedding_config = config["embedding_network"],
        random_seed = RANDOM_SEED,
    script: "scripts/train_embedding_network.py"


rule train_npe_on_embeddings:
    message:
        "Training posterior estimator with existing embeddings..."
    input:
        zarr = rules.setup_training.output.zarr,
        network = rules.train_embedding_net.output.network,
    output:
        network = os.path.join(PROJECT_DIR, "pretrain_normalizing_flow"),
    log:
        tensorboard = os.path.join(LOG_DIR, "pretrain_normalizing_flow"),
    threads: 4
    resources: **GPU_RESOURCES
    params:
        optimizer = config["optimizer"],
        batch_size = config["batch_size"],
        use_cache = config["use_cache"],
        learning_rate = config["learning_rate"],
        max_num_epochs = config["max_num_epochs"],
        stop_after_epochs = config["stop_after_epochs"],
        clip_max_norm = config["clip_max_norm"],
        packed_sequence = config["packed_sequence"],
        random_seed = RANDOM_SEED,
    script: "scripts/train_npe_on_embeddings.py"


rule train_npe_on_features:
    message:
        "Training posterior estimator on features..."
    input:
        zarr = rules.setup_training.output.zarr,
        done = rules.process_all.input,
    output:
        embedding_net = os.path.join(PROJECT_DIR, "embedding_network"),
        normalizing_flow = os.path.join(PROJECT_DIR, "normalizing_flow"),
    log:
        tensorboard = os.path.join(LOG_DIR, "embedding_normalizing_flow"),
    threads: 4
    resources: **GPU_RESOURCES
    params:
        optimizer = config["optimizer"],
        batch_size = config["batch_size"],
        use_cache = config["use_cache"],
        learning_rate = config["learning_rate"],
        max_num_epochs = config["max_num_epochs"],
        stop_after_epochs = config["stop_after_epochs"],
        clip_max_norm = config["clip_max_norm"],
        packed_sequence = config["packed_sequence"],
        embedding_config = config["embedding_network"],
        random_seed = RANDOM_SEED,
    script: "scripts/train_npe_on_features.py"


rule plot_diagnostics:
    message:
        "Plotting diagnostics..."
    input:
        zarr = rules.setup_training.output.zarr,
        embedding_net = rules.train_all.input.embedding_net,
        normalizing_flow = rules.train_all.input.normalizing_flow,
    output:
        calibration = os.path.join(PLOT_DIR, "posterior_calibration.png"),
        concentration = os.path.join(PLOT_DIR, "posterior_concentration.png"),
        expectation = os.path.join(PLOT_DIR, "posterior_expectation.png"),
        at_prior_mean = os.path.join(PLOT_DIR, "posterior_at_prior_mean.png"),
        at_prior_low = os.path.join(PLOT_DIR, "posterior_at_prior_low.png"),
        at_prior_high = os.path.join(PLOT_DIR, "posterior_at_prior_high.png"),
    threads: 4
    resources: **GPU_RESOURCES
    params:
        batch_size = config["batch_size"],
        use_cache = config["use_cache"],
        packed_sequence = config["packed_sequence"],
        simulator_config = config["simulator"],
        random_seed = RANDOM_SEED,
    script: "scripts/plot_diagnostics.py"


rule plot_simulation_stats:
    message:
        "Plotting simulation stats..."
    input:
        zarr = rules.setup_training.output.zarr,
    output:
        stats_hist = os.path.join(PLOT_DIR, "simulation_stats.png"),
        stats_vs_params_pairplot = os.path.join(PLOT_DIR, "stats_vs_params_pairplot.png"),
        stats_heatmaps = os.path.join(PLOT_DIR, "stats_heatmaps.png"),
    params:
        simulator_config = config["simulator"],
        tree_dir = TREE_DIR,
    threads: 10
    resources: **CPU_RESOURCES
    script: "scripts/plot_simulation_stats.py"


rule clean_setup:
    params:
        setup = rules.setup_training.output,
    shell: "rm -rf {params.setup}"


rule clean_simulation:
    params:
        simulation = rules.simulate_all.input,
        tree_dir = TREE_DIR,
    shell: "rm -rf {params.simulation} {params.tree_dir}/*.trees"


rule clean_processing:
    params:
        processing = rules.process_all.input,
    shell: "rm -rf {params.processing}"


rule clean_training:
    params:
        plot_dir = PLOT_DIR,
        networks = rules.train_all.input,
    shell: "rm -rf {params.networks} {params.plot_dir}"


rule clean_all:
    params:
        dir = [TREE_DIR, TENSOR_DIR, PLOT_DIR, LOG_DIR],
        networks = rules.train_all.input,
    shell: "rm -rf {params.dir} {params.networks}"
