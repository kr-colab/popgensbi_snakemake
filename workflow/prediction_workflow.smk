"""
Predict posteriors across a windowed VCF, using trained normalizing flow
and embedding network
"""

import os
import glob
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

# Directory structure
VCF_PATH = config["prediction"].get("vcf")
VCF_PREFIX = os.path.basename(VCF_PATH)
VCF_DIR = os.path.join(PROJECT_DIR, VCF_PREFIX)
TENSOR_DIR = os.path.join(VCF_DIR, "tensors")
TREE_DIR = os.path.join(VCF_DIR, "trees")
PLOT_DIR = os.path.join(VCF_DIR, "plots")

# Divvy up windows into chunks
N_CHUNK = config["prediction"].get("n_chunk")


localrules: 
    clean_all,


scattergather:
    split = N_CHUNK,


rule predict_all:
    input:
        predictions = os.path.join(PLOT_DIR, "posteriors-across-windows.png"),


rule process_all:
    input:
        done = gather.split(os.path.join(TREE_DIR, "{scatteritem}-process.done")), 


rule infer_all:
    input:
        done = gather.split(os.path.join(TREE_DIR, "{scatteritem}-infer.done")), 


rule convert_vcf:
    message: 
        "Converting {input.vcf} to zarr..."
    input:
        vcf = VCF_PATH,
    output:
        icf = temp(directory(os.path.join(VCF_DIR, "icf"))),
        vcz = directory(os.path.join(VCF_DIR, "vcz")),
    threads: 4
    resources: **CPU_RESOURCES
    shell:
        """
        vcf2zarr explode -Q -p {threads} -f {input.vcf} {output.icf}
        vcf2zarr encode -Q -p {threads} -f {output.icf} {output.vcz}
        """


rule setup_prediction:
    message: 
        "Setting up prediction pipeline and creating zarr storage..."
    input:
        vcz = rules.convert_vcf.output.vcz,
    output:
        yaml = scatter.split(os.path.join(TREE_DIR, "{scatteritem}.yaml")),
        zarr = directory(os.path.join(TENSOR_DIR, "zarr")),
    threads: 4
    resources: **CPU_RESOURCES
    params:
        simulator_config = config["simulator"],
        prediction_config = config["prediction"],
    script:
        "scripts/setup_prediction.py"


rule infer_batch:
    message:
        "Inferring batch {wildcards.scatteritem} of tree sequences..."
    input:
        vcz = rules.convert_vcf.output.vcz,
        zarr = rules.setup_prediction.output.zarr,
        yaml = os.path.join(TREE_DIR, "{scatteritem}.yaml")
    output:
        done = touch(os.path.join(TREE_DIR, "{scatteritem}-infer.done"))
    threads: 1
    resources: **CPU_RESOURCES
    params:
        output_dir = TREE_DIR,
    script:
        "scripts/infer_batch.py"


rule process_batch:
    message:
        "Processing batch {wildcards.scatteritem} of tree sequences..."
    input:
        zarr = rules.setup_prediction.output.zarr,
        yaml = os.path.join(TREE_DIR, "{scatteritem}.yaml"),
        done = rules.infer_batch.output.done,
    output:
        done = touch(os.path.join(TREE_DIR, "{scatteritem}-process.done")),
    threads: 1
    resources: **CPU_RESOURCES
    params:
        processor_config = config["processor"],
    script: 
        "scripts/process_batch.py"


rule predict_windows:
    message:
        "Predicting targets across windows..."
    input:
        done = gather.split(os.path.join(TREE_DIR, "{scatteritem}-process.done")),
        zarr = rules.setup_prediction.output.zarr,
        embedding_net = EMBEDDING_NET,
        normalizing_flow = NORMALIZING_FLOW,
    output:
        predictions = rules.predict_all.input.predictions,
    threads: 4
    resources: **GPU_RESOURCES
    params:
        batch_size = config["batch_size"],
        use_cache = config["use_cache"],
        packed_sequence = config["packed_sequence"],
        simulator_config = config["simulator"],
        random_seed = RANDOM_SEED,
    script: 
        "scripts/predict_windows.py"


rule clean_setup:
    params:
        setup = rules.setup_prediction.output,
    shell: "rm -rf {params.setup}"


rule clean_inferred:
    params:
        inferred = rules.infer_all.input,
        tree_dir = TREE_DIR,
    shell: "rm -rf {params.simulation} {params.tree_dir}/*.trees"


rule clean_processing:
    params:
        processing = rules.process_all.input,
    shell: "rm -rf {params.processing}"


rule clean_all:
    params:
        vcf_dir = VCF_DIR,
    shell: "rm -rf {params.vcf_dir}"

