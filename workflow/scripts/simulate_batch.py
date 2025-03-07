"""
Simulate tree sequences
"""

import os
import zarr
import yaml
import numpy as np

import ts_simulators

indices = yaml.safe_load(open(snakemake.input.yaml))["indices"]
output_dir = os.path.dirname(snakemake.input.yaml)

config = snakemake.params.simulator_config
model = config["class_name"]
simulator = getattr(ts_simulators, model)(config)

min_snps = 1
sync = zarr.sync.ProcessSynchronizer(output_dir)
root = zarr.open(snakemake.input.zarr, "a", synchronizer=sync)
for i in indices:
    seed = root.seed[i]
    ts, theta = simulator(seed=seed)
    if ts.num_sites < min_snps:
        raise ValueError(
            f"Insufficient mutations in simulation ({ts.num_sites} < {min_snps}), "
            f"adjust prior and/or use rejection sampling inside simulator"
        )
    root.targets[i] = theta
    root.targets_shape[i] = theta.shape
    ts.dump(os.path.join(output_dir, f"{i}.trees"))
