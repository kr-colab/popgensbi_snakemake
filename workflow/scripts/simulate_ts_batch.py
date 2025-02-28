import os
import zarr
import numpy as np
import warnings

import ts_simulators

# Get parameters from snakemake
batch_id = snakemake.params.batch_id
batch_size = snakemake.params.batch_size
config = snakemake.params.simulator_config

# Calculate batch start
batch_start = batch_id * batch_size

# Get output directory for this batch
output_dir = os.path.dirname(snakemake.output.done)

# Initialize simulator
model = config["class_name"]
simulator = getattr(ts_simulators, model)(config)

# Simulate and save tree sequences in batch
max_attempts = 1000
root = zarr.open(snakemake.input.zarr, "rw")
for i in range(batch_size):
    idx = batch_start + i
    if idx >= root.seed.size:
        continue  # Skip if we've reached the end of the simulations

    # Try simulating until the tree sequence contains more than one site.
    seed = root.seed[idx]
    for _ in range(max_attempts):
        ts, theta = simulator(seed=seed)
        if ts.num_sites > 1:
            break
        seed += 1
    else:
        raise ValueError(
            f"Zero mutations returned for simulation {idx} after {max_attempts} attempts"
        )
    
    # Save tree sequence
    ts_path = os.path.join(output_dir, f"{idx}.trees")
    with open(ts_path, "wb") as ts_file:
        ts.dump(ts_file)
    
    # Save parameters
    root.targets[idx] = theta
    root.targets_shape[idx] = theta.shape

# Create done file
with open(snakemake.output.done, "w") as handle:
    handle.write("done")
