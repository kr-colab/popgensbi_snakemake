import os
import zarr
import numpy as np

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
root = zarr.open(snakemake.input.zarr, "rw")
for i in range(batch_size):
    idx = batch_start + i
    if idx >= root.seed.size:
        continue  # Skip if we've reached the end of the simulations

    # Simulate tree sequence
    ts, theta = simulator(seed=root.seed[idx])
    
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
