import torch
import msprime
import os
import numpy as np
from ts_simulators import *

# Get parameters from snakemake
batch_id = snakemake.params.batch_id
batch_size = snakemake.params.batch_size
datadir = snakemake.params.datadir

# Calculate batch start
batch_start = batch_id * batch_size

# Create output directory for this batch
batch_dir = os.path.join(datadir, f"test_batch_{batch_id}")
os.makedirs(batch_dir, exist_ok=True)

# Initialize simulator
demog_model = snakemake.params.demog_model
simulator = MODEL_LIST[demog_model](snakemake)

# Sample all parameters for the batch at once
thetas = simulator.prior.sample((batch_size,))

# Simulate and save tree sequences in batch
for i in range(batch_size):
    idx = batch_start + i
    theta = thetas[i]
    
    # Simulate tree sequence
    ts = simulator(theta)
    
    # Save tree sequence
    ts_path = os.path.join(datadir, f"test_{idx}.trees")
    with open(ts_path, "wb") as ts_file:
        ts.dump(ts_file)
    
    # Save parameters
    theta_path = os.path.join(datadir, f"test_theta_{idx}.npy")
    theta_np = theta.squeeze().cpu().numpy()
    np.save(theta_path, theta_np)

# Create done file
with open(snakemake.output.trees, "w") as f:
    f.write("done")