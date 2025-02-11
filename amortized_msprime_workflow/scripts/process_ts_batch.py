import os
import numpy as np
from ts_processors import *
import tskit
import ray

# Initialize ray once at the start of processing
if not ray.is_initialized():
    num_cpus = 3  # or another appropriate number
    ray.init(num_cpus=num_cpus)

# Get parameters
batch_id = snakemake.params.batch_id
batch_size = snakemake.params.batch_size
datadir = snakemake.params.datadir
datasubdir = snakemake.params.datasubdir

# Calculate batch start
batch_start = batch_id * batch_size

try:
    # Create output directory
    outdir = os.path.join(datadir, datasubdir)
    os.makedirs(outdir, exist_ok=True)

    # Initialize processor
    ts_processor = PROCESSOR_LIST[snakemake.params.ts_processor](snakemake)

    # Process all tree sequences in batch
    for i in range(batch_size):
        idx = batch_start + i
        
        # Determine if this is a test batch
        is_test = "test" in snakemake.output.done
        prefix = "test_" if is_test else ""
        
        # Load tree sequence
        ts_path = os.path.join(datadir, f"{prefix}{idx}.trees")
        if not os.path.exists(ts_path):
            continue  # Skip if we've reached the end of the simulations
        
        with open(ts_path, "rb") as ts_file:
            ts = tskit.load(ts_file)
        
        # Process and save features
        features = ts_processor(ts)
        np.save(os.path.join(outdir, f"{prefix}x_{idx}.npy"), features.numpy())

    # Create done file
    with open(snakemake.output.done, "w") as f:
        f.write("done")

finally:
    # Shutdown ray when done
    if ray.is_initialized():
        ray.shutdown() 