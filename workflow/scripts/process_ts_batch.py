import os
import numpy as np
import zarr
import tskit
import ts_processors
import glob

# Get parameters
batch_id = snakemake.params.batch_id
batch_size = snakemake.params.batch_size
config = snakemake.params.processor_config

# Calculate batch start
batch_start = batch_id * batch_size

# Get input directory
input_dir = os.path.dirname(snakemake.input.done)

# Open zarr store
root = zarr.open(snakemake.input.zarr, "rw")
# Initialize processor
class_name = config["class_name"]
ts_processor = getattr(ts_processors, class_name)(config)

# List and sort all tree sequence files in the input directory.
tree_files = sorted(glob.glob(os.path.join(input_dir, "*.trees")))

# Process all tree sequences in batch
for i in range(batch_size):
    idx = batch_start + i
    
    if idx >= len(tree_files):
        continue  # No more tree sequences
    ts_path = tree_files[idx]
    
    with open(ts_path, "rb") as ts_file:
        ts = tskit.load(ts_file)
    
    # Process features
    features = ts_processor(ts)
    # Write to zarr store
    root.features[idx] = features.flatten()
    root.features_shape[idx] = features.shape

# Create done file
with open(snakemake.output.done, "w") as handle:
    handle.write("done")
