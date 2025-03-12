"""
Set up simulation design for training
"""

import os
import numpy as np
import zarr
import yaml
from math import ceil
from numcodecs import Blosc

zarr_path = snakemake.output.zarr
split_sizes = snakemake.params.split_sizes
split_names = snakemake.params.split_names
random_seed = snakemake.params.random_seed

n_sims = np.sum(split_sizes)
n_chunks = min(len(snakemake.output.yaml), n_sims)
chunk_size = ceil(n_sims / n_chunks)
rng = np.random.default_rng(random_seed)

store = zarr.DirectoryStore(zarr_path)
root = zarr.group(store=store, overwrite=True)
# TODO: currently not using compression b/c speed
# codec = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)

# random seed per instance
root.create_dataset(name="seed", shape=n_sims, dtype='i8') 
root.seed[:] = rng.integers(2 ** 32 - 1, size=n_sims)

# create indices for splits
idx = 0
for i, n in zip(split_sizes, split_names):
    root.create_dataset(name=n, shape=i, dtype='i8') 
    root[n][:] = np.arange(idx, idx + i)
    idx = idx + i

# store stuff as flattened arrays to allow variation in shape
zarr_kwargs = {"chunks": n_chunks, "chunk_size": chunk_size, "shape": n_sims}
for what in ["features", "targets"]:
    root.create_dataset(f'{what}', dtype='array:f4', **zarr_kwargs)
    root.create_dataset(f'{what}_shape', dtype='array:i4', **zarr_kwargs)
    # chunks/chunk_size must be set appropriately to avoid collisions when writing

# allocate simulations to jobs via a yaml intermediate
# - indices must be contiguous to avoid collisions when writing
# - must create a file for "extra" chunks if n_sims <= len(output.yaml)
for i, yml in enumerate(snakemake.output.yaml):
    indices = np.arange(i * chunk_size, min(n_sims, (i + 1) * chunk_size))
    chunk_config = {
        "indices": indices.tolist(),
    }
    yaml.dump(chunk_config, open(yml, "w"), default_flow_style=True)

