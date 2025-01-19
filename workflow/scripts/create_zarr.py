import os
import numpy as np
import zarr
from numcodecs import Blosc

# Get parameters from snakemake
zarr_path = snakemake.output.zarr
split_sizes = snakemake.params.split_sizes
split_names = snakemake.params.split_names
chunk_size = snakemake.params.chunk_size
random_seed = snakemake.params.random_seed
fields = snakemake.params.fields

n_sims = np.sum(split_sizes)
rng = np.random.default_rng(random_seed)

# Create zarr arrays with compression
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
store = zarr.DirectoryStore(zarr_path)
root = zarr.group(store=store)

# Random seed per instance
root.create_dataset(
    name="seed", 
    shape=n_sims, 
    chunks=chunk_size, 
    dtype='i8', 
    compressor=compressor,
)
root.seed[:] = rng.integers(2 ** 32 - 1, size=n_sims)

# Create indices for splits
idx = 0
for i, n in zip(split_sizes, split_names):
    root.create_dataset(
        name=n,
        shape=i,
        chunks=chunk_size, 
        dtype='i8', 
        compressor=compressor,
    )
    root[n][:] = np.arange(idx, idx + i)
    idx = idx + i

# Store stuff as flattened arrays to allow variation in shape
for what in fields:
    root.create_dataset(
        f'{what}',  
        shape=n_sims,
        chunks=chunk_size,
        dtype='array:f4',
        compressor=compressor,
    )
    root.create_dataset(
        f'{what}_shape',
        shape=n_sims,
        chunks=chunk_size,
        dtype='array:i4',
        compressor=compressor,
    )
