import os
import numpy as np
import zarr
from numcodecs import Blosc
from concurrent.futures import ProcessPoolExecutor
import math

def load_and_write_chunk(args):
    """Load and write a chunk of data to zarr arrays"""
    start_idx, end_idx, datadir, datasubdir, prefix, x_array, theta_array = args
    
    for i in range(start_idx, end_idx):
        x = np.load(os.path.join(datadir, datasubdir, f"{prefix}x_{i}.npy"))
        theta = np.load(os.path.join(datadir, f"{prefix}theta_{i}.npy"))
        x_array[i] = x
        theta_array[i] = theta

def create_zarr_dataset(output_path, datadir, datasubdir, indices, batch_size=512, prefix="", n_workers=8):
    """Create a zarr dataset from numpy files with chunks aligned to batch size"""
    
    # Load first file to get shapes
    x_shape = np.load(os.path.join(datadir, datasubdir, f"{prefix}x_0.npy")).shape
    theta_shape = np.load(os.path.join(datadir, f"{prefix}theta_0.npy")).shape
    
    # Create zarr arrays with compression
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store)
    
    # Chunk size matches batch_size for first dimension, full size for feature dimensions
    x_chunks = (batch_size, *x_shape)
    theta_chunks = (batch_size, *theta_shape)
    
    x_array = root.create_dataset('x', 
                                shape=(len(indices), *x_shape),
                                chunks=x_chunks,
                                dtype=np.float32,
                                compressor=compressor)
    
    theta_array = root.create_dataset('theta',
                                    shape=(len(indices), *theta_shape),
                                    chunks=theta_chunks,
                                    dtype=np.float32,
                                    compressor=compressor)
    
    # Split work into chunks for parallel processing
    n_total = len(indices)
    chunk_size = math.ceil(n_total / n_workers)
    chunks = []
    
    for i in range(n_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_total)
        if start_idx < end_idx:  # ensure we don't create empty chunks
            chunks.append((start_idx, end_idx, datadir, datasubdir, prefix, x_array, theta_array))
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(load_and_write_chunk, chunks))

# Get parameters from snakemake
datadir = snakemake.params.datadir
datasubdir = snakemake.params.datasubdir
n_sims = snakemake.params.n_sims
batch_size = snakemake.params.batch_size
prefix = snakemake.params.prefix
n_workers = snakemake.threads if hasattr(snakemake, 'threads') else 8

# Create dataset
create_zarr_dataset(
    snakemake.output.zarr,
    datadir,
    datasubdir,
    range(n_sims),
    batch_size=batch_size,
    prefix=prefix,
    n_workers=n_workers
) 