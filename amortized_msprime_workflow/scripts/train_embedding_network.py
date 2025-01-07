"""Training implementation for embedding network using disk-based data loading."""

import os
import pickle
import time
import logging
import sys
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import zarr

from embedding_networks import (
    ExchangeableCNN,
    ExchangeableCNN_OG,
)
from ts_simulators import MODEL_LIST

# Configure logging
log_file = snakemake.log[0]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Get number of threads from SLURM environment, defaulting to 1 if not set
N_THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
N_WORKERS = max(1, N_THREADS - 1)

# Device configuration
try:
    if hasattr(snakemake.params, "device") and snakemake.params.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
except Exception as e:
    logger.error(f"Error setting up device: {e}")
    raise

class ZarrDataset(torch.utils.data.Dataset):
    """Dataset that loads from zarr files efficiently"""
    def __init__(self, zarr_path, indices=None, preload_chunks=True):
        self.root = zarr.open(zarr_path, mode='r')
        self.x_data = self.root['x']
        self.theta_data = self.root['theta']
        
        self.indices = indices if indices is not None else range(len(self.x_data))
        self.x_shape = self.x_data.shape[1:]
        self.theta_shape = self.theta_data.shape[1:]

        if preload_chunks:
            logger.info("Preloading data into CPU memory...")
            self.x_cache = torch.from_numpy(self.x_data[self.indices]).float()
            self.theta_cache = torch.from_numpy(self.theta_data[self.indices]).float()
            logger.info("Data preloaded")
            self.use_cache = True
        else:
            self.use_cache = False

    def __getitem__(self, idx):
        if self.use_cache:
            return self.theta_cache[idx].clone(), self.x_cache[idx].clone()
        else:
            actual_idx = self.indices[idx]
            x = torch.from_numpy(self.x_data[actual_idx]).float()
            theta = torch.from_numpy(self.theta_data[actual_idx]).float()
            return theta, x

    def __len__(self):
        return len(self.indices)

def train_embedding_network(
    train_zarr_path: str,
    n_train: int,
    training_batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    stop_after_epochs: int,
    clip_max_norm: float,
    show_train_summary: bool = True,
    preload_chunks: bool = True,
) -> nn.Module:
    """Training loop for embedding network using zarr files."""
    
    logger.info("Initializing training...")
    start_time = time.time()

    # Create train/validation split
    all_indices = np.arange(n_train)
    np.random.shuffle(all_indices)
    n_validation = int(n_train * validation_fraction)
    validation_indices = all_indices[:n_validation]
    training_indices = all_indices[n_validation:]
    
    # Create datasets
    logger.info("Creating datasets...")
    dataset_start = time.time()
    train_dataset = ZarrDataset(train_zarr_path, indices=training_indices, preload_chunks=preload_chunks)
    val_dataset = ZarrDataset(train_zarr_path, indices=validation_indices, preload_chunks=preload_chunks)
    logger.info(f"Dataset creation took {time.time() - dataset_start:.2f}s")
    
    # TODO: Implement training logic
    # This should include:
    # 1. Setting up the embedding network
    # 2. Defining the loss function
    # 3. Training loop with validation
    # 4. Early stopping
    # 5. Model checkpointing
    
    return None  # Replace with trained embedding network

# Get parameters from snakemake
datadir = snakemake.params.datadir
datasubdir = snakemake.params.datasubdir
ts_processor = snakemake.params.ts_processor
posteriordir = snakemake.params.posteriordir
posteriorsubdir = snakemake.params.posteriorsubdir
ensemble = snakemake.params.ensemble
n_train = int(snakemake.params.n_train)

# Get training parameters from config
batch_size = snakemake.params.batch_size
learning_rate = snakemake.params.learning_rate
validation_fraction = snakemake.params.validation_fraction
stop_after_epochs = snakemake.params.stop_after_epoch
clip_max_norm = snakemake.params.clip_max_norm

# Set up tensorboard logging
log_dir = os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "embedding_logs", f"rep_{ensemble}")

# Train embedding network
embedding_network = train_embedding_network(
    train_zarr_path=os.path.join(datadir, datasubdir, "train.zarr"),
    n_train=n_train,
    training_batch_size=batch_size,
    learning_rate=learning_rate,
    validation_fraction=validation_fraction,
    stop_after_epochs=stop_after_epochs,
    clip_max_norm=clip_max_norm,
    show_train_summary=True
)

# Save results
training_info = {
    "n_train": n_train,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "validation_fraction": validation_fraction,
    "stop_after_epochs": stop_after_epochs,
    "clip_max_norm": clip_max_norm
}

with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"embedding_network_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(embedding_network, f)
with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"embedding_training_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(training_info, f) 