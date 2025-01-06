"""Custom training implementation for Neural Posterior Estimation using disk-based data loading."""

import os
import pickle
import time
import logging
import sys
from copy import deepcopy
from typing import Optional, Dict, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from sbi.inference import SNPE
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import posterior_nn
from sbi.utils.sbiutils import x_shape_from_simulation
from torch.distributions import MultivariateNormal, Uniform
import zarr

from embedding_networks import (
    ExchangeableCNN,
    ExchangeableCNN_OG,
)
from ts_simulators import MODEL_LIST

# Configure logging to both file and stdout
log_file = snakemake.log[0]  # Access log file directly from snakemake.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_STOP_EPOCHS = 20
DEFAULT_CLIP_NORM = 5.0
MAX_EPOCHS = 2**31 - 1

# Get number of threads from SLURM environment, defaulting to 1 if not set
N_THREADS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
# Use threads-1 for workers, minimum of 1
N_WORKERS = max(1, N_THREADS - 1)

# Device configuration with error handling
try:
    if hasattr(snakemake.params, "device") and snakemake.params.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
except Exception as e:
    logger.error(f"Error setting up device: {e}")
    raise

class ZarrDataset(Dataset):
    """Dataset that loads from zarr files efficiently"""
    def __init__(self, zarr_path, indices=None, preload_chunks=True):
        self.root = zarr.open(zarr_path, mode='r')
        self.x_data = self.root['x']
        self.theta_data = self.root['theta']
        
        # Use all indices if none provided
        self.indices = indices if indices is not None else range(len(self.x_data))
        
        # Get shapes
        self.x_shape = self.x_data.shape[1:]
        self.theta_shape = self.theta_data.shape[1:]

        # Preload data into CPU memory if requested
        if preload_chunks:
            logger.info("Preloading data into CPU memory...")
            # Keep data on CPU
            self.x_cache = torch.from_numpy(self.x_data[self.indices]).float()
            self.theta_cache = torch.from_numpy(self.theta_data[self.indices]).float()
            logger.info("Data preloaded")
            self.use_cache = True
        else:
            self.use_cache = False

    def __getitem__(self, idx):
        if self.use_cache:
            # Return CPU tensors
            return self.theta_cache[idx].clone(), self.x_cache[idx].clone()
        else:
            # Load from zarr and return CPU tensors
            actual_idx = self.indices[idx]
            x = torch.from_numpy(self.x_data[actual_idx]).float()
            theta = torch.from_numpy(self.theta_data[actual_idx]).float()
            return theta, x

    def __len__(self):
        return len(self.indices)

def train_on_disk(
    inference,
    train_zarr_path: str,
    n_train: int,
    training_batch_size: int = 512,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.1,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 2**31 - 1,
    clip_max_norm: Optional[float] = 5.0,
    calibration_kernel: Optional[Callable] = None,
    resume_training: bool = False,
    force_first_round_loss: bool = False,
    discard_prior_samples: bool = False,
    retrain_from_scratch: bool = False,
    show_train_summary: bool = True,
    preload_chunks: bool = True,
    dataloader_kwargs: Optional[Dict] = None,
) -> nn.Module:
    """Optimized training loop using zarr files."""
    
    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    
    logger.info("Initializing training...")
    start_time = time.time()

    # Create train/validation split
    all_indices = np.arange(n_train)
    np.random.shuffle(all_indices)
    n_validation = int(n_train * validation_fraction)
    validation_indices = all_indices[:n_validation]
    training_indices = all_indices[n_validation:]
    
    logger.info("Creating datasets...")
    dataset_start = time.time()
    # Create datasets using zarr with preloading to CPU
    train_dataset = ZarrDataset(train_zarr_path, indices=training_indices, preload_chunks=preload_chunks)
    val_dataset = ZarrDataset(train_zarr_path, indices=validation_indices, preload_chunks=preload_chunks)
    logger.info(f"Dataset creation took {time.time() - dataset_start:.2f}s")
    
    # Optimize DataLoader settings with pin_memory for efficient async transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=N_WORKERS,
        pin_memory=True,  # Required for efficient non-blocking transfers
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_batch_size * 2,
        shuffle=False,
        num_workers=max(1, N_WORKERS // 2),
        pin_memory=True,  # Required for efficient non-blocking transfers
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        **dataloader_kwargs
    )

    # Initialize neural network if needed
    logger.info("Setting up neural network...")
    net_start = time.time()
    if inference._neural_net is None or retrain_from_scratch:
        # Load first batch to get shapes (already on GPU)
        first_theta, first_x = next(iter(train_loader))
        inference._neural_net = inference._build_neural_net(
            first_theta,  # Already on GPU
            first_x,  # Already on GPU
        )
        inference._x_shape = x_shape_from_simulation(first_x)

    # Move network to device
    inference._neural_net.to(device)
    logger.info(f"Neural network setup took {time.time() - net_start:.2f}s")
    
    # Get proposal for current round
    proposal = inference._proposal_roundwise[inference._round]
    
    # Initialize or resume training state
    if not resume_training:
        optimizer = torch.optim.Adam(
            list(inference._neural_net.parameters()),
            lr=learning_rate
        )
        epoch, val_log_prob = 0, float("-Inf")
    else:
        optimizer = inference.optimizer
        epoch = inference.epoch
        val_log_prob = inference._val_log_prob

    # Initialize training summaries
    if not hasattr(inference, "_summary"):
        inference._summary = {
            "training_log_probs": [],
            "validation_log_probs": [],
            "epochs_trained": [],
            "best_validation_log_prob": [],
            "epoch_durations_sec": []
        }

    best_val_log_prob = float("-Inf")
    epochs_since_last_improvement = 0

    # Add timing tracking
    start_time = time.time()
    total_batches = 0
    
    while epoch <= max_num_epochs:
        epoch_start = time.time()
        batch_count = 0
        
        # Training phase
        inference._neural_net.train()
        train_log_probs_sum = 0
        
        for batch_thetas, batch_xs in train_loader:
            batch_start = time.time()
            
            # Asynchronous transfer to GPU
            batch_thetas = batch_thetas.to(device, non_blocking=True)
            batch_xs = batch_xs.to(device, non_blocking=True)
            
            # Create masks on GPU directly
            batch_masks = torch.ones(batch_xs.shape[0], device=device)
            
            optimizer.zero_grad(set_to_none=True)
            
            train_losses = inference._loss(
                batch_thetas,
                batch_xs,
                batch_masks,
                proposal=proposal,
                calibration_kernel=lambda x: torch.ones(batch_thetas.shape[0], device=device),
                force_first_round_loss=force_first_round_loss
            )
            train_loss = torch.mean(train_losses)
            train_log_probs_sum -= train_losses.sum().item()
            
            train_loss.backward()
            if clip_max_norm is not None:
                clip_grad_norm_(inference._neural_net.parameters(), max_norm=clip_max_norm)
            optimizer.step()

            batch_count += 1
            if batch_count % 10 == 0:
                current_speed = batch_count / (time.time() - epoch_start)
                logger.info(f"Epoch {epoch}, Batch {batch_count}: "
                          f"Speed = {current_speed:.2f} batches/sec, "
                          f"Loss = {train_loss.item():.4f}")

            # Clean up
            del train_losses, train_loss, batch_thetas, batch_xs, batch_masks

        # End of epoch timing
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, "
                   f"Average speed: {batch_count/epoch_time:.2f} batches/sec")

        # Validation phase
        inference._neural_net.eval()
        val_log_probs_sum = 0
        
        with torch.no_grad():
            for batch_thetas, batch_xs in val_loader:
                # Asynchronous transfer to GPU
                batch_thetas = batch_thetas.to(device, non_blocking=True)
                batch_xs = batch_xs.to(device, non_blocking=True)
                batch_masks = torch.ones(batch_xs.shape[0], device=device)
                
                val_losses = inference._loss(
                    batch_thetas,
                    batch_xs,
                    batch_masks,
                    proposal=proposal,
                    calibration_kernel=lambda x: torch.ones(batch_thetas.shape[0], device=device),
                    force_first_round_loss=force_first_round_loss
                )
                val_log_probs_sum -= val_losses.sum().item()

                # Clean up
                del val_losses, batch_thetas, batch_xs, batch_masks

        # Calculate metrics
        train_log_prob = train_log_probs_sum / len(train_loader.dataset)
        val_log_prob = val_log_probs_sum / len(val_loader.dataset)

        # Update summaries
        inference._summary["training_log_probs"].append(train_log_prob)
        inference._summary["validation_log_probs"].append(val_log_prob)
        inference._summary["epoch_durations_sec"].append(time.time() - epoch_start)

        # Early stopping logic
        if val_log_prob > best_val_log_prob:
            best_val_log_prob = val_log_prob
            epochs_since_last_improvement = 0
            best_model_state_dict = inference._neural_net.state_dict().copy()
        else:
            epochs_since_last_improvement += 1

        if epochs_since_last_improvement >= stop_after_epochs:
            break

        # Update progress
        if inference._show_progress_bars:
            inference._maybe_show_progress(inference._show_progress_bars, epoch)

        epoch += 1

    # Restore best model
    if best_model_state_dict is not None:
        inference._neural_net.load_state_dict(best_model_state_dict)

    # Update final summaries
    inference._summary["epochs_trained"].append(epoch)
    inference._summary["best_validation_log_prob"].append(best_val_log_prob)
    inference._summarize(round_=inference._round)

    if show_train_summary:
        print(inference._describe_round(inference._round, inference._summary))

    # Clear gradients
    inference._neural_net.zero_grad(set_to_none=True)

    return deepcopy(inference._neural_net)

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

# Create directories if they don't exist
if not os.path.isdir(posteriordir):
    os.mkdir(posteriordir)
if not os.path.isdir(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}")):
    os.mkdir(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}"))

# Initialize simulator and prior
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
prior = simulator.prior

# Get first batch to determine shapes
first_x = np.load(os.path.join(datadir, datasubdir, "x_0.npy"), mmap_mode='r')
xs_shape = first_x.shape

# Set up embedding network
if snakemake.params.embedding_net == "ExchangeableCNN":
    if ts_processor == "three_channel_feature_matrices":
        embedding_net = ExchangeableCNN(channels=3).to(device)
    elif ts_processor == "dinf_multiple_pops":
        embedding_net = ExchangeableCNN(unmasked_x_shps=[(2, v, snakemake.params.n_snps) for v in simulator.samples.values()]).to(device)
    else:
        embedding_net = ExchangeableCNN().to(device)
elif snakemake.params.embedding_net == "ExchangeableCNN_OG":
    if ts_processor == "three_channel_feature_matrices":
        embedding_net = ExchangeableCNN_OG(channels=3).to(device)
    elif ts_processor == "dinf_multiple_pops":
        embedding_net = ExchangeableCNN_OG(unmasked_x_shps=[(2, v, snakemake.params.n_snps) for v in simulator.samples.values()]).to(device)
    else:
        embedding_net = ExchangeableCNN_OG().to(device)
elif snakemake.params.embedding_net == "MLP":
    from sbi.neural_nets.embedding_nets import FCEmbedding
    embedding_net = FCEmbedding(input_dim = xs_shape[-1]).to(device)
elif snakemake.params.embedding_net == "CNN":
    from sbi.neural_nets.embedding_nets import CNNEmbedding
    embedding_net = CNNEmbedding(input_shape=xs_shape[1:]).to(device)
elif snakemake.params.embedding_net =="Identity":
    embedding_net = torch.nn.Identity().to(device)

# Set up neural density estimator
normalizing_flow_density_estimator = posterior_nn(
    model="maf_rqs",
    z_score_x="none",
    embedding_net=embedding_net
)

# Set up tensorboard logging
log_dir = os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "sbi_logs", f"rep_{ensemble}")

# Initialize inference
inference = SNPE(
    prior=prior,
    density_estimator=normalizing_flow_density_estimator,
    device=device.type,
    show_progress_bars=True,
    summary_writer=SummaryWriter(log_dir=log_dir)
)

# Initialize inference with first round
inference._data_round_index = [0]  # Initialize with round 0
inference._proposal_roundwise = [prior]  # Use prior as the first proposal

# Use zarr paths directly
train_zarr_path = os.path.join(datadir, datasubdir, "train.zarr")

# Train using zarr files
posterior_estimator = train_on_disk(
    inference,
    train_zarr_path,
    n_train,
    training_batch_size=batch_size,
    learning_rate=learning_rate,
    validation_fraction=validation_fraction,
    stop_after_epochs=stop_after_epochs,
    clip_max_norm=clip_max_norm,
    show_train_summary=True
)

# Create posterior
posterior = DirectPosterior(
    posterior_estimator=posterior_estimator, 
    prior=prior, 
    device=device.type
)

# Save results
with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"inference_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(inference, f)
with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"posterior_estimator_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior_estimator, f)
with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"posterior_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior, f)