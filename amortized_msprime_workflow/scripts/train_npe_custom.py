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
    def __init__(self, zarr_path, indices=None, preload_chunks=True, ragged_x=False):
        self.root = zarr.open(zarr_path, mode='r')
        self.x_data = self.root['x']
        self.theta_data = self.root['theta']
        self.collate_fn = lambda x: pack_sequence(x, enforce_sorted=False) if ragged_x else torch.stack
        
        # Use all indices if none provided
        self.indices = indices if indices is not None else range(len(self.x_data))
        
        # Get shapes
        self.x_shape = self.root.attrs['x_shape']
        self.theta_shape = self.root.attrs['theta_shape']

        # Preload data into CPU memory if requested
        if preload_chunks:
            logger.info("Preloading data into CPU memory...")
            # Keep data on CPU
            self.x_cache = [
                torch.from_numpy(x.reshape(-1, *self.x_shape)).float()
                for x in self.x_data[self.indices]
            ]
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
            x = torch.from_numpy(self.x_data[actual_idx].reshape(-1, *self.x_shape)).float()
            theta = torch.from_numpy(self.theta_data[actual_idx]).float()
            return theta, x

    def __len__(self):
        return len(self.indices)

    def make_collate_fn(self):
        return lambda data: tuple(
            torch.stack((d[0] for d in data)),
            self.collate_fn((d[1] for d in data)),
        )
            

def train_on_disk(
    inference,
    train_zarr_path: str,
    n_train: int,
    embedding_net: torch.Module = torch.Identity(),
    training_batch_size: int = 512,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.2,
    stop_after_epochs: int = 20,
    max_num_epochs: int = 2**31 - 1,
    clip_max_norm: Optional[float] = 5.0,
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
    
    validation_batch_size = training_batch_size * 2
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=max(1, N_WORKERS // 2),
        pin_memory=True,  # Required for efficient non-blocking transfers
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        **dataloader_kwargs
    )

    embedding_net.to(device)
    first_theta, first_x = next(iter(train_loader))
    first_ex = embedding_net(first_x)

    class Model(LightningModule):
        def __init__(self):
            self.embedding_net = embedding_net
            self.density_estimator = inference._build_neural_net(first_theta, first_ex)

        def loss(self, thetas, xs):
            """
            Use uncorrected (first-round) loss, as there's only a single training round:
            https://github.com/sbi-dev/sbi/blob/9152e933871eee48a40be6f67b9f802a37c41b69/sbi/inference/trainers/npe/npe_base.py#L606
            """
            embedding = self.embedding_net.embedding(xs)
            losses = self.density_estimator._loss(thetas, embedding)
            return -torch.mean(losses).sum().item() #TODO why sum

        def training_step(self, data):
            loss = self.loss(*data)
            self.log("train_loss", loss, batch_size=training_batch_size, sync_dist=True)
            return loss

        def validation_step(self, data):
            loss = self.loss(*data)
            self.log("val_loss", loss, batch_size=validation_batch_size, sync_dist=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=learning_rate)


    model = Model()
    logger = TensorBoardLogger(embedding_path, version=model_name)
    save_best_model = ModelCheckpoint(
        dirpath=...,
        filename=os.path.basename(checkpoint_path).removesuffix(".ckpt"), 
        save_top_k=1, 
        monitor='val_loss',
        enable_version_counter=False,
    )
    stop_early = EarlyStopping(
        monitor="val_loss", 
        mode="min", 
        patience=stop_after_epochs,
    )
    trainer = Trainer(
        max_epochs=max_num_epochs,
        accelerator=..., #'cuda',
        devices=..., #[args.gpu],
        default_root_dir=..., #embedding_path,
        gradient_clip_val=clip_max_norm,
        logger=logger,
        callbacks=[save_best_model, stop_early],
    )
    logger.info("Training model...")
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )


    # return deepcopy of neural net

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
else:
    raise ValueError("Embedding network not implemented")

# Set up neural density estimator
normalizing_flow_density_estimator = posterior_nn(
    model="maf_rqs",
    z_score_x="none",
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

## Initialize inference with first round
#inference._data_round_index = [0]  # Initialize with round 0
#inference._proposal_roundwise = [prior]  # Use prior as the first proposal

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
