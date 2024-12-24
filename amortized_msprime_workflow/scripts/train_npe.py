import os
import pickle
import multiprocessing as mp
from functools import partial

from embedding_networks import *
from ts_simulators import *
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import posterior_nn
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter

# Use cuda if available unless we specifically asked for cpu in config
if hasattr(snakemake.params, "device") and snakemake.params.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_file_worker(file_path, mmap_mode='r'):
    """Worker function to load a single file"""
    return np.load(file_path, mmap_mode=mmap_mode)

def load_data_files(data_dir, datasubdir, n_train, batch_size=1000, num_workers=4):
    """
    Load data files efficiently using memory mapping, batch processing, and parallel loading.
    
    Args:
        data_dir: Directory containing the data files
        datasubdir: Subdirectory for x files
        n_train: Number of training samples to load
        batch_size: Size of batches to load at once
        num_workers: Number of parallel workers for loading files
    """
    n_train = int(n_train)
    
    # First, memory map all files to get shapes without loading
    first_x = np.load(os.path.join(data_dir, datasubdir, "x_0.npy"), mmap_mode='r')
    first_theta = np.load(os.path.join(data_dir, "theta_0.npy"), mmap_mode='r')
    
    # Pre-allocate tensors
    xs = torch.empty((n_train, *first_x.shape), dtype=torch.float32, device=device)
    thetas = torch.empty((n_train, *first_theta.shape), dtype=torch.float32, device=device)
    
    # Initialize process pool
    with mp.Pool(num_workers) as pool:
        # Load data in batches
        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_indices = range(start_idx, end_idx)
            
            # Prepare file paths for this batch
            x_paths = [os.path.join(data_dir, datasubdir, f"x_{i}.npy") for i in batch_indices]
            theta_paths = [os.path.join(data_dir, f"theta_{i}.npy") for i in batch_indices]
            
            # Load files in parallel
            x_batch_arrays = pool.map(partial(load_file_worker, mmap_mode='r'), x_paths)
            theta_batch_arrays = pool.map(partial(load_file_worker, mmap_mode='r'), theta_paths)
            
            # Stack arrays and convert to tensors
            x_batch = np.stack(x_batch_arrays)
            theta_batch = np.stack(theta_batch_arrays)
            
            # Transfer to device
            xs[start_idx:end_idx] = torch.from_numpy(x_batch).to(torch.float32)
            thetas[start_idx:end_idx] = torch.from_numpy(theta_batch).to(torch.float32)
            
            # Force garbage collection after each batch
            del x_batch, theta_batch, x_batch_arrays, theta_batch_arrays
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return thetas, xs


datadir = snakemake.params.datadir
datasubdir = snakemake.params.datasubdir
ts_processor = snakemake.params.ts_processor
posteriordir = snakemake.params.posteriordir
posteriorsubdir = snakemake.params.posteriorsubdir
ensemble = snakemake.params.ensemble
n_train = snakemake.params.n_train

if not os.path.isdir(posteriordir):
    os.mkdir(posteriordir)
if not os.path.isdir(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}")):
    os.mkdir(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}"))

thetas, xs = load_data_files(datadir, datasubdir, n_train)
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
prior = simulator.prior

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
    embedding_net = FCEmbedding(input_dim = xs.shape[-1]).to(device)
elif snakemake.params.embedding_net == "CNN":
    from sbi.neural_nets.embedding_nets import CNNEmbedding
    embedding_net = CNNEmbedding(input_shape=xs.shape[1:]).to(device)
elif snakemake.params.embedding_net =="Identity":
    embedding_net = torch.nn.Identity().to(device)

normalizing_flow_density_estimator = posterior_nn(
    model="maf_rqs",
    z_score_x="none",
    embedding_net=embedding_net
)
# get the log directory for tensorboard summary writer
log_dir = os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", "sbi_logs", f"rep_{ensemble}")
writer = SummaryWriter(log_dir=log_dir)
inference = SNPE(
    prior=prior,
    density_estimator=normalizing_flow_density_estimator,
    device=device.type,
    show_progress_bars=True,
    summary_writer=writer,
)

inference = inference.append_simulations(thetas, xs, proposal = prior)

posterior_estimator = inference.append_simulations(thetas, xs).train(
    show_train_summary=True,
    retrain_from_scratch=True,
    validation_fraction=0.2,
)

writer.close()
posterior = DirectPosterior(
    posterior_estimator=posterior_estimator, 
    prior=prior, 
    device=device.type)

with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"inference_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(inference, f)
# save posterior estimator (this contains trained normalizing flow - can be used for fine-turning?)
with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"posterior_estimator_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior_estimator, f)
with open(os.path.join(posteriordir, posteriorsubdir, f"n_train_{n_train}", f"posterior_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior, f)


