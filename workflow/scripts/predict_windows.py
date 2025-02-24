import os
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch import Tensor
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from sbi.inference import DirectPosterior
from sbi.utils import BoxUniform

import ts_simulators
from data_handlers import ZarrDataset
from utils import get_least_busy_gpu
rng = np.random.default_rng(snakemake.params.random_seed)

# Determine the device
if torch.cuda.is_available():
    best_gpu = get_least_busy_gpu()
    device = f"cuda:{best_gpu}"
    devices = [best_gpu]  # Set devices to the least busy GPU
else:
    device = "cpu"
    devices = 1  # Ensure CPU compatibility
    
    
num_samples = 1000
# Get simulator config
simulator_config = snakemake.params.simulator_config
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding_net = torch.load(
            snakemake.input.embedding_net,
            weights_only=False,
        )
        self.normalizing_flow = torch.load(
            snakemake.input.normalizing_flow, 
            weights_only=False,
        )

    def predict_step(self, data):
        _, features = data
        embeddings = self.embedding_net(features)
        # TODO: prior parameters have to be on device for SBI code to work
        # TODO: what if prior isn't uniform
        prior = BoxUniform(
            simulator.prior.base_dist.low.to(self.device), 
            simulator.prior.base_dist.high.to(self.device),
        )
        posterior = DirectPosterior(
            posterior_estimator=self.normalizing_flow,
            prior=prior,
            device=self.device,
        )
        samples = posterior.sample_batched(
            [num_samples], 
            x=embeddings, 
            show_progress_bars=False,
        ).permute(1, 2, 0) # dimensions are (batch, parameter, npe sample)
        return samples

# Sample posteriors on GPU
dataset = ZarrDataset(
    snakemake.input.zarr, 
    split="vcf_windows",
    packed_sequence=snakemake.params.packed_sequence,
    use_cache=snakemake.params.use_cache,
)
loader = DataLoader(
    dataset,
    batch_size=snakemake.params.batch_size,
    shuffle=False,
    num_workers=snakemake.threads,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
    collate_fn=dataset.make_collate_fn(),
)

model = Model()
trainer = Trainer(
    accelerator="gpu" if device.startswith("cuda") else "cpu",
    devices=devices,
    logger=False,
)
samples = trainer.predict(model=model, dataloaders=loader)
# samples is a list of tensors, each with shape (batch_size, num_parameters, num_samples)
# Save samples to zarr

with zarr.open(snakemake.input.zarr, mode="a") as zarr_store:
    if "predictions" in zarr_store:
        del zarr_store["predictions"]  # Remove existing dataset
    zarr_store["predictions"] = samples