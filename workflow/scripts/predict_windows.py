import os
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from math import ceil
from torch import Tensor
from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from sbi.inference import DirectPosterior
from sbi.utils import BoxUniform

import ts_simulators
from data_handlers import ZarrDataset
from utils import get_least_busy_gpu
rng = np.random.default_rng(snakemake.params.random_seed)
torch.manual_seed(snakemake.params.random_seed)


# Determine the device
if torch.cuda.is_available():
    best_gpu = get_least_busy_gpu()
    device = f"cuda:{best_gpu}"
    devices = [best_gpu]  # Set devices to the least busy GPU
else:
    device = "cpu"
    devices = 1  # Ensure CPU compatibility
    
    
# Get simulator config
simulator_config = snakemake.params.simulator_config
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)

num_samples = 1000

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
samples = torch.cat(samples).cpu().numpy() # (num_windows, num_parameters, num_samples)
zarr.save(f"{snakemake.input.zarr}/predictions", samples)

# sanity check: posteriors across windows for each parameter
# TODO: this needs work
#  - violin plot probably would be better? 
#  - but becomes much harder to see with lots of windows
#  - heatmap "fades out" if posterior is too diffuse
root = zarr.open(f"{snakemake.input.zarr}", "r")
n_windows, n_params, n_samples = samples.shape
n_grid = 51  # grid size for heatmap
inch_per_window = 1 / 5
inch_per_param = 3
contig_breaks = np.flatnonzero(root.window_contig[:-1] != root.window_contig[1:])
fig, axs = plt.subplots(
    n_params, 1, 
    figsize=(ceil(inch_per_window * n_windows), ceil(inch_per_param * n_params)),
    constrained_layout=True, 
    squeeze=False, 
    sharex=True,
)
for i, ax in enumerate(axs.ravel()):
    subset = samples[:, i]
    y_min = simulator.prior.base_dist.low[i].item()
    y_max = simulator.prior.base_dist.high[i].item()
    #sns.violinplot(subset.T, inner=None, ax=ax, color="dodgerblue")
    #ax.plot(np.arange(n_windows), subset.mean(axis=1), color="dodgerblue")
    y_breaks = np.linspace(y_min, y_max, n_grid)
    y_grid = np.digitize(samples[:, i], y_breaks) - 1
    assert np.all(np.logical_and(y_grid > -1, y_grid < n_grid - 1))
    y_grid = np.apply_along_axis(np.bincount, axis=1, arr=y_grid, minlength=n_grid - 1)
    y_grid = y_grid.astype(float) / n_samples
    sns.heatmap(y_grid.T, cbar=False, ax=ax, xticklabels=False, yticklabels=False)
    for b in contig_breaks:
        ax.axvline(x=b + 1, color="red", linestyle="dashed")
    # TODO: format numeric tick labels nicely, somehow
    ax.set_yticks(np.arange(n_grid)[::10], [f"{x:.2e}" for x in y_breaks][::10])
    ax.set_xticks(np.arange(n_windows)[::10], np.arange(n_windows)[::10])
    ax.invert_yaxis()
    ax.set_title(simulator.parameters[i])
fig.supylabel("Posterior samples")
fig.supxlabel("Window")
plt.savefig(f"{snakemake.output.predictions}")

