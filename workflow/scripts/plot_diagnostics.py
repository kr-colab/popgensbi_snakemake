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
    
# Get information from prior
simulator_config = snakemake.params.simulator_config
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)
parameters = simulator.parameters # TODO: where to enforce this field always existing?
prior_mean = simulator.prior.mean.numpy()
prior_stdd = simulator.prior.stddev.numpy()

# Get means and credibility intervals for the test set
num_samples = 1000
num_quantiles = 10
alpha_grid = np.linspace(0, 0.5, num_quantiles + 2)[1:-1]

# Lightning wrapper around the whole kaboodle
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
        targets, features = data
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
        return samples, targets


# Sample posteriors on GPU
dataset = ZarrDataset(
    snakemake.input.zarr, 
    split="sbi_test",
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
posterior_draws = torch.cat([p[0] for p in samples]).cpu().numpy()
true_values = torch.cat([p[1] for p in samples]).cpu().numpy()

# Get prior intervals
prior_lower = np.quantile(true_values, alpha_grid, axis=0)
prior_upper = np.quantile(true_values, 1 - alpha_grid, axis=0)

# Get posterior coverage across grid of interval widths
posterior_means = posterior_draws.mean(axis=-1)
posterior_lower = np.quantile(posterior_draws, alpha_grid, axis=-1).transpose(1, 2, 0)
posterior_upper = np.quantile(posterior_draws, 1 - alpha_grid, axis=-1).transpose(1, 2, 0)
posterior_coverage = np.logical_and(
    true_values[:, :, np.newaxis] >= posterior_lower,
    true_values[:, :, np.newaxis] <= posterior_upper,
).mean(axis=0)

# Sanity check: posterior means vs truth
cols = 3
rows = int(np.ceil(len(parameters) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
for i, ax in enumerate(np.ravel(axs)):
    if i < len(parameters):
        mx = np.mean(true_values[:, i])
        ax.scatter(true_values[:, i], posterior_means[:, i])
        ax.axline((mx, mx), slope=1, color="black", linestyle="--")
        ax.set_title(parameters[i])
    else:
        ax.set_axis_off()
fig.supxlabel("Truth")
fig.supylabel("Posterior mean")
plt.savefig(snakemake.output.expectation)
plt.clf()

# Sanity check: posterior coverage vs expected coverage
cols = 3
rows = int(np.ceil(len(parameters) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
for i, ax in enumerate(np.ravel(axs)):
    if i < len(parameters):
        ax.scatter(1 - alpha_grid * 2, posterior_coverage[i], color="red")
        ax.axline((0, 0), (1, 1), color="black", linestyle="--")
        ax.set_title(parameters[i])
    else:
        ax.set_axis_off()
fig.supxlabel("Expected coverage")
fig.supylabel("Observed coverage")
plt.savefig(snakemake.output.calibration)
plt.clf()

# Sanity check: posterior coverage vs expected coverage
cols = 3
rows = int(np.ceil(len(parameters) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
for i, ax in enumerate(np.ravel(axs)):
    if i < len(parameters):
        # TODO: this assumes a uniform prior
        prior_width = prior_upper[:, i] - prior_lower[:, i]
        posterior_width = posterior_upper[:, i] - posterior_lower[:, i]
        concentration = posterior_width / prior_width[np.newaxis, :]
        ax.scatter(1 - alpha_grid * 2, concentration.mean(axis=0), color="red")
        ax.axhline(y=1, color="black", linestyle="--")
        ax.set_ylim(0, 1.1)
        ax.set_title(parameters[i])
    else:
        ax.set_axis_off()
fig.supxlabel("Expected coverage")
fig.supylabel("Posterior width / prior width")
plt.savefig(snakemake.output.concentration)
plt.clf()

# Sanity check: posterior at various points in parameter space
def contour_at_point(z, path, levels=[0.25, 0.5, 0.75]):
    z_scores = (true_values - prior_mean[np.newaxis, :]) / prior_stdd[np.newaxis, :]
    closest = np.power(z_scores - z, 2).sum(axis=1).argmin().item()
    samples_at_instance = posterior_draws[closest]
    true_values_at_instance = true_values[closest]
    dim = len(parameters)
    fig, axs = plt.subplots(dim, dim, figsize=(dim * 2, dim * 2))
    # TODO: what if prior is not uniform
    lower = simulator.prior.base_dist.low.numpy()
    upper = simulator.prior.base_dist.high.numpy()
    for i in range(dim):
        for j in range(dim):
            if i == j:
                a, b = lower[i], upper[i]
                sns.kdeplot(x=samples_at_instance[i], ax=axs[i, j])
                axs[i, j].axvline(x=true_values_at_instance[i], color="black")
                axs[i, j].set_xlim(a, b)
                axs[i, j].set_yticks([], [])
                axs[i, j].set_ylabel("")
                axs[i, j].set_xticks([a, b], [a, b])
                axs[i, j].set_xlabel(parameters[i])
            elif j > i:
                ai, bi = lower[i], upper[i]
                aj, bj = lower[j], upper[j]
                sns.kdeplot(
                    x=samples_at_instance[j], 
                    y=samples_at_instance[i], 
                    ax=axs[i, j], 
                    levels=levels,
                )
                axs[i, j].plot(
                    true_values_at_instance[j], 
                    true_values_at_instance[i], 
                    "o", 
                    color="black",
                )
                axs[i, j].set_xlim(aj, bj)
                axs[i, j].set_ylim(ai, bi)
                axs[i, j].set_xticks([], [])
                axs[i, j].set_xlabel("")
                axs[i, j].set_yticks([], [])
                axs[i, j].set_ylabel("")
            else:
                axs[i, j].set_axis_off()
    plt.savefig(path)
    plt.clf()

contour_at_point(np.zeros(len(simulator.parameters)), snakemake.output.at_prior_mean)
contour_at_point(np.full(len(simulator.parameters), -2), snakemake.output.at_prior_low)
contour_at_point(np.full(len(simulator.parameters), 2), snakemake.output.at_prior_high)
