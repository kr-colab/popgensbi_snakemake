import os
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import LightningModule, Trainer, LightningDataModule

import embedding_networks
from data_handlers import ZarrDataset
from utils import get_least_busy_gpu

torch.manual_seed(snakemake.params.random_seed)


# Determine the device
if torch.cuda.is_available():
    best_gpu = get_least_busy_gpu()
    device = f"cuda:{best_gpu}"
    devices = [best_gpu]  # Set devices to the least busy GPU
else:
    device = "cpu"
    devices = 1  # Ensure CPU compatibility


# Lightning wrapper around data loaders
class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        self.dataset = {}
        for split in ["pre_train", "pre_val"]:
            self.dataset[split] = ZarrDataset(
                snakemake.input.zarr,
                split=split,
                packed_sequence=snakemake.params.packed_sequence,
                use_cache=snakemake.params.use_cache,
            )

    @staticmethod
    def _dataloader(dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=snakemake.params.batch_size,
            shuffle=shuffle,
            num_workers=snakemake.threads,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
            collate_fn=dataset.make_collate_fn(),
        )

    def train_dataloader(self):
        return self._dataloader(self.dataset["pre_train"])

    def val_dataloader(self):
        return self._dataloader(self.dataset["pre_val"], shuffle=False)


# Initialize model
embedding_config = snakemake.params.embedding_config.copy()
class_name = embedding_config.pop("class_name")
if class_name == "SummaryStatisticsEmbedding":
    # Remove all unnecessary parameters
    embedding_config = {}  # Clear the config for this specific class
embedding_net = getattr(embedding_networks, class_name)(**embedding_config)

# TODO: this is not particularily efficient as "setup" is run again by the
# trainer, on a brand-new instantiation of the class. If we're caching stuff
# then this involves costly decompression. Need to read up on the sequence of
# events here.
datamodule = DataModule()
datamodule.setup()

# Determine output dimension, target scaling
_, first_x = datamodule.dataset["pre_train"][0] # skips packed_sequence
first_output = embedding_net(first_x.unsqueeze(0))
embedding_dim = first_output.size(-1)
target_mean = torch.mean(datamodule.dataset["pre_train"].theta, axis=0)
target_prec = 1.0 / torch.std(datamodule.dataset["pre_train"].theta, axis=0)

# Lightning wrapper around embedding net
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_prec", target_prec) 
        self.embedding_net = embedding_net
        self.projection = torch.nn.Linear(embedding_dim, target_mean.numel())
        self.mseloss = torch.nn.MSELoss(reduction="none")

    def loss(self, target, features, log=None):
        embedding = self.embedding_net(features)
        prediction = self.projection(embedding)
        target = (target - self.target_mean) * self.target_prec
        mseloss = self.mseloss(target, prediction).mean(axis=0)
        loss = mseloss.sum()
        if log is not None:
            for i, x in enumerate(mseloss):
                self.log(f"{log}_{i}", x, batch_size=snakemake.params.batch_size, sync_dist=True)
            self.log(f"{log}", loss, batch_size=snakemake.params.batch_size, sync_dist=True)
        return loss

    def training_step(self, data):
        return self.loss(*data, log="train_loss")

    def validation_step(self, data):
        return self.loss(*data, log="val_loss")

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, snakemake.params.optimizer)
        return optimizer(self.parameters(), lr=snakemake.params.learning_rate)


# Train model
model = Model()
logger = TensorBoardLogger(
    snakemake.log.tensorboard,
    version=f"{class_name}"
        f"-{snakemake.params.optimizer}"
        f"-{snakemake.params.learning_rate}"
        f"-{snakemake.params.batch_size}",
)
save_best_model = ModelCheckpoint(
    dirpath=os.path.dirname(snakemake.output.network),
    filename=os.path.basename(snakemake.output.network),
    save_top_k=1, 
    monitor='val_loss',
    enable_version_counter=False,
)
stop_early = EarlyStopping(
    monitor="val_loss", 
    mode="min", 
    patience=snakemake.params.stop_after_epochs,
)
trainer = Trainer(
    max_epochs=snakemake.params.max_num_epochs,
    accelerator="gpu" if device.startswith("cuda") else "cpu",
    devices=devices,
    default_root_dir=os.path.dirname(snakemake.output.network),
    gradient_clip_val=snakemake.params.clip_max_norm,
    logger=logger,
    callbacks=[save_best_model, stop_early],
)
trainer.fit(model=model, datamodule=datamodule)

# Save best model
best_model = Model.load_from_checkpoint(f"{snakemake.output.network}.ckpt")
torch.save(best_model.embedding_net.cpu(), snakemake.output.network)

## Sanity check: estimator vs truth
#import matplotlib.pyplot as plt
#
#predictions = trainer.predict(
#    model=best_model,
#    dataloaders=loader["pre_test"],
#)
#targets = torch.cat([x[0] for x in predictions])
#predictions = torch.cat([x[1] for x in predictions])
#
#cols = 3
#rows = int(np.ceil(target_mean.numel() / cols))
#fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
#for i, ax in enumerate(np.ravel(axs)):
#    if i < target_mean.numel():
#        a = target_mean[i]
#        ax.scatter(targets[:, i].numpy(), predictions[:, i].numpy())
#        ax.axline((a, a), slope=1, color="black", linestyle="--")
#        # TODO: need to load in correct simulator
#        #ax.set_title(simulator.parameter_names)
#    else:
#        ax.set_axis_off()
#fig.supxlabel("Truth")
#fig.supylabel("Neural net estimator")
#plt.savefig(snakemake.output.sanity_check)
