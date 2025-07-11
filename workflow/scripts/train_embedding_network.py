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

# A100 Optimizations
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matmul
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for convolutions
torch.backends.cudnn.benchmark = True  # Optimize cudnn
torch.set_float32_matmul_precision("medium")  # Use TF32 precision

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
        # Use optimal number of workers (2) based on our findings
        num_workers = getattr(snakemake.params, 'num_workers', 2)
        
        return DataLoader(
            dataset,
            batch_size=snakemake.params.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
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

datamodule = DataModule()
datamodule.setup()

# Determine output dimension, target scaling
_, first_x = datamodule.dataset["pre_train"][0]  # skips packed_sequence
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
        self.automatic_optimization = True

    def loss(self, target, features, log=None):
        embedding = self.embedding_net(features)
        prediction = self.projection(embedding)
        
        target = (target - self.target_mean) * self.target_prec
        mseloss = self.mseloss(target, prediction).mean(axis=0)
        loss = mseloss.sum()
        
        if log is not None:
            for i, x in enumerate(mseloss):
                self.log(
                    f"{log}_{i}",
                    x,
                    batch_size=snakemake.params.batch_size,
                    sync_dist=True,
                )
            self.log(
                f"{log}", loss, batch_size=snakemake.params.batch_size, sync_dist=True
            )
        return loss

    def training_step(self, data):
        return self.loss(*data, log="train_loss")

    def validation_step(self, data):
        return self.loss(*data, log="val_loss")

    def configure_optimizers(self):
        optimizer_name = snakemake.params.optimizer
        weight_decay = getattr(snakemake.params, 'weight_decay', 0.0)
        
        optimizer_class = getattr(torch.optim, optimizer_name)
        if optimizer_name == "Adam" and weight_decay > 0:
            # Use AdamW for weight decay with Adam
            return torch.optim.AdamW(
                self.parameters(), 
                lr=snakemake.params.learning_rate,
                weight_decay=weight_decay
            )
        else:
            return optimizer_class(self.parameters(), lr=snakemake.params.learning_rate)
    


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
    monitor="val_loss",
    enable_version_counter=False,
)
stop_early = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=snakemake.params.stop_after_epochs,
)


# Use mixed precision by default on CUDA
precision = "16-mixed" if device.startswith("cuda") else "32"

trainer = Trainer(
    max_epochs=snakemake.params.max_num_epochs,
    accelerator="gpu" if device.startswith("cuda") else "cpu",
    devices=devices,
    default_root_dir=os.path.dirname(snakemake.output.network),
    gradient_clip_val=snakemake.params.clip_max_norm,
    logger=logger,
    callbacks=[save_best_model, stop_early],
    # GPU optimizations
    precision=precision,
    deterministic=False,
    benchmark=True,
)
trainer.fit(model=model, datamodule=datamodule)

# Save best model
best_model = Model.load_from_checkpoint(f"{snakemake.output.network}.ckpt")
torch.save(best_model.embedding_net.cpu(), snakemake.output.network)