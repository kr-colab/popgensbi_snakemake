import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import LightningModule, Trainer, LightningDataModule
from sbi.neural_nets import posterior_nn

import embedding_networks
from data_handlers import ZarrDataset

torch.manual_seed(snakemake.params.random_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data loaders
class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        self.dataset = {}
        for split in ["sbi_train", "sbi_val"]:
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
        return self._dataloader(self.dataset["sbi_train"])

    def val_dataloader(self):
        return self._dataloader(self.dataset["sbi_val"], shuffle=False)


# Set up embedding net
embedding_config = snakemake.params.embedding_config.copy()
class_name = embedding_config.pop("class_name")
embedding_net = getattr(embedding_networks, class_name)(**embedding_config)

# TODO: this is not particularily efficient as "setup" is run again by the
# trainer, on a brand-new instantiation of the class. If we're caching stuff
# then this involves costly decompression. Need to read up on the sequence of
# events here.
datamodule = DataModule()
datamodule.setup()

# Initialize model and determine output dimension
# TODO: make various things, like "sbi_model", settable through config
sbi_model = "maf_rqs"
approximator = posterior_nn(model=sbi_model, z_score_x="none")
_, first_x = datamodule.dataset["sbi_train"][0] # skips packed_sequence
first_ex = embedding_net(first_x.unsqueeze(0)) # used to get latent dimension
theta = datamodule.dataset["sbi_train"].theta # used to get z-score scaling

# Lightning wrapper around embedding net + normalizing flow
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding_net = embedding_net
        self.normalizing_flow = approximator(theta, first_ex)

    def loss(self, target, features, log=None):
        embedding = self.embedding_net(features)
        loss = self.normalizing_flow.loss(target, embedding).mean()
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
    version=f"{class_name}-{sbi_model}"
        f"-{snakemake.params.optimizer}"
        f"-{snakemake.params.learning_rate}"
        f"-{snakemake.params.batch_size}",
)
save_best_model = ModelCheckpoint(
    dirpath=os.path.dirname(snakemake.output.embedding_net),
    filename=os.path.basename(snakemake.output.embedding_net),
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
    accelerator=device,
    devices=1, # TODO: enable multi-gpu training
    default_root_dir=os.path.dirname(snakemake.output.embedding_net),
    gradient_clip_val=snakemake.params.clip_max_norm,
    logger=logger,
    callbacks=[save_best_model, stop_early],
)
trainer.fit(model=model, datamodule=datamodule)

# Save best model
best_model = Model.load_from_checkpoint(f"{snakemake.output.embedding_net}.ckpt")
torch.save(best_model.embedding_net.cpu(), snakemake.output.embedding_net)
torch.save(best_model.normalizing_flow.cpu(), snakemake.output.normalizing_flow)

