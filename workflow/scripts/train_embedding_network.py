import os
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import LightningModule, Trainer

import embedding_networks
from data_handlers import ZarrDataset

torch.manual_seed(snakemake.params.random_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data loaders
dataset = {}
loader = {}
for split in ["pre_val", "pre_train"]:
    dataset[split] = ZarrDataset(
        snakemake.input.zarr, 
        split=split,
        packed_sequence=snakemake.params.packed_sequence,
    )
    loader[split] = DataLoader(
        dataset[split],
        batch_size=snakemake.params.batch_size,
        shuffle="train" in split,
        num_workers=snakemake.threads,
        pin_memory=True,  # Required for efficient non-blocking transfers
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=dataset[split].make_collate_fn(),
    )

# Normalise targets
target_mean = torch.mean(dataset["pre_train"].theta, axis=0)
target_prec = 1.0 / torch.std(dataset["pre_train"].theta, axis=0)

# Initialize model and determine output dimension
embedding_config = snakemake.params.embedding_config.copy()
class_name = embedding_config.pop("class_name")
embedding_net = getattr(embedding_networks, class_name)(**embedding_config)
_, first_x = dataset["pre_train"][0] # skips packed_sequence
first_output = embedding_net(first_x.unsqueeze(0))
embedding_dim = first_output.size(-1)

# Lightning wrapper
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

    def predict_step(self, data):
        target, features = data
        embedding = self.embedding_net(features)
        prediction = self.projection(embedding) / self.target_prec + self.target_mean
        return target, prediction

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
    accelerator=device,
    devices=1, # TODO: enable multi-GPU training
    default_root_dir=os.path.dirname(snakemake.output.network),
    gradient_clip_val=snakemake.params.clip_max_norm,
    logger=logger,
    callbacks=[save_best_model, stop_early],
)
trainer.fit(
    model=model, 
    train_dataloaders=loader["pre_train"], 
    val_dataloaders=loader["pre_val"],
)

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
