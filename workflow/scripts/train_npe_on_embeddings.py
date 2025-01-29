import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import LightningModule, Trainer
from sbi.neural_nets import posterior_nn

from data_handlers import ZarrDataset, ArrayDataset

torch.manual_seed(snakemake.params.random_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Lightning wrapper around pretrained embedding network
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding_net = torch.load(snakemake.input.network)

    def predict_step(self, data):
        target, features = data
        return target, self.embedding_net(features)


embedding_model = Model()
trainer = Trainer(
    accelerator=device,
    devices=1,
    default_root_dir=os.path.dirname(snakemake.input.network),
    logger=False,
)

# Embed training and validation set
loader = {}
for split in ["sbi_train", "sbi_val"]:
    zarr_dataset = ZarrDataset(
        snakemake.input.zarr, 
        split=split,
        packed_sequence=snakemake.params.packed_sequence,
        use_cache=snakemake.params.use_cache,
    )
    zarr_loader = DataLoader(
        zarr_dataset,
        batch_size=snakemake.params.batch_size,
        shuffle=False,
        num_workers=snakemake.threads,
        pin_memory=True,  # Required for efficient non-blocking transfers
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
        collate_fn=zarr_dataset.make_collate_fn(),
    )
    embeddings = trainer.predict(model=embedding_model, dataloaders=zarr_loader)
    loader[split] = DataLoader(
        ArrayDataset(
            targets=torch.cat([x[0] for x in embeddings]).cpu().numpy(),
            features=torch.cat([x[1] for x in embeddings]).cpu().numpy(),
        ),
        batch_size=snakemake.params.batch_size,
        shuffle="train" in split,
        num_workers=snakemake.threads,
        drop_last=True,
    )

# Initialize model and determine output dimension
# TODO: make various things, like "sbi_model", settable through config
sbi_model = "maf_rqs"
approximator = posterior_nn(model=sbi_model, z_score_x="none")
first_theta, first_x = next(iter(loader["sbi_train"]))

# Lightning wrapper around normalizing flow
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.normalizing_flow = approximator(first_theta, first_x)

    def loss(self, target, embedding, log=None):
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
    version=f"{sbi_model}"
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
    devices=1, # TODO: enable multi-gpu training
    default_root_dir=os.path.dirname(snakemake.output.network),
    gradient_clip_val=snakemake.params.clip_max_norm,
    logger=logger,
    callbacks=[save_best_model, stop_early],
)
trainer.fit(
    model=model, 
    train_dataloaders=loader["sbi_train"], 
    val_dataloaders=loader["sbi_val"],
)

# Save best model
best_model = Model.load_from_checkpoint(f"{snakemake.output.network}.ckpt")
torch.save(best_model.normalizing_flow.cpu(), snakemake.output.network)

