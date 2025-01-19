import os
import numpy as np
import torch
import zarr

from torch.utils.data import DataLoader
from lightning import LightningModule, Trainer
from data_handlers import ZarrDataset

# Lightning wrapper around pretrained network
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding_net = torch.load(snakemake.input.network)

    def predict_step(self, data):
        target, features = data
        return self.embedding_net(features)

model = Model()
trainer = Trainer(
    accelerator=snakemake.params.device,
    devices=[snakemake.params.device_id],
    default_root_dir=os.path.dirname(snakemake.input.network),
    logger=False,
)

# Embed training set and stash in zarr
root = zarr.open(snakemake.input.zarr, "rw")
for split in ["sbi_train", "sbi_val", "sbi_test"]:
    dataset = ZarrDataset(snakemake.input.zarr, split=split)
    loader = DataLoader(
        dataset,
        batch_size=snakemake.params.batch_size,
        shuffle=False,
        num_workers=snakemake.threads,
        pin_memory=True,  # Required for efficient non-blocking transfers
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        collate_fn=dataset.make_collate_fn(),
    )
    embeddings = torch.cat(trainer.predict(model=model, dataloaders=loader))
    embeddings = embeddings.cpu().numpy()
    for i, x in zip(dataset.indices, embeddings):
        root.embeddings[i] = x
        root.embeddings_shape[i] = x.shape

with open(snakemake.output.done, "w") as handle:
    handle.write("done")
