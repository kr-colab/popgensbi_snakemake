import argparse
import os
import numpy as np
import yaml
import torch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import LightningModule, Trainer
from networks import ExchangeableCNN, RNN

# --- lib --- #

class Data(Dataset):
    def __init__(self, path, names, ragged=None, max_snps=None):
        self.path = path
        self.seeds = np.load(os.path.join(path, "seeds.npy"))
        self.names = names
        self.ragged = ragged
        if self.ragged is None:
            self.ragged = [False] * len(self.names)
        self.max_snps = max_snps

    def __len__(self):
        return self.seeds.size

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        return tuple(
            torch.load(os.path.join(self.path, f"{seed}.{n}.pt")) 
            for n in self.names
        )

    def make_collate_fn(self):
        return lambda data: tuple(
            pack_sequence(tuple(d[i][:self.max_snps] for d in data), enforce_sorted=False) if ragged
            else torch.stack(tuple(d[i] for d in data))
            for i, ragged in enumerate(self.ragged)
        )


# --- impl --- #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file")
    parser.add_argument("--refit", action="store_true", help="Refit model")
    parser.add_argument("--no-optim", action="store_true", help="Skip optimization")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--cpus", type=int, default=10, help="Number of CPUs for data loader")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    sim_name = f"{config['snps']}"
    for n in config["samples"].values():
        sim_name += f"-{n}"

    model_name = \
        f"{config['model']['prefix']}-"       \
        f"{config['model']['optimizer']}-"    \
        f"{config['model']['batch-size']}-"   \
        f"{config['model']['learning-rate']}" \

    data_path = os.path.join(config["path"], sim_name)
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")
    embedding_path = os.path.join(data_path, "embedding_network")
    checkpoint_path = os.path.join(embedding_path, model_name + ".ckpt")
    frozen_model_path = os.path.join(embedding_path, f"{model_name}.pt")
    assert os.path.exists(train_path) 
    assert os.path.exists(val_path) 
    assert os.path.exists(test_path) 
    if not os.path.exists(embedding_path): os.makedirs(embedding_path)

    # data
    batch_size = config["model"]["batch-size"]
    tensor_names = ["genotypes", "parameters"]
    tensor_ragged = [config["snps"] == "all", False]
    train_data = Data(train_path, tensor_names, ragged=tensor_ragged, max_snps=config["model"]["rnn-max-snps"])
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=train_data.make_collate_fn(),
        shuffle=True,
        num_workers=args.cpus,
    )
    val_data = Data(val_path, tensor_names, ragged=tensor_ragged, max_snps=config["model"]["rnn-max-snps"])
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=val_data.make_collate_fn(),
        shuffle=False,
        num_workers=args.cpus,
    )
    test_data = Data(test_path, tensor_names, ragged=tensor_ragged, max_snps=config["model"]["rnn-max-snps"])
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=test_data.make_collate_fn(),
        shuffle=False,
        num_workers=args.cpus,
    )
    
    # normalise targets
    rng = np.random.default_rng(config["seed"])
    target_mean = []
    target_prec = []
    target_name = []
    for p, v in config["prior"].items():
        distr = getattr(rng, v["func"])
        x = distr(**v["args"], size=1000)
        mean = np.mean(x)
        std = np.std(x)
        target_mean.append(mean)
        target_prec.append(1.0 / std if std > 0 else 0.0)
        target_name.append(p)
    target_mean = np.array(target_mean, dtype=np.float32)
    target_prec = np.array(target_prec, dtype=np.float32)

    # lightning wrapper
    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.register_buffer("target_mean", Tensor(target_mean))
            self.register_buffer("target_prec", Tensor(target_prec)) 
            if config["snps"] == "all":
                self.model = RNN(
                    input_size=np.sum([v for v in config["samples"].values()]) + 1, 
                    output_size=len(target_mean),
                )
            else:
                self.model = ExchangeableCNN(
                    latent_dim=len(target_mean),
                    unmasked_x_shps=[(2, v, config["snps"]) for v in config["samples"].values()]
                )
            self.mseloss = torch.nn.MSELoss(reduction="none")
    
        def loss(self, features, target, log=None):
            prediction = self.model(features)
            target = (target.squeeze(-1) - self.target_mean) * self.target_prec
            mseloss = self.mseloss(target, prediction).mean(axis=0)
            loss = mseloss.sum()
            if log is not None:
                for i, x in enumerate(mseloss):
                    self.log(f"{log}_{i}", x, batch_size=batch_size, sync_dist=True)
                self.log(f"{log}", loss, batch_size=batch_size, sync_dist=True)
            return loss
    
        def training_step(self, data):
            return self.loss(*data, log="train_loss")
    
        def validation_step(self, data):
            return self.loss(*data, log="val_loss")

        def predict_step(self, data):
            features, target = data
            prediction = self.model(features) / self.target_prec + self.target_mean
            return prediction, target
    
        def configure_optimizers(self):
            optimizer = getattr(torch.optim, config["model"]["optimizer"])
            return optimizer(self.parameters(), lr=config["model"]["learning-rate"])

    # fit model
    if not args.no_optim:
        model = Model()
        logger = TensorBoardLogger(embedding_path, version=model_name)
        save_best_model = ModelCheckpoint(
            dirpath=os.path.dirname(checkpoint_path), 
            filename=os.path.basename(checkpoint_path).removesuffix(".ckpt"), 
            save_top_k=1, 
            monitor='val_loss',
            enable_version_counter=False,
        )
        trainer = Trainer(
            max_epochs=config["model"]["max-epochs"], 
            accelerator='cuda',
            devices=[args.gpu],
            default_root_dir=embedding_path,
            logger=logger,
            callbacks=[save_best_model],
        )
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader,
            ckpt_path=None if args.refit or not os.path.exists(checkpoint_path) \
                else checkpoint_path,
        )

    # sanity check: predictions on test
    model = Model()
    trainer = Trainer(
        accelerator='cuda',
        devices=[args.gpu],
        default_root_dir=embedding_path,
    )
    batched_predictions = trainer.predict(
        model=model,
        dataloaders=test_loader,
        ckpt_path=checkpoint_path,
    )
    predictions = torch.concatenate([d[0] for d in batched_predictions])
    targets = torch.concatenate([d[1] for d in batched_predictions])

    import matplotlib.pyplot as plt
    cols = 3
    rows = int(np.ceil(len(target_name) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        if i < target_mean.size:
            a = target_mean[i]
            ax.scatter(targets[:, i].numpy(), predictions[:, i].numpy())
            ax.axline((a, a), slope=1, color="black", linestyle="--")
            ax.set_title(target_name[i])
        else:
            ax.set_axis_off()
    fig.supxlabel("Truth")
    fig.supylabel("Neural net estimator")
    plt.savefig(os.path.join(embedding_path, f"{model_name}.test-predictions.png"))

    # save out model
    torch.save(model.model, frozen_model_path)

