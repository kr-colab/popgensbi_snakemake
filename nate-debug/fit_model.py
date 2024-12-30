import argparse
import os
import numpy as np
import yaml
import torch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import LightningModule, Trainer
from networks import ExchangeableCNN

# --- lib --- #

class Data(Dataset):
    def __init__(self, path, names):
        self.path = path
        self.seeds = np.load(os.path.join(path, "seeds.npy"))
        self.names = names

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
            torch.stack(tuple(d[i] for d in data)) 
            for i, _ in enumerate(self.names)
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
    embedding_path = os.path.join(data_path, "embedding")
    checkpoint_path = os.path.join(embedding_path, model_name + ".ckpt")
    frozen_model_path = os.path.join(embedding_path, "model.pt")
    assert os.path.exists(train_path) 
    assert os.path.exists(val_path) 
    assert os.path.exists(test_path) 
    if not os.path.exists(embedding_path): os.makedirs(embedding_path)

    # data
    tensor_names = ["genotypes", "parameters"]
    train_data = Data(train_path, tensor_names)
    train_loader = DataLoader(
        train_data,
        batch_size=config["model"]["batch-size"],
        collate_fn=train_data.make_collate_fn(),
        shuffle=True,
        num_workers=args.cpus,
    )
    val_data = Data(val_path, tensor_names)
    val_loader = DataLoader(
        val_data,
        batch_size=config["model"]["batch-size"],
        collate_fn=val_data.make_collate_fn(),
        shuffle=False,
        num_workers=args.cpus,
    )
    test_data = Data(test_path, tensor_names)
    test_loader = DataLoader(
        test_data,
        batch_size=config["model"]["batch-size"],
        collate_fn=test_data.make_collate_fn(),
        shuffle=False,
        num_workers=args.cpus,
    )
    
    # normalise targets
    rng = np.random.default_rng(config["seed"])
    target_mean = []
    target_prec = []
    for v in config["prior"].values():
        distr = getattr(rng, v["func"])
        x = distr(**v["args"], size=1000)
        mean = np.mean(x)
        std = np.std(x)
        target_mean.append(mean)
        target_prec.append(1.0 / std if std > 0 else 0.0)
    target_mean = np.array(target_mean, dtype=np.float32)
    target_prec = np.array(target_prec, dtype=np.float32)

    # lightning wrapper
    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.register_buffer("target_mean", Tensor(target_mean))
            self.register_buffer("target_prec", Tensor(target_prec)) 
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
                    self.log(f"{log}_{i}", x, sync_dist=True)
                self.log(f"{log}", loss, sync_dist=True)
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

    # predictions on test
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
    parameter_names = ["N_A", "N_YRI", "N_CEU_1", "N_CEU_2", "m", "T", "Tp"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        a, b = target_mean[i + 1], target_mean[i + 1] + 1 / target_prec[i + 1]
        ax.scatter(targets[:, i + 1].numpy(), predictions[:, i + 1].numpy())
        ax.axline((a, a), (b, b), color="black", linestyle="--")
        ax.set_title(parameter_names[i + 1])
    fig.supxlabel("Truth")
    fig.supylabel("Neural net estimator")
    plt.savefig(os.path.join(embedding_path, "test-predictions.png"))

    # save out model
    torch.save(model.model, frozen_model_path)

