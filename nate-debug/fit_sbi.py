import argparse
import os
import numpy as np
import yaml
import torch
import warnings
import sbi
import pickle
import ray

warnings.simplefilter(action='ignore', category=FutureWarning)

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import LightningModule, Trainer
from networks import ExchangeableCNN

from sbi.inference import SNPE
from sbi.utils import BoxUniform, posterior_nn
from sbi.inference.posteriors import DirectPosterior
from torch.utils.tensorboard import SummaryWriter

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
    torch.manual_seed(config["seed"])

    sim_name = f"{config['snps']}"
    for n in config["samples"].values():
        sim_name += f"-{n}"

    model_name = \
        f"{config['model']['prefix']}-"       \
        f"{config['model']['optimizer']}-"    \
        f"{config['model']['batch-size']}-"   \
        f"{config['model']['learning-rate']}" \

    data_path = os.path.join(config["path"], sim_name)
    train_path = os.path.join(data_path, "sbi", "train")
    val_path = os.path.join(data_path, "sbi", "val")
    test_path = os.path.join(data_path, "sbi", "test")
    sbi_path = os.path.join(data_path, "sbi", "fit")
    frozen_model_path = os.path.join(data_path, "embedding", "model.pt")
    estimator_path = os.path.join(sbi_path, "posterior_estimator.pkl")
    posterior_path = os.path.join(sbi_path, "posterior.pkl")
    assert os.path.exists(train_path) 
    assert os.path.exists(val_path) 
    assert os.path.exists(test_path) 
    assert os.path.exists(frozen_model_path) 
    if not os.path.exists(sbi_path): os.makedirs(sbi_path)

    # prior
    low = []
    high = []
    keep = []
    for v in config["prior"].values():
        keep.append(v["args"]["low"] != v["args"]["high"])
        low.append(v["args"]["low"])
        high.append(v["args"]["high"])
    prior = BoxUniform(low=torch.tensor(low)[keep], high=torch.tensor(high)[keep], device="cpu")

    # data
    tensor_names = ["genotypes", "parameters"]
    train_data = Data(train_path, tensor_names)
    train_loader = DataLoader(
        train_data,
        batch_size=config["model"]["batch-size"],
        collate_fn=train_data.make_collate_fn(),
        shuffle=False,
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
    
    # lightning wrapper
    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.load(frozen_model_path, weights_only=False)
    
        def predict_step(self, data):
            features, target = data
            embedding = self.model.embedding(features)
            return embedding, target

    # feature embedding
    model = Model()
    trainer = Trainer(
        accelerator='cuda',
        devices=[args.gpu],
        default_root_dir=sbi_path,
    )

    train_predictions = trainer.predict(
        model=model,
        dataloaders=train_loader,
    )
    train_xs = torch.concatenate([d[0] for d in train_predictions])
    train_thetas = torch.concatenate([d[1] for d in train_predictions])[:, keep]

    val_predictions = trainer.predict(
        model=model,
        dataloaders=val_loader,
    )
    val_xs = torch.concatenate([d[0] for d in val_predictions])
    val_thetas = torch.concatenate([d[1] for d in val_predictions])[:, keep]

    test_predictions = trainer.predict(
        model=model,
        dataloaders=test_loader,
    )
    test_xs = torch.concatenate([d[0] for d in test_predictions])
    test_thetas = torch.concatenate([d[1] for d in test_predictions])[:, keep]

    # sbi training
    if not args.no_optim:
        thetas = torch.cat([train_thetas, val_thetas], dim=0)
        xs = torch.cat([train_xs, val_xs], dim=0)

        normalizing_flow_density_estimator = posterior_nn(
            model="maf_rqs",
            z_score_x="none",
            #embedding_net=embedding_net
        )
        log_dir = os.path.join(sbi_path, "logs")
        writer = SummaryWriter(log_dir=log_dir)
        inference = SNPE(
            prior=prior,
            density_estimator=normalizing_flow_density_estimator,
            device="cpu", #model.device(),
            show_progress_bars=True,
            summary_writer=writer,
        )
        inference = inference.append_simulations(
            thetas,
            xs, 
            #proposal=prior,
        )
        estimator = inference.train(
            show_train_summary=True,
            retrain_from_scratch=True,
            validation_fraction=config["sbi"]["val-frac"],
        )
        posterior = DirectPosterior(
            posterior_estimator=estimator, 
            prior=prior, 
            device="cpu",
        )
        pickle.dump(estimator, open(estimator_path, "wb"))
        pickle.dump(posterior, open(posterior_path, "wb"))
    else:
        estimator = pickle.load(open(estimator_path, "rb"))
        posterior = pickle.load(open(posterior_path, "rb"))

    # sanity check: posterior means on test
    ray.init(num_cpus=config["cpus"])

    @ray.remote
    def posterior_mean(x, num_samples=100):
        return posterior.sample([num_samples], x=x, show_progress_bars=False).mean(axis=0)

    test_means = []
    job_list = [posterior_mean.remote(x, 100) for x in test_xs]
    test_means = torch.stack(ray.get(job_list))

    import matplotlib.pyplot as plt
    parameter_names = ["N_A", "N_YRI", "N_CEU_1", "N_CEU_2", "m", "T", "Tp"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        a, b = low[i + 1], high[i + 1]
        ax.scatter(test_thetas[:, i].numpy(), test_means[:, i].numpy())
        ax.axline((a, a), (b, b), color="black", linestyle="--")
        ax.set_title(parameter_names[i + 1])
    fig.supxlabel("Truth")
    fig.supylabel("Posterior mean")
    plt.savefig(os.path.join(sbi_path, "test-posterior-means.png"))
    plt.clf()

    # sanity check: posterior coverage/concentration on test
    @ray.remote
    def posterior_intervals(x, alpha_grid, num_samples=1000):
        #samples = prior.sample([num_samples]).numpy()  # double sanity check: prior should be perfectly calibrated
        samples = posterior.sample([num_samples], x=x, show_progress_bars=False).numpy()
        lower = np.full((samples.shape[1], alpha_grid.size), np.nan)
        upper = np.full((samples.shape[1], alpha_grid.size), np.nan)
        for i, alpha in enumerate(alpha_grid):
            lower[:, i] = np.quantile(samples, alpha, axis=0)
            upper[:, i] = np.quantile(samples, 1 - alpha, axis=0)
        return lower, upper

    test_cover = []
    num_alpha = 10
    alpha_grid = np.linspace(0, 0.5, num_alpha + 2)[1:-1]
    job_list = [posterior_intervals.remote(x, alpha_grid, 1000) for x in test_xs]
    test_cover = ray.get(job_list)
    test_lower = np.stack([d[0] for d in test_cover])
    test_upper = np.stack([d[1] for d in test_cover])
    test_truth = test_thetas.numpy()
    test_cover = np.logical_and(
        test_truth[:, :, np.newaxis] >= test_lower,
        test_truth[:, :, np.newaxis] <= test_upper,
    ).sum(axis=0) / test_truth.shape[0]
    print(test_cover)

    import matplotlib.pyplot as plt
    parameter_names = ["N_A", "N_YRI", "N_CEU_1", "N_CEU_2", "m", "T", "Tp"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        ax.scatter(1 - alpha_grid * 2, test_cover[i], color="red")
        ax.axline((0, 0), (1, 1), color="black", linestyle="--")
        ax.set_title(parameter_names[i + 1])
    fig.supxlabel("Expected coverage")
    fig.supylabel("Observed coverage")
    plt.savefig(os.path.join(sbi_path, "test-posterior-coverage.png"))
    plt.clf()

    import matplotlib.pyplot as plt
    parameter_names = ["N_A", "N_YRI", "N_CEU_1", "N_CEU_2", "m", "T", "Tp"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        prior_width = (high[i + 1] - low[i + 1]) * (1 - alpha_grid * 2)
        posterior_width = test_upper[:, i] - test_lower[:, i]
        concentration = posterior_width / prior_width[np.newaxis, :]
        ax.scatter(1 - alpha_grid * 2, concentration.mean(axis=0), color="red")
        ax.axhline(y=1, color="black", linestyle="--")
        ax.set_ylim(0, 2)
        ax.set_title(parameter_names[i + 1])
    fig.supxlabel("Expected coverage")
    fig.supylabel("Posterior width / prior width")
    plt.savefig(os.path.join(sbi_path, "test-posterior-concentration.png"))
    plt.clf()
