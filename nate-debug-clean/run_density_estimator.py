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
from torch.nn.utils.rnn import pack_sequence
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import LightningModule, Trainer

from sbi.inference import SNPE
from sbi.utils import BoxUniform, posterior_nn
from sbi.inference.posteriors import DirectPosterior
from torch.utils.tensorboard import SummaryWriter

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
    train_path = os.path.join(data_path, "sbi")
    test_path = os.path.join(data_path, "test")
    sbi_path = os.path.join(data_path, "density_network")
    embedding_path = os.path.join(data_path, "embedding_network")
    frozen_model_path = os.path.join(embedding_path, f"{model_name}.pt")
    estimator_path = os.path.join(sbi_path, f"{model_name}.estimator.pkl")
    posterior_path = os.path.join(sbi_path, f"{model_name}.posterior.pkl")
    assert os.path.exists(train_path) 
    assert os.path.exists(test_path) 
    assert os.path.exists(frozen_model_path) 
    if not os.path.exists(sbi_path): os.makedirs(sbi_path)

    # prior
    target_low = []
    target_high = []
    target_name = []
    for p, v in config["prior"].items():
        assert v["func"] == "uniform"
        target_low.append(v["args"]["low"])
        target_high.append(v["args"]["high"])
        target_name.append(p)
    prior = BoxUniform(low=torch.tensor(target_low), high=torch.tensor(target_high), device="cpu")

    # data
    tensor_names = ["genotypes", "parameters"]
    tensor_ragged = [config["snps"] == "all", False]
    train_data = Data(train_path, tensor_names, ragged=tensor_ragged, max_snps=config["model"]["rnn-max-snps"])
    train_loader = DataLoader(
        train_data,
        batch_size=config["model"]["batch-size"],
        collate_fn=train_data.make_collate_fn(),
        shuffle=False,
        num_workers=args.cpus,
    )
    test_data = Data(test_path, tensor_names, ragged=tensor_ragged, max_snps=config["model"]["rnn-max-snps"])
    test_loader = DataLoader(
        test_data,
        batch_size=config["model"]["batch-size"],
        collate_fn=test_data.make_collate_fn(),
        shuffle=False,
        num_workers=args.cpus,
    )
    
    # feature embedding
    class Model(LightningModule):
        def __init__(self):
            super().__init__()
            self.model = torch.load(frozen_model_path, weights_only=False)
    
        def predict_step(self, data):
            features, target = data
            embedding = self.model.embedding(features)
            return embedding, target

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
    train_thetas = torch.concatenate([d[1] for d in train_predictions])

    test_predictions = trainer.predict(
        model=model,
        dataloaders=test_loader,
    )
    test_xs = torch.concatenate([d[0] for d in test_predictions])
    test_thetas = torch.concatenate([d[1] for d in test_predictions])

    # sbi training
    if not args.no_optim:
        normalizing_flow_density_estimator = posterior_nn(
            model="maf_rqs",
            z_score_x="none",
        )
        log_dir = os.path.join(sbi_path, "logs")
        writer = SummaryWriter(log_dir=log_dir)
        inference = SNPE(
            prior=prior,
            density_estimator=normalizing_flow_density_estimator,
            device="cpu",
            show_progress_bars=True,
            summary_writer=writer,
        )
        inference = inference.append_simulations(
            train_thetas,
            train_xs, 
        )
        estimator = inference.train(
            show_train_summary=True,
            retrain_from_scratch=True,
            validation_fraction=0.2,
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
    cols = 3
    rows = int(np.ceil(len(target_name) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        if i < len(target_name):
            a, b = target_low[i], target_high[i]
            ax.scatter(test_thetas[:, i].numpy(), test_means[:, i].numpy())
            ax.axline((a, a), (b, b), color="black", linestyle="--")
            ax.set_title(target_name[i])
        else:
            ax.set_axis_off()
    fig.supxlabel("Truth")
    fig.supylabel("Posterior mean")
    plt.savefig(os.path.join(sbi_path, f"{model_name}.test-posterior-means.png"))
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

    import matplotlib.pyplot as plt
    cols = 3
    rows = int(np.ceil(len(target_name) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        if i < len(target_name):
            ax.scatter(1 - alpha_grid * 2, test_cover[i], color="red")
            ax.axline((0, 0), (1, 1), color="black", linestyle="--")
            ax.set_title(target_name[i])
        else:
            ax.set_axis_off()
    fig.supxlabel("Expected coverage")
    fig.supylabel("Observed coverage")
    plt.savefig(os.path.join(sbi_path, f"{model_name}.test-posterior-coverage.png"))
    plt.clf()

    import matplotlib.pyplot as plt
    cols = 3
    rows = int(np.ceil(len(target_name) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), constrained_layout=True)
    for i, ax in enumerate(np.ravel(axs)):
        if i < len(target_name):
            prior_width = (target_high[i] - target_low[i]) * (1 - alpha_grid * 2)
            posterior_width = test_upper[:, i] - test_lower[:, i]
            concentration = posterior_width / prior_width[np.newaxis, :]
            ax.scatter(1 - alpha_grid * 2, concentration.mean(axis=0), color="red")
            ax.axhline(y=1, color="black", linestyle="--")
            ax.set_ylim(0, 2)
            ax.set_title(target_name[i])
        else:
            ax.set_axis_off()
    fig.supxlabel("Expected coverage")
    fig.supylabel("Posterior width / prior width")
    plt.savefig(os.path.join(sbi_path, f"{model_name}.test-posterior-concentration.png"))
    plt.clf()

    #sanity check: look at bivariate marginals for test instance closest to prior mean
    import matplotlib.pyplot as plt
    import seaborn as sns

    def contour_at_point(z, path, num_samples=1000):
        scaled_thetas = (test_thetas - prior.mean.unsqueeze(0)) / prior.stddev.unsqueeze(0)
        at_prior_mean = (scaled_thetas - z).pow(2).sum(dim=1).argmin().item()
        samples_at_prior_mean = posterior.sample([num_samples], x=test_xs[at_prior_mean], show_progress_bars=False)
        dim = len(target_name)
        fig, axs = plt.subplots(dim, dim, figsize=(dim * 2, dim * 2))
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    a, b = target_low[i], target_high[i]
                    sns.kdeplot(x=samples_at_prior_mean[:, i], ax=axs[i, j])
                    axs[i, j].axvline(x=test_thetas[at_prior_mean].numpy()[i], color="black")
                    axs[i, j].set_xlim(a, b)
                    axs[i, j].set_yticks([], [])
                    axs[i, j].set_ylabel("")
                    axs[i, j].set_xticks([a, b], [a, b])
                    axs[i, j].set_xlabel(target_name[i])
                elif j > i:
                    ai, bi = target_low[i], target_high[i]
                    aj, bj = target_low[j], target_high[j]
                    sns.kdeplot(x=samples_at_prior_mean[:, j], y=samples_at_prior_mean[:, i], ax=axs[i, j], levels=[0.25, 0.5, 0.75])
                    axs[i, j].plot(test_thetas[at_prior_mean].numpy()[j], test_thetas[at_prior_mean].numpy()[i], "o", color="black")
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

    contour_at_point(torch.zeros_like(prior.mean.unsqueeze(0)), os.path.join(sbi_path, f"{model_name}.test-posterior-at-prior-mean-contour.png"))
    contour_at_point(torch.full_like(prior.mean.unsqueeze(0), -2), os.path.join(sbi_path, f"{model_name}.test-posterior-at-prior-low-contour.png"))
    contour_at_point(torch.full_like(prior.mean.unsqueeze(0), 2), os.path.join(sbi_path, f"{model_name}.test-posterior-at-prior-high-contour.png"))

