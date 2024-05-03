import glob
import os
import pickle

from embedding_networks import ExchangeableCNN
from simulators import AraTha_2epoch_simulator
import defopt
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import posterior_nn


def load_data_files(data_dir: str):
    """
    Function that loads the simulated data from a given directory, converts to
    torch and reshapes as necessary

    :param data_dir: path to the directory storing simulated data files

    :returns: Tuple of torch.tensors of paramters, thetas, and simulated data, xs
    """
    data_files = glob.glob(f"{data_dir}/rep*.npy")
    xs = []
    thetas = []
    for df in data_files:
        simdata = np.load(df, allow_pickle=True).item()
        xs.append(simdata["x"])
        thetas.append(simdata["theta"])
    Nsims = thetas[-1].shape[0]
    Nreps = len(thetas)
    theta_dim = thetas[-1].shape[-1]
    x_shape = xs[-1].shape[1:]
    xs = torch.from_numpy(np.array(xs).reshape((Nsims * Nreps, *x_shape)))
    thetas = torch.from_numpy(np.array(thetas).reshape((Nsims * Nreps, theta_dim)))
    return thetas, xs


def main(
    data_dir: str,
    outdir: str,
    *,
    prefix: str = "",
):

    thetas, xs = load_data_files(data_dir)
    simulator = AraTha_2epoch_simulator()
    prior = simulator.prior

    embedding_net = ExchangeableCNN().cuda()
    normalizing_flow_density_estimator = posterior_nn(
        model="maf_rqs",
        z_score_x="none",
        embedding_net=embedding_net,
        hidden_features=64,
        num_transforms=6,
    )
    inference = SNPE(
        prior,
        density_estimator=normalizing_flow_density_estimator,
        device="cuda",
        show_progress_bars=True,
    )
    posterior_estimator = inference.append_simulations(thetas, xs).train(
        show_train_summary=True,
        retrain_from_scratch=True,
        validation_fraction=0.2,
    )
    posterior = DirectPosterior(posterior_estimator, prior, device="cuda")


    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if prefix != "":
        prefix += "_"
    pkl_file = f"{outdir}/{prefix}posterior.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(posterior, f)

if __name__ == "__main__":
    defopt.run(main)
