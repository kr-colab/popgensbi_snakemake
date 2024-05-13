import glob
import os
import pickle

from embedding_networks import ExchangeableCNN
from ts_simulators import AraTha_2epoch_simulator
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import posterior_nn
from natsort import natsorted

def load_data_files(data_dir, rounds):
    """
    Function that loads the simulated data from a given directory, converts to
    torch and reshapes as necessary

    :param data_dir: path to the directory storing simulated data files

    :returns: Tuple of torch.tensors of paramters, thetas, and simulated data, xs
    """
    x_files_all = glob.glob(os.path.join(data_dir, f"round_{rounds}/", "x_*.npy"))
    # Making sure that we use 0 to n_train_sims-1
    x_files = natsorted(x_files_all)[:snakemake.params.n_train_sims]
    xs = []
    thetas = []
    for xf in x_files:
        xs.append(np.load(xf))
        # Separate out the variable nambe (what * is)
        var = os.path.basename(xf)[2:-4]
        # Use it to find the corresponding theta file
        thetas.append(np.load(os.path.join(data_dir, f"round_{rounds}/", f"theta_{var}.npy")))
    xs = torch.from_numpy(np.array(xs))
    thetas = torch.from_numpy(np.array(thetas))
    return thetas, xs


datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
rounds = snakemake.params.rounds

thetas, xs = load_data_files(datadir, rounds)
if snakemake.params.demog_model == "AraTha_2epoch":
    simulator = AraTha_2epoch_simulator(snakemake)
prior = simulator.prior
if snakemake.params.embedding_net == "ExchangeableCNN":
    embedding_net = ExchangeableCNN().cuda()
normalizing_flow_density_estimator = posterior_nn(
    model="maf_rqs",
    z_score_x="none",
    embedding_net=embedding_net,
    hidden_features=64,
    num_transforms=6,
)
inference = SNPE(
    prior=prior,
    density_estimator=normalizing_flow_density_estimator,
    device="cuda",
    show_progress_bars=True,
)
posterior_estimator = inference.append_simulations(thetas, xs).train(
    show_train_summary=True,
    retrain_from_scratch=True,
    validation_fraction=0.2,
)
posterior = DirectPosterior(
    posterior_estimator=posterior_estimator, 
    prior=prior, 
    device="cuda")


if not os.path.isdir(os.path.join(posteriordir, f"round_{rounds}")):
    os.mkdir(os.path.join(posteriordir, f"round_{rounds}"))
pkl_file = os.path.join(posteriordir, f"round_{rounds}/", "posterior.pkl")
with open(pkl_file, "wb") as f:
    pickle.dump(posterior, f)

# Draw set of thetas from the posterior for the next round of simulations
x_obs = np.load(os.path.join(datadir, "x_obs.npy"))
x_obs = x_obs.reshape(1, *x_obs.shape)
proposal = posterior.set_default_x(torch.from_numpy(x_obs))
thetas = proposal.sample((snakemake.params.n_train_sims,))
np.save(os.path.join(datadir, f"round_{rounds}/", "thetas.npy"), thetas.cpu().numpy())
