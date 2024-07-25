import glob
import os
import pickle

from embedding_networks import *
from sbi.neural_nets.embedding_nets import *
from ts_simulators import *
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import posterior_nn
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter

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
ensemble = snakemake.params.ensemble

if not os.path.isdir(os.path.join(posteriordir, f"round_{rounds}")):
    os.mkdir(os.path.join(posteriordir, f"round_{rounds}"))

thetas, xs = load_data_files(datadir, rounds)
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
prior = simulator.prior

if snakemake.params.embedding_net == "ExchangeableCNN":
    if snakemake.params.ts_processor == "three_channel_feature_matrices":
        embedding_net = ExchangeableCNN(channels=3).cuda()
    elif snakemake.params.ts_processor == "dinf_multiple_pops":
        embedding_net = ExchangeableCNN(unmasked_x_shps=[(2, v, snakemake.params.n_snps) for v in simulator.samples.values()]).cuda()
    else:
        embedding_net = ExchangeableCNN().cuda()
elif snakemake.params.embedding_net == "MLP":
    embedding_net = FCEmbedding(input_dim = xs.shape[-1]).cuda()
elif snakemake.params.embedding_net == "CNN":
    embedding_net = CNNEmbedding(input_shape=xs.shape[1:]).cuda()
elif snakemake.params.embedding_net == "SPIDNA":
    embedding_net=SPIDNA(num_output=thetas.shape[-1]).cuda()
elif snakemake.params.embedding_net == "FlagelNN":
    embedding_net=FlagelNN(SNP_max=xs.shape[-1], nindiv=xs.shape[-2], num_params=thetas.shape[-1]).cuda()

normalizing_flow_density_estimator = posterior_nn(
    model="maf_rqs",
    z_score_x="none",
    embedding_net=embedding_net,
    hidden_features=64,
    num_transforms=6,
)
# get the log directory for tensorboard summary writer
log_dir = os.path.join(posteriordir, f"round_{rounds}", "sbi_logs", f"rep_{ensemble}")
writer = SummaryWriter(log_dir=log_dir)
inference = SNPE(
    prior=prior,
    density_estimator=normalizing_flow_density_estimator,
    device="cuda",
    show_progress_bars=True,
    summary_writer=writer,
)
posterior_estimator = inference.append_simulations(thetas, xs).train(
    show_train_summary=True,
    retrain_from_scratch=True,
    validation_fraction=0.2,
)

writer.close()
posterior = DirectPosterior(
    posterior_estimator=posterior_estimator, 
    prior=prior, 
    device="cuda")

pkl_file = os.path.join(posteriordir, f"round_{rounds}/", f"posterior_{ensemble}.pkl")
with open(pkl_file, "wb") as f:
    pickle.dump(posterior, f)


