import os
import pickle

from dadi_simulators import *
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import posterior_nn
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter

def load_data_files(data_dir, n_train):
    n_train = int(n_train)
    xs = []
    thetas = []
    for n in range(n_train):
        xs.append(np.load(os.path.join(data_dir, f"fs_{n}.npy")))
        thetas.append(np.load(os.path.join(data_dir, f"theta_{n}.npy")))
    xs = torch.from_numpy(np.array(xs)).to(torch.float32)
    thetas = torch.from_numpy(np.array(thetas)).to(torch.float32)
    return thetas, xs

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
n_train = snakemake.params.n_train
ensemble = snakemake.params.ensemble
embedding_net = snakemake.params.embedding_net

print(n_train)

if not os.path.isdir(posteriordir):
    os.mkdir(posteriordir)
if not os.path.isdir(os.path.join(posteriordir, f"n_train_{n_train}")):
    os.mkdir(os.path.join(posteriordir, f"n_train_{n_train}"))

thetas, xs = load_data_files(datadir, n_train)
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
prior = simulator.prior
ns = simulator.ns

if embedding_net == "CNN":
    from sbi.neural_nets.embedding_nets import CNNEmbedding
    if len(ns) == 2:
    # using default setup of CNN embedding network in sbi
        embedding_net = CNNEmbedding(
        input_shape=(ns[0]+1, ns[1]+1),
        )
elif embedding_net == "Identity":
    embedding_net = torch.nn.Identity().cuda()
elif embedding_net == "MLP":
    from sbi.neural_nets.embedding_nets import FCEmbedding
    embedding_net = FCEmbedding(
        input_dim=ns[0]+1
    )


normalizing_flow_density_estimator = posterior_nn(
        model="maf_rqs",
        z_score_x="none",
        embedding_net=embedding_net)

log_dir = os.path.join(posteriordir, "sbi_logs", f"n_train_{n_train}", f"rep_{ensemble}")
writer = SummaryWriter(log_dir=log_dir)

inference = SNPE(
    prior=prior, 
    density_estimator=normalizing_flow_density_estimator, 
    show_progress_bars=True, 
    summary_writer=writer)

thetas, xs = load_data_files(datadir, n_train)
inference = inference.append_simulations(thetas, xs, proposal=prior)

posterior_estimator = inference.train(show_train_summary=True, 
    force_first_round_loss=True, 
    validation_fraction=0.2)

writer.close()
posterior = DirectPosterior(posterior_estimator=posterior_estimator, prior=prior)

with open(os.path.join(posteriordir, f"n_train_{n_train}", f"inference_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(inference, f)
# save posterior estimator (this contains trained normalizing flow - can be used for fine-turning?)
with open(os.path.join(posteriordir, f"n_train_{n_train}", f"posterior_estimator_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior_estimator, f)
with open(os.path.join(posteriordir, f"n_train_{n_train}", f"posterior_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior, f)
