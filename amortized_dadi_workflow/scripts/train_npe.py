import glob
import os
import pickle

from sbi.neural_nets.embedding_nets import *
from dadi_simulators import *
import numpy as np
import torch
from sbi.inference import SNPE
from sbi.inference.posteriors import DirectPosterior
from sbi.utils import posterior_nn
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter

def load_data_files(data_dir, sim_rounds):
    xs = []
    thetas = []
    x_files_all = glob.glob(os.path.join(data_dir, f"sim_round_{sim_rounds}/", "fs_*.npy"))
    x_files = natsorted(x_files_all)
    for xf in x_files:
        xs.append(np.load(xf))
        var = os.path.basename(xf)[3:-4]
        thetas.append(np.load(os.path.join(data_dir, f"sim_round_{sim_rounds}/", f"theta_{var}.npy")))

    xs = torch.from_numpy(np.array(xs)).to(torch.float32)
    thetas = torch.from_numpy(np.array(thetas)).to(torch.float32)
    return thetas, xs

datadir = snakemake.params.datadir
posteriordir = snakemake.params.posteriordir
sim_rounds = snakemake.params.sim_rounds
ensemble = snakemake.params.ensemble
embedding_net = snakemake.params.embedding_net

if not os.path.isdir(posteriordir):
    os.mkdir(posteriordir)
if not os.path.isdir(os.path.join(posteriordir, f"sim_round_{sim_rounds}")):
    os.mkdir(os.path.join(posteriordir, f"sim_round_{sim_rounds}"))

thetas, xs = load_data_files(datadir, sim_rounds)
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
prior = simulator.prior
ns = simulator.ns

if embedding_net == "CNN":
    if len(ns) == 2:
    # using default setup of CNN embedding network in sbi
        embedding_net = CNNEmbedding(
        input_shape=(ns[0]+1, ns[1]+1),
        )
elif embedding_net == "Identity":
    embedding_net = torch.nn.Identity().cuda()
elif embedding_net == "MLP":
    embedding_net = FCEmbedding(
        input_dim=ns[0]+1
    )


normalizing_flow_density_estimator = posterior_nn(
        model="maf_rqs",
        z_score_x="none",
        embedding_net=embedding_net)

log_dir = os.path.join(posteriordir, "sbi_logs", f"sim_round_{sim_rounds}", f"rep_{ensemble}")
writer = SummaryWriter(log_dir=log_dir)

inference = SNPE(
    prior=prior, 
    density_estimator=normalizing_flow_density_estimator, 
    device="cuda",
    show_progress_bars=True, 
    summary_writer=writer)

for n in range(int(sim_rounds)+1):
    thetas, xs = load_data_files(datadir, n)
    inference = inference.append_simulations(thetas, xs, proposal=prior)

posterior_estimator = inference.train(show_train_summary=True, 
    force_first_round_loss=True, 
    validation_fraction=0.2)

writer.close()
posterior = DirectPosterior(posterior_estimator=posterior_estimator, prior=prior, device="cuda")

with open(os.path.join(posteriordir, f"sim_round_{sim_rounds}", f"inference_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(inference, f)
# save posterior estimator (this contains trained normalizing flow - can be used for fine-turning?)
with open(os.path.join(posteriordir, f"sim_round_{sim_rounds}", f"posterior_estimator_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior_estimator, f)
with open(os.path.join(posteriordir, f"sim_round_{sim_rounds}", f"posterior_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior, f)
