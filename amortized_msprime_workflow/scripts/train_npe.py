import glob
import os
import pickle

from embedding_networks import *
from ts_simulators import *
import numpy as np
import torch
from sbi.inference import NPE
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter

# Use cuda if available unless we specifically asked for cpu in config
if hasattr(snakemake.params, "device") and snakemake.params.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data_files(data_dir, ts_processor, sim_rounds):
    xs = []
    thetas = []
    x_files_all = glob.glob(os.path.join(data_dir, ts_processor, f"sim_round_{sim_rounds}/", "x_*.npy"))
    x_files = natsorted(x_files_all)
    for xf in x_files:
        xs.append(np.load(xf))
        var = os.path.basename(xf)[2:-4]
        thetas.append(np.load(os.path.join(data_dir, f"sim_round_{sim_rounds}/", f"theta_{var}.npy")))

    xs = torch.from_numpy(np.array(xs)).to(torch.float32).to(device)
    thetas = torch.from_numpy(np.array(thetas)).to(torch.float32).to(device)
    return thetas, xs


datadir = snakemake.params.datadir
ts_processor = snakemake.params.ts_processor
posteriordir = snakemake.params.posteriordir
sim_rounds = snakemake.params.sim_rounds
ensemble = snakemake.params.ensemble

if not os.path.isdir(posteriordir):
    os.mkdir(posteriordir)
if not os.path.isdir(os.path.join(posteriordir, ts_processor, f"sim_round_{sim_rounds}")):
    os.mkdir(os.path.join(posteriordir, ts_processor, f"sim_round_{sim_rounds}"))

thetas, xs = load_data_files(datadir, ts_processor, sim_rounds)
simulator = MODEL_LIST[snakemake.params.demog_model](snakemake)
prior = simulator.prior

if snakemake.params.embedding_net == "ExchangeableCNN":
    if ts_processor == "three_channel_feature_matrices":
        embedding_net = ExchangeableCNN(channels=3).to(device)
    elif ts_processor == "dinf_multiple_pops":
        embedding_net = ExchangeableCNN(unmasked_x_shps=[(2, v, snakemake.params.n_snps) for v in simulator.samples.values()]).to(device)
    else:
        embedding_net = ExchangeableCNN().to(device)
elif snakemake.params.embedding_net == "MLP":
    from sbi.neural_nets.embedding_nets import FCEmbedding
    embedding_net = FCEmbedding(input_dim = xs.shape[-1]).to(device)
elif snakemake.params.embedding_net == "CNN":
    from sbi.neural_nets.embedding_nets import CNNEmbedding
    embedding_net = CNNEmbedding(input_shape=xs.shape[1:]).to(device)
elif snakemake.params.embedding_net =="Identity":
    embedding_net = torch.nn.Identity().to(device)

normalizing_flow_density_estimator = posterior_nn(
    model="maf_rqs",
    z_score_x="none",
    embedding_net=embedding_net
)
# get the log directory for tensorboard summary writer
log_dir = os.path.join(posteriordir, ts_processor, f"sim_round_{sim_rounds}", "sbi_logs", f"rep_{ensemble}")
writer = SummaryWriter(log_dir=log_dir)
inference = NPE(
    prior=prior,
    density_estimator=normalizing_flow_density_estimator,
    device=device.type,
    show_progress_bars=True,
    summary_writer=writer,
)

for n in range(int(sim_rounds)+1):
    thetas, xs = load_data_files(datadir, ts_processor, n)
    inference = inference.append_simulations(thetas, xs, proposal=prior)

posterior_estimator = inference.append_simulations(thetas, xs).train(
    show_train_summary=True,
    retrain_from_scratch=True,
    validation_fraction=0.2,
)

writer.close()
posterior = DirectPosterior(
    posterior_estimator=posterior_estimator, 
    prior=prior, 
    device=device.type)

with open(os.path.join(posteriordir, ts_processor, f"sim_round_{sim_rounds}", f"inference_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(inference, f)
# save posterior estimator (this contains trained normalizing flow - can be used for fine-turning?)
with open(os.path.join(posteriordir, ts_processor, f"sim_round_{sim_rounds}", f"posterior_estimator_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior_estimator, f)
with open(os.path.join(posteriordir, ts_processor, f"sim_round_{sim_rounds}", f"posterior_rep_{ensemble}.pkl"), "wb") as f:
    pickle.dump(posterior, f)


