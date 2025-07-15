import zarr
import numpy as np
import sys
import argparse
import yaml
import os

# put popgensbi_snakemake scripts in the load path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "scripts"))
import ts_simulators
import ts_processors
from utils import get_least_busy_gpu

parser = argparse.ArgumentParser()
parser.add_argument("--configfile", type=str, default="npe-config-alt/ooa2T12-1000g-rnn.yaml")
parser.add_argument("--output-path", type=str, default="/home/natep/public_html/popgensbi/ooa2T12/rnn/obs/combined-posteriors.png")
args = parser.parse_args()

config = yaml.safe_load(open(args.configfile))
vcf_config = config["prediction"]
simulator_config = config["simulator"]

# build paths to predictions, data
project_dir = config["project_dir"]
uid = (
    f"{config['simulator']['class_name']}-"
    f"{config['processor']['class_name']}-"
    f"{config['embedding_network']['class_name']}-"
    f"{int(config['random_seed'])}-"
    f"{int(config['n_train'])}"
)
uid += "-sep" if config["train_embedding_net_separately"] else "-e2e"
work_dir = os.path.join(project_dir, uid, os.path.basename(vcf_config["vcf"]))

# instantiate simulator
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)

# pull out posterior samples
pred_zarr = zarr.open(os.path.join(work_dir, "tensors", "zarr"))
data_zarr = zarr.open(os.path.join(work_dir, "vcz"))
post_vars = pred_zarr.predictions[:].var(axis=-1)
post_means = pred_zarr.predictions[:].mean(axis=-1)
post_weights = post_vars / post_vars.sum(axis=0)[None, :]
post_all = pred_zarr.predictions[:].transpose(1, 0, 2).reshape(len(simulator.parameters), -1)


# plot posterior means
import matplotlib.pyplot as plt
rows = post_means.shape[-1]
cols = post_means.shape[-1]
fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*2.5), constrained_layout=True)
for i in range(post_means.shape[-1]):
    for j in range(post_means.shape[-1]):
        if i == j:
            axs[i, j].hist(post_all[i], bins=1000)
            axs[i, j].spines['left'].set_visible(False)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].set_yticklabels([])
            axs[i, j].tick_params(axis='y', length=0)
            axs[i, j].set_ylabel(simulator.parameters[i])
        elif i < j:
            axs[i, j].hexbin(post_all[j], post_all[i], cmap="terrain")
        else:
            axs[i, j].set_visible(False)
plt.savefig(args.output_path)

