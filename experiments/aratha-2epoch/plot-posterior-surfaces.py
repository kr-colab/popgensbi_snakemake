import argparse
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    "Plot bivariate posteriors output by `compare-posteriors.py`"
)
parser.add_argument(
    "--inpaths", 
    help="Directories containing likelihood surfaces, etc output by `compare-posteriors.py`",
    type=str, nargs="+", required=True,
)
parser.add_argument("--outpath", help="Path to output plot", type=str, required=True)
parser.add_argument("--vmin", help="Truncate surfaces at this lower bound for plotting surface", type=float, default=-15)
args = parser.parse_args()


# --- plot log-likelihood/log-probability surfaces ---
labels = {
    "MomentsGodambe": "moments-sfs",
    "ABC-SMC": "abc-smc",
    "ABC-Rejection": "abc-rejection",
    "SummaryStatisticsEmbedding": "npe-sfs",
    "ExchangeableCNN": "npe-cnn",
    "RNN": "npe-rnn",
}
plt.clf()
rows = len(args.inpaths)
cols = len(
    pickle.load(
        open(os.path.join(args.inpaths[0], "posterior-surfaces.pkl"), "rb")
    ).get("surfaces")
)
fig, axs = plt.subplots(
    rows, cols, 
    figsize=(cols * 2, rows * 2), 
    constrained_layout=True, 
    sharex=True,
    sharey=True,
    squeeze=False,
)

for j, inpath in enumerate(args.inpaths):

    fit = pickle.load(open(os.path.join(inpath, "moments-fit.pkl"), "rb"))
    params = pickle.load(open(os.path.join(inpath, "generative-params.pkl"), "rb"))
    surf = pickle.load(open(os.path.join(inpath, "posterior-surfaces.pkl"), "rb"))
    breaks_nu = surf["breaks_nu"]
    breaks_T = surf["breaks_T"]
    surfaces = surf["surfaces"]

    for i, name in enumerate(surfaces):
        grid_logprob = surfaces[name]
        if name == "MomentsGodambe":
            # ad hoc rescaling for loglikelihood surface to make
            # colorscale comparable with posteriors: first, rescale
            # so that curvature matches GIM, then normalize
            scaling = np.max(fit["GIM"] / fit["FIM"])
            grid_logprob /= scaling
        grid_logprob -= np.log(np.sum(np.exp(grid_logprob)))
        img = axs[j, i].pcolormesh(
            breaks_nu, breaks_T,
            grid_logprob.reshape(breaks_T.size - 1, breaks_nu.size - 1).T, 
            cmap="terrain", vmin=args.vmin,
        )
        axs[j, i].plot(*params, "o", color="red", markersize=4)
        if j == 0: axs[j, i].set_title(labels[name])
        #plt.colorbar(img, ax=axs[j, i], label="log posterior")
fig.supxlabel(r"Bottleneck severity ($\nu$)")
fig.supylabel(r"Time of bottleneck ($T$)")
plt.savefig(f"{args.outpath}")
