import os
import pickle

import corner
import defopt
import matplotlib.pyplot as plt
import numpy as np
import torch
from ts_simulators import AraTha_2epoch_simulator
from ts_processors import *
def main(
    data_dir: str,
    outdir: str):

    simulator = AraTha_2epoch_simulator()
    bounds = simulator.bounds
    theta_star = simulator.true_values
    ts_processor = dinf_extract(n_snps=2000)
    with open(f"{outdir}/posterior.pkl", "rb") as f:
        posterior = pickle.load(f)   

    try:
        ts_star = simulator(list(theta_star.values()))
        x_obs = ts_processor(ts_star).unsqueeze(0)
        samples = posterior.sample((10_000,), x=x_obs.cuda(), show_progress_bars=True).cpu().numpy()

        np.save(f"{outdir}/default_obs_samples.npy", samples)
        _ = corner.corner(
            samples,
            labels=list(bounds.keys()),
            truths=list(theta_star.values()),
            range=list(bounds.values()),
        )
        plt.savefig(f"{outdir}/default_obs_corner.png")
    except KeyError:
        print(
            "No Default parameter set found -- Not sampling and plotting posterior distribution..."
        )


if __name__ == "__main__":
    defopt.run(main)
