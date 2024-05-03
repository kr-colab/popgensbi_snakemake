import os
import pickle

import corner
import defopt
import matplotlib.pyplot as plt
import numpy as np
import torch
from simulators import AraTha_2epoch_simulator

def main(
    data_dir: str,
    outdir: str,
    *,
    prefix: str = "",
):

    simulator = AraTha_2epoch_simulator()
    bounds = simulator.bounds
    theta_star = simulator.true_values
    with open(f"{outdir}/{prefix}posterior.pkl", "rb") as f:
        posterior = pickle.load(f)   

    try:
        x_obs = simulator(list(theta_star.values())).unsqueeze(0)
        samples = posterior.sample((10_000,), x=x_obs.cuda(), show_progress_bars=True).cpu().numpy()

        np.save(f"{outdir}/{prefix}default_obs_samples.npy", samples)
        _ = corner.corner(
            samples,
            labels=list(bounds.keys()),
            truths=list(theta_star.values()),
            range=list(bounds.values()),
        )
        plt.savefig(f"{outdir}/{prefix}default_obs_corner.png")
    except KeyError:
        print(
            "No Default parameter set found -- Not sampling and plotting posterior distribution..."
        )


if __name__ == "__main__":
    defopt.run(main)
