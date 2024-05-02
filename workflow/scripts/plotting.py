import os
import pickle

import corner
import defopt
import matplotlib.pyplot as plt
import numpy as np
import torch

def main(
    data_dir: str,
    outdir: str,
    *,
    prefix: str = "",
):

    simulator = AraTha_2epoch_simulator()
    bounds = simulator.bounds
    thata_star = simulator.true_values
    posterior = pickle.load(f"{outdir}/{prefix}")

    try:
        x_obs = simulator(theta).unsqueeze(0)
        samples = posterior.sample((10_000,), x=x_obs.cuda(), show_progress_bars=True).cpu().numpy()

        np.save(f"{outdir}/{prefix}default_obs_samples.npy", samples)
        _ = corner.corner(
            samples,
            labels=[k for k in bounds.keys()],
            truths=[v for v in theta_star.items()],
            range=[v for _, v in bounds.items()],
        )
        plt.savefig(f"{outdir}/{prefix}default_obs_corner.png")
    except KeyError:
        print(
            "No Default parameter set found -- Not sampling and plotting posterior distribution..."
        )


if __name__ == "__main__":
    defopt.run(main)
