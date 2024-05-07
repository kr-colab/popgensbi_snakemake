import tskit
import os
import defopt
from ts_processors import *
import torch
from ts_simulators import *
import numpy as np

def main(num_simulations: int, outdir: str):
    with open(f"{outdir}/{num_simulations}.trees", "rb") as ts_file:
        ts = tskit.load(ts_file)

    # Todo : config parameter that chooses dinf_extract as an option and an if statement (arg == dinf_extract)

    simulator = AraTha_2epoch_simulator()
    n_sample = simulator.n_sample
    processor = dinf_extract(n_snps=2000)
    x = processor(ts)
    # x is tensor, so change it to numpy first and save it as .npy
    x = x.squeeze().cpu().numpy()
    np.save(f"{outdir}/x_{num_simulations}.npy",x)
 
if __name__ == "__main__":
    defopt.run(main)
