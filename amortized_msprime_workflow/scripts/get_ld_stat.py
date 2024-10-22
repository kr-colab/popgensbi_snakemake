# get a segment of a tree sequence and calculate the LD statistics and save it.
# will use the ld stat later to average over.

import tskit
import os
import torch
from ts_simulators import *
from ts_processors import *
import numpy as np
import moments
import pickle

datadir = snakemake.params.datadir
num_simulations = snakemake.params.num_simulations
ts_processor = snakemake.params.ts_processor
n_seg_idx = int(snakemake.params.n_seg_idx) # segment index
n_segs = int(snakemake.params.n_segs) # total number of segments

with open(f"{datadir}/{num_simulations}.trees", "rb") as ts_file:
    ts = tskit.load(ts_file)

vcf_name = os.path.join(datadir, ts_processor, "{0}_seg_{1}.vcf".format(num_simulations, n_seg_idx))
site_mask = np.ones(ts.num_sites, dtype=bool)

seg_len = int(ts.sequence_length / n_segs)
print(seg_len)

# unmask segments we want
for site in ts.sites():
    if site.position > n_seg_idx * seg_len:
        if site.position < (n_seg_idx + 1) * seg_len:
            site_mask[site.id] = 0
with open(vcf_name, "w+") as fout:
    ts.write_vcf(fout, site_mask=site_mask)
if os.path.exists(f"{vcf_name}.gz"):
    os.system(f"rm {vcf_name}.gz")
os.system(f"gzip {vcf_name}")

# tood write rec map file and popfile based on ts

ld_stat = moments.LD.Parsing.compute_ld_statistics(
    str(vcf_name)+'.gz',
    rec_map_file=os.path.join(datadir, "rec_map_file.txt"),
    pop_file=os.path.join(datadir, "pop_file.txt"),
    pops=["YRI", "CEU"],
    r_bins=[0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    report=True)
os.system("rm " + str(vcf_name)+'.gz')
os.system("rm " + str(vcf_name)[:-3] + "h5")
with open(f"{datadir}/{ts_processor}/ld_stat_{num_simulations}_{n_seg_idx}.pkl", "wb") as f:
    pickle.dump(ld_stat, f)
