"""
Infer trees from windowed VCF
"""

import tsinfer
import zarr
import yaml
import numpy as np
import os

output_dir = os.path.dirname(snakemake.input.yaml)
vcz = zarr.open(snakemake.input.vcz, "r")

sync = zarr.sync.ProcessSynchronizer(output_dir)
root = zarr.open(snakemake.input.zarr, "a", synchronizer=sync)

chunk_config = yaml.safe_load(open(snakemake.input.yaml))
indices = chunk_config["indices"]
target_size = chunk_config["target_size"]

sample_mask = vcz.individuals_population[:] == -1
sample_population = vcz.individuals_population.get_mask_selection(~sample_mask)
site_mask = np.full(vcz.variant_position.size, True)
for i in indices:
    ts_path = os.path.join(output_dir, f"{i}.trees")
    start, end = root.window_start[i], root.window_end[i]
    left, right = root.window_left[i], root.window_right[i]
    contig = root.window_contig[i]
    site_mask[start:end] = False
    # implicitly uses `individuals_population`, `populations_metadata` from vcz
    variant_data = tsinfer.VariantData(
        vcz,
        sample_mask=sample_mask,
        site_mask=site_mask,
        ancestral_state=vcz.ancestral_state.get_mask_selection(~site_mask),
        sequence_length=vcz.contig_length[contig],
    )
    its = tsinfer.infer(
        variant_data, 
        progress_monitor=False, 
        post_process=False,  # avoids trimming to first/last sites
        num_threads=snakemake.threads,
    )
    its = tsinfer.post_process(its, erase_flanks=False)
    its = its.keep_intervals([[left, right]], simplify=False).trim()
    # TODO: what happens to non-segregating sites?
    # what about sites without ancestral states?
    # will this cause assertion below to fail?
    # ideally we keep everything, and leave filtering to the processor
    assert right - left == its.sequence_length
    assert its.num_sites == variant_data.num_sites
    assert np.allclose(its.individuals_population, sample_population)
    its.dump(ts_path)
    site_mask[start:end] = True
    # dummy targets for dimensionality, and to appease the dataloader
    target = np.full(target_size, np.nan)
    root.targets[i] = target
    root.targets_shape[i] = target.shape

