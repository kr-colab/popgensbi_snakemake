"""
Set up the VCF prediction pipeline 
"""

import zarr
import tskit
import os
import yaml
import numpy as np
from pysam import FastaFile
from numcodecs import Blosc
from math import ceil
import ts_simulators


vcz = zarr.open(snakemake.input.vcz)
n_variants = vcz.variant_position.size
n_samples = vcz.sample_id.size
n_chunks = len(snakemake.output.yaml)

prediction_config = snakemake.params.prediction_config.copy()
ancestral_fasta_path = prediction_config.get("ancestral_fasta")
population_map_path = prediction_config.get("population_map")
#window_type = prediction_config["window_type"]
#window_size = prediction_config["window_size"]


# check the sample configuration in the data against that from the simulator
simulator_config = snakemake.params.simulator_config.copy()
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)
ts_sim, theta_sim = simulator(seed=1024)

population_id = [p.metadata["name"] for p in ts_sim.populations()]
population_coding = {p: i for i, p in enumerate(population_id)}
population_map = yaml.safe_load(open(population_map_path))
sample_map = {sid: i for i, sid in enumerate(vcz.sample_id)}
sample_population = np.full(n_samples, -1)
for ind, pop in population_map.items():
    assert pop in population_coding, f"Population {pop} not found in simulated instance"
    assert ind in sample_map, f"Sample {ind} not found in VCF"
    sample_population[sample_map[ind]] = population_coding[pop]
    # if a sample from the VCF is not found in the population map, it will be
    # coded as -1 and will subsequently be omitted when building trees

sample_counts_sim = np.bincount(
    ts_sim.individuals_population, 
    minlength=len(population_coding),
)
sample_counts = np.bincount(
    sample_population[sample_population > -1], 
    minlength=len(population_coding),
)
assert np.allclose(sample_counts, sample_counts_sim), \
    f"Sample counts mismatch between simulation ({sample_counts_sim}) and data ({sample_counts})"

schema = tskit.MetadataSchema.permissive_json()
populations_metadata = [schema.validate_and_encode_row({"name": p}) for p in population_id]
zarr.save(f"{snakemake.input.vcz}/individuals_population", sample_population)
zarr.save(f"{snakemake.input.vcz}/populations_metadata", populations_metadata)


# add ancestral state information to vcz
ancestral_fasta = FastaFile(ancestral_fasta_path)
ancestral_state = np.full(vcz.variant_position.size, "N", dtype="U1")
for i, contig in enumerate(vcz.contig_id):
    fasta = np.fromiter(ancestral_fasta.fetch(reference=contig), dtype="U1")
    contig_mask = vcz.variant_contig[:] == i
    position = vcz.variant_position.get_mask_selection(contig_mask)
    ancestral_state[contig_mask] = np.char.upper(fasta[position - 1])
zarr.save(f"{snakemake.input.vcz}/ancestral_state", ancestral_state)


# split each contig in the vcz into windows
variant_window = np.full(n_variants, -1)
window_size = prediction_config["window_size"]
assert isinstance(window_size, (float, int))
min_snps_per_window = prediction_config["min_snps_per_window"]
# NB: implementing windowing for a fixed number of variants has a few gotchas,
# one of which is that these should be segregating in the subset of samples.
# Further, these will likely be further filtered by the processor class. To
# make the windowing mechanism transparent, we keep it to "sequence windows"
# specified two ways: a bed file that the user provides; or a window size in bp
# that is triggered by `windows=[float|int]`.
window_start = []
window_end = []
window_left = []
window_right = []
window_contig = []
for i, contig_length in enumerate(vcz.contig_length):
    contig_mask = vcz.variant_contig[:] == i
    assert np.any(contig_mask), f"No variants on {i}th contig"
    contig_offset = contig_mask.argmax()
    position = vcz.variant_position.get_mask_selection(contig_mask)
    assert np.all(np.diff(position)), f"Variants are out of order"
    #if window_type == "bed":
    #    assert False, "TODO"
    #elif window_type == "sequence":
    breaks = np.append(
        np.arange(0, contig_length, window_size), 
        contig_length,
    )
    left, right = breaks[:-1], breaks[1:]
    # ---
    start = contig_offset + np.searchsorted(position, left, side='left')
    end = contig_offset + np.searchsorted(position, right, side='left')
    mask = end - start >= min_snps_per_window  # filter to minimum number of variants
    window_start.extend(start[mask])
    window_end.extend(end[mask])
    window_left.extend(left[mask])
    window_right.extend(right[mask])
    window_contig.extend([i] * np.sum(mask))
window_start = np.array(window_start)
window_end = np.array(window_end)
window_left = np.array(window_left)
window_right = np.array(window_right)
n_windows = window_start.size
# TODO: don't save this stuff, just put pointers/bounds into zarr below
#zarr.save(f"{snakemake.input.vcz}/variant_window", variant_window)
#zarr.save(f"{snakemake.input.vcz}/window_left", np.array(window_left))
#zarr.save(f"{snakemake.input.vcz}/window_right", np.array(window_right))
#zarr.save(f"{snakemake.input.vcz}/window_contig", np.array(window_contig))


# assign windows to chunks via a yaml intermediate
# - indices must be contiguous to avoid collisions when writing
# - must create a file for "extra" chunks if n_sims <= len(output.yaml)
n_chunks = min(n_windows, n_chunks)
chunk_size = ceil(n_windows / n_chunks)
for i, yml in enumerate(snakemake.output.yaml):
    indices = np.arange(i * chunk_size, min(n_windows, (i + 1) * chunk_size))
    chunk_config = {
        "indices": indices.tolist(),
        "target_size": theta_sim.size,
    }
    yaml.dump(
        chunk_config,
        open(yml, "w"),
        default_flow_style=True,
    )


# create a separate zarr to store features, predictions per window
store = zarr.DirectoryStore(snakemake.output.zarr)
root = zarr.group(store=store, overwrite=True) 
# TODO: currently not using compression b/c speed
# codec = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
zarr_kwargs = {"chunks": n_chunks, "chunk_size": chunk_size, "shape": n_windows}
for what in ["features", "targets"]:
    root.create_dataset(f'{what}', dtype='array:f4', **zarr_kwargs)
    root.create_dataset(f'{what}_shape', dtype='array:i4', **zarr_kwargs)

zarr.save(f"{snakemake.output.zarr}/window_start", window_start)
zarr.save(f"{snakemake.output.zarr}/window_end", window_end)
zarr.save(f"{snakemake.output.zarr}/window_left", window_left)
zarr.save(f"{snakemake.output.zarr}/window_right", window_right)
zarr.save(f"{snakemake.output.zarr}/window_contig", window_contig)
zarr.save(f"{snakemake.output.zarr}/vcf_windows", np.arange(n_windows)) # "split" indices
