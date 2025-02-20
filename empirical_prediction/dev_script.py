import ray
import msprime
import numpy as np
import subprocess
import zarr
import tsinfer
import os

############# simulate a vcf file with msprime #############
# simulate a vcf file with msprime

sequence_length = 10_000_000
recombination_rate = 1e-8
mutation_rate = 1e-8

ts = msprime.sim_ancestry(
    samples=10,
    population_size=1000,
    recombination_rate=recombination_rate,
    sequence_length=sequence_length,
    random_seed=42
)

ts = msprime.sim_mutations(ts, rate=mutation_rate)

name = "test"
with open(f"{name}.vcf", "w") as vcf_file:
    ts.write_vcf(vcf_file)
# write out ancestral alleles
np.save(f"{name}-AA.npy", [s.ancestral_state for s in ts.sites()])

###### this is the code to convert the vcf file to a zarr file #############
# Compress the VCF file using bgzip
subprocess.run(["bgzip", "-c", f"{name}.vcf"], stdout=open(f"{name}.vcf.gz", "wb"))
# Index the compressed VCF file using tabix
subprocess.run(["tabix", "-p", "vcf", f"{name}.vcf.gz"])
# Explode the VCF file into Zarr format
subprocess.run(["vcf2zarr", "explode", "-f",f"{name}.vcf.gz", f"{name}.icf"])
# Encode the exploded Zarr file
subprocess.run(["vcf2zarr", "encode", "-f",f"{name}.icf", f"{name}.vcz"])
ds = zarr.open(f"{name}.vcz")

############# this is the code to infer the tree sequence #############

# create a directory to store the tree sequences
output_dir = "empirical_prediction/tree_sequences"
os.makedirs(output_dir, exist_ok=True)

# load ancestral alleles, won't be needed with real data
ancestral_states = np.load(f"{name}-AA.npy")

# Create windows that cover the entire sequence length
window_size = 100000
site_mask = np.ones(len(ds.variant_position[:]), dtype=bool)  # Start with all True

# Initialize Ray
ray.init()

@ray.remote
def infer_window_tree_sequence(name, output_dir, window_start, ds, ancestral_states, site_mask):
    """
    Infer tree sequence for a specific window using Ray
    
    Parameters:
    - name: base name for output files
    - output_dir: directory to save tree sequences
    - window_start: start of the current window
    - ds: zarr dataset
    - ancestral_states: ancestral states array
    - site_mask: boolean mask for site selection
    
    Returns:
    - Path to the saved tree sequence file
    """
    # Mask out variants within the current window
    window_end = min(window_start + window_size, sequence_length)
    window_mask = (ds.variant_position[:] >= window_start) & (ds.variant_position[:] < window_end)
    current_site_mask = site_mask.copy()
    current_site_mask[window_mask] = False

    # Define the VariantData object
    window_vdata = tsinfer.VariantData(
        f"{name}.vcz",
        ancestral_state=ancestral_states[current_site_mask == False],
        site_mask=current_site_mask,
    )

    # Infer the tree sequence
    output_path = f"{output_dir}/{name}-{window_start}.trees"
    tsinfer.infer(window_vdata, 
                  progress_monitor=True, 
                  num_threads=1).dump(output_path)
    
    return output_path

# Prepare Ray tasks
window_tasks = []
for i in range(0, sequence_length, window_size):
    task = infer_window_tree_sequence.remote(
        name, 
        output_dir, 
        i, 
        ds, 
        ancestral_states, 
        site_mask
    )
    window_tasks.append(task)

# Wait for all tasks to complete and collect results
tree_sequence_paths = ray.get(window_tasks)

# Optional: print out the paths of generated tree sequences
print("Generated tree sequences:")
for path in tree_sequence_paths:
    print(path)

# Shutdown Ray when done
ray.shutdown()
