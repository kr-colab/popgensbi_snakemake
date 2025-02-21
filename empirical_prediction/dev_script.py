import ray
import msprime
import numpy as np
import subprocess
import zarr
import tsinfer
import tskit
import os
import argparse  # Import argparse for command-line arguments

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Process VCF and infer tree sequences.")
parser.add_argument("--window_size", type=int, required=True, help="Size of the window for processing.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output files.")
parser.add_argument("--window_type", type=str, default="variant_index", choices=["variant_index", "physical_position"], help="Type of windowing: variant_index or physical_position.")
parser.add_argument("--report", action='store_true', help="Generate a report about the tree sequences generated.")
args = parser.parse_args()

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
subprocess.run(["vcf2zarr", "explode", "-p", "4", "-f",f"{name}.vcf.gz", f"{name}.icf"])
# Encode the exploded Zarr file
subprocess.run(["vcf2zarr", "encode", "-p", "4", "-f",f"{name}.icf", f"{name}.vcz"])
ds = zarr.open(f"{name}.vcz")

############# this is the code to infer the tree sequence #############

# create a directory to store the tree sequences
output_dir = args.output_dir  # Use the output_dir from command-line arguments
os.makedirs(output_dir, exist_ok=True)

# load ancestral alleles, won't be needed with real data
ancestral_states = np.load(f"{name}-AA.npy")

# Initialize Ray
ray.init(logging_level="ERROR")

@ray.remote
def infer_window_tree_sequence(name, output_dir, window_start, window_end, ds, ancestral_states, site_mask, window_type):
    """
    Infer tree sequence for a specific window based on variant indices or physical position
    
    Parameters:
    - name: base name for output files
    - output_dir: directory to save tree sequences
    - window_start: start index or position of the window
    - window_end: end index or position of the window
    - ds: zarr dataset
    - ancestral_states: ancestral states array
    - site_mask: boolean mask for site selection
    - window_type: type of windowing ('variant_index' or 'physical_position')
    
    Returns:
    - Path to the saved tree sequence file
    """
    # Suppress warnings
    import warnings
    warnings.simplefilter("ignore")
    # Create a copy of the site mask
    current_site_mask = site_mask.copy()

    if window_type == "variant_index":
        # Mask out variants within the current window based on variant indices
        window_variant_mask = (np.arange(len(ds.variant_position[:])) >= window_start) & \
                               (np.arange(len(ds.variant_position[:])) < window_end) 
        # Temp physical positions for the window based on variant indices
        temp_physical_position_left = ds.variant_position[window_start]
        temp_physical_position_right = ds.variant_position[window_end-1] + 1
        
    elif window_type == "physical_position":
        # Mask out variants within the current window based on physical positions
        window_variant_mask = (ds.variant_position[:] >= window_start) & \
                               (ds.variant_position[:] < window_end)
    else:
        raise ValueError(f"Unknown window_type: {window_type}")

    current_site_mask[window_variant_mask] = False

    # Define the VariantData object
    window_vdata = tsinfer.VariantData(
        f"{name}.vcz",
        ancestral_state=ancestral_states[current_site_mask == False],
        site_mask=current_site_mask,
    )

    # Infer the tree sequence
    output_path = f"{output_dir}/{name}-variants_{window_start}_{window_end}.trees"
    its = tsinfer.infer(window_vdata,
                  progress_monitor=False,
                  num_threads=1)
    # Keep the interval and trim the tree sequence
    if window_type == "variant_index":
        print(f"Keeping interval {temp_physical_position_left} to {temp_physical_position_right}")
        window_ts = its.keep_intervals([[temp_physical_position_left, temp_physical_position_right]]).trim()
    else:
        print(f"Keeping interval {window_start} to {window_end}")
        window_ts = its.keep_intervals([[window_start, window_end]]).trim()
    window_ts.dump(output_path)

    return output_path

site_mask = np.ones(len(ds.variant_position[:]), dtype=bool)  # Start with all True
window_tasks = []

if args.window_type == "variant_index":
    # Windowing by variant index
    variant_window_size = args.window_size
    for i in range(0, len(ds.variant_position[:]), variant_window_size):
        window_start = i
        window_end = min(i + variant_window_size, len(ds.variant_position[:]))
        task = infer_window_tree_sequence.remote(
            name,
            output_dir,
            window_start,
            window_end,
            ds,
            ancestral_states,
            site_mask,
            args.window_type
        )
        window_tasks.append(task)

elif args.window_type == "physical_position":
    # Windowing by physical position
    physical_window_size = args.window_size
    sequence_length = ds.variant_position[-1] if len(ds.variant_position) > 0 else 0 # Use last position as sequence length
    for window_start in range(0, int(sequence_length), physical_window_size):
        window_end = min(window_start + physical_window_size, sequence_length)
        task = infer_window_tree_sequence.remote(
            name,
            output_dir,
            window_start,
            window_end,
            ds,
            ancestral_states,
            site_mask,
            args.window_type
        )
        window_tasks.append(task)

else:
    raise ValueError(f"Unknown window_type: {args.window_type}")


# Wait for all tasks to complete and collect results
tree_sequence_paths = ray.get(window_tasks)

# Optional: print out the paths of generated tree sequences
if args.report:
    print("Generated tree sequences:")
    for path in tree_sequence_paths:
        ts = tskit.load(path)
        print(f"{path}: {ts.num_trees} trees, {ts.num_mutations} mutations, {ts.num_individuals} individuals, {ts.num_nodes} nodes")

# Shutdown Ray when done
ray.shutdown()
