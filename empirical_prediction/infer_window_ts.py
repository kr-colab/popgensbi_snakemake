import ray
import numpy as np
import tsinfer
import tskit
import os
import zarr

############# this is the code to infer the tree sequence #############

# Get parameters from snakemake
window_size = snakemake.params.window_size
output_dir = snakemake.params.output_dir
window_type = snakemake.params.window_type
report = snakemake.params.report
input_vcf = snakemake.input.vcf

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load ancestral alleles
ancestral_states = np.load(snakemake.config["ancestral_states"])

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
        ds,
        ancestral_state=ancestral_states[current_site_mask == False],
        site_mask=current_site_mask,
    )

    # Infer the tree sequence
    output_path = f"{output_dir}/window_{window_start}_{window_end}.trees"
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

ds = zarr.open(snakemake.input.zarr)
name = os.path.basename(input_vcf)[0]
site_mask = np.ones(len(ds.variant_position[:]), dtype=bool)  # Start with all True
window_tasks = []

if window_type == "variant_index":
    # Windowing by variant index
    variant_window_size = window_size
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
            window_type
        )
        window_tasks.append(task)

elif window_type == "physical_position":
    # Windowing by physical position
    physical_window_size = window_size
    sequence_length = ds.variant_position[-1] if len(ds.variant_position) > 0 else 0
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
            window_type
        )
        window_tasks.append(task)

else:
    raise ValueError(f"Unknown window_type: {window_type}")


# Wait for all tasks to complete and collect results
tree_sequence_paths = ray.get(window_tasks)

# Optional: print out the paths of generated tree sequences
if report:
    print("Generated tree sequences:")
    for path in tree_sequence_paths:
        ts = tskit.load(path)
        print(f"{path}: {ts.num_trees} trees, {ts.num_mutations} mutations, {ts.num_individuals} individuals, {ts.num_nodes} nodes")

# Shutdown Ray when done
ray.shutdown()

# Create a sentinel file to mark completion
with open(os.path.join(output_dir, "done.txt"), "w") as f:
    pass
