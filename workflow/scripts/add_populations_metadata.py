import zarr
import tskit
import json
import numpy as np


ds = zarr.load(snakemake.input.zarr)

populations_csv_path = snakemake.params.popn_csv_path
populations_array = np.genfromtxt(populations_csv_path, delimiter=",", dtype=str, skip_header=1)
pop_dict = {row[0]: row[1] for row in populations_array}
pop_names = list(pop_dict.values())

# save the population data in json format
schema = json.dumps(tskit.MetadataSchema.permissive_json().schema).encode()
zarr.save(snakemake.input.zarr + "/populations_metadata_schema", schema)
metadata = [
    json.dumps({"name": pop, "description": "population this sample comes from"}).encode()
    for pop in pop_names
]
zarr.save(snakemake.input.zarr + "/populations_metadata", metadata)

# Now assign each diploid sample to a population
num_individuals = ds["sample_id"].shape[0]
individuals_population = np.full(num_individuals, tskit.NULL, dtype=np.int32)
# loop through each row in the populations array
for i, sample_id in enumerate(ds["sample_id"]):
    pop_id = pop_dict[sample_id]
    individuals_population[i] = pop_names.index(pop_id)

zarr.save(snakemake.input.zarr + "/individuals_population", individuals_population)
