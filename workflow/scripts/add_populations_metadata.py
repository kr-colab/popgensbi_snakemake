import zarr
import tskit
import json
import numpy as np


ds = zarr.load(snakemake.input.zarr)

populations = snakemake.params.populations
pop_names = list(populations.keys())
# save the population data in json format
schema = json.dumps(tskit.MetadataSchema.permissive_json().schema).encode()
zarr.save(snakemake.input.zarr + "/populations_metadata_schema", schema)
metadata = [
    json.dumps({"name": pop, "description": "The country from which this sample comes"}).encode()
    for pop in populations.keys()
]
zarr.save(snakemake.input.zarr + "/populations_metadata", metadata)

# Now assign each diploid sample to a population
num_individuals = ds["sample_id"].shape[0]
individuals_population = np.full(num_individuals, tskit.NULL, dtype=np.int32)
for pop, pop_size in populations.items():
    for i in range(pop_size):
        individuals_population[i] = pop_names.index(pop)
zarr.save(snakemake.input.zarr + "/individuals_population", individuals_population)
