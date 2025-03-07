"""
Extract features from simulated tree sequences
"""

import os
import numpy as np
import zarr
import yaml
import tskit
import ts_processors

indices = yaml.safe_load(open(snakemake.input.yaml))["indices"]
input_dir = os.path.dirname(snakemake.input.yaml)

config = snakemake.params.processor_config
class_name = config["class_name"]
ts_processor = getattr(ts_processors, class_name)(config)

sync = zarr.sync.ProcessSynchronizer(input_dir)
root = zarr.open(snakemake.input.zarr, "a", synchronizer=sync)
for i in indices:
    ts_path = os.path.join(input_dir, f"{i}.trees")
    ts = tskit.load(ts_path)
    features = ts_processor(ts)
    root.features[i] = features.flatten()
    root.features_shape[i] = features.shape
    assert root.features[i].size == features.size
