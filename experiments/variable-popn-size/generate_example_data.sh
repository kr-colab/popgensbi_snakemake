#!/bin/bash
# Generate example VCF data for variable population size experiment

# Create output directory
mkdir -p example_data/variable_popn_size

# Generate simulated VCF using the simulate-vcf.py utility
python resources/util/simulate-vcf.py \
  --outpath "example_data/variable_popn_size" \
  --window-size 1000000 \
  --configfile experiments/variable-popn-size/npe-config/variable_popn_size_cnn.yaml

echo "Example data generated in example_data/variable_popn_size/"