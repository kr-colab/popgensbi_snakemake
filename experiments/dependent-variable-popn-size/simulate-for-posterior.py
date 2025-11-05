#!/usr/bin/env python3
"""
Simulate tree sequences and parameters for posterior analysis.

This script simulates data using msprime directly (without popgensbi simulator classes)
and outputs:
1. A tree sequence file (.trees)
2. A parameters file (.yaml) compatible with plot-variable-popsize-posterior.py

Available models:
- VariablePopulationSize: Single population with multiple size changes
- DependentVariablePopulationSize: Single population with variable recombination rate
"""

import argparse
import numpy as np
import msprime
import demes
import tskit
import yaml
import sys
import os


def simulate_variable_population_size(params, config):
    """Simulate variable population size model."""
    # Extract population sizes (in log10 space) and recombination rate
    pop_sizes_log10 = np.array(params[:-1])
    recomb_rate = params[-1]
    pop_sizes = 10 ** pop_sizes_log10
    
    # Calculate change times using exponential spacing
    num_epochs = len(pop_sizes)
    max_time = config.get("max_time", 100000)
    time_rate = config.get("time_rate", 0.1)
    
    # Calculate change times (same logic as original)
    win = np.logspace(2, np.log10(max_time), num_epochs)
    win[0] = 0
    change_times = np.around(win).astype(int)[1:]  # Exclude time 0
    
    # Build demography
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=float(pop_sizes[0]))
    
    # Add population size changes
    for i in range(1, len(pop_sizes)):
        demography.add_population_parameters_change(
            time=change_times[i-1],
            initial_size=float(pop_sizes[i]),
            population="pop"
        )
    
    # Simulate
    samples = {"pop": config["samples"]["pop"]}
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=config["sequence_length"],
        recombination_rate=recomb_rate,
        random_seed=config["seed"],
        ploidy=2,  # Ensure diploid individuals
    )
    ts = msprime.sim_mutations(ts, rate=config["mutation_rate"], random_seed=config["seed"])
    
    return ts, change_times.tolist()


# Model configurations
MODEL_CONFIGS = {
    "VariablePopulationSize": {
        "params": None,  # Dynamic based on num_epochs
        "defaults": {
            "samples": {"pop": 21},
            "sequence_length": 2e6,
            "mutation_rate": 1e-8,
            "num_epochs": 21,
            "max_time": 130000,
            "time_rate": 0.03,
        },
        "simulate": simulate_variable_population_size,
    },
    "DependentVariablePopulationSize": {
        "params": None,  # Dynamic based on num_epochs
        "defaults": {
            "samples": {"pop": 21},
            "sequence_length": 2e6,
            "mutation_rate": 1e-8,
            "num_epochs": 21,
            "max_time": 130000,
            "time_rate": 0.03,
        },
        "simulate": simulate_variable_population_size,
    }
    
}


def main():
    parser = argparse.ArgumentParser(
        description="Simulate tree sequences for posterior analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Demographic model to simulate"
    )
    parser.add_argument(
        "--params",
        type=float,
        nargs="+",
        required=True,
        help="Model parameters (see model documentation for order)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files"
    )
    parser.add_argument(
        "--sequence-length",
        type=float,
        help="Sequence length (overrides default)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        help="Mutation rate (overrides default)"
    )
    parser.add_argument(
        "--recombination-rate",
        type=float,
        help="Recombination rate (overrides default for non-variable models)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples per population (overrides default)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of epochs for VariablePopulationSize model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Get model configuration
    model_config = MODEL_CONFIGS[args.model]
    config = model_config["defaults"].copy()
    
    # Override defaults with command-line arguments
    if args.sequence_length:
        config["sequence_length"] = args.sequence_length
    if args.mutation_rate:
        config["mutation_rate"] = args.mutation_rate
    if args.recombination_rate and args.model != "VariablePopulationSize":
        config["recombination_rate"] = args.recombination_rate
    if args.samples:
        # Update samples for all populations
        for pop in config["samples"]:
            config["samples"][pop] = args.samples
    if args.num_epochs:
        config["num_epochs"] = args.num_epochs
    
    config["seed"] = args.seed
    
    # Check parameter count
    num_epochs = config.get("num_epochs", 21)
    expected_params = num_epochs + 1  # N1, N2, ..., Nn, recomb_rate
    param_names = [f"log10_N_{i+1}" for i in range(num_epochs)] + ["recombination_rate"]
    
    if len(args.params) != expected_params:
        print(f"Error: {args.model} model requires {expected_params} parameters:")
        print(f"  {', '.join(param_names)}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Simulate
    print(f"Simulating {args.model} model...")
    simulate_fn = model_config["simulate"]
    ts, change_times = simulate_fn(args.params, config)
    
    N = 10 ** args.params[0]
    theta = 4 * N * config["mutation_rate"] * config["sequence_length"]
    
    # Save tree sequence
    ts_path = os.path.join(args.output_dir, "simulated.trees")
    ts.dump(ts_path)
    print(f"Saved tree sequence to {ts_path}")
    print(f"  Samples: {ts.num_samples}")
    print(f"  Sequence length: {ts.sequence_length}")
    print(f"  Mutations: {ts.num_mutations}")
    print(f"  Trees: {ts.num_trees}")
    
    # Create parameters file
    params_data = {
        "theta": theta,
        "change_times": change_times,
        "true_params": args.params,
        "simulator_config": {
            "class_name": args.model
        },
        "num_epochs": 21
    }
    
    
    params_path = os.path.join(args.output_dir, "parameters.yaml")
    with open(params_path, "w") as f:
        yaml.dump(params_data, f, default_flow_style=False)
    
    print(f"\nSaved parameters to {params_path}")
    print(f"  theta: {theta}")
    print(f"  change_times: {change_times}")
    print(f"  true_params: {args.params}")
    
    # Print example usage
    print(f"\nTo analyze the posterior, run:")
    print(f"python plot-variable-popsize-posterior.py \\")
    print(f"    --configfile <your_config.yaml> \\")
    print(f"    --tree-sequence {ts_path} \\")
    print(f"    --parameters {params_path} \\")
    print(f"    --outpath <output_directory>")


if __name__ == "__main__":
    main()