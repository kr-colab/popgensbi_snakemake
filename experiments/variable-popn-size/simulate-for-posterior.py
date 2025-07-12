#!/usr/bin/env python3
"""
Simulate tree sequences and parameters for posterior analysis.

This script simulates data using msprime directly (without popgensbi simulator classes)
and outputs:
1. A tree sequence file (.trees)
2. A parameters file (.yaml) compatible with plot-variable-popsize-posterior.py

Available models:
- YRI_CEU: Two-population model with migration
- AraTha_2epoch: Single population with one size change
- VariablePopulationSize: Single population with multiple size changes
- recombination_rate: Single population with variable recombination rate
"""

import argparse
import numpy as np
import msprime
import demes
import tskit
import yaml
import sys
import os


def simulate_yri_ceu(params, config):
    """Simulate YRI-CEU two-population model with migration."""
    N_A, N_YRI, N_CEU_initial, N_CEU_final, M, Tp, T = params
    
    # Build demography using demes
    graph = demes.Builder()
    graph.add_deme(
        "ancestral", 
        epochs=[dict(start_size=N_A, end_time=Tp + T)]
    )
    graph.add_deme(
        "AMH", 
        ancestors=["ancestral"], 
        epochs=[dict(start_size=N_YRI, end_time=T)],
    )
    graph.add_deme(
        "CEU", 
        ancestors=["AMH"], 
        epochs=[dict(start_size=N_CEU_initial, end_size=N_CEU_final)],
    )
    graph.add_deme(
        "YRI", 
        ancestors=["AMH"], 
        epochs=[dict(start_size=N_YRI)],
    )
    graph.add_migration(demes=["CEU", "YRI"], rate=M)
    
    demog = msprime.Demography.from_demes(graph.resolve())
    
    # Simulate
    ts = msprime.sim_ancestry(
        config["samples"],
        demography=demog,
        sequence_length=config["sequence_length"],
        recombination_rate=config["recombination_rate"],
        random_seed=config["seed"],
        ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=config["mutation_rate"], random_seed=config["seed"])
    
    # No change times for this model
    change_times = None
    
    return ts, change_times


def simulate_aratha_2epoch(params, config):
    """Simulate AraTha 2-epoch model (single population with size change)."""
    nu, T = params
    
    # Get ancestral population size (default from stdpopsim model)
    N_A = 7420747  # From stdpopsim AraTha African2Epoch_1H18 model
    
    # Calculate actual population sizes and time
    N_current = nu * N_A
    time_change = T * 2 * N_A
    
    # Build demography
    demography = msprime.Demography()
    demography.add_population(name="pop", initial_size=N_current)
    demography.add_population_parameters_change(
        time=time_change,
        initial_size=N_A,
        population="pop"
    )
    
    # Simulate
    samples = {"pop": config["samples"]["SouthMiddleAtlas"]}
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=config["sequence_length"],
        recombination_rate=config["recombination_rate"],
        random_seed=config["seed"],
        ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=config["mutation_rate"], random_seed=config["seed"])
    
    # Change times for plotting
    change_times = [time_change]
    
    return ts, change_times


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


def simulate_recombination_rate(params, config):
    """Simulate single population with variable recombination rate."""
    recomb_rate = params[0]
    
    # Simulate with fixed population size
    ts = msprime.sim_ancestry(
        config["samples"],
        population_size=config["pop_size"],
        sequence_length=config["sequence_length"],
        recombination_rate=recomb_rate,
        random_seed=config["seed"],
        ploidy=2,
    )
    ts = msprime.sim_mutations(ts, rate=config["mutation_rate"], random_seed=config["seed"])
    
    # No change times for this model
    change_times = None
    
    return ts, change_times


# Model configurations
MODEL_CONFIGS = {
    "YRI_CEU": {
        "params": ["N_A", "N_YRI", "N_CEU_initial", "N_CEU_final", "M", "Tp", "T"],
        "defaults": {
            "samples": {"YRI": 10, "CEU": 10},
            "sequence_length": 10e6,
            "recombination_rate": 1.5e-8,
            "mutation_rate": 1.5e-8,
        },
        "simulate": simulate_yri_ceu,
    },
    "AraTha_2epoch": {
        "params": ["nu", "T"],
        "defaults": {
            "samples": {"SouthMiddleAtlas": 10},
            "sequence_length": 10e6,
            "recombination_rate": 1.5e-8,
            "mutation_rate": 1.5e-8,
        },
        "simulate": simulate_aratha_2epoch,
    },
    "VariablePopulationSize": {
        "params": None,  # Dynamic based on num_epochs
        "defaults": {
            "samples": {"pop": 10},
            "sequence_length": 10e6,
            "mutation_rate": 1.5e-8,
            "num_epochs": 5,
            "max_time": 100000,
            "time_rate": 0.1,
        },
        "simulate": simulate_variable_population_size,
    },
    "recombination_rate": {
        "params": ["recombination_rate"],
        "defaults": {
            "samples": {0: 10},
            "sequence_length": 1e6,
            "mutation_rate": 1.5e-8,
            "pop_size": 1e4,
        },
        "simulate": simulate_recombination_rate,
    },
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
    if args.num_epochs and args.model == "VariablePopulationSize":
        config["num_epochs"] = args.num_epochs
    
    config["seed"] = args.seed
    
    # Check parameter count
    if args.model == "VariablePopulationSize":
        num_epochs = config.get("num_epochs", 5)
        expected_params = num_epochs + 1  # N1, N2, ..., Nn, recomb_rate
        param_names = [f"log10_N_{i+1}" for i in range(num_epochs)] + ["recombination_rate"]
    else:
        param_names = model_config["params"]
        expected_params = len(param_names)
    
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
    
    # Calculate theta (4 * N * mu * L for diploid)
    # For models with variable population size, use the first epoch size
    if args.model == "VariablePopulationSize":
        N = 10 ** args.params[0]
    elif args.model == "AraTha_2epoch":
        N_A = 7420747
        N = args.params[0] * N_A
    elif args.model == "YRI_CEU":
        N = args.params[0]  # Ancestral population size
    else:  # recombination_rate model
        N = config["pop_size"]
    
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
    }
    
    # Add model-specific information
    if args.model == "VariablePopulationSize":
        params_data["num_epochs"] = config.get("num_epochs", 5)
    
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