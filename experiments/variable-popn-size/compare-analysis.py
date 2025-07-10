#!/usr/bin/env python3
"""
Unified comparison tool for analyzing posteriors from multiple models.

This script provides a complete workflow for simulating data once and comparing
posterior distributions from different trained models.
"""

import argparse
import subprocess
import os
import sys
import yaml


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {description} failed")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare posterior distributions from multiple popgensbi models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models on existing tree sequence:
  %(prog)s --configs model1.yaml model2.yaml --tree-sequence data.trees --parameters params.yaml --output comparison/

  # Simulate and compare:
  %(prog)s --configs model1.yaml model2.yaml --simulate --model VariablePopulationSize --params 4.0 4.2 3.8 4.5 4.1 1.5e-8 --output comparison/

  # With custom labels:
  %(prog)s --configs cnn.yaml rnn.yaml --labels "CNN" "RNN" --simulate --model VariablePopulationSize --params 4.0 4.2 3.8 4.5 4.1 1.5e-8 --output comparison/
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Paths to popgensbi config files for models to compare"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for comparison results"
    )
    
    # Model labels
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each model (defaults to processor-embedding names)"
    )
    
    # Input data options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--tree-sequence",
        help="Path to existing tree sequence file"
    )
    input_group.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate new data"
    )
    
    # Parameters for existing data
    parser.add_argument(
        "--parameters",
        help="Path to parameters file (required with --tree-sequence)"
    )
    
    # Simulation parameters
    sim_group = parser.add_argument_group("simulation options")
    sim_group.add_argument(
        "--model",
        choices=["YRI_CEU", "AraTha_2epoch", "VariablePopulationSize", "recombination_rate"],
        help="Model to simulate (required with --simulate)"
    )
    sim_group.add_argument(
        "--params",
        type=float,
        nargs="+",
        help="Model parameters (required with --simulate)"
    )
    sim_group.add_argument(
        "--num-epochs",
        type=int,
        help="Number of epochs for VariablePopulationSize model"
    )
    sim_group.add_argument(
        "--sim-seed",
        type=int,
        default=42,
        help="Random seed for simulation (default: 42)"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples (default: 1000)"
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=100,
        help="Number of samples to plot (default: 100)"
    )
    parser.add_argument(
        "--analysis-seed",
        type=int,
        default=1024,
        help="Random seed for analysis (default: 1024)"
    )
    
    # Python executable
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current python)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.tree_sequence and not args.parameters:
        parser.error("--parameters is required when using --tree-sequence")
    
    if args.simulate and (not args.model or not args.params):
        parser.error("--model and --params are required when using --simulate")
    
    if args.labels and len(args.labels) != len(args.configs):
        parser.error("Number of labels must match number of configs")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Handle data input
    if args.simulate:
        # Create directory for simulation
        sim_dir = os.path.join(args.output, "simulated_data")
        os.makedirs(sim_dir, exist_ok=True)
        
        # Build simulation command
        sim_cmd = [
            args.python,
            os.path.join(script_dir, "simulate-for-posterior.py"),
            "--model", args.model,
            "--params"
        ] + [str(p) for p in args.params] + [
            "--output-dir", sim_dir,
            "--seed", str(args.sim_seed)
        ]
        
        # Add optional simulation parameters
        if args.num_epochs:
            sim_cmd.extend(["--num-epochs", str(args.num_epochs)])
        
        # Extract simulation parameters from first config if available
        with open(args.configs[0], 'r') as f:
            config = yaml.safe_load(f)
        
        if "simulator" in config:
            sim_config = config["simulator"]
            if "sequence_length" in sim_config:
                sim_cmd.extend(["--sequence-length", str(sim_config["sequence_length"])])
            if "mutation_rate" in sim_config:
                sim_cmd.extend(["--mutation-rate", str(sim_config["mutation_rate"])])
            if "samples" in sim_config:
                # Get sample size from first population
                sample_size = list(sim_config["samples"].values())[0]
                sim_cmd.extend(["--samples", str(sample_size)])
        
        # Run simulation
        run_command(sim_cmd, "Simulating data")
        
        tree_sequence = os.path.join(sim_dir, "simulated.trees")
        parameters = os.path.join(sim_dir, "parameters.yaml")
    else:
        tree_sequence = args.tree_sequence
        parameters = args.parameters
    
    # Build comparison command
    comparison_cmd = [
        args.python,
        os.path.join(script_dir, "compare-posteriors.py"),
        "--configs"
    ] + args.configs + [
        "--tree-sequence", tree_sequence,
        "--parameters", parameters,
        "--output", args.output,
        "--posterior-samples", str(args.posterior_samples),
        "--seed", str(args.analysis_seed)
    ]
    
    # Add labels if provided
    if args.labels:
        comparison_cmd.extend(["--labels"] + args.labels)
    
    # Run comparison
    run_command(comparison_cmd, "Comparing posterior distributions")
    
    print(f"\nComparison complete! Results saved to: {args.output}")
    print(f"  - Comparison plots: {args.output}/*.png")
    print(f"  - Results data: {args.output}/comparison-results.pkl")
    
    if args.simulate:
        print(f"  - Simulated data: {sim_dir}/")
    
    # List all generated plots
    print("\nGenerated plots:")
    plot_files = [
        "comparison-posterior-means.png",
        "comparison-marginals.png",
        "comparison-metrics.png",
        "comparison-uncertainty.png",
        "comparison-ribbons.png"
    ]
    for plot_file in plot_files:
        plot_path = os.path.join(args.output, plot_file)
        if os.path.exists(plot_path):
            print(f"  - {plot_file}")


if __name__ == "__main__":
    main()