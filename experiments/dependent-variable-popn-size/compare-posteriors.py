#!/usr/bin/env python3
"""
Compare posterior distributions from multiple trained models.

This script runs tree sequences through different trained models
and creates comparison plots showing how different architectures perform
on the same data.

Modified version:
- For each config, it will first look for
  <OUTPUT>/simulated_data_<LABEL>/simulated.trees
  where <LABEL> comes from --labels or from the config filename.
- If that file does not exist, it falls back to the global --tree-sequence.
"""

import argparse
import numpy as np
import tskit
import sys
import os
import yaml
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

from torch import Tensor
from sbi.inference import DirectPosterior
from sbi.utils import BoxUniform

# Add popgensbi scripts to load path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
while not os.path.exists(os.path.join(project_root, "workflow", "scripts")):
    parent = os.path.dirname(project_root)
    if parent == project_root:
        raise RuntimeError("Could not find popgensbi project root")
    project_root = parent
sys.path.insert(0, os.path.join(project_root, "workflow", "scripts"))
import ts_processors
from utils import get_least_busy_gpu


if torch.cuda.is_available():
    device = "cuda:0"
    torch.cuda.set_device(0)
else:
    device = "cpu"


def load_model_and_process(config_path, ts, device):
    """Load a trained model and process the tree sequence."""
    config = yaml.safe_load(open(config_path))

    # Extract model components from config
    project_dir = config["project_dir"]
    random_seed = int(config["random_seed"])
    n_train = int(config["n_train"])
    train_separately = bool(config["train_embedding_net_separately"])
    simulator_name = config["simulator"]["class_name"]
    processor_config = config["processor"]
    processor_name = processor_config["class_name"]
    embedding_config = config["embedding_network"]
    embedding_name = embedding_config["class_name"]

    # Create processor
    processor = getattr(ts_processors, processor_name)(processor_config)

    # Build model path
    path = (
        f"{project_dir}/"
        f"{simulator_name}-"
        f"{processor_name}-"
        f"{embedding_name}-"
        f"{random_seed}-"
        f"{n_train}"
    )
    if train_separately:
        path += "-sep/pretrain_"
    else:
        path += "-e2e/"

    # Debug: Print the actual model path being loaded
    print(f"\nLoading model from: {path}")
    print(f"  Config path: {config_path}")
    print(f"  Model type: {processor_name}-{embedding_name}")

    # Load models
    embedding_net = torch.load(
        f"{path}embedding_network",
        weights_only=False,
        map_location=device,
    ).to(device)
    normalizing_flow = torch.load(
        f"{path}normalizing_flow",
        weights_only=False,
        map_location=device,
    ).to(device)

    # Set to eval mode
    embedding_net.eval()
    normalizing_flow.eval()

    # Process data
    processed_data = processor(ts)
    features = Tensor(processed_data).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = embedding_net(features).to(device)

    return {
        "embedding_net": embedding_net,
        "normalizing_flow": normalizing_flow,
        "embedding": embedding,
        "processor": processor,
        "config": config,
        "model_name": f"{processor_name}-{embedding_name}",
        "processed_data": processed_data,
    }


def plot_step_function(
    times, sizes, ax, label=None, color=None, alpha=1.0, linewidth=2, linestyle="-"
):
    """Plot population size as step function backward in time."""
    assert (
        len(times) == len(sizes) + 1
    ), f"times (len={len(times)}) must be one longer than sizes (len={len(sizes)})"
    for i in range(len(sizes)):
        ax.hlines(
            sizes[i],
            times[i],
            times[i + 1],
            colors=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=label if i == 0 else None,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare posterior distributions from multiple trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models:
  %(prog)s --configs model1.yaml model2.yaml --tree-sequence data.trees --parameters params.yaml --output comparison/

  # Compare with custom labels:
  %(prog)s --configs model1.yaml model2.yaml --labels "CNN" "RNN" --tree-sequence data.trees --parameters params.yaml --output comparison/
        """,
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Paths to config files for trained models to compare",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for each model (defaults to config base names)",
    )
    parser.add_argument(
        "--tree-sequence",
        type=str,
        required=True,
        help="Path to tree sequence file (.trees) for legacy mode. "
             "If per-config simulations exist in output/simulated_data_<LABEL>/, "
             "those will be used instead.",
    )
    parser.add_argument(
        "--parameters", type=str, required=True, help="Path to parameters file (YAML)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for comparison plots",
    )
    parser.add_argument(
        "--seed", type=int, default=1024, help="Random seed (default: 1024)"
    )
    parser.add_argument(
        "--posterior-samples",
        type=int,
        default=1000,
        help="Number of posterior samples (default: 1000)",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=100,
        help="Number of samples to plot (default: 100)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.labels and len(args.labels) != len(args.configs):
        parser.error("Number of labels must match number of configs")

    # Determine labels for models
    if args.labels:
        labels = args.labels
    else:
        labels = [
            os.path.splitext(os.path.basename(cfg))[0]
            for cfg in args.configs
        ]

    # Set random seeds
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load parameters (shared "true" parameters)
    with open(args.parameters, "r") as f:
        params_data = yaml.safe_load(f)

    theta = params_data["theta"]
    change_times = params_data["change_times"]
    true_params = params_data["true_params"]

    n_epochs = len(true_params) - 1
    n_params = len(true_params)

    # Extract true values
    true_pop_sizes_log10 = true_params[:-1]
    true_pop_sizes = 10 ** np.array(true_pop_sizes_log10)
    true_recomb_rate = true_params[-1]

    # Load all models and generate posteriors
    models = []
    all_posterior_samples = []

    for i, (config_path, label) in enumerate(zip(args.configs, labels)):
        print(f"\nProcessing model {i+1}/{len(args.configs)}: {config_path}")
        print(f"  Using label: {label}")

        # Prefer per-config tree sequence:
        # <output>/simulated_data_<LABEL>/simulated.trees
        ts_path = os.path.join(
            args.output,
            f"simulated_data_{label}",
            "simulated.trees",
        )

        if os.path.exists(ts_path):
            print(f"  Found per-config tree sequence at {ts_path}")
        else:
            # Fall back to the global --tree-sequence
            ts_path = args.tree_sequence
            print(f"  Per-config tree sequence not found, using global: {ts_path}")

        ts = tskit.load(ts_path)
        print(
            f"  Loaded tree sequence: {ts.num_samples} samples, "
            f"{ts.sequence_length} bp"
        )

        # Load model and process this specific tree sequence
        model_data = load_model_and_process(config_path, ts, device)

        # grab simulator and prior bounds
        simulator_config = model_data["config"]["simulator"]
        import ts_simulators

        simulator = getattr(ts_simulators, simulator_config["class_name"])(
            simulator_config
        )

        # Set label to our label (override processor-embedding name)
        model_data["label"] = label

        # Create posterior (use simulator's prior directly like in plot_diagnostics.py)
        prior = BoxUniform(
            simulator.prior.base_dist.low.to(device),
            simulator.prior.base_dist.high.to(device),
        )
        posterior = DirectPosterior(
            posterior_estimator=model_data["normalizing_flow"],
            prior=prior,
            device=device,
        )

        # Sample from posterior
        posterior_samples = (
            posterior.sample(
                [args.posterior_samples],
                x=model_data["embedding"],
                show_progress_bars=False,
            )
            .cpu()
            .numpy()
        )

        # Store results
        model_data["posterior_samples"] = posterior_samples
        model_data["posterior_mean"] = posterior_samples.mean(axis=0)
        model_data["posterior_std"] = posterior_samples.std(axis=0)

        models.append(model_data)
        all_posterior_samples.append(posterior_samples)

    # Define consistent colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # Times for step plots
    times_with_one = np.concatenate([[1], np.array(change_times), [100000]])

    # Plot 1: Posterior mean comparison with credible intervals
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(1, figsize=(12, 8), constrained_layout=True)

    # Plot true parameters
    plot_step_function(
        times_with_one, true_pop_sizes, ax, label="True", color="black", linewidth=3
    )

    # Plot each model's posterior
    for i, model in enumerate(models):
        # Use mean of log10 values, then convert to linear for plotting
        posterior_mean_log10 = np.mean(model["posterior_samples"][:, :-1], axis=0)
        posterior_mean_pop_sizes = 10**posterior_mean_log10

        # Get population sizes in linear scale for credible intervals
        posterior_pop_sizes = 10 ** model["posterior_samples"][:, :-1]

        # Plot mean
        plot_step_function(
            times_with_one,
            posterior_mean_pop_sizes,
            ax,
            label=model["label"],
            color=colors[i],
            linewidth=2.5,
        )

        # Add credible intervals
        lower = np.percentile(posterior_pop_sizes, 2.5, axis=0)
        upper = np.percentile(posterior_pop_sizes, 97.5, axis=0)

        for j in range(len(times_with_one) - 1):
            ax.fill_between(
                [times_with_one[j], times_with_one[j + 1]],
                [lower[j], lower[j]],
                [upper[j], upper[j]],
                alpha=0.2,
                color=colors[i],
            )

    ax.set_xlabel("Time (generations ago)")
    ax.set_ylabel("Population size")
    ax.set_xlim(1, max(change_times) * 1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.title("Posterior Mean Comparison Across Models")
    plt.savefig(f"{args.output}/comparison-posterior-means.png", dpi=150)

    # Plot 2: Marginal distributions comparison
    n_cols = n_params
    n_rows = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Parameter labels
    param_labels = [f"log10(N{i+1})" for i in range(n_epochs)] + ["Recomb rate"]

    for row, model in enumerate(models):
        # Plot population size marginals
        for col in range(n_epochs):
            ax = axes[row, col]
            # The posterior samples are already in log10 space for population sizes
            posterior_log10 = model["posterior_samples"][:, col]

            sns.histplot(
                posterior_log10, bins=30, kde=True, ax=ax, color=colors[row], alpha=0.5
            )

            # Add true value
            ax.axvline(
                true_pop_sizes_log10[col],
                color="black",
                linestyle="--",
                linewidth=2,
                label="True",
            )

            # Add posterior mean - show mean of log10 values
            mean_val = np.mean(model["posterior_samples"][:, col])
            ax.axvline(
                mean_val, color=colors[row], linestyle="-", linewidth=2, label="Mean"
            )

            if row == 0:
                ax.set_title(param_labels[col])
            if col == 0:
                ax.set_ylabel(f"{model['label']}\nDensity")
            if row == n_rows - 1:
                ax.set_xlabel("Value")

            if row == 0 and col == 0:
                ax.legend()

        # Plot recombination rate marginal
        ax = axes[row, -1]
        sns.histplot(
            model["posterior_samples"][:, -1],
            bins=30,
            kde=True,
            ax=ax,
            color=colors[row],
            alpha=0.5,
        )
        ax.axvline(true_recomb_rate, color="black", linestyle="--", linewidth=2)
        ax.axvline(
            model["posterior_mean"][-1], color=colors[row], linestyle="-", linewidth=2
        )

        if row == 0:
            ax.set_title(param_labels[-1])
        if row == n_rows - 1:
            ax.set_xlabel("Value")

    plt.suptitle("Marginal Distributions Comparison")
    plt.savefig(f"{args.output}/comparison-marginals.png", dpi=150)

    # Plot 3: Performance metrics comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Calculate metrics for each model
    model_names = [m["label"] for m in models]
    pop_size_errors = []
    recomb_errors = []

    for model in models:
        # Use mean of log10 values, then convert to linear
        posterior_mean_log10 = np.mean(model["posterior_samples"][:, :-1], axis=0)
        posterior_mean_pop_sizes = 10**posterior_mean_log10

        # Mean relative error for population sizes
        mre = np.mean(
            np.abs((posterior_mean_pop_sizes - true_pop_sizes) / true_pop_sizes)
        )
        pop_size_errors.append(mre)

        # Relative error for recombination rate
        recomb_error = (
            abs(model["posterior_mean"][-1] - true_recomb_rate) / true_recomb_rate
        )
        recomb_errors.append(recomb_error)

    # Plot population size errors
    x = np.arange(len(models))
    ax1.bar(x, np.array(pop_size_errors) * 100, color=colors[: len(models)])
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Mean Relative Error (%)")
    ax1.set_title("Population Size Estimation Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot recombination rate errors
    ax2.bar(x, np.array(recomb_errors) * 100, color=colors[: len(models)])
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Relative Error (%)")
    ax2.set_title("Recombination Rate Estimation Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Performance Comparison")
    plt.savefig(f"{args.output}/comparison-metrics.png", dpi=150)

    # Plot 4: Uncertainty comparison (credible interval widths)
    fig, ax = plt.subplots(1, figsize=(10, 6), constrained_layout=True)

    width = 0.8 / len(models)
    x = np.arange(n_epochs + 1)

    for i, model in enumerate(models):
        ci_widths = []

        # Calculate CI widths for population sizes
        for j in range(n_epochs):
            posterior_vals = 10 ** model["posterior_samples"][:, j]
            lower = np.percentile(posterior_vals, 2.5)
            upper = np.percentile(posterior_vals, 97.5)
            # Normalize by geometric mean (10^mean of log values)
            geometric_mean = 10 ** np.mean(model["posterior_samples"][:, j])
            ci_width = (upper - lower) / geometric_mean
            ci_widths.append(ci_width)

        # Calculate CI width for recombination rate
        lower = np.percentile(model["posterior_samples"][:, -1], 2.5)
        upper = np.percentile(model["posterior_samples"][:, -1], 97.5)
        ci_width = (upper - lower) / model["posterior_mean"][-1]
        ci_widths.append(ci_width)

        # Plot bars
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, ci_widths, width, label=model["label"], color=colors[i])

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Normalized CI Width")
    ax.set_title("Uncertainty Comparison (95% CI Width / Mean)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{i+1}" for i in range(n_epochs)] + ["Recomb"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(f"{args.output}/comparison-uncertainty.png", dpi=150)

    # Plot 5: Ribbon plot comparison - each model gets its own subplot
    n_models = len(models)
    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(12, 6 * n_models),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    for i, (model, ax) in enumerate(zip(models, axes)):
        # Extract posterior samples for this model
        posterior_pop_sizes = 10 ** model["posterior_samples"][:, :-1]

        # Compute quantiles for each epoch
        lower_95 = np.percentile(posterior_pop_sizes, 2.5, axis=0)
        upper_95 = np.percentile(posterior_pop_sizes, 97.5, axis=0)
        lower_80 = np.percentile(posterior_pop_sizes, 10, axis=0)
        upper_80 = np.percentile(posterior_pop_sizes, 90, axis=0)
        lower_50 = np.percentile(posterior_pop_sizes, 25, axis=0)
        upper_50 = np.percentile(posterior_pop_sizes, 75, axis=0)

        def step_ribbon(times, lower, upper):
            # times: (n,) array of change times (including present)
            # lower, upper: (n-1,) arrays of quantiles for each epoch
            step_times = np.empty(2 * (len(times) - 1))
            step_lower = np.empty_like(step_times)
            step_upper = np.empty_like(step_times)
            for j in range(len(times) - 1):
                step_times[2 * j] = times[j]
                step_times[2 * j + 1] = times[j + 1]
                step_lower[2 * j] = lower[j]
                step_lower[2 * j + 1] = lower[j]
                step_upper[2 * j] = upper[j]
                step_upper[2 * j + 1] = upper[j]
            return step_times, step_lower, step_upper

        # Plot 95% ribbon
        st, sl, su = step_ribbon(times_with_one, lower_95, upper_95)
        ax.fill_between(st, sl, su, color=colors[i], alpha=0.15, label="95% CI")

        # Plot 80% ribbon
        st, sl, su = step_ribbon(times_with_one, lower_80, upper_80)
        ax.fill_between(st, sl, su, color=colors[i], alpha=0.25, label="80% CI")

        # Plot 50% ribbon
        st, sl, su = step_ribbon(times_with_one, lower_50, upper_50)
        ax.fill_between(st, sl, su, color=colors[i], alpha=0.35, label="50% CI")

        # Overlay true population size
        plot_step_function(
            times_with_one, true_pop_sizes, ax, label="True", color="black", linewidth=3
        )

        # Add model label
        ax.set_title(f"{model['label']} - Posterior Distribution", fontsize=14)
        ax.set_ylabel("Population size")
        ax.set_xlim(1, max(change_times) * 1.2)
        ax.set_ylim(5e1, 5e5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (generations ago)")
    plt.suptitle("Posterior Ribbon Plots Comparison", fontsize=16)
    plt.savefig(f"{args.output}/comparison-ribbons.png", dpi=150)

    # Save comparison results
    comparison_results = {
        "true_params": true_params,
        "change_times": change_times,
        "models": [],
    }

    for model in models:
        comparison_results["models"].append(
            {
                "label": model["label"],
                "config_path": model["config"]["project_dir"],
                "posterior_samples": model["posterior_samples"],
                "posterior_mean": model["posterior_mean"],
                "posterior_std": model["posterior_std"],
            }
        )

    with open(f"{args.output}/comparison-results.pkl", "wb") as f:
        pickle.dump(comparison_results, f)

    # Print summary statistics
    print(f"\nComparison complete! Results saved to {args.output}/")
    print("\nModel Performance Summary:")
    print("-" * 60)
    print(f"{'Model':<20} {'Pop Size MRE':<15} {'Recomb RE':<15}")
    print("-" * 60)

    for i, model in enumerate(models):
        print(
            f"{model['label']:<20} {pop_size_errors[i]:<15.2} {recomb_errors[i]:<15.2}"
        )

    print("\nTrue parameters:")
    print(f"  Population sizes (log10): {true_pop_sizes_log10}")
    print(f"  Recombination rate: {true_recomb_rate:.2e}")


if __name__ == "__main__":
    main()
