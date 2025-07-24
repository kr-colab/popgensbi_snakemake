#!/usr/bin/env python3
"""
Create 3x3 panel plot from existing comparison results.

This script reads the posterior samples from compare-analysis.py outputs
and creates a unified panel plot.
"""

import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import yaml

def plot_step_function(times, sizes, ax, label=None, color=None, alpha=1.0, linewidth=2, linestyle="-"):
    """Plot population size as step function backward in time."""
    assert len(times) == len(sizes) + 1, f"times (len={len(times)}) must be one longer than sizes (len={len(sizes)})"
    for i in range(len(sizes)):
        ax.hlines(sizes[i], times[i], times[i+1], 
                  colors=color, linewidth=linewidth, alpha=alpha, 
                  linestyle=linestyle, label=label if i == 0 else None)


def create_ribbon_plot(ax, posterior_samples, true_pop_sizes, change_times, color='blue', 
                       title=None, show_legend=True, ylim=(1e2, 1e5)):
    """Create a ribbon plot for a single model on the given axis."""
    # Extract population sizes from posterior samples
    posterior_pop_sizes = 10 ** posterior_samples[:, :-1]
    
    # Times for step plots
    times_with_one = np.concatenate([[1], np.array(change_times), [100000]])

    # Compute quantiles for each epoch
    lower_95 = np.percentile(posterior_pop_sizes, 2.5, axis=0)
    upper_95 = np.percentile(posterior_pop_sizes, 97.5, axis=0)
    lower_80 = np.percentile(posterior_pop_sizes, 10, axis=0)
    upper_80 = np.percentile(posterior_pop_sizes, 90, axis=0)
    lower_50 = np.percentile(posterior_pop_sizes, 25, axis=0)
    upper_50 = np.percentile(posterior_pop_sizes, 75, axis=0)
    
    def step_ribbon(times, lower, upper):
        step_times = np.empty(2 * (len(times) - 1))
        step_lower = np.empty_like(step_times)
        step_upper = np.empty_like(step_times)
        for j in range(len(times) - 1):
            step_times[2*j] = times[j]
            step_times[2*j+1] = times[j+1]
            step_lower[2*j] = lower[j]
            step_lower[2*j+1] = lower[j]
            step_upper[2*j] = upper[j]
            step_upper[2*j+1] = upper[j]
        return step_times, step_lower, step_upper
    
    # Plot ribbons
    st, sl, su = step_ribbon(times_with_one, lower_95, upper_95)
    ax.fill_between(st, sl, su, color=color, alpha=0.15, label="95% CI" if show_legend else None)
    
    st, sl, su = step_ribbon(times_with_one, lower_80, upper_80)
    ax.fill_between(st, sl, su, color=color, alpha=0.25, label="80% CI" if show_legend else None)
    
    st, sl, su = step_ribbon(times_with_one, lower_50, upper_50)
    ax.fill_between(st, sl, su, color=color, alpha=0.35, label="50% CI" if show_legend else None)
    
    # Overlay true population size
    plot_step_function(times_with_one, true_pop_sizes, ax,
                      label="True" if show_legend else None, color="black", linewidth=3)
    
    # Styling
    ax.set_xlim(100, max(change_times) * 1.2)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title, fontsize=20)
    
    if show_legend:
        ax.legend(loc='best', fontsize=15, frameon=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create 3x3 panel plot from comparison results"
    )
    
    parser.add_argument(
        "--comparison-dirs",
        nargs="+",
        required=True,
        help="Directories containing comparison results"
    )
    parser.add_argument(
        "--scenario-labels",
        nargs="+",
        required=True,
        help="Labels for each scenario"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output filename for panel plot"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[15, 12],
        help="Figure size (width height)"
    )
    
    args = parser.parse_args()
    
    # Load results from each comparison directory
    all_results = []
    model_labels = None
    
    for comp_dir in args.comparison_dirs:
        # Load comparison results
        with open(os.path.join(comp_dir, "comparison-results.pkl"), "rb") as f:
            results = pickle.load(f)
        
        all_results.append(results)
        
        # Get model labels from first result
        if model_labels is None:
            model_labels = [m["label"] for m in results["models"]]
    
    # Create panel plot
    n_scenarios = len(all_results)
    n_models = len(model_labels)
    
    fig, axes = plt.subplots(n_scenarios, n_models, 
                            figsize=args.figsize,
                            constrained_layout=True, sharex=True, sharey=True)
    
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)
    elif n_models == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colors
    colors = plt.cm.Accent(np.linspace(0, 1, n_models))

    # Plot each scenario and model
    for scenario_idx, results in enumerate(all_results):
        true_params = results["true_params"]
        change_times = results["change_times"]
        n_epochs = len(true_params) - 1
        
        # Extract true values
        true_pop_sizes_log10 = true_params[:-1]
        true_pop_sizes = 10 ** np.array(true_pop_sizes_log10)
        
        for model_idx, model_data in enumerate(results["models"]):
            ax = axes[scenario_idx, model_idx]
            
            create_ribbon_plot(
                ax,
                model_data["posterior_samples"],
                true_pop_sizes,
                change_times,
                color=colors[model_idx],
                title=None,
                show_legend=(scenario_idx == 0 and model_idx == 0),
                ylim=(1e2, 1e5)
            )
            
            # Add labels
            if scenario_idx == 0:
                ax.set_title(model_data["label"], fontsize=18)
    
    # Add superlabels
    fig.supylabel("Population size", fontsize=20)
    fig.supxlabel("Generations ago", fontsize=20)
    
    # Add scenario labels on the right
    for i, scenario_label in enumerate(args.scenario_labels):
        axes[i, n_models-1].yaxis.set_label_position("right")
        axes[i, n_models-1].set_ylabel(scenario_label, rotation=270, va="center", fontsize=18, labelpad=20)
    
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Panel plot saved to: {args.output}")


if __name__ == "__main__":
    main()