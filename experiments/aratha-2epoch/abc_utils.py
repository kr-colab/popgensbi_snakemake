"""
ABC utilities for AraTha-2epoch experiment.

This module provides functions for:
1. Calculating summary statistics from tree sequences
2. Running SMC-ABC inference using PyMC
3. Estimating posterior surfaces from ABC samples
"""

import numpy as np
import tskit
import torch
import sys
import os

# Add workflow scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "scripts"))
import ts_simulators


def calculate_summary_stats(ts, sample_sets=None):
    """
    Calculate summary statistics from a tree sequence.

    Statistics calculated:
    - Diversity (π): Nucleotide diversity
    - Segregating sites (S): Number of segregating sites
    - Tajima's D: Neutrality test statistic

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence to calculate statistics from
    sample_sets : list, optional
        List of sample sets for population-specific statistics.
        If None, uses all samples.

    Returns
    -------
    np.ndarray
        Array of shape (3,) containing [π, S, D]
    """
    if sample_sets is None:
        # Use all samples
        sample_sets = [ts.samples()]

    # Diversity (π)
    diversity = ts.diversity(sample_sets=sample_sets, mode='site')
    if isinstance(diversity, np.ndarray):
        diversity = diversity[0]  # Take first element if array

    # Segregating sites (S)
    segregating_sites = ts.num_sites

    # Tajima's D
    tajimas_d = ts.Tajimas_D(sample_sets=sample_sets, mode='site')
    if isinstance(tajimas_d, np.ndarray):
        tajimas_d = tajimas_d[0]  # Take first element if array

    # Handle NaN values (can occur with very few sites)
    if np.isnan(tajimas_d):
        tajimas_d = 0.0

    return np.array([diversity, float(segregating_sites), tajimas_d])


def standardize_stats(stats, obs_stats):
    """
    Standardize statistics by observed values.

    This is necessary because different statistics have vastly different scales:
    - Diversity: ~10^-5 to 10^-3
    - Segregating sites: 0-1000s
    - Tajima's D: -2 to +2

    Parameters
    ----------
    stats : np.ndarray
        Statistics to standardize
    obs_stats : np.ndarray
        Observed statistics for normalization

    Returns
    -------
    np.ndarray
        Standardized statistics
    """
    # Avoid division by zero
    epsilon = 1e-10
    return (stats - obs_stats) / (np.abs(obs_stats) + epsilon)


def abc_distance(sim_stats, obs_stats):
    """
    Calculate distance between simulated and observed statistics.

    Uses Euclidean distance on standardized statistics.

    Parameters
    ----------
    sim_stats : np.ndarray
        Simulated summary statistics
    obs_stats : np.ndarray
        Observed summary statistics

    Returns
    -------
    float
        Distance metric
    """
    standardized = standardize_stats(sim_stats, obs_stats)
    return np.sqrt(np.sum(standardized ** 2))


def create_abc_simulator(simulator_config, obs_stats):
    """
    Create a simulator function compatible with PyMC's pm.Simulator.

    Parameters
    ----------
    simulator_config : dict
        Configuration dictionary for the simulator
    obs_stats : np.ndarray
        Observed summary statistics for standardization

    Returns
    -------
    callable
        Function that takes rng, size, and parameters, returns summary statistics
    """
    def simulator_func(rng, size, nu, T):
        """
        Simulator function for PyMC ABC.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator from PyMC
        size : int
            Number of simulations (for PyMC interface, usually 1)
        nu : float or array
            Bottleneck severity parameter
        T : float or array
            Bottleneck timing parameter

        Returns
        -------
        np.ndarray
            Summary statistics (not standardized - PyMC handles distance)
        """
        # Extract scalar values from arrays if needed
        nu_val = float(np.atleast_1d(nu)[0])
        T_val = float(np.atleast_1d(T)[0])

        # Create simulator config with fixed parameters
        config = simulator_config.copy()
        config['nu'] = [nu_val, nu_val]
        config['T'] = [T_val, T_val]

        # Create simulator
        simulator = ts_simulators.AraTha_2epoch(config)

        # Generate random seed from rng
        seed = rng.integers(0, 2**31 - 1)

        # Run simulation
        ts, theta = simulator(seed=seed)

        # Calculate summary statistics
        stats = calculate_summary_stats(ts)

        return stats

    return simulator_func


def run_abc_smc(obs_ts, simulator_config, prior_bounds,
                n_samples=1000, seed=1024, num_cores=4):
    """
    Run SMC-ABC inference and return samples.

    Parameters
    ----------
    obs_ts : tskit.TreeSequence
        Observed tree sequence
    simulator_config : dict
        Configuration for the simulator
    prior_bounds : np.ndarray
        Array of shape (2, 2) with [[nu_low, nu_high], [T_low, T_high]]
    n_samples : int, default=1000
        Number of posterior samples to draw
    seed : int, default=1024
        Random seed for reproducibility
    num_cores : int, default=4
        Number of CPU cores for parallel computation

    Returns
    -------
    samples : np.ndarray
        Posterior samples of shape (n_samples, 2) for [nu, T]
    trace : arviz.InferenceData
        PyMC trace object with full inference details
    """
    import pymc as pm
    from pymc.smc.kernels import IMH

    # Calculate observed summary stats
    obs_stats = calculate_summary_stats(obs_ts)
    print(f"Observed summary statistics: π={obs_stats[0]:.6f}, S={obs_stats[1]:.0f}, D={obs_stats[2]:.4f}")

    # Create simulator function
    sim_func = create_abc_simulator(simulator_config, obs_stats)

    # PyMC model
    with pm.Model() as model:
        # Priors matching NPE priors
        nu = pm.Uniform('nu', lower=prior_bounds[0, 0], upper=prior_bounds[0, 1])
        T = pm.Uniform('T', lower=prior_bounds[1, 0], upper=prior_bounds[1, 1])

        # ABC Simulator
        # Note: PyMC's Simulator handles the distance calculation internally
        sim = pm.Simulator('sim', sim_func,
                          params=[nu, T],
                          distance='gaussian',
                          sum_stat='sort',
                          epsilon=1.0,  # Will be adapted by SMC
                          observed=obs_stats)

        # SMC-ABC sampling
        print(f"Running SMC-ABC with {n_samples} samples...")
        trace = pm.sample_smc(
            draws=n_samples,
            kernel=IMH,
            random_seed=seed,
            cores=num_cores,
            progressbar=True,
        )

    # Extract samples
    samples = np.column_stack([
        trace.posterior['nu'].values.flatten(),
        trace.posterior['T'].values.flatten()
    ])

    print(f"ABC inference complete. Generated {samples.shape[0]} samples.")
    print(f"Posterior mean: nu={samples[:, 0].mean():.4f}, T={samples[:, 1].mean():.4f}")

    return samples, trace


def estimate_abc_surface(samples, breaks_nu, breaks_T):
    """
    Estimate log-probability surface from ABC samples using KDE.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples of shape (n_samples, 2)
    breaks_nu : np.ndarray
        Grid breaks for nu parameter
    breaks_T : np.ndarray
        Grid breaks for T parameter

    Returns
    -------
    log_prob : np.ndarray
        Log-probability values on grid, flattened to match grid format
    """
    from scipy.stats import gaussian_kde

    # Create KDE from samples
    kde = gaussian_kde(samples.T)

    # Create grid
    grid_nu = (breaks_nu[1:] + breaks_nu[:-1]) / 2
    grid_T = (breaks_T[1:] + breaks_T[:-1]) / 2

    grid = np.array([[nu, T] for nu in grid_nu for T in grid_T])

    # Evaluate KDE on grid
    log_prob = kde.logpdf(grid.T)

    # Normalize (subtract max for numerical stability)
    log_prob -= np.max(log_prob)

    return log_prob
