import os
import zarr
import numpy as np
import tskit
import yaml
import ray
import argparse

parser = argparse.ArgumentParser("Plot observed and posterior summary statistics across windows")
parser.add_argument("--configfile", type=str, help="Path to config file", default="npe-config/DroMel_CO_FR_rnn.yaml")
parser.add_argument("--num-cpus", type=int, default=50)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--output-prefix", type=str, default="/sietch_colab/data_share/popgen_npe/dromel-isolation/plots/rnn-")
args = parser.parse_args()

if not os.path.exists(os.path.dirname(args.output_prefix)): os.makedirs(os.path.dirname(args.output_prefix))
config = yaml.safe_load(open(args.configfile))

# paths 
N_TRAIN = int(config["n_train"])
RANDOM_SEED = int(config["random_seed"])
TRAIN_SEPARATELY = bool(config["train_embedding_net_separately"])
SIMULATOR = config["simulator"]["class_name"]
PROCESSOR = config["processor"]["class_name"]
EMBEDDING = config["embedding_network"]["class_name"]
UID = f"{SIMULATOR}-{PROCESSOR}-{EMBEDDING}-{RANDOM_SEED}-{N_TRAIN}"
UID += "-sep" if TRAIN_SEPARATELY else "-e2e"
PROJECT_DIR = os.path.join(config["project_dir"], UID)
VCF_PATH = config["prediction"].get("vcf")
VCF_PREFIX = os.path.basename(VCF_PATH)
VCF_DIR = os.path.join(PROJECT_DIR, VCF_PREFIX)
TENSOR_DIR = os.path.join(VCF_DIR, "tensors")
TREE_DIR = os.path.join(VCF_DIR, "trees")

# load zarr with windows, predictions
pred_zarr = zarr.load(os.path.join(TENSOR_DIR, "zarr"))


# the strategy is:
#   - for a given posterior sample, rescale parameter estimates so as to exactly match diversity/segsites
#     with the observed data (analogous to what dadi does).
#   - given the rescaled model, simulate a new tree sequence
#   - calculate site statistics from the simulation and compare to observed data
recombination_rate = config["simulator"]["recombination_rate"]
haploid = config["simulator"]["haploid"]
samples = config["simulator"]["samples"]
sequence_length = config["simulator"]["sequence_length"]
mutation_rate = config["simulator"]["mutation_rate"]


def demography(unscaled_params, scale=1.0):
    """
    Build the model given parameter estimates. Rescale expected "theta" according to
    `scale`. Return demography object and rescaled parameters (in log10 space)
    """
    import msprime
    N_ANC, N_CO, N_FR0, N_FR1, T, m_CO_FR, m_FR_CO = (10 ** unscaled_params)
    # rescaling
    N_ANC *= scale
    N_CO *= scale
    N_FR0 *= scale
    N_FR1 *= scale
    T *= scale
    m_CO_FR /= scale
    m_FR_CO /= scale
    scaled_params = np.array([N_ANC, N_CO, N_FR0, N_FR1, T, m_CO_FR, m_FR_CO])
    # model definition
    G_FR = np.log(N_FR1 / N_FR0) / T
    demogr = msprime.Demography()
    demogr.add_population(name="CO", initial_size=N_CO)
    demogr.add_population(name="FR", initial_size=N_FR1, growth_rate=G_FR)
    demogr.add_population(name="ANC", initial_size=N_ANC)
    demogr.migration_matrix = np.array([[0, m_CO_FR, 0], [m_FR_CO, 0, 0], [0, 0, 0]])
    demogr.add_population_split(time=T, derived=["CO", "FR"], ancestral="ANC")
    return demogr, np.log10(scaled_params)


def get_rescaled_demography(unscaled_params, target_divergence):
    """
    Rescale an already built model to match theta with some target value
    """
    demogr, _ = demography(unscaled_params)
    debugger = demogr.debug()
    expected_divergence = \
        2 * debugger.mean_coalescence_time(lineages={"CO":1, "FR":1}, max_iter=100) * mutation_rate 
    theta_scale = target_divergence / expected_divergence
    scaled_demogr, scaled_params = demography(unscaled_params, theta_scale)
    #print(
    #    target_divergence,
    #    expected_divergence,
    #    2 * scaled_demogr.debug().mean_coalescence_time(lineages={"CO":1, "FR":1}) * mutation_rate,
    #)
    return scaled_demogr, scaled_params

 
def simulate_arg(demogr, seed):
    """
    Note that we draw a random recombination rate from the prior
    """
    import msprime
    rng = np.random.default_rng(seed)
    r = rng.uniform(recombination_rate[0], recombination_rate[1])
    seeds = rng.integers(2 ** 32 - 1, size=2)
    sample_sets = [
        msprime.SampleSet(n, population=p, ploidy=1 if haploid else 2)
        for p, n in samples.items()
    ]
    ts = msprime.sim_ancestry(
        sample_sets, 
        demography=demogr, 
        sequence_length=sequence_length,
        recombination_rate=r,
        random_seed=seeds[0],
    )
    ts = msprime.sim_mutations(
        ts, 
        rate=mutation_rate, 
        random_seed=seeds[1],
    )
    return ts


def calc_stats(ts):
    """
    Calculate some summary statistics to compare
    """
    sample_sets = [ts.samples(population=0), ts.samples(population=1)]
    fst_CO_FR = ts.Fst(sample_sets, indexes=[(0, 1)]).item()
    div_CO_FR = ts.divergence(sample_sets, indexes=[(0, 1)]).item()
    taj_CO_FR = ts.Tajimas_D()
    div_CO, div_FR = ts.diversity(sample_sets).squeeze().tolist()
    seg_CO, seg_FR = ts.segregating_sites(sample_sets).squeeze().tolist()
    taj_CO, taj_FR = ts.Tajimas_D(sample_sets).squeeze().tolist()
    return fst_CO_FR, div_CO_FR, taj_CO_FR, seg_CO, seg_FR, div_CO, div_FR, taj_CO, taj_FR


@ray.remote
def simulated_stats(unscaled_params, target_diversity, seed):
    """
    Rescale, simulate, calculate stats
    """
    demogr, scaled_params = get_rescaled_demography(unscaled_params, target_diversity)
    ts = simulate_arg(demogr, seed)
    stats = calc_stats(ts)
    return stats, scaled_params


# get observed stats
window_start = pred_zarr["window_start"]
window_end = pred_zarr["window_end"]

obs_cache = f"{args.output_prefix}cache-obs.npy"
if not os.path.exists(obs_cache) or args.overwrite:
    obs_stats = []
    for i, (a, b) in enumerate(zip(window_start, window_end)):
        f = os.path.join(TREE_DIR, f"{i}.trees")
        ts = tskit.load(f)
        obs_stats.append(calc_stats(ts))
    obs_stats = np.stack(obs_stats)
    np.save(obs_cache, obs_stats)
else:
    obs_stats = np.load(obs_cache)


# extract observed diversgence per window
obs_divergence = obs_stats[:, 1]

# simulate from rescaled model
unscaled_params = pred_zarr["predictions"]
assert window_start.size == unscaled_params.shape[0]

exp_cache = f"{args.output_prefix}cache-exp.npy"
if not os.path.exists(exp_cache) or args.overwrite:
    ray.init(num_cpus=args.num_cpus)
    rng = np.random.default_rng(RANDOM_SEED)
    exp_stats = []
    for w, div in enumerate(obs_divergence):
        seeds = rng.integers(2 ** 10 - 1, size=100)
        joblist = [
            simulated_stats.remote(unscaled_params[w, :, i], div, s) for i, s in enumerate(seeds)
        ]
        stats, params = zip(*ray.get(joblist))
        print(f"window {w}, obs divergence {div}, sim divergence {np.stack(stats)[:,1].mean()}")
        exp_stats.append(np.stack(stats))
    exp_stats = np.stack(exp_stats)
    np.save(exp_cache, exp_stats)
else:
    exp_stats = np.load(exp_cache)


# plot
import matplotlib.pyplot as plt
import seaborn as sns
labels = ["Fst (CO,FR)", "SegSites (CO)", "SegSites (FR)", "Tajima's D (CO)", "Tajima's D (FR)"]
exp_stats = exp_stats[..., [0,3,4,7,8]]
obs_stats = obs_stats[..., [0,3,4,7,8]]
rows = exp_stats.shape[-1]
assert len(labels) == rows
fig, axs = plt.subplots(rows, 1, figsize=(5, rows * 1.5), constrained_layout=True, sharex=True)
for i, ax in enumerate(axs.ravel()):
    #axs[i].boxplot([x for x in exp_stats[:, :, i]], 
    #    showfliers=False, 
    #    medianprops={"color":"gray"}, 
    #    boxprops={"color":"gray"}, 
    #    whiskerprops={"color":"gray"}
    #)
    sns.boxplot(
        [x for x in exp_stats[:, :, i]], 
        whis=(5,95),
        fliersize=0,
        color="lightgrey",
        linecolor="grey",
        ax=ax,
    )
    if labels[i][:3] == "Fst":
        ax.set_ylim(0, 0.4)
    elif labels[i][:3] == "Seg":
        ax.set_ylim(0, 0.04)
    elif labels[i][:3] == "Taj":
        ax.set_ylim(-1.5, 1.00)
    ax.plot(
        np.arange(window_start.size), 
        obs_stats[:, i], "o", 
        color="firebrick", markersize=2, zorder=9,
    )
    ax.set_ylabel(labels[i])
    ax.set_xticklabels([])
fig.supxlabel("Window")
plt.savefig(f"{args.output_prefix}stats-over-windows.png")

