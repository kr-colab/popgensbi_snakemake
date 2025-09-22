import os
import zarr
import numpy as np
import tskit
import yaml
import ray
import argparse

parser = argparse.ArgumentParser("Plot posteriors after rescaling parameters by fitting theta analytically")
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

# simulator settings
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


# TODO: might be better to use moments, msprime routine seems downwards biased
#def expected_theta(demogr, total_mutation_rate, sample_sizes):
#    import moments
#    sfs = moments.Spectrum.from_demes(
#        demogr.to_demes(),
#        ...
#    )
#    return sfs.sum()


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


@ray.remote
def rescale_posterior(unscaled_params, target_diversity):
    """
    Rescale, simulate, calculate stats
    """
    demogr, scaled_params = get_rescaled_demography(unscaled_params, target_diversity)
    return scaled_params


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

par_cache = f"{args.output_prefix}cache-par.npy"
if not os.path.exists(par_cache) or args.overwrite:
    ray.init(num_cpus=args.num_cpus)
    rng = np.random.default_rng(RANDOM_SEED)
    rescaled_params = []
    for w, div in enumerate(obs_divergence):
        joblist = [
            rescale_posterior.remote(unscaled_params[w, :, i], div) 
            for i in range(unscaled_params.shape[-1])
        ]
        params = np.stack(ray.get(joblist))
        rescaled_params.append(np.stack(params))
    rescaled_params = np.stack(rescaled_params)
    np.save(par_cache, rescaled_params)
else:
    rescaled_params = np.load(par_cache)


# dump posterior means, medians, etc in logspace and in natural units
#flat_rescaled_params = rescaled_params.reshape(-1, rescaled_params.shape[-1])
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
labels = ["N_ANC", "N_CO", "N_FR_PRE", "N_FR_POST", "T_SPLIT", "M_CO_FR", "M_FR_CO"]

post_means = rescaled_params.mean(axis=1).mean(axis=0)
post_quantiles = np.quantile(rescaled_params, quantiles, axis=1).T.mean(axis=1)
summary_csv = open(f"{args.output_prefix}posterior-summary-log10.csv", "w")
summary_csv.write("parameter,post_mean,")
for q in quantiles: summary_csv.write(f"post_quant_{q},")
summary_csv.write("\n")
for i, nm in enumerate(labels):
    summary_csv.write(f"{nm},{post_means[i]},")
    for q in post_quantiles[i]:
        summary_csv.write(f"{q},")
    summary_csv.write("\n")
summary_csv.close()

post_means = (10 ** rescaled_params).mean(axis=1).mean(axis=0)
post_quantiles = np.quantile(10 ** rescaled_params, quantiles, axis=1).T.mean(axis=1)
summary_csv = open(f"{args.output_prefix}posterior-summary.csv", "w")
summary_csv.write("parameter,post_mean,")
for q in quantiles: summary_csv.write(f"post_quant_{q},")
summary_csv.write("\n")
for i, nm in enumerate(labels):
    summary_csv.write(f"{nm},{post_means[i]},")
    for q in post_quantiles[i]:
        summary_csv.write(f"{q},")
    summary_csv.write("\n")
summary_csv.close()

pretty_labels = [
    r"Size of ancestral",
    r"Size of CO",
    r"Size of FR (ancient)",
    r"Size of FR (current)",
    r"Split time",
    r"Migr CO $\rightarrow$ FR",
    r"Migr FR $\rightarrow$ CO",
]
summary_tex = open(f"{args.output_prefix}posterior-summary.tex", "w")
summary_tex.write(r"\begin{tabular}{lccccc}" + "\n")
summary_tex.write(r" & & & \multicolumn{4}{c}{Quantiles} \\" + "\n")
summary_tex.write(r" & Mean & Median & 0.05 & 0.25 & 0.75 & 0.95 \\" + "\n")
for i, nm in enumerate(pretty_labels):
    quants = post_quantiles[i]
    summary_tex.write(nm + " & ")
    if "Migr" in nm:
        summary_tex.write(f"{post_means[i]:.2e} & ")
        summary_tex.write(f"{post_quantiles[i, 2]:.2e} & ")
        summary_tex.write(f"{post_quantiles[i, 0]:.2e} & ")
        summary_tex.write(f"{post_quantiles[i, 1]:.2e} & ")
        summary_tex.write(f"{post_quantiles[i, 3]:.2e} & ")
        summary_tex.write(f"{post_quantiles[i, 4]:.2e} " + r"\\ ")
    else:
        summary_tex.write(f"{int(post_means[i]):,d} & ")
        summary_tex.write(f"{int(post_quantiles[i, 2]):,d} & ")
        summary_tex.write(f"{int(post_quantiles[i, 0]):,d} & ")
        summary_tex.write(f"{int(post_quantiles[i, 1]):,d} & ")
        summary_tex.write(f"{int(post_quantiles[i, 3]):,d} & ")
        summary_tex.write(f"{int(post_quantiles[i, 4]):,d} " + r"\\ ")
    summary_tex.write("\n")
summary_tex.write(r"\end{tabular}" + "\n")
summary_tex.close()


# subsample for plots
n_samp = 100 # samples per window
rescaled_params = 10 ** rescaled_params[:, :n_samp, :]

# plot
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
labels = [
    "Size of\nancestral",
    r"Size of CO",
    "Size of FR\n(ancient)",
    "Size of FR\n(current)",
    r"Split time",
    "Migr\nCO $\\rightarrow$ FR",
    "Migr\nFR $\\rightarrow$ CO",
]
lb = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e-9, 1e-9])
ub = np.array([1e7, 1e7, 1e7, 1e7, 1e7, 1e-2, 1e-2])
ls = np.array([False, False, False, False, False, False, False])
cols = len(labels)
rows = len(labels)
assert len(labels) == rescaled_params.shape[-1]
fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), constrained_layout=False)
for i in range(len(labels)):
    for j in range(len(labels)):
        if i < j:
            axs[i, j].set_yscale("log")
            axs[i, j].set_xscale("log")
            axs[i, j].set_ylim(lb[i], ub[i])
            axs[i, j].set_xlim(lb[j], ub[j])
            if j < len(labels) - 1:
                axs[i, j].tick_params(axis='x', bottom=False, labelbottom=False)
                axs[i, j].tick_params(axis='y', left=False, labelleft=False)
            else:
                axs[i, j].tick_params(axis='x', bottom=False, labelbottom=False)
                axs[i, j].tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)
                if labels[i].startswith("Migr"):
                    axs[i, j].set_yticks(10.0 ** np.array([-7, -4]))
                else:
                    axs[i, j].set_yticks(10.0 ** np.array([4, 6]))
            sns.kdeplot(
                x=rescaled_params[:, :, j].flatten(),
                y=rescaled_params[:, :, i].flatten(), 
                cmap="terrain",
                ax=axs[i, j],
            )
        elif i == j:
            axs[i, j].set_xscale("log")
            sns.kdeplot(x=rescaled_params[:, :, i].flatten(), color="grey", fill=True, ax=axs[i, j])
            axs[i, j].set_xlabel(labels[j])
            axs[i, j].set_ylabel(None)
            if labels[j].startswith("Migr"):
                axs[i, j].set_xticks(10.0 ** np.array([-7, -4]))
            else:
                axs[i, j].set_xticks(10.0 ** np.array([4, 6]))
            axs[i, j].set_xlim(lb[j], ub[j])
            axs[i, j].tick_params(axis='x', labelbottom=True)
            axs[i, j].tick_params(axis='y', labelleft=False, left=False)
        else:
            axs[i, j].set_visible(False)
fig.subplots_adjust(hspace=0.0, wspace=0.0)
plt.savefig(f"{args.output_prefix}rescaled-joint-posteriors.png")

