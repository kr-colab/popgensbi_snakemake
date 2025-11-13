import argparse
import numpy as np
import tskit
import sys
import os
import yaml
import torch
import moments
import stdpopsim
import ray
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import seaborn as sns
import pickle

from torch import Tensor
from sbi.inference import DirectPosterior
from sbi.utils import BoxUniform

# put popgensbi_snakemake scripts in the load path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "scripts"))
import ts_simulators
import ts_processors
from utils import get_least_busy_gpu

# ABC utilities
import abc_utils


#if torch.cuda.is_available():
#    best_gpu = get_least_busy_gpu()
#    device = f"cuda:{best_gpu}"
#else:
#    device = "cpu"
# TODO: for some reason DirectPosterior.sample is using tensors on multiple
# GPUs and erroring out, if the above is used to get a free device. Need to
# sort out why this is happening. Seems to work fine outside of snakemake.
device = "cuda:0"  # DEBUG: hardcoded for now


parser = argparse.ArgumentParser(
    "Plot bivariate posteriors from normalizing flow(s) and profile log-likelihood "
    "surface from a Poisson random-field model of the SFS (AraTha 2-epoch model)."
)
parser.add_argument(
    "--configfile", 
    help="Config file(s) used for snakemake training workflow: trained neural "
    "network from each will be used for plotting, but data will "
    "be simulated using first configfile", 
    type=str, nargs="+", required=True,
)
parser.add_argument("--outpath", help="Path to output directory", type=str, required=True)
parser.add_argument(
    "--params", 
    help="Parameter values to use for simulation. Plots vary "
    "pairs of parameters while integrating/profiling the rest out.",
    type=float, nargs="+", required=True,
)
parser.add_argument("--vmin", help="Truncate loglikelihood at this lower bound for plotting surface", type=float, default=-600)
parser.add_argument("--sequence-length", help="Use this sequence length rather than what is in the config", type=float, default=None)
parser.add_argument("--skip-sanity-checks", action="store_true", help="Skip visual checks for model misspecification")
parser.add_argument("--skip-bootstrap", action="store_true", help="Skip parametric bootstrap")
parser.add_argument("--seed", help="Random seed passed to simulator", type=int, default=1024)
parser.add_argument("--grid-size", help="Grid size for plots", type=int, default=25)
parser.add_argument("--posterior-samples", help="Number of samples to draw from posterior/bootstrap", type=int, default=1000)
parser.add_argument("--num-cpus", help="Number of workers for parallel computation", type=int, default=20)
args = parser.parse_args()


ray.init(num_cpus=args.num_cpus)
rng = np.random.default_rng(args.seed)


# set up simulator from first config file
if not os.path.exists(args.outpath): os.makedirs(args.outpath)
config = yaml.safe_load(open(args.configfile[0]))
simulator_config = config["simulator"]
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)
prior_low = simulator.prior.base_dist.low.numpy()
prior_high = simulator.prior.base_dist.high.numpy()
assert len(simulator.parameters) == len(args.params), \
    f"Model needs {len(simulator.parameters)} parameters but {len(args.params)} were supplied"
if args.sequence_length is not None:
    simulator_config["sequence_length"] = args.sequence_length
for name, value in zip(simulator.parameters, args.params):
    print(f"Using {name} = {value}")
    simulator_config[name] = [value, value]
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)
ts, theta = simulator(args.seed)


# set up diffusion approximation
sequence_length = simulator_config["sequence_length"]
sample_sizes = simulator_config["samples"]
population_map = {pop.metadata["name"]: pop.id for pop in ts.populations()}
#mutation_rate = simulator_config["mutation_rate"]
#recombination_rate = simulator_config["recombination_rate"]
# currently the AraTha model doesn't use mutation/recombination from the config, so:
contig = stdpopsim.get_species("AraTha").get_contig(length=sequence_length)
mutation_rate = contig.mutation_rate
recombination_rate = contig.recombination_map.mean_rate


def empirical_sfs(ts):
    sample_sets = []
    for pop in sample_sizes.keys():
        pop_id = population_map[pop]
        sample_id = np.flatnonzero(
            np.logical_and(
                ts.nodes_population == pop_id, 
                np.bitwise_and(ts.nodes_flags, tskit.NODE_IS_SAMPLE)
            )
        )
        sample_sets.append(sample_id)
    sfs = ts.allele_frequency_spectrum(
        sample_sets=sample_sets, 
        polarised=True, 
        span_normalise=False,
    )
    return moments.Spectrum(sfs)


def demographic_model(theta: np.ndarray) -> stdpopsim.DemographicModel:
    import msprime
    N_A = 746148.0
    nu, T = theta
    demog = msprime.Demography()
    demog.add_population(
        name="SouthMiddleAtlas", 
        initial_size=nu * N_A,
    )
    demog.add_population_parameters_change(
        time=2 * N_A * T, 
        population="SouthMiddleAtlas", 
        initial_size=N_A,
    )
    return demog, N_A


def diffusion_sfs(theta: np.ndarray) -> np.ndarray:
    """
    Get expected SFS under a diffusion approximation
    """
    demography, N_A = demographic_model(theta)
    return moments.Spectrum.from_demes(
        demography.to_demes(),
        sample_sizes=[n * 2 for n in sample_sizes.values()],
        sampled_demes=[p for p in sample_sizes.keys()],
        theta=N_A * sequence_length * mutation_rate * 4,
    )


def simulated_sfs(theta, seed: int = None) -> moments.Spectrum:
    """
    Get expected SFS under the coalescent by simulation
    """
    import msprime  # avoid serialization issues
    model, _ = demographic_model(theta)
    ts = msprime.sim_ancestry(
        samples=sample_sizes,
        demography=model,
        recombination_rate=recombination_rate,
        sequence_length=sequence_length,
        discrete_genome=False,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(
        ts, 
        rate=mutation_rate, 
        discrete_genome=False,
        random_seed=seed + 1,
    )
    return empirical_sfs(ts)


def loglikelihood(theta: np.ndarray, obs_sfs: np.ndarray) -> float:
    fit_sfs = diffusion_sfs(theta)
    return np.sum(np.log(fit_sfs) * obs_sfs - fit_sfs)


def optimize(
    theta: np.ndarray,
    obs_sfs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bobyqa: bool = True,
    nan_on_fail: bool = False,
) -> np.ndarray:
    import nlopt

    def obj(pars, grad):
        if grad.size > 0:
            grad[:] = nd.Gradient(
                lambda x: loglikelihood(x, obs_sfs), 
                step=1e-4, richardson_terms=2
            )(pars)
        return loglikelihood(pars, obs_sfs)

    opt = nlopt.opt(nlopt.LN_BOBYQA if bobyqa else nlopt.LD_LBFGS, theta.size)
    opt.set_max_objective(obj)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_ftol_rel(1e-8)
    try:
        theta_hat = opt.optimize(theta)
    except:
        theta_hat = np.full_like(theta, np.nan) if nan_on_fail else theta.copy()
    on_boundary = (
        np.any(np.isclose(theta_hat, lower)) or
        np.any(np.isclose(theta_hat, upper))
    )
    return theta_hat, on_boundary


@ray.remote
def simulate_parallel(seed: int, pars: np.ndarray) -> np.ndarray:
    return simulated_sfs(pars, seed=seed)


@ray.remote
def loglikelihood_parallel(theta: np.ndarray, obs_sfs: np.ndarray) -> float:
    return loglikelihood(theta, obs_sfs)


@ray.remote
def optimize_parallel(sfs: np.ndarray, pars: np.ndarray) -> np.ndarray:
    mle, _ = optimize(pars, sfs, prior_low, prior_high, nan_on_fail=True)
    return mle


if not args.skip_sanity_checks:
    # SANITY CHECK: do diffusion and demographic model agree?
    ck_seed = rng.integers(2 ** 32 - 1, size=100)
    ck_dif_sfs = diffusion_sfs(np.array(args.params))
    ck_sim_sfs = sum(
        ray.get([simulate_parallel.remote(seed, np.array(args.params)) for seed in ck_seed])
    ) / ck_seed.size
    plt.clf()
    fig, axs = plt.subplots(1, figsize=(4, 4), constrained_layout=True)
    axs.plot(ck_dif_sfs, ck_sim_sfs, "o", color="red")
    axs.axline(
        (np.mean(ck_dif_sfs), np.mean(ck_dif_sfs)), 
        (np.mean(ck_dif_sfs) + 1e-8, np.mean(ck_dif_sfs) + 1e-8),
        linestyle="dashed",
        color="black",
    )
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel("diffusion sfs")
    axs.set_ylabel("simulated sfs (manual)")
    plt.savefig(f"{args.outpath}/diffusion-vs-simulation.png")
    
    
    # SANITY CHECK: do diffusion and already-implemented simulator agree?
    ck_seed = rng.integers(2 ** 32 - 1, size=100)
    ck_mod_sfs = sum(
        [empirical_sfs(simulator(seed)[0]) for seed in ck_seed]
    ) / ck_seed.size
    plt.clf()
    fig, axs = plt.subplots(1, figsize=(4, 4), constrained_layout=True)
    axs.plot(ck_dif_sfs, ck_mod_sfs, "o", color="red")
    axs.axline(
        (np.mean(ck_dif_sfs), np.mean(ck_dif_sfs)), 
        (np.mean(ck_dif_sfs) + 1e-8, np.mean(ck_dif_sfs) + 1e-8),
        linestyle="dashed",
        color="black",
    )
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel("diffusion sfs")
    axs.set_ylabel("simulated sfs (implm)")
    plt.savefig(f"{args.outpath}/diffusion-vs-model.png")


# --- loglikelihood over parameter grid from moments ---
grid_size = args.grid_size
breaks_nu = np.linspace(prior_low[0], prior_high[0], grid_size + 1)
breaks_T = np.linspace(prior_low[1], prior_high[1], grid_size + 1)
grid = np.array([
    (nu, T) for nu in (breaks_nu[1:] + breaks_nu[:-1]) / 2
    for T in (breaks_T[1:] + breaks_T[:-1]) / 2
])
obs_sfs = empirical_sfs(ts)
grid_loglik = np.array(ray.get([loglikelihood_parallel.remote(x, obs_sfs) for x in grid]))
grid_loglik -= np.max(grid_loglik)
mle, on_boundary = optimize(grid[np.argmax(grid_loglik)], obs_sfs, prior_low, prior_high) # refine mle
if on_boundary: print("MLE is on boundary of prior, GIM / FIM will be wrong")
hessian = -nd.Hessian(
    lambda x: loglikelihood(x, obs_sfs), 
    step=1e-4, richardson_terms=4,
)(mle)


# --- parametric bootstrapping ---
bootstrap_sfs = ray.get([
    simulate_parallel.remote(seed, mle) for seed 
    in rng.integers(2 ** 32 - 1, size=args.posterior_samples)
]) 
if not args.skip_bootstrap:
    bootstrap_samples = np.stack(
        ray.get([
            optimize_parallel.remote(sfs, mle) for sfs 
            in bootstrap_sfs
        ]) 
    )


# --- godambe information matrix ---
# see: https://www.stat.umn.edu/geyer/8112/notes/equations.pdf
@ray.remote
def hessian_parallel(sfs, pars):
    return -nd.Hessian(
        lambda x: loglikelihood(x, sfs), 
        step=1e-4, richardson_terms=4,
    )(pars)

@ray.remote
def score_parallel(sfs, pars):
    return nd.Gradient(
        lambda x: loglikelihood(x, sfs), 
        step=1e-4, richardson_terms=4,
    )(pars)

#boot_hessian = ray.get([hessian_parallel.remote(sfs, mle) for sfs in bootstrap_sfs])
#jac_score = np.mean(np.stack(boot_hessian), axis=0)
boot_score = ray.get([score_parallel.remote(sfs, mle) for sfs in bootstrap_sfs])
var_score = np.cov(np.stack(boot_score).T)
godambe = hessian @ np.linalg.inv(var_score) @ hessian
godambe_samples = rng.multivariate_normal(
    mean=mle, cov=np.linalg.inv(godambe), size=args.posterior_samples,
)
pickle.dump(
    {"MLE": mle, "GIM": np.linalg.inv(godambe), "FIM": np.linalg.inv(hessian)},
    open(f"{args.outpath}/moments-fit.pkl", "wb"),
)


# --- ABC inference using summary statistics ---
print("Running ABC-SMC inference...")
abc_samples, abc_trace = abc_utils.run_abc_smc(
    obs_ts=ts,
    simulator_config=simulator_config,
    prior_bounds=np.column_stack([prior_low, prior_high]),
    n_samples=args.posterior_samples,
    seed=args.seed,
    num_cores=args.num_cpus,
)


# DEBUG: check against the moments implementation
#from moments.Godambe import GIM_uncert
#from moments.Godambe import FIM_uncert
#hessian_ck = FIM_uncert(
#    func_ex=lambda x, ns: diffusion_sfs(x), 
#    p0=mle, 
#    data=moments.Spectrum(obs_sfs), 
#    eps=1e-4,
#    multinom=False, 
#)
#godambe_ck = GIM_uncert(
#    func_ex=lambda x, ns: diffusion_sfs(x), 
#    all_boot=bootstrap_sfs,
#    p0=mle, 
#    data=moments.Spectrum(obs_sfs), 
#    eps=1e-4,
#    multinom=False, 
#    return_GIM=True,
#)
#print(np.sqrt(np.diag(np.linalg.inv(hessian))))
#print(hessian_ck)
#print(np.sqrt(np.diag(np.linalg.inv(godambe))))
#print(godambe_ck[0])

    
# --- plot log-likelihood/log-probability surfaces ---
text_offset = np.array([-(prior_high[0] - prior_low[0]) * 0.01, 0])
plt.clf()
rows = 1
cols = len(args.configfile) + 2  # +1 for moments, +1 for ABC
fig, axs = plt.subplots(
    rows, cols,
    figsize=(cols * 5, rows * 4),
    constrained_layout=True,
    squeeze=False,
)

img = axs[0, 0].pcolormesh(
    breaks_nu, breaks_T,
    grid_loglik.reshape(grid_size, grid_size).T, 
    cmap="terrain", 
    vmin=args.vmin, 
)
axs[0, 0].plot(*args.params, "o", color="red", markersize=4)
axs[0, 0].text(
    *(args.params + text_offset), s="True", 
    color="red", ha="right", va="center", size=10,
)
axs[0, 0].plot(*mle, "o", color="green", markersize=4)
axs[0, 0].text(
    *(mle + text_offset), s="MLE", 
    color="green", ha="right", va="center", size=10,
)
axs[0, 0].set_title("moments")
plt.colorbar(img, ax=axs[0, 0], label="log likelihood")

# data used for contour plot, further down
posterior_surface = []
posterior_samples = []
posterior_labels = []
posterior_surface.append(grid_loglik)
if not args.skip_bootstrap:
    posterior_samples.append(bootstrap_samples)
    posterior_labels.append("MomentsBootstrap")
else:
    posterior_samples.append(godambe_samples)
    posterior_labels.append("MomentsGodambe")

# Add ABC to posterior data
posterior_samples.append(abc_samples)
posterior_labels.append("ABC-SMC")
abc_surface = abc_utils.estimate_abc_surface(abc_samples, breaks_nu, breaks_T)
posterior_surface.append(abc_surface)

# Plot ABC panel
abc_mean = abc_samples.mean(axis=0)
img = axs[0, 1].pcolormesh(
    breaks_nu, breaks_T,
    abc_surface.reshape(grid_size, grid_size).T,
    cmap="terrain",
)
axs[0, 1].plot(*args.params, "o", color="red", markersize=4)
axs[0, 1].text(
    *(args.params + text_offset), s="True",
    color="red", ha="right", va="center", size=10,
)
axs[0, 1].plot(*abc_mean, "o", color="green", markersize=4)
axs[0, 1].text(
    *(abc_mean + text_offset), s=r"$\mathbb{E}[\theta]$",
    color="green", ha="right", va="center", size=10,
)
axs[0, 1].set_title("ABC-SMC")
plt.colorbar(img, ax=axs[0, 1], label="log posterior")

# plot the log posteriors
for i, configfile in enumerate(args.configfile):
    config = yaml.safe_load(open(configfile))

    # set up processor, neural networks
    project_dir = config["project_dir"]
    random_seed = int(config["random_seed"])
    n_train = int(config["n_train"])
    train_separately = bool(config["train_embedding_net_separately"])
    simulator_config = config["simulator"]
    simulator_name = simulator_config["class_name"]
    processor_config = config["processor"]
    processor_name = processor_config["class_name"]
    embedding_config = config["embedding_network"]
    embedding_name = embedding_config["class_name"]

    processor = getattr(ts_processors, processor_name)(processor_config)
    
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
            
    embedding_net = torch.load(
        f"{path}embedding_network",
        weights_only=False,
    ).to(device)
    normalizing_flow = torch.load(
        f"{path}normalizing_flow",
        weights_only=False,
    ).to(device)
    
    features = Tensor(processor(ts)).unsqueeze(0).to(device)
    embedding = embedding_net(features).to(device)
    posterior = DirectPosterior(
        posterior_estimator=normalizing_flow,
        prior=BoxUniform(
            Tensor(prior_low).to(device), 
            Tensor(prior_high).to(device),
        ),
        device=device,
    )

    # stash samples for contours
    torch.manual_seed(args.seed)
    posterior_samples.append(
        posterior.sample(
            [args.posterior_samples],
            x=embedding,
            show_progress_bars=False,
        ).cpu().numpy()
    )
    posterior_labels.append(f"{embedding_name}")

    # normalizing flow logprob across grid
    grid_logprob = posterior.log_prob(
        Tensor(grid).to(device), 
        x=embedding,
    ).detach().cpu().numpy()
    grid_logprob -= np.max(grid_logprob)
    posterior_surface.append(grid_logprob)
    mean = posterior_samples[-1].mean(axis=0)

    img = axs[0, i + 2].pcolormesh(
        breaks_nu, breaks_T,
        grid_logprob.reshape(grid_size, grid_size).T,
        cmap="terrain",
    )
    axs[0, i + 2].plot(*args.params, "o", color="red", markersize=4)
    axs[0, i + 2].text(
        *(args.params + text_offset), s="True",
        color="red", va="center", ha="right", size=10,
    )
    axs[0, i + 2].plot(*mean, "o", color="green", markersize=4)
    axs[0, i + 2].text(
        *(mean + text_offset), s=r"$\mathbb{E}[\theta]$",
        color="green", va="center", ha="right", size=10,
    )
    axs[0, i + 2].set_title(f"{embedding_name}")
    plt.colorbar(img, ax=axs[0, i + 2], label="log posterior")
fig.supxlabel(r"Bottleneck severity ($\nu$)")
fig.supylabel(r"Time of bottleneck ($T$)")
plt.savefig(f"{args.outpath}/loglik-surface.png")


# save bootstrap/posterior samples
pickle.dump(
    np.array(args.params),
    open(f"{args.outpath}/generative-params.pkl", "wb"),
)
pickle.dump(
    {k: v for k, v in zip(posterior_labels, posterior_samples)},
    open(f"{args.outpath}/posterior-samples.pkl", "wb"),
)
pickle.dump(
    {
        "breaks_nu": breaks_nu, 
        "breaks_T": breaks_T,
        "surfaces": {
            k: v.reshape(grid_size, grid_size) 
            for k, v in zip(posterior_labels, posterior_surface)
        },
    },
    open(f"{args.outpath}/posterior-surfaces.pkl", "wb"),
)


# --- plot parametric bootstrap and godambe ---
plt.clf()
rows = 1
cols = len(posterior_samples)
fig, axs = plt.subplots(
    rows, cols, 
    figsize=(cols * 4, rows * 4), 
    constrained_layout=True, 
    squeeze=False,
)

levels = [0.5, 0.75, 0.95]
for i, (samples, label) in enumerate(zip(posterior_samples, posterior_labels)):
    post_mean = np.nanmean(samples, axis=0)
    sns.kdeplot(
        x=samples.T[0],
        y=samples.T[1],
        levels=levels,
        color="lightgray",
        ax=axs[0, i],
    )
    axs[0, i].plot(*post_mean, "o", markersize=4, color="black")
    axs[0, i].text(
        *(post_mean + text_offset), s=r"$\mathbb{E}[\theta]$", 
        color="black", va="center", ha="right", size=10,
    )
    axs[0, i].plot(*args.params, "o", markersize=4, color="red")
    axs[0, i].text(
        *(args.params + text_offset), s="True", 
        color="red", va="center", ha="right", size=10,
    )
    axs[0, i].set_xlim(prior_low[0], prior_high[0])
    axs[0, i].set_ylim(prior_low[1], prior_high[1])
    axs[0, i].set_title(f"{label}")
fig.supxlabel(r"Bottleneck severity ($\nu$)")
fig.supylabel(r"Time of bottleneck ($T$)")
plt.savefig(f"{args.outpath}/posterior-contours.png")
