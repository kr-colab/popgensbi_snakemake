import dadi
import tskit
import numpy as np
import os

datadir = snakemake.params.datadir
num_simulations = snakemake.params.num_simulations

pts = 100
func = dadi.Demographics1D.two_epoch
func_ex = dadi.Numerics.make_extrap_log_func(func)
upper_bound = [1, 1.5]
lower_bound = [1e-2, 1e-2]
p0 = [1, 1]

def find_dadi_MLE(obs_sfs):
    '''
    Repeat optimization 100 times to get MLE
    Return MLE and composite-loglikelihood
    '''
    MLEs = []
    ll_list = []
    ns = obs_sfs.sample_sizes
    for i in range(100):
        p0_perturb = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)
        popt = dadi.Inference.optimize(p0_perturb, obs_sfs, func_ex, pts, lower_bound=lower_bound, upper_bound=upper_bound, verbose=len(p0_perturb), maxiter=3)
        MLEs.append(popt)

        model = func_ex(list(popt), ns, pts)
        ll_model = dadi.Inference.ll_multinom(model, obs_sfs)
        ll_list.append(ll_model)

    best_idx = np.argmax(ll_list)
    return MLEs[best_idx], ll_list[best_idx]



import dadi.Godambe

all_boot = []
n_boot = int(snakemake.params.n_rep_dadi)
for i in range(n_boot):
    fs_sample = np.load(os.path.join(datadir, f"test_{num_simulations}_sfs_rep_{i}.npy"))
    fs_sample = dadi.Spectrum(fs_sample)
    all_boot.append(fs_sample)

# load test ts and use its sfs to find MLE
ts_test = tskit.load(os.path.join(datadir, f"test_{num_simulations}.trees"))
sfs_test = ts_test.allele_frequency_spectrum(
            sample_sets = [ts_test.samples(population=i) for i in range(ts_test.num_populations) if len(ts_test.samples(population=i))>0], 
            windows = None, 
            mode = 'site', 
            span_normalise = False, 
            polarised = True)
sfs_test = sfs_test / sum(sfs_test.flatten())
sfs_test = dadi.Spectrum(sfs_test)
MLE, ll = find_dadi_MLE(sfs_test)
# Godambe analysis to get uncertainty. Use bigger epsilon because eps=0.01 seems to give a singular cov matrix
uncerts, GIM, H = dadi.Godambe.GIM_uncert(func_ex, (pts,), all_boot,
list(MLE), sfs_test, log=True, multinom=False, return_GIM=True, boot_theta_adjusts=[1 for i in range(n_boot)], eps=0.05)
np.save(os.path.join(datadir, f"test_MLE_{num_simulations}.npy"), np.array(MLE))
np.save(os.path.join(datadir, f"test_uncerts_{num_simulations}.npy"), np.array(uncerts))
np.save(os.path.join(datadir, f"test_GIM_{num_simulations}.npy"), np.array(GIM))