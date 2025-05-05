"""
Compare NPE against one another and against a baseline that fits the site
frequency spectrum via "moments" 
"""

import numpy as np
import glob

configfile: "experiment-config.yaml"

GPU_RESOURCES = config["gpu_resources"]
PROJECT_DIR = config["project_dir"]
CONFIG_DIR = config["config_dir"]
NUM_WORKERS = int(config["num_workers"])
RANDOM_SEED = int(config["random_seed"])
NUM_SIMULATIONS = int(config["num_simulations"])
SIZE_BOUNDS = np.array(config["size_bounds"])
TIME_BOUNDS = np.array(config["time_bounds"])

SIMULATION_DIR = os.path.join(PROJECT_DIR, "sims")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")


localrules:
    shard_simulation_grid,
    calculate_coverage,


scattergather:
    split = NUM_WORKERS


rule all:
    input:
        mean_scatterplot = os.path.join(OUTPUT_DIR, "mean-scatterplot.png"),
        mode_scatterplot = os.path.join(OUTPUT_DIR, "mode-scatterplot.png"),
        mean_boxplot = os.path.join(OUTPUT_DIR, "mean-boxplot.png"),
        mode_boxplot = os.path.join(OUTPUT_DIR, "mode-boxplot.png"),
        coverage_plot = os.path.join(OUTPUT_DIR, "posterior-coverage.png"),
        concentr_plot = os.path.join(OUTPUT_DIR, "posterior-concentration.png"),
        posterior_plot = os.path.join(OUTPUT_DIR, "posterior-surfaces.png"), 


rule shard_simulation_grid:
    output:
        params = scatter.split(os.path.join(SIMULATION_DIR, "{scatteritem}.txt"))
    run:
        rng = np.random.default_rng(RANDOM_SEED)
        pars = np.stack([
            rng.uniform(*SIZE_BOUNDS, size=NUM_SIMULATIONS),
            rng.uniform(*TIME_BOUNDS, size=NUM_SIMULATIONS),
        ], axis=-1)
        seed = rng.integers(2 ** 32 - 1, size=NUM_SIMULATIONS)
        worker = np.repeat(
            np.arange(NUM_WORKERS), 
            np.ceil(NUM_SIMULATIONS / NUM_WORKERS)
        )[:NUM_SIMULATIONS]
        for i in range(NUM_WORKERS):
            handle = open(output.params[i], "w")
            for s, x in zip(seed[worker == i], pars[worker == i]):
                handle.write(f"{s}\t{x[0]}\t{x[1]}\n")
            handle.close()

        
rule compare_posteriors:
    input:
        params = os.path.join(SIMULATION_DIR, "{scatteritem}.txt"),
        config = glob.glob(os.path.join(CONFIG_DIR, "*.yaml")),
    output:
        done = touch(os.path.join(SIMULATION_DIR, "{scatteritem}.done")),
    threads: 4
    resources:
        **GPU_RESOURCES,
    shell:
        """
        while read SEED NU T; do
            OUTDIR="`dirname {output.done}`/$SEED"
            python compare-posteriors.py \
              --skip-bootstrap \
              --skip-sanity-checks \
              --num-cpus {threads} \
              --params $NU $T \
              --seed $SEED \
              --configfile {input.config} \
              --outpath $OUTDIR
        done < {input.params}
        """


rule calculate_coverage:
    """
    Calculate CI coverage in various parts of parameter space. Output array
    has dimensions that are (popsize_bin, time_bin, coverage_quantile)
    """
    input:
        done = gather.split(os.path.join(SIMULATION_DIR, "{scatteritem}.done")),
    output:
        coverage_plot = rules.all.input.coverage_plot,
        concentr_plot = rules.all.input.concentr_plot,
    run:
        import pickle
        grid_size = 10
        size_grid = np.linspace(*SIZE_BOUNDS, grid_size + 1)
        time_grid = np.linspace(*TIME_BOUNDS, grid_size + 1)
        alpha_grid = np.linspace(0.0, 0.5, grid_size + 2)[1:-1]
        true_values = []
        coverage = None
        for file in input.done:
            base = os.path.dirname(file)
            for line in open(file.removesuffix(".done") + ".txt"):
                seed, *_ = line.split()
                sdir = f"{base}/{seed}"
                true = pickle.load(open(f"{sdir}/generative-params.pkl", "rb"))
                samp = pickle.load(open(f"{sdir}/posterior-samples.pkl", "rb"))
                if coverage is None: 
                    coverage = {
                        k: np.zeros((grid_size + 1, grid_size + 1, grid_size, 2))
                        for k in samp
                    }
                    concentr = {
                        k: np.zeros((grid_size + 1, grid_size + 1, grid_size, 2))
                        for k in samp
                    }
                    pointest = {k: list() for k in samp}
                    totals = {
                        k: np.zeros((grid_size + 1, grid_size + 1, grid_size, 2))
                        for k in samp
                    }
                nidx = np.digitize(true[0], size_grid) - 1
                tidx = np.digitize(true[1], time_grid) - 1
                for k, v in samp.items():  # get coverage and increment
                    x = v[np.all(~np.isnan(v), axis=1)]
                    x[:, 0] = np.clip(x[:, 0], SIZE_BOUNDS[0], SIZE_BOUNDS[1])
                    x[:, 1] = np.clip(x[:, 1], TIME_BOUNDS[0], TIME_BOUNDS[1])
                    lowr = np.quantile(x, alpha_grid, axis=0)
                    uppr = np.quantile(x, 1 - alpha_grid, axis=0)
                    covr = np.logical_and(
                        true[np.newaxis, :] > lowr,
                        true[np.newaxis, :] < uppr,
                    )
                    coverage[k][nidx, tidx] += covr
                    concentr[k][nidx, tidx] += (uppr - lowr)
                    totals[k][nidx, tidx] += 1
                    if "Moments" in k:
                        fit = pickle.load(open(f"{sdir}/moments-fit.pkl", "rb"))
                        pointest[k].append(fit["MLE"])
                    else:
                        pointest[k].append(x.mean(axis=0))
                true_values.append(true)
        for k in pointest: pointest[k] = np.array(pointest[k])
        true_values = np.array(true_values)
        prior_concentr = \
            np.quantile(true_values, 1 - alpha_grid, axis=0) - \
            np.quantile(true_values, alpha_grid, axis=0)
        global_coverage = {
            k: coverage[k].sum(axis=(0, 1)) / totals[k].sum(axis=(0, 1)) 
            for k in coverage
        }
        global_concentr = {
            k: concentr[k].sum(axis=(0, 1)) / totals[k].sum(axis=(0, 1)) 
            for k in concentr
        }
        import matplotlib.pyplot as plt
        colors = [plt.get_cmap("tab10")(x) for x in np.linspace(0, 1, 10)]
        labels = {
            "MomentsGodambe": "moments-sfs", "ExchangeableCNN": "npe-cnn", 
            "RNN": "npe-rnn", "SummaryStatisticsEmbedding": "npe-sfs",
        }
        # coverage
        rows = 1
        cols = 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), 
            constrained_layout=True, sharey=True, sharex=True,
        )
        for i, ax in enumerate(axs):
            ax.axline((0.5, 0.5), slope=1, linestyle="dashed", color="black")
            for name, color in zip(global_coverage, colors):
                ax.plot(
                    1 - 2 * alpha_grid, global_coverage[name][:, i], 
                    "-o", label=labels[name],
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
        axs[0].set_title(r"$\nu$ (AraTha_2epoch)")
        axs[1].set_title(r"$T$ (AraTha_2epoch)")
        fig.supxlabel("Expected coverage")
        fig.supylabel("Empirical coverage")
        plt.savefig(output.coverage_plot)
        # concentration
        plt.clf()
        rows = 1
        cols = 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), 
            constrained_layout=True, sharey=True, sharex=True,
        )
        for i, ax in enumerate(axs):
            for name, color in zip(global_concentr, colors):
                ax.plot(
                    1 - 2 * alpha_grid, 
                    global_concentr[name][:, i] / prior_concentr[:, i], 
                    "-o", label=labels[name],
                )
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.legend()
        axs[0].set_title(r"$\nu$ (AraTha_2epoch)")
        axs[1].set_title(r"$T$ (AraTha_2epoch)")
        fig.supxlabel("Expected coverage")
        fig.supylabel("Posterior/prior interval width")
        plt.savefig(output.concentr_plot)


rule calculate_mse:
    """
    Calculate MSE of parameter estimates vs truth
    """
    input:
        done = gather.split(os.path.join(SIMULATION_DIR, "{scatteritem}.done")),
    output:
        mode_scatterplot = rules.all.input.mode_scatterplot,
        mean_scatterplot = rules.all.input.mean_scatterplot,
        mode_boxplot = rules.all.input.mode_boxplot,
        mean_boxplot = rules.all.input.mean_boxplot,
    run:
        import pickle
        true_values = []
        mode_estimates = None
        for file in input.done:
            base = os.path.dirname(file)
            for line in open(file.removesuffix(".done") + ".txt"):
                seed, *_ = line.split()
                sdir = f"{base}/{seed}"
                true = pickle.load(open(f"{sdir}/generative-params.pkl", "rb"))
                surf = pickle.load(open(f"{sdir}/posterior-surfaces.pkl", "rb"))
                nu = (surf["breaks_nu"][:-1] + surf["breaks_nu"][1:]) / 2
                T = (surf["breaks_T"][:-1] + surf["breaks_T"][1:]) / 2
                surfaces = surf["surfaces"]
                if mode_estimates is None: 
                    mode_estimates = {k: [] for k in surfaces}
                    mean_estimates = {k: [] for k in surfaces}
                for k, v in surfaces.items():  # get coverage and increment
                    z = np.exp(v) / np.sum(np.exp(v))
                    i, j = np.unravel_index(v.argmax(), v.shape)
                    mode = np.array([nu[i], T[j]])
                    mean = np.array([np.sum(nu[:, None] * z), np.sum(T[None, :] * z)])
                    mode_estimates[k].append(mode)
                    mean_estimates[k].append(mean)
                true_values.append(true)
        for k in mode_estimates: mode_estimates[k] = np.array(mode_estimates[k])
        for k in mean_estimates: mean_estimates[k] = np.array(mean_estimates[k])
        true_values = np.array(true_values)
        # mode estimates
        import matplotlib.pyplot as plt
        plt.clf()
        cols = len(mode_estimates)
        rows = 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), 
            constrained_layout=True,
        )
        panels = ["MomentsGodambe", "ExchangeableCNN", "RNN", "SummaryStatisticsEmbedding"]
        labels = ["moments-SFS", "npe-CNN", "npe-RNN", "npe-SFS"]
        for i, k in enumerate(panels):
            for j in range(2):
                mse = np.mean((true_values[:, j] - mode_estimates[k][:, j]) ** 2)
                axs[j, i].plot(true_values[:, j], mode_estimates[k][:, j], "o", markersize=2)
                axs[j, i].text(
                    0.01, 0.99, f"mse: {mse:.3f}", 
                    ha="left", va="top", transform=axs[j, i].transAxes,
                )
                axs[j, i].axline((0, 0), slope=1, linestyle="dashed", color="black")
                axs[j, i].set_title(labels[i])
            axs[0, i].set_xlim(SIZE_BOUNDS[0], SIZE_BOUNDS[1])
            axs[0, i].set_ylim(SIZE_BOUNDS[0], SIZE_BOUNDS[1])
            axs[1, i].set_xlim(TIME_BOUNDS[0], TIME_BOUNDS[1])
            axs[1, i].set_ylim(TIME_BOUNDS[0], TIME_BOUNDS[1])
            axs[0, i].set_ylabel(r"$\nu$ (posterior mode)")
            axs[1, i].set_ylabel(r"$T$ (posterior mode)")
            axs[0, i].set_xlabel(r"$\nu$ (truth)")
            axs[1, i].set_xlabel(r"$T$ (truth)")
        plt.savefig(output.mode_scatterplot)
        # mean estimates
        import matplotlib.pyplot as plt
        plt.clf()
        cols = len(mean_estimates)
        rows = 2
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), 
            constrained_layout=True,
        )
        prior_mean = np.array([np.mean(SIZE_BOUNDS), np.mean(TIME_BOUNDS)])
        panels = ["MomentsGodambe", "ExchangeableCNN", "RNN", "SummaryStatisticsEmbedding"]
        labels = ["moments-SFS", "npe-CNN", "npe-RNN", "npe-SFS"]
        for i, k in enumerate(panels):
            for j in range(2):
                mse = np.mean((true_values[:, j] - mean_estimates[k][:, j]) ** 2)
                baseline = np.mean((true_values[:, j] - prior_mean[j]) ** 2)
                axs[j, i].plot(true_values[:, j], mean_estimates[k][:, j], "o", markersize=2)
                axs[j, i].text(
                    0.01, 0.99, f"mse: {mse:.3f}, prior-mse: {baseline:.3f}", 
                    ha="left", va="top", transform=axs[j, i].transAxes,
                )
                axs[j, i].axline((0, 0), slope=1, linestyle="dashed", color="black")
                axs[j, i].set_title(labels[i])
            axs[0, i].set_xlim(SIZE_BOUNDS[0], SIZE_BOUNDS[1])
            axs[0, i].set_ylim(SIZE_BOUNDS[0], SIZE_BOUNDS[1])
            axs[1, i].set_xlim(TIME_BOUNDS[0], TIME_BOUNDS[1])
            axs[1, i].set_ylim(TIME_BOUNDS[0], TIME_BOUNDS[1])
            axs[0, i].set_ylabel(r"$\nu$ (posterior mean)")
            axs[1, i].set_ylabel(r"$T$ (posterior mean)")
            axs[0, i].set_xlabel(r"$\nu$ (truth)")
            axs[1, i].set_xlabel(r"$T$ (truth)")
        plt.savefig(output.mean_scatterplot)
        # boxplots (posterior mean)
        import matplotlib.pyplot as plt
        plt.clf()
        cols = 2
        rows = 1
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), 
            constrained_layout=True, sharey=True,
        )
        panels = ["MomentsGodambe", "ExchangeableCNN", "RNN", "SummaryStatisticsEmbedding"]
        labels = ["moments-SFS", "npe-CNN", "npe-RNN", "npe-SFS"]
        errs_nu = []
        errs_T = []
        for i, k in enumerate(panels):
            errs_nu.append(np.sqrt((true_values[:, 0] - mean_estimates[k][:, 0]) ** 2))
            errs_T.append(np.sqrt((true_values[:, 1] - mean_estimates[k][:, 1]) ** 2))
        axs[0].boxplot(errs_nu, tick_labels=labels, showfliers=False)
        axs[0].axhline(y=prior_mean[0], linestyle="dashed", color="black")
        axs[0].set_title(r"Bottleneck severity ($\nu$)")
        axs[1].boxplot(errs_T, tick_labels=labels, showfliers=False)
        axs[1].axhline(y=prior_mean[1], linestyle="dashed", color="black")
        axs[1].set_title(r"Bottleneck timing ($T$)")
        fig.supylabel("|posterior mean - truth|")
        plt.savefig(output.mean_boxplot)
        # boxplots (posterior mode)
        import matplotlib.pyplot as plt
        plt.clf()
        cols = 2
        rows = 1
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), 
            constrained_layout=True, sharey=True,
        )
        panels = ["MomentsGodambe", "ExchangeableCNN", "RNN", "SummaryStatisticsEmbedding"]
        labels = ["moments-SFS", "npe-CNN", "npe-RNN", "npe-SFS"]
        errs_nu = []
        errs_T = []
        for i, k in enumerate(panels):
            errs_nu.append(np.sqrt((true_values[:, 0] - mode_estimates[k][:, 0]) ** 2))
            errs_T.append(np.sqrt((true_values[:, 1] - mode_estimates[k][:, 1]) ** 2))
        axs[0].boxplot(errs_nu, tick_labels=labels, showfliers=False)
        axs[0].axhline(y=prior_mean[0], linestyle="dashed", color="black")
        axs[0].set_title(r"Bottleneck severity ($\nu$)")
        axs[1].boxplot(errs_T, tick_labels=labels, showfliers=False)
        axs[1].axhline(y=prior_mean[1], linestyle="dashed", color="black")
        axs[1].set_title(r"Bottleneck timing ($T$)")
        fig.supylabel("|posterior mode - truth|")
        plt.savefig(output.mode_boxplot)
            

rule posterior_surfaces_data:
    input:
        config = glob.glob(os.path.join(CONFIG_DIR, "*.yaml")),
    output:
        surf = touch(os.path.join(OUTPUT_DIR, "posterior-surfaces.png")),
    threads: 40
    resources:
        **GPU_RESOURCES,
    shell:
        """
        OUTDIR="`dirname {output.surf}`"
        INPATHS=""
        while IFS=$" " read nu T; do
          OUTPATH="$OUTDIR/sim-"$nu"-"$T
          INPATHS+=" $OUTPATH"
          python compare-posteriors.py \
            --skip-bootstrap \
            --skip-sanity-checks \
            --num-cpus {threads} \
            --grid-size 50 \
            --params $nu $T \
            --seed 1024 \
            --configfile {input.config} \
            --outpath "$OUTPATH"
        done <<< $"0.1 0.8\n0.5 0.4\n0.9 0.05"
        python plot-posterior-surfaces.py \
          --inpaths $INPATHS \
          --outpath {output.surf}
        #rm -r $INPATHS
        """

