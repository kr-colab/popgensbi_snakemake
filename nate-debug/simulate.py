import numpy as np
import ray
import os
import glob
import torch
import argparse
import yaml

from collections import OrderedDict


# --- lib --- #

def draw_from_prior(seed, prior):
    rng = np.random.default_rng(seed)
    sample = OrderedDict()
    for p, v in prior.items():
        distr = getattr(rng, v["func"])
        sample[p] = distr(**v["args"])
    return sample


def demographic_model(N_A=10000, N_YRI=18800, N_CEU_initial=724, N_CEU_final=17640, m=4.65e-5, Tp=7260, T=2240):
    import demes
    import msprime
    graph = demes.Builder()
    graph.add_deme(
        "ancestral", 
        epochs=[dict(start_size=N_A, end_time=Tp + T)],
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
    graph.add_migration(demes=["CEU", "YRI"], rate=m)
    return msprime.Demography.from_demes(graph.resolve())


def simulate_msprime(samples, demography, mutation_rate=1.5e-8, recombination_rate=1.5e-8, sequence_length=1e7, seed=None):
    import msprime
    ts = msprime.sim_ancestry(
        samples,
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(
        ts, 
        rate=mutation_rate, 
        random_seed=seed,
    )
    return ts


def extract_tensors_cnn(ts, num_snps=500, maf_threshold=0.05, phased=False, ploidy=2, **kwargs):
    import dinf
    from dinf.misc import ts_individuals
    pop_names = [pop.metadata['name'] for pop in ts.populations()]
    sampled_pop_names = [pop for pop in pop_names if len(ts_individuals(ts, pop)) > 0]
    individuals = {name: ts_individuals(ts, name) for name in sampled_pop_names}
    num_individuals = {name: len(individuals[name]) for name in sampled_pop_names}
    extractor = dinf.feature_extractor.MultipleHaplotypeMatrices(
        num_individuals=num_individuals, 
        num_loci={pop: num_snps for pop in sampled_pop_names},
        ploidy={pop: ploidy for pop in sampled_pop_names},
        global_phased=phased,
        global_maf_thresh=maf_threshold,
    )
    feature_matrices = extractor.from_ts(ts, individuals=individuals)
    max_num_individuals = max([num_individuals[pop] for pop in sampled_pop_names])
    for pop in individuals.keys():  # pad missing individuals with -1
        feature_matrices[pop] = torch.from_numpy(feature_matrices[pop])
        num_individuals = feature_matrices[pop].shape[0]
        if num_individuals < max_num_individuals:
            feature_matrices[pop] = torch.nn.functional.pad(
                feature_matrices[pop], 
                (0, 0, 0, 0, 0, max_num_individuals - num_individuals), 
                "constant", 
                -1
            )
    output_mat = torch.stack([v for v in feature_matrices.values()]).permute(0, 3, 1, 2)
    # the dimensions here are, (population, genotypes | positions, individuals, snps)
    return output_mat


def extract_tensors_rnn(ts, phased=False, **kwargs):
    geno = ts.genotype_matrix()
    if not phased:
        diploid_map = np.zeros((ts.num_samples, ts.num_individuals))
        for i, ind in enumerate(ts.individuals()):
            diploid_map[ind.nodes, i] = 1.0
        geno = geno @ diploid_map
    pos = ts.sites_position / ts.sequence_length
    geno = np.concatenate([geno, pos.reshape(ts.num_sites, -1)], axis=-1)
    return torch.Tensor(geno).float()



# --- impl --- #

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    if config["snps"] == "all":
        extract_tensors = extract_tensors_rnn
    else:
        extract_tensors = extract_tensors_cnn

    sim_name = f"{config['snps']}"
    for n in config["samples"].values():
        sim_name += f"-{n}"

    data_path = os.path.join(config["path"], sim_name)
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")
    for path in (train_path, val_path, test_path):
        if not os.path.exists(path): 
            os.makedirs(path)
        else:
            if not args.no_clean:
                for f in glob.glob(os.path.join(path, "*.pt")): os.remove(f)
                for f in glob.glob(os.path.join(path, "*.npy")): os.remove(f)
    
    rng = np.random.default_rng(config["seed"])
    ray.init(num_cpus=config["cpus"])
    
    @ray.remote
    def worker(seed, path):
        prefix = os.path.join(path, f"{seed}")
        params = draw_from_prior(seed, config["prior"])
        trees = simulate_msprime(
            config["samples"], 
            demographic_model(**params), 
            sequence_length=config["sequence-length"], 
            seed=seed + 1,
        )
        features = extract_tensors(
            trees,
            num_snps=config["snps"],
        )
        torch.save(features, f"{prefix}.genotypes.pt")
        parameters = torch.from_numpy(np.array(list(params.values()))).float()
        torch.save(parameters, f"{prefix}.parameters.pt")

    train_seed_array = rng.integers(2 ** 32 - 1, size=config["reps"]["train"])
    job_list = [worker.remote(seed, train_path) for seed in train_seed_array]
    ray.get(job_list)
    train_seed_path = os.path.join(train_path, "seeds.npy")
    if os.path.exists(train_seed_path):
        train_seed_array = np.append(np.load(train_seed_path), train_seed_array)
    np.save(train_seed_path, train_seed_array)

    val_seed_array = rng.integers(2 ** 32 - 1, size=config["reps"]["val"])
    job_list = [worker.remote(seed, val_path) for seed in val_seed_array]
    ray.get(job_list)
    val_seed_path = os.path.join(val_path, "seeds.npy")
    if os.path.exists(val_seed_path):
        val_seed_array = np.append(np.load(val_seed_path), val_seed_array)
    np.save(val_seed_path, val_seed_array)

    test_seed_array = rng.integers(2 ** 32 - 1, size=config["reps"]["test"])
    job_list = [worker.remote(seed, test_path) for seed in test_seed_array]
    ray.get(job_list)
    test_seed_path = os.path.join(test_path, "seeds.npy")
    if os.path.exists(test_seed_path):
        test_seed_array = np.append(np.load(test_seed_path), test_seed_array)
    np.save(test_seed_path, test_seed_array)

    if "sbi" in config:
        sbi_train_path = os.path.join(data_path, "sbi", "train")
        sbi_val_path = os.path.join(data_path, "sbi", "val")
        sbi_test_path = os.path.join(data_path, "sbi", "test")
        for path in (sbi_train_path, sbi_val_path, sbi_test_path):
            if not os.path.exists(path): 
                os.makedirs(path)
            else:
                if not args.no_clean:
                    for f in glob.glob(os.path.join(path, "*.pt")): os.remove(f)
                    for f in glob.glob(os.path.join(path, "*.npy")): os.remove(f)

        train_seed_array = rng.integers(2 ** 32 - 1, size=config["sbi"]["train"])
        job_list = [worker.remote(seed, sbi_train_path) for seed in train_seed_array]
        ray.get(job_list)
        train_seed_path = os.path.join(sbi_train_path, "seeds.npy")
        if os.path.exists(train_seed_path):
            train_seed_array = np.append(np.load(train_seed_path), train_seed_array)
        np.save(train_seed_path, train_seed_array)

        val_seed_array = rng.integers(2 ** 32 - 1, size=config["sbi"]["val"])
        job_list = [worker.remote(seed, sbi_val_path) for seed in val_seed_array]
        ray.get(job_list)
        val_seed_path = os.path.join(sbi_val_path, "seeds.npy")
        if os.path.exists(val_seed_path):
            val_seed_array = np.append(np.load(val_seed_path), val_seed_array)
        np.save(val_seed_path, val_seed_array)

        test_seed_array = rng.integers(2 ** 32 - 1, size=config["sbi"]["test"])
        job_list = [worker.remote(seed, sbi_test_path) for seed in test_seed_array]
        ray.get(job_list)
        test_seed_path = os.path.join(sbi_test_path, "seeds.npy")
        if os.path.exists(test_seed_path):
            test_seed_array = np.append(np.load(test_seed_path), test_seed_array)
        np.save(test_seed_path, test_seed_array)
