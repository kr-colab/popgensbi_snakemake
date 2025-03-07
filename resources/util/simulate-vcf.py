import argparse
import numpy as np
import gzip
import sys
import os
import yaml
import torch
import allel
import pysam
import pysam.bcftools

# put popgensbi_snakemake scripts in the load path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "workflow", "scripts"))
import ts_simulators

parser = argparse.ArgumentParser(
    "Simulate data from one of the implemented models, using a random parameter "
    "draw from the prior. Outputs bgzipped VCFs, ancestral fasta, population metadata, "
    "and true parameter values. Assumes the simulator has a `sequence_length` "
    "attribute that can be overridden."
)
parser.add_argument("--configfile", help="Config file used for snakemake training workflow", type=str, required=True)
parser.add_argument("--outpath", help="Path to output directory", type=str, required=True)
parser.add_argument("--num-contig", help="Number of contigs in VCF (eg chromosomes)", type=int, default=3)
parser.add_argument("--sequence-length", help="Size of contigs (eg chromosome length)", type=float, default=10e6)
parser.add_argument("--seed", help="Random seed passed to simulator", type=int, default=1024)
parser.add_argument("--no-centromere", help="Don't mimic a centromere", action="store_true")
args = parser.parse_args()


# parse the config, instantiate the simulator
if not os.path.exists(args.outpath): os.makedirs(args.outpath)
config = yaml.safe_load(open(args.configfile))
simulator_config = config["simulator"]
simulator = getattr(ts_simulators, simulator_config["class_name"])(simulator_config)
assert hasattr(simulator, "sequence_length"), "Simulator lacks `sequence_length` attribute"
simulator.sequence_length = args.sequence_length


# condition the simulator on particular parameter values
torch.manual_seed(args.seed)
theta = simulator.prior.sample()
for i, p in enumerate(simulator.parameters):  # set prior to point mass
    setattr(simulator, p, [theta[i].item()] * 2)


# simulate data
fa_path = os.path.join(args.outpath, f"anc.fa")
vcf_path = os.path.join(args.outpath, f"snp.vcf")
tmp_path = []
fa = open(fa_path, "w")
for chrom in range(args.num_contig):
    ts, _ = simulator(args.seed)
    if not args.no_centromere:
        ts = ts.delete_intervals(np.array([[0.4, 0.6]]) * ts.sequence_length)
    indv_names = [f"IND{i}" for i in range(ts.num_individuals)]
    tmp_path.append(os.path.join(args.outpath, f"chr{chrom}.vcf"))
    ts.write_vcf(
        open(tmp_path[-1], "w"),
        contig_id=f"chr{chrom}",
        individual_names=indv_names,
    )
    # ancestral fasta
    vcf = allel.read_vcf(tmp_path[-1])
    pos = vcf["variants/POS"]
    anc = vcf["variants/REF"]
    fasta = np.full(int(ts.sequence_length), "N")
    fasta[pos - 1] = anc
    fa.write(f">chr{chrom}\n")
    fa.write("".join(fasta) + "\n")
fa.close()


# compress, index, etc
pysam.tabix_compress(fa_path, fa_path + ".gz", force=True)
pysam.bcftools.concat(*tmp_path, "-o", vcf_path, catch_stdout=False)
pysam.tabix_index(vcf_path, preset="vcf", force=True)
for vcf in tmp_path: os.remove(vcf)
os.remove(fa_path)


# parameters
pars = open(os.path.join(args.outpath, f"pars.txt"), "w")
for p, x in zip(simulator.parameters, theta):
    pars.write(f"{p}\t{x}\n")
pars.close()


# population metadata
pop_map = {i: p.metadata["name"] for i, p in enumerate(ts.populations())}
meta = open(os.path.join(args.outpath, f"popmap.yaml"), "w")
for ind in ts.individuals():
    i, p = ind.id, ind.population
    meta.write(f"{indv_names[i]}: \"{pop_map[p]}\"\n")
meta.close()
