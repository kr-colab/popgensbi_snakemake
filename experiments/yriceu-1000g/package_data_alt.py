import tszip
import numpy as np
import os
import pysam
import pysam.bcftools

out_dir = "/sietch_colab/data_share/popgen_npe/ooa2T12-1000g/data"
in_dir = "/sietch_colab/data_share/hg1kg/tsinfer-trees/working"
anc_fa_dir = "/sietch_colab/data_share/hg38_anc/human_ancestor_bcgm/"

n_samp = 50  # number of diploids per pop
window_size = 1e7  # window size in bp
        
if not os.path.exists(out_dir): os.makedirs(out_dir)

anc_path = f"{out_dir}/fa"
vcf_path = f"{out_dir}/vcf"
bed_file = open(f"{out_dir}/bed", "w")
pop_file = open(f"{out_dir}/yml", "w")
anc_file = open(anc_path, "w")

tmp_path = []
for chrom_id in range(1, 23):
    chrom_id = f"chr{chrom_id}"
    for arm in ["p", "q"]:
        ts_path = f"{in_dir}/{chrom_id}{arm}.tsz"
        if not os.path.exists(ts_path): continue
        print(f"{ts_path}", flush=True)

        ts = tszip.load(ts_path)
        ceu = np.array([i.nodes for i in ts.individuals() if i.metadata["Population code"] == "CEU"])
        yri = np.array([i.nodes for i in ts.individuals() if i.metadata["Population code"] == "YRI"])
        sample_sets = [ceu[:n_samp].flatten(), yri[:n_samp].flatten()]
        
        ts = ts.simplify(samples=np.concatenate(sample_sets))
        left = ts.edges_left.min()
        right = ts.edges_right.max()
        windows = np.arange(left, right, window_size)
        
        # write bed of windows to use
        for a, b in zip(windows[:-1], windows[1:]):
            bed_file.write(f"{chrom_id}\t{int(a)}\t{int(b)}\n")
        
        # write population map
        individual_names = []
        for ind in ts.individuals():
            pop = ind.metadata["Population code"]
            pop = "AFR" if pop == "YRI" else "EUR"
            ids = ind.metadata["Sample name"]
            if len(tmp_path) == 0: pop_file.write(f"{ids}: {pop}\n")
            individual_names.append(ids)
        
        # write VCF
        tmp_path.append(f"{out_dir}/{chrom_id}{arm}.vcf")
        ts.write_vcf(
            open(tmp_path[-1], "w"), 
            contig_id=chrom_id, 
            individual_names=individual_names,
        )

    # concatenate ancestral fasta
    anc_file.write(open(f"{anc_fa_dir}/{chrom_id}.fa").read())


pop_file.close()
bed_file.close()
anc_file.close()

# compress/index ancestral fasta
pysam.tabix_compress(anc_path, anc_path + ".gz", force=True)
os.remove(anc_path)

# compress/index vcf
pysam.bcftools.concat(*tmp_path, "-o", vcf_path, catch_stdout=False)
pysam.tabix_index(vcf_path, preset="vcf", force=True, keep_original=False)
for vcf in tmp_path: os.remove(vcf)
