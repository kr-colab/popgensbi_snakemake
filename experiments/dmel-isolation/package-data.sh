set -e

VCF_DIR="/sietch_colab/data_share/drosophila_melanogaster/vcf_output/"
OUT_DIR="/sietch_colab/data_share/popgen_npe/dromel-isolation/data"

# Parse subset of DroMel samples from VCF, remove nonsegregating
# and multiallelic sites
mkdir -p $OUT_DIR
rm -f $OUT_DIR/sample_list.txt
for VCF in $VCF_DIR/*.vcf; do

CHROM=`echo $VCF | sed -rn 's/.+(Chr[A-Z|0-9]+).+/\1/p'`

if [ ! -f $OUT_DIR/sample_list.txt ]; then
  bcftools query -l $VCF | grep -E "(CO)|(FR)" > $OUT_DIR/sample_list.txt
fi

# subset to two populations
bcftools view -a -S $OUT_DIR/sample_list.txt $VCF >subset_unfilt.vcf

# rename chromosome
echo "1 $CHROM" >tmp.txt
bcftools annotate --rename-chr tmp.txt subset_unfilt.vcf -o tmp.vcf
mv tmp.vcf subset_unfilt.vcf
rm tmp.txt

# remove sites that are nonsegregating, or where the alternate state is just "N" (*)
bcftools view -i 'ALT != "*" && ALT != "."' subset_unfilt.vcf >subset_multi.vcf

# remove multiallelic sites and biallelic sites with missing data
bcftools view --max-alleles 2 subset_multi.vcf >subset_biall.vcf

# sanity check: alternate and ref allele counts
for f in subset_multi.vcf subset_biall.vcf; do
awk '!/^#/ {print $5}' $f | sort | uniq -c >$f.alt_counts
awk '!/^#/ {print $4}' $f | sort | uniq -c >$f.ref_counts
done

# the adjusted mutation rate is mu * biallelic / (biallelic + multiallelic)
# so save out the total number of segregating sites and biallelic/nonmissing sites
# per chromosome
TOTAL=`awk '{sm += $1}END{print sm}' subset_multi.vcf.alt_counts`
RETAINED=`awk '{sm += $1}END{print sm}' subset_biall.vcf.alt_counts`

# compress and index
bgzip -c subset_biall.vcf >$OUT_DIR/$CHROM.vcf.gz
tabix $OUT_DIR/$CHROM.vcf.gz
rm subset_*vcf*

echo $CHROM $TOTAL $RETAINED 

done >$OUT_DIR/proportion_retained.txt

awk '{tot+=$2; keep+=$3}END{print keep/tot}' $OUT_DIR/proportion_retained.txt >$OUT_DIR/mu_adjust.txt


# Make synthetic ancestral chromosome with reference allele as ancestral state
FA_DIR="/sietch_colab/data_share/drosophila_melanogaster/dpgp_ancestor/"
for VCF in $OUT_DIR/*.vcf.gz; do

CHROM=`echo $VCF | sed -rn 's/.+(Chr[A-Z|0-9]+).+/\1/p'`
LENGTH=`zcat $VCF | awk '/^##contig=/ {print; exit}' | sed -rn 's/.+length=([0-9]+).+/\1/p'`

printf '
from sys import stdout
import allel
vcf = "%s"
chrom = "%s"
length = int(%s)
vcf = allel.read_vcf(vcf)
fa = ["A"] * length
for p, x in zip(vcf["variants/POS"], vcf["variants/REF"]): fa[p - 1] = x
stdout.write(f">{chrom}\\n")
stdout.write("".join(fa) + "\\n")
' $VCF $CHROM $LENGTH | python >$OUT_DIR/$CHROM.fa
bgzip -f $OUT_DIR/$CHROM.fa


done


# yaml with population per sample
while read SAMPLE; do
  echo "$SAMPLE: ${SAMPLE:0:2}"
done<$OUT_DIR/sample_list.txt >$OUT_DIR/population_map.yaml


# 300kb windows
for VCF in $OUT_DIR/Chr*.vcf.gz; do
CHROM=`echo $VCF | sed -rn 's/.+(Chr[A-Z|0-9]+).+/\1/p'`

printf '
import allel
import numpy as np
chrom = "%s"
vcf = "%s"
min_segsites = 1000
vcf = allel.read_vcf(vcf)
pos = vcf["variants/POS"]
frq = vcf["calldata/GT"][:, :, 0].sum(axis=1)
pos = pos[np.logical_and(frq > 1, frq < len(vcf["samples"]) - 1)]
grid = np.arange(pos.min() - 1, pos.max() + 1, 3e5)
segsites = np.bincount(np.digitize(pos - 1, grid))
for a, b, s in zip(grid[:-1], grid[1:], segsites):
    if s >= min_segsites:
      print(f"{chrom}\t{int(a)}\t{int(b)}")
' $CHROM $VCF | python >$OUT_DIR/$CHROM.windows.bed

done


## combine TODO
#bcftools concat -o $OUT_DIR/all.vcf.gz $OUT_DIR/Chr*.vcf.gz 
#cat $OUT_DIR/Chr*.windows.bed >$OUT_DIR/all.windows.bed
