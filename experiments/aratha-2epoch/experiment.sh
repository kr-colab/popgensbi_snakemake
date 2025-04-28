PWD=`realpath "$0"`
PWD=`dirname $PWD`

# train NPE models
for CONFIG in $PWD/npe-config/*.yaml; do
  snakemake --configfile $CONFIG \
    --snakefile $PWD/../../workflow/training-workflow.smk
done

# coverage experiment and figures
snakemake --configfile $PWD/experiment-config.yaml \
  --snakefile $PWD/experiment.smk

