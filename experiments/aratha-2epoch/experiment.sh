set -e 

PWD=`realpath "$0"`
PWD=`dirname $PWD`

# train NPE models
for CONFIG in $PWD/npe-config/*.yaml; do
  snakemake --configfile $CONFIG --jobs 50 \
    --snakefile $PWD/../../workflow/training_workflow.smk
done

# coverage experiment and figures
snakemake --configfile $PWD/experiment-config.yaml \
  --snakefile $PWD/experiment.smk

