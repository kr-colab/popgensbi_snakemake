set -e 

PWD=`realpath "$0"`
PWD=`dirname $PWD`

# train NPE models
#for CONFIG in $PWD/npe-config/*.yaml; do
#  snakemake --configfile $CONFIG --jobs 50 \
#    --snakefile $PWD/../../workflow/training_workflow.smk
#done

# predict in 1000g windows
for CONFIG in $PWD/npe-config/*cnn.yaml; do
  snakemake --configfile $CONFIG --jobs 50 \
    --snakefile $PWD/../../workflow/prediction_workflow.smk
done


