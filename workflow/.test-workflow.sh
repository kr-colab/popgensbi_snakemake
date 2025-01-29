#'
#' Quickly test that workflow runs through with various options
#' 
set -e

rm -rf .test-workflow && mkdir .test-workflow

# --- base config --- #
CONFIG='
cpu_resources:
  runtime: "2h"
  mem_mb: 16000

gpu_resources:
  runtime: "4h"
  mem_mb: 50000
  gpus: 1
  slurm_partition: "kerngpu,gpu"
  slurm_extra: "--gres=gpu:1 --constraint=a100"

random_seed: 1
chunk_size: 100
n_train: 100
n_val: 50
n_test: 50

optimizer: "Adam"
batch_size: 10
learning_rate: 0.0005
max_num_epochs: 10
stop_after_epochs: 5
clip_max_norm: 5
'


# --- simulator options --- #
YRI_CEU_CONFIG='
simulator:
  class_name: "YRI_CEU"
  # OPTIONAL CLASS KWARGS:
  sequence_length: 100000
  mutation_rate: 1.5e-8
  recombination_rate: 1.5e-8
  samples:
    YRI: 10
    CEU: 10
'

# --- architecture specific options --- #
# NB: some of these will depend on simulation size

RNN_CONFIG='
packed_sequence: True

processor:
  class_name: "genotypes_and_distances"
  # OPTIONAL CLASS KWARGS:
  max_snps: 100
  phased: False
  min_freq: 0.00
  max_freq: 1.00

embedding_network:
  class_name: "RNN"
  # OPTIONAL CLASS KWARGS:
  input_size: 21
  output_size: 64
'

CNN_CONFIG='
packed_sequence: False

processor:
  class_name: "dinf_multiple_pops"
  # OPTIONAL CLASS KWARGS:
  n_snps: 100
  maf_thresh: 0.05

embedding_network:
  class_name: "ExchangeableCNN_IN"
  # OPTIONAL CLASS KWARGS:
  output_dim: 64
  input_rows: [10, 10]
  input_cols: [100, 100]
'

# --- training mode options --- #
E2E_CONFIG='
train_embedding_net_separately: False
'

SEP_CONFIG='
train_embedding_net_separately: True
'

# --- memory management options --- #
MEM_CONFIG='
use_cache: True
'

DISK_CONFIG='
use_cache: False
'

# --- test runs --- #
for SIM in YRI_CEU; do
  for ARCH in RNN CNN; do
    for STRATEGY in E2E SEP; do
      for CACHE in MEM DISK; do
        BASE=.test-workflow/$SIM-$ARCH-$STRATEGY-$CACHE
        YAML=${BASE}.yaml
        LOG=${BASE}.log
        echo "${CONFIG}" >$YAML
        echo "project_dir: \"${BASE}\"" >>$YAML
        TEMP=${SIM}_CONFIG
        eval "echo \"\${$TEMP}\" >>$YAML"
        TEMP=${ARCH}_CONFIG
        eval "echo \"\${$TEMP}\" >>$YAML"
        TEMP=${STRATEGY}_CONFIG
        eval "echo \"\${$TEMP}\" >>$YAML"
        TEMP=${CACHE}_CONFIG
        eval "echo \"\${$TEMP}\" >>$YAML"
        snakemake --jobs 2 --configfile $YAML &>$LOG
      done
    done
  done
done



