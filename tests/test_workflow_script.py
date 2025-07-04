import subprocess
import pytest
import os
import shutil
import tempfile
import re

# Base configuration
config = '''
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
'''

# Simulator configuration
yri_ce_config = '''
simulator:
  class_name: "YRI_CEU"
  sequence_length: 100000
  mutation_rate: 1.5e-8
  recombination_rate: 1.5e-8
  samples:
    YRI: 10
    CEU: 10
'''

# Architecture configurations
rnn_config = '''
packed_sequence: True

processor:
  class_name: "genotypes_and_distances"
  max_snps: 100
  phased: False
  min_freq: 0.00
  max_freq: 1.00

embedding_network:
  class_name: "RNN"
  input_size: 21
  output_size: 64
'''

cnn_config = '''
packed_sequence: False

processor:
  class_name: "cnn_extract"
  n_snps: 100
  maf_thresh: 0.05

embedding_network:
  class_name: "ExchangeableCNN_IN"
  output_dim: 64
  input_rows: [10, 10]
  input_cols: [100, 100]
'''

sum_stats_config = '''
packed_sequence: False

processor:
  class_name: "tskit_sfs"
  sample_sets: [YRI, CEU]
  windows: null
  mode: "site"
  span_normalise: True
  polarised: True
  
embedding_network:
  class_name: "SummaryStatisticsEmbedding"
'''

sum_stats_config_2 = '''
packed_sequence: False

processor:
  class_name: "tskit_windowed_sfs_plus_ld"
  window_size: 1000000
  span_normalise: False
  polarised: True
  
embedding_network:
  class_name: "SummaryStatisticsEmbedding"
'''

# Variable Population Size Simulator configuration
variable_popn_config = '''
simulator:
  class_name: "VariablePopulationSize"
  sequence_length: 1000000
  mutation_rate: 1.5e-8
  num_time_windows: 3
  max_time: 100000
  time_rate: 0.1
  samples:
    pop: 10
'''

# Variable Population Size Simulator configuration
dependent_popn_config = '''
simulator:
  class_name: "DependentVariablePopulationSize"
  sequence_length: 2000000
  mutation_rate: 1e-8
  num_time_windows: 21
  max_time: 130000
  time_rate: 0.06
  samples:
    pop: 25
'''

SPIDNA_config = '''
packed_sequence: False

processor:
  class_name: "SPIDNA_processor"
  n_snps: 2000
  maf_thresh: 0.05
  relative_position: True
  
embedding_network:
  class_name: "SPIDNA_embedding_network"
  output_dim: 64
  num_block: 6
  num_feature: 10
'''

# Training strategy configurations
e2e_config = '''
train_embedding_net_separately: False
'''

sep_config = '''
train_embedding_net_separately: True
'''

# Cache configurations
mem_config = '''
use_cache: True
'''

disk_config = '''
use_cache: False
'''

def setup_test_environment(test_name):
    """Set up the test environment by copying and modifying the Snakefile and scripts."""
    # Create test directories with unique name
    test_workflow_dir = f".test-workflow-{test_name}"
    if os.path.exists(test_workflow_dir):
        shutil.rmtree(test_workflow_dir)
    os.makedirs(test_workflow_dir)
    
    # Copy Snakefile to test directory and modify it
    test_snakefile = os.path.join(test_workflow_dir, "Snakefile")
    with open("workflow/Snakefile", 'r') as f:
        snakefile_content = f.read()
    
    # Remove the configfile line if present
    modified_content = '\n'.join(
        line for line in snakefile_content.split('\n')
        if not line.strip().startswith('configfile:')
    )
    
    with open(test_snakefile, 'w') as f:
        f.write(modified_content)

    # Copy scripts directory
    workflow_scripts_dir = "workflow/scripts"
    test_scripts_dir = os.path.join(test_workflow_dir, "scripts")
    if os.path.exists(workflow_scripts_dir):
        shutil.copytree(workflow_scripts_dir, test_scripts_dir)
    
    return test_workflow_dir

def cleanup_test_environment(test_name):
    """Clean up the test environment."""
    test_workflow_dir = f".test-workflow-{test_name}"
    if os.path.exists(test_workflow_dir):
        shutil.rmtree(test_workflow_dir)

def check_workflow_completion(log_file):
    """Check if the workflow completed successfully by examining the log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return False
            # Check second to last line for completion status
            completion_line = lines[-2].strip()
            # Match pattern "X of X steps (100%) done" where X can be any number
            completion_pattern = r'(\d+) of \1 steps \(100%\) done'
            return bool(re.search(completion_pattern, completion_line))
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return False

def _run_workflow_with_config(base_config, sim_config, arch_config, strategy_config, cache_config, test_workflow_dir):
    """Helper function to run a single workflow configuration."""
    base = f"{sim_config['class_name']}-{arch_config['name']}"
    yaml_file = f"{base}.yaml"
    log_file = f"{base}.log"
    full_base = os.path.join(test_workflow_dir, base)

    # Create the YAML configuration file
    yaml_path = os.path.join(test_workflow_dir, yaml_file)
    with open(yaml_path, 'w') as f:
        f.write(base_config)
        f.write(f'project_dir: "{full_base}"\n')
        f.write(sim_config['config'])
        f.write(arch_config['config'])
        f.write(strategy_config)
        f.write(cache_config)

    # First, try to unlock if there's a stale lock
    subprocess.run(
        ["snakemake", "--unlock", "--configfile", yaml_file, "--snakefile", "Snakefile"],
        capture_output=True,
        text=True,
        cwd=test_workflow_dir
    )

    # Then run the workflow
    result = subprocess.run(
        ["snakemake", "--jobs", "2", "--configfile", yaml_file, "--snakefile", "Snakefile"],
        capture_output=True,
        text=True,
        cwd=test_workflow_dir
    )

    # Write output to log file
    log_path = os.path.join(test_workflow_dir, log_file)
    with open(log_path, 'w') as f:
        f.write(result.stdout)
        f.write(result.stderr)

    # Check return code and completion
    assert result.returncode == 0, f"Snakemake workflow failed for {full_base}. Error: {result.stderr}"
    assert check_workflow_completion(log_path), f"Workflow did not complete successfully for {full_base}. Check {log_file} for details."

# Define test configurations
strategies = [
    ('E2E', e2e_config),
    ('SEP', sep_config)
]

cache_configs = [
    ('MEM', mem_config),
    ('DISK', disk_config)
]

# Create test parameter combinations
test_params = [
    (strategy_name, strategy_config, cache_name, cache_config)
    for strategy_name, strategy_config in strategies
    for cache_name, cache_config in cache_configs
]

@pytest.mark.parametrize("strategy_name,strategy_config,cache_name,cache_config", test_params)
def test_cnn_workflow(strategy_name, strategy_config, cache_name, cache_config):
    """Test the CNN workflow configuration."""
    test_name = f"cnn-{strategy_name}-{cache_name}"
    try:
        test_workflow_dir = setup_test_environment(test_name)
        _run_workflow_with_config(
            base_config=config,
            sim_config={
                'class_name': 'YRI_CEU',
                'config': yri_ce_config
            },
            arch_config={
                'name': f'CNN-{strategy_name}-{cache_name}',
                'config': cnn_config
            },
            strategy_config=strategy_config,
            cache_config=cache_config,
            test_workflow_dir=test_workflow_dir
        )
    finally:
        cleanup_test_environment(test_name)

@pytest.mark.parametrize("strategy_name,strategy_config,cache_name,cache_config", test_params)
def test_rnn_workflow(strategy_name, strategy_config, cache_name, cache_config):
    """Test the RNN workflow configuration."""
    test_name = f"rnn-{strategy_name}-{cache_name}"
    try:
        test_workflow_dir = setup_test_environment(test_name)
        _run_workflow_with_config(
            base_config=config,
            sim_config={
                'class_name': 'YRI_CEU',
                'config': yri_ce_config
            },
            arch_config={
                'name': f'RNN-{strategy_name}-{cache_name}',
                'config': rnn_config
            },
            strategy_config=strategy_config,
            cache_config=cache_config,
            test_workflow_dir=test_workflow_dir
        )
    finally:
        cleanup_test_environment(test_name)

@pytest.mark.parametrize("strategy_name,strategy_config,cache_name,cache_config", test_params)
def test_summary_stats_workflow(strategy_name, strategy_config, cache_name, cache_config):
    """Test the Summary Statistics workflow configuration."""
    test_name = f"sumstats-{strategy_name}-{cache_name}"
    try:
        test_workflow_dir = setup_test_environment(test_name)
        _run_workflow_with_config(
            base_config=config,
            sim_config={
                'class_name': 'YRI_CEU',
                'config': yri_ce_config
            },
            arch_config={
                'name': f'SUMMARY_STATS-{strategy_name}-{cache_name}',
                'config': sum_stats_config
            },
            strategy_config=strategy_config,
            cache_config=cache_config,
            test_workflow_dir=test_workflow_dir
        )
    finally:
        cleanup_test_environment(test_name) 

@pytest.mark.parametrize("strategy_name,strategy_config,cache_name,cache_config", test_params)
def test_spidna_workflow(strategy_name, strategy_config, cache_name, cache_config):
    """Test the SPIDNA workflow configuration."""
    test_name = f"spidna-{strategy_name}-{cache_name}"
    try:
        test_workflow_dir = setup_test_environment(test_name)
        _run_workflow_with_config(
            base_config=config,
            sim_config={
                'class_name': 'VariablePopulationSize',
                'config': variable_popn_config
            },
            arch_config={
                'name': f'SPIDNA-{strategy_name}-{cache_name}',
                'config': SPIDNA_config
            },
            strategy_config=strategy_config,
            cache_config=cache_config,
            test_workflow_dir=test_workflow_dir
        )
    finally:
        cleanup_test_environment(test_name)
        
        
@pytest.mark.parametrize("strategy_name,strategy_config,cache_name,cache_config", test_params)
def test_summary_stats2_workflow(strategy_name, strategy_config, cache_name, cache_config):
    """Test the Summary Statistics workflow configuration."""
    test_name = f"sumstats-{strategy_name}-{cache_name}"
    try:
        test_workflow_dir = setup_test_environment(test_name)
        _run_workflow_with_config(
            base_config=config,
            sim_config={
                'class_name': 'VariablePopulationSize',
                'config': variable_popn_config
            },
            arch_config={
                'name': f'SUMMARY_STATS-{strategy_name}-{cache_name}',
                'config': sum_stats_config_2
            },
            strategy_config=strategy_config,
            cache_config=cache_config,
            test_workflow_dir=test_workflow_dir
        )
    finally:
        cleanup_test_environment(test_name)


@pytest.mark.parametrize("strategy_name,strategy_config,cache_name,cache_config", test_params)
def test_summary_stats3_workflow(strategy_name, strategy_config, cache_name, cache_config):
    """Test the Summary Statistics workflow configuration."""
    test_name = f"sumstats-{strategy_name}-{cache_name}"
    try:
        test_workflow_dir = setup_test_environment(test_name)
        _run_workflow_with_config(
            base_config=config,
            sim_config={
                'class_name': 'DependentVariablePopulationSize',
                'config': dependent_popn_config
            },
            arch_config={
                'name': f'SUMMARY_STATS-{strategy_name}-{cache_name}',
                'config': sum_stats_config_2
            },
            strategy_config=strategy_config,
            cache_config=cache_config,
            test_workflow_dir=test_workflow_dir
        )
    finally:
        cleanup_test_environment(test_name)  
