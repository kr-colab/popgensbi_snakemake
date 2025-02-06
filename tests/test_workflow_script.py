import subprocess
import pytest
import os
import shutil
import tempfile
import re

def setup_test_environment():
    """Set up the test environment by copying and modifying the Snakefile and scripts."""
    # Create test directories
    test_workflow_dir = ".test-workflow"
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
    
    return test_workflow_dir, test_snakefile

def cleanup_test_environment():
    """Clean up the test environment."""
    test_workflow_dir = ".test-workflow"
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

def run_test_workflow():
    """Run the test workflow script and return the completion status."""
    try:
        test_workflow_dir, test_snakefile = setup_test_environment()

        # Define the configuration and options
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

        e2e_config = '''
train_embedding_net_separately: False
'''

        sep_config = '''
train_embedding_net_separately: True
'''

        mem_config = '''
use_cache: True
'''

        disk_config = '''
use_cache: False
'''

        # --- test runs --- #
        for sim in ["YRI_CEU"]:
            for arch in ["RNN", "CNN"]:
                for strategy in ["E2E", "SEP"]:
                    for cache in ["MEM", "DISK"]:
                        base = f"{sim}-{arch}-{strategy}-{cache}"
                        yaml_file = f"{base}.yaml"
                        log_file = f"{base}.log"
                        full_base = os.path.join(test_workflow_dir, base)

                        # Create the YAML configuration file
                        yaml_path = os.path.join(test_workflow_dir, yaml_file)
                        with open(yaml_path, 'w') as f:
                            f.write(config)
                            f.write(f'project_dir: "{full_base}"\n')
                            f.write(yri_ce_config)
                            f.write(rnn_config if arch == "RNN" else cnn_config)
                            f.write(e2e_config if strategy == "E2E" else sep_config)
                            f.write(mem_config if cache == "MEM" else disk_config)

                        # First, try to unlock if there's a stale lock
                        subprocess.run(
                            ["snakemake", "--unlock", "--configfile", yaml_file, "--snakefile", "Snakefile"],
                            capture_output=True,
                            text=True,
                            cwd=test_workflow_dir  # Run from test directory
                        )

                        # Then run the workflow
                        result = subprocess.run(
                            ["snakemake", "--jobs", "2", "--configfile", yaml_file, "--snakefile", "Snakefile"],
                            capture_output=True,
                            text=True,
                            cwd=test_workflow_dir  # Run from test directory
                        )

                        # Write output to log file
                        log_path = os.path.join(test_workflow_dir, log_file)
                        with open(log_path, 'w') as f:
                            f.write(result.stdout)
                            f.write(result.stderr)

                        # Check return code
                        assert result.returncode == 0, f"Snakemake workflow failed for {full_base}. Error: {result.stderr}"
                        
                        # Check completion status in log file
                        assert check_workflow_completion(log_path), f"Workflow did not complete successfully for {full_base}. Check {log_file} for details."

    finally:
        # Clean up test environment
        cleanup_test_environment()

def test_workflow():
    """Test the Snakemake workflow for successful completion."""
    run_test_workflow() 