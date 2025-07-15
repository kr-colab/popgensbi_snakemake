#!/bin/bash
# Example usage of simulate-for-posterior.py for different models

# Example 1: YRI_CEU model
# Parameters: N_A, N_YRI, N_CEU_initial, N_CEU_final, M, Tp, T
echo "Simulating YRI_CEU model..."
python simulate-for-posterior.py \
    --model YRI_CEU \
    --params 1e4 2e4 5e3 3e4 1e-4 1e4 2e4 \
    --output-dir simulated_data/yri_ceu \
    --seed 123

# Example 2: AraTha_2epoch model  
# Parameters: nu (ratio of current to ancestral size), T (time of change, scaled)
echo -e "\nSimulating AraTha_2epoch model..."
python simulate-for-posterior.py \
    --model AraTha_2epoch \
    --params 0.5 0.8 \
    --output-dir simulated_data/aratha_2epoch \
    --seed 456

# Example 3: VariablePopulationSize model (5 epochs)
# Parameters: log10(N1), log10(N2), log10(N3), log10(N4), log10(N5), recomb_rate
echo -e "\nSimulating VariablePopulationSize model (5 epochs)..."
python simulate-for-posterior.py \
    --model VariablePopulationSize \
    --params 4.0 4.2 3.8 4.5 4.1 1.5e-8 \
    --output-dir simulated_data/variable_popsize_5epoch \
    --num-epochs 5 \
    --seed 789

# Example 4: VariablePopulationSize model (3 epochs)
# Parameters: log10(N1), log10(N2), log10(N3), recomb_rate
echo -e "\nSimulating VariablePopulationSize model (3 epochs)..."
python simulate-for-posterior.py \
    --model VariablePopulationSize \
    --params 4.0 4.5 3.8 1.5e-8 \
    --output-dir simulated_data/variable_popsize_3epoch \
    --num-epochs 3 \
    --seed 101112

# Example 5: recombination_rate model
# Parameters: recombination_rate
echo -e "\nSimulating recombination_rate model..."
python simulate-for-posterior.py \
    --model recombination_rate \
    --params 1e-8 \
    --output-dir simulated_data/recomb_rate \
    --seed 131415

# Example with custom parameters
echo -e "\nSimulating with custom sequence length and mutation rate..."
python simulate-for-posterior.py \
    --model VariablePopulationSize \
    --params 4.0 4.2 3.8 4.5 4.1 1.5e-8 \
    --output-dir simulated_data/custom_params \
    --sequence-length 50000000 \
    --mutation-rate 2e-8 \
    --samples 20 \
    --num-epochs 5 \
    --seed 161718