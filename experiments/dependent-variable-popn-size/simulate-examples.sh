#!/bin/bash
# Example usage of simulate-for-posterior.py for different models

# Example 1: VariablePopulationSize model (21 epochs)
# Parameters: log10(N1), log10(N2), log10(N3), ..., log10(N21), recomb_rate
echo -e "\nSimulating VariablePopulationSize model (21 epochs)..."
python simulate-for-posterior.py \
    --model VariablePopulationSize \
    --params 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.1 1e-8 \
    --output-dir simulated_data/variable_popsize \
    --num-epochs 21 \
    --seed 1233

# Example 2: DependentVariablePopulationSize model (21 epochs)
# Parameters: log10(N1), log10(N2), log10(N3),..., log10(N21), recomb_rate
echo -e "\nSimulating DependentVariablePopulationSize model (21 epochs)..."
python simulate-for-posterior.py \
    --model DependentVariablePopulationSize \
    --params 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.1 1e-8 \
    --output-dir simulated_data/dependent_popsize \
    --num-epochs 21 \
    --seed 20202

# Example 3: DependentVariablePopulationSize model (21 epochs)
# Parameters: log10(N1), log10(N2), log10(N3),..., log10(N21), recomb_rate
echo -e "\nSimulating DependentVariablePopulationSize model with longer sequence length..."
python simulate-for-posterior.py \
    --model DependentVariablePopulationSize \
    --params 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.0 4.2 3.8 4.5 4.1 4.1 1e-8\
    --sequence-length 2e8 \
    --output-dir simulated_data/dependent_popsize_longsequence \
    --num-epochs 21 \
    --seed 202
