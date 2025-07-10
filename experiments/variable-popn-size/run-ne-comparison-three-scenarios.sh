#!/bin/bash
# Script to compare CNN, RNN, and SPIDNA across three population size scenarios:
# 1. Abrupt change
# 2. Linear decline  
# 3. Linear growth

# Navigate to the posterior analysis directory
cd /home/adkern/popgensbi_snakemake/tools/posterior_analysis

# Create output directory
OUTPUT_DIR="ne_comparison_three_scenarios"
mkdir -p $OUTPUT_DIR

# Common parameters
NUM_EPOCHS=15
POSTERIOR_SAMPLES=1000
SIM_SEED=42
ANALYSIS_SEED=666

# Configuration files
CNN_CONFIG="../../experiments/variable-popn-size/npe-config/variable_popn_size_cnn.yaml"
RNN_CONFIG="../../experiments/variable-popn-size/npe-config/variable_popn_size_rnn.yaml"
SPIDNA_CONFIG="../../experiments/variable-popn-size/npe-config/variable_popn_size_spidna.yaml"

echo "=== Using compare-analysis.py for each scenario (this works!) ==="

# Scenario 1: Abrupt population size change
echo -e "\n1. Abrupt population size change..."
python compare-analysis.py \
    --configs $CNN_CONFIG $RNN_CONFIG $SPIDNA_CONFIG \
    --labels "CNN" "RNN" "SPIDNA" \
    --simulate \
    --model VariablePopulationSize \
    --params 4.0 4.0 4.0 4.0 4.0 4.0 4.0 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 1.5e-8 \
    --num-epochs $NUM_EPOCHS \
    --output $OUTPUT_DIR/abrupt_change \
    --posterior-samples $POSTERIOR_SAMPLES \
    --sim-seed $SIM_SEED \
    --analysis-seed $((ANALYSIS_SEED + 100))

# Scenario 2: Linear decline in population size
echo -e "\n2. Linear decline in population size..."
python compare-analysis.py \
    --configs $CNN_CONFIG $RNN_CONFIG $SPIDNA_CONFIG \
    --labels "CNN" "RNN" "SPIDNA" \
    --simulate \
    --model VariablePopulationSize \
    --params 4.0 3.9 3.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3.0 2.9 2.8 2.7 2.5 1.5e-8 \
    --num-epochs $NUM_EPOCHS \
    --output $OUTPUT_DIR/linear_decline \
    --posterior-samples $POSTERIOR_SAMPLES \
    --sim-seed $((SIM_SEED + 1)) \
    --analysis-seed $((ANALYSIS_SEED + 200))

# Scenario 3: Linear growth in population size
echo -e "\n3. Linear growth in population size..."
python compare-analysis.py \
    --configs $CNN_CONFIG $RNN_CONFIG $SPIDNA_CONFIG \
    --labels "CNN" "RNN" "SPIDNA" \
    --simulate \
    --model VariablePopulationSize \
    --params 2.5 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0 1.5e-8 \
    --num-epochs $NUM_EPOCHS \
    --output $OUTPUT_DIR/linear_growth \
    --posterior-samples $POSTERIOR_SAMPLES \
    --sim-seed $((SIM_SEED + 2)) \
    --analysis-seed $((ANALYSIS_SEED + 300))

echo -e "\n=== Creating 3x3 panel plot from results ==="

# Create the panel plot using the saved results
python create-3x3-panel.py \
    --comparison-dirs $OUTPUT_DIR/abrupt_change \
                      $OUTPUT_DIR/linear_decline \
                      $OUTPUT_DIR/linear_growth \
    --scenario-labels "Abrupt Change" "Linear Decline" "Linear Growth" \
    --output $OUTPUT_DIR/ne_comparison_3x3_panel.png \
    --figsize 15 12

echo -e "\nComparison complete! Results saved to $OUTPUT_DIR/"
echo "Main figure: $OUTPUT_DIR/ne_comparison_3x3_panel.png"
echo ""
echo "Individual comparison results for each scenario:"
echo "  - $OUTPUT_DIR/abrupt_change/"
echo "  - $OUTPUT_DIR/linear_decline/"
echo "  - $OUTPUT_DIR/linear_growth/"