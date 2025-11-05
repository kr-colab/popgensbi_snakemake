#!/bin/bash
# Script to compare the variable and dependent prior across six population size scenarios:
# 1. Medium
# 2. Large
# 3. Decline
# 4. Expansion
# 5. Bottleneck
# 6. Zigzag

# Navigate to the experiments directory (scripts are now here)
# cd /home/adkern/popgensbi_snakemake/experiments/variable-popn-size

# Create output directory
OUTPUT_DIR="ne_comparison_6_scenarios"
mkdir -p $OUTPUT_DIR

# Common parameters
NUM_EPOCHS=21
POSTERIOR_SAMPLES=10
SIM_SEED=42
ANALYSIS_SEED=666

# Configuration files
V_CONFIG="npe-config/variable_popnSize_windowed_afs_ld.yaml"
D_CONFIG="npe-config/dependent_popnSize_windowed_afs_ld.yaml"
DL_CONFIG="npe-config/dependent_popnSize_windowed_afs_ld_longGENOME.yaml"

echo "=== Using compare-analysis.py for each scenario (this works!) ==="

# Scenario 1: Medium
# echo -e "\n1. Medium..."
# python compare-analysis.py \
#     --configs $V_CONFIG $D_CONFIG $DL_CONFIG \
#     --labels "Varianble" "Dependent" "Dependent_longGENOME" \
#     --simulate \
#     --params 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 3.7 1.5e-8 \
#     --num-epochs $NUM_EPOCHS \
#     --output $OUTPUT_DIR/medium \
#     --posterior-samples $POSTERIOR_SAMPLES \
#     --sim-seed $SIM_SEED \
#     --analysis-seed $((ANALYSIS_SEED + 100))

# # Scenario 2: Large
echo -e "\n2. Large..."
python compare-analysis.py \
    --configs  $V_CONFIG $D_CONFIG $DL_CONFIG \
    --labels "Varianble" "Dependent" "Dependent_longGENOME" \
    --simulate \
    --params 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 4.7 1.5e-8 \
    --num-epochs $NUM_EPOCHS \
    --output $OUTPUT_DIR/large \
    --posterior-samples $POSTERIOR_SAMPLES \
    --sim-seed $((SIM_SEED + 1)) \
    --analysis-seed $((ANALYSIS_SEED + 200))

# Scenario 3: Decline
# echo -e "\n3. Decline..."
# python compare-analysis.py \
#     --configs  $V_CONFIG $D_CONFIG $DL_CONFIG \
#     --labels "Varianble" "Dependent" "Dependent_longGENOME" \
#     --simulate \
#     --params 2.5 2.5 3 3 3 3 3.2 3.4 3.6 3.8 4 4.2 4.6 4.6 4.6 4.6 4.6 4.6 4.6 4.6 4.6 1.5e-8 \
#     --num-epochs $NUM_EPOCHS \
#     --output $OUTPUT_DIR/decline \
#     --posterior-samples $POSTERIOR_SAMPLES \
#     --sim-seed $((SIM_SEED + 2)) \
#     --analysis-seed $((ANALYSIS_SEED + 300))

# Scenario 4: Expansion
# echo -e "\n4. Expansion..."
# python compare-analysis.py \
#     --configs  $V_CONFIG $D_CONFIG $DL_CONFIG \
#     --labels "Varianble" "Dependent" "Dependent_longGENOME" \
#     --simulate \
#     --params 4.7 4.7 4.7 4.6 4.6 4.5 4.4 4.3 4 3.7 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 1.5e-8 \
#     --num-epochs $NUM_EPOCHS \
#     --output $OUTPUT_DIR/expansion \
#     --posterior-samples $POSTERIOR_SAMPLES \
#     --sim-seed $((SIM_SEED + 2)) \
#     --analysis-seed $((ANALYSIS_SEED + 300))

# # Scenario 5: Bottleneck
# echo -e "\n5. Bottleneck..."
# python compare-analysis.py \
#     --configs  $V_CONFIG $D_CONFIG $DL_CONFIG \
#     --labels "Varianble" "Dependent" "Dependent_longGENOME" \
#     --simulate \
#     --params 4.8 4.8 4.8 4.8 4.8 4.8 4.8 4.8 4.8 4.5 4.15 3.8 4.3 4.8 4.55 4.3 4.05 3.8 3.8 3.8 3.8 1.5e-8 \
#     --num-epochs $NUM_EPOCHS \
#     --output $OUTPUT_DIR/bottleneck \
#     --posterior-samples $POSTERIOR_SAMPLES \
#     --sim-seed $((SIM_SEED + 2)) \
#     --analysis-seed $((ANALYSIS_SEED + 300))

# # Scenario 6: Zigzag
# echo -e "\n6. Zigzag..."
# python compare-analysis.py \
#     --configs  $V_CONFIG $D_CONFIG $DL_CONFIG \
#     --labels "Varianble" "Dependent" "Dependent_longGENOME" \
#     --simulate \
#     --params 4.8 4.8 4.8 4.5 4.15 3.8 4.15 4.5 4.8 4.5 4.15 3.8 4.3 4.8 4.55 4.3 4.05 3.8 3.8 3.8 3.8 1.5e-8 \
#     --num-epochs $NUM_EPOCHS \
#     --output $OUTPUT_DIR/zigzag \
#     --posterior-samples $POSTERIOR_SAMPLES \
#     --sim-seed $((SIM_SEED + 2)) \
#     --analysis-seed $((ANALYSIS_SEED + 300))

echo -e "\n=== Creating 3x6 panel plot from results ==="

# Create the panel plot using the saved results
python plot-6-scenarios.py \
    --comparison-dirs $OUTPUT_DIR/medium \
                      $OUTPUT_DIR/large \
                      $OUTPUT_DIR/decline \
                      $OUTPUT_DIR/expansion \
                      $OUTPUT_DIR/bottleneck \
                      $OUTPUT_DIR/zigzag \
    --scenario-labels "medium" "large" "decline" "expansion" "bottleneck" "zigzag" \
    --output $OUTPUT_DIR/ne_comparison_6-scenarios.png \
    --figsize 15 12

echo -e "\nComparison complete! Results saved to $OUTPUT_DIR/"
echo "Main figure: $OUTPUT_DIR/ne_comparison_6_scenarios.png"
echo ""
echo "Individual comparison results for each scenario:"
echo "  - $OUTPUT_DIR/medium/"
echo "  - $OUTPUT_DIR/large/"
echo "  - $OUTPUT_DIR/decline/"
echo "  - $OUTPUT_DIR/expansion/"
echo "  - $OUTPUT_DIR/bottleneck/"
echo "  - $OUTPUT_DIR/zigzag/"