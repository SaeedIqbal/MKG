#!/bin/bash
# ==============================================================================
# Reproduce Table 9: Memory Lifecycle & Active Forgetting over 50 Tasks
# ==============================================================================

CONFIG="configs/experiments/table9_memory_lifecycle.yaml"
SEED=42
DEVICE="cuda:0"

echo "Starting Table 9 Evaluation: 50-Task Memory Lifecycle"

# 1. Full MKG (With Active Forgetting)
echo "-------------------------------------------------------------------"
echo "Running Full MKG (Active Forgetting Enabled)"
echo "-------------------------------------------------------------------"
python main.py \
    --config $CONFIG \
    --model mkg \
    --seed $SEED \
    --device $DEVICE \
    --log_dir logs/table9_memory_lifecycle/mkg_full

# 2. MKG w/o Active Forgetting (Ablation)
echo "-------------------------------------------------------------------"
echo "Running MKG (Ablation: w/o Active Forgetting)"
echo "-------------------------------------------------------------------"
python main.py \
    --config $CONFIG \
    --model mkg \
    --seed $SEED \
    --override model.rho_max=1.0 model.epsilon_q=0.0 \
    --device $DEVICE \
    --log_dir logs/table9_memory_lifecycle/mkg_no_forgetting

# 3. Dynamic Expansion Baselines (Physical Bloat)
for MODEL in "alade_snn" "progressive_snn"; do
    echo "-------------------------------------------------------------------"
    echo "Running $MODEL (Physical Expansion Baseline)"
    echo "-------------------------------------------------------------------"
    python main.py \
        --config $CONFIG \
        --model $MODEL \
        --seed $SEED \
        --device $DEVICE \
        --log_dir logs/table9_memory_lifecycle/${MODEL}
done

echo "Table 9 evaluation complete."