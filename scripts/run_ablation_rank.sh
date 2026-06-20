#!/bin/bash
# ==============================================================================
# Reproduce Table 6: Ablation - Impact of SpaLRD Rank (r)
# Sweep: r ∈ {8, 16, 32, 64}
# ==============================================================================

CONFIG="configs/experiments/table6_spalrd_rank.yaml"
RANKS=(8 16 32 64)
MODELS=("mkg" "snn_lora")
SEED=42
DEVICE="cuda:0"

echo "Starting Table 6 Evaluation: SpaLRD Rank Sensitivity"

for MODEL in "${MODELS[@]}"; do
    for R in "${RANKS[@]}"; do
        echo "-------------------------------------------------------------------"
        echo "Running $MODEL | Rank (r): $R"
        echo "-------------------------------------------------------------------"
        
        python main.py \
            --config $CONFIG \
            --model $MODEL \
            --seed $SEED \
            --override model.rank=$R \
            --device $DEVICE \
            --log_dir logs/table6_ablation_rank/${MODEL}_r${R}
    done
done

echo "Table 6 evaluation complete."