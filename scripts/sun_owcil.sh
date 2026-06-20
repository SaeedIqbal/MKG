#!/bin/bash
# ==============================================================================
# Reproduce Table 1: OWCIL and OOD Detection on Split-ImageNet-100 + ImageNet-O
# ==============================================================================

CONFIG="configs/experiments/table1_owcil.yaml"
SEEDS=(42 43 44)
MODELS=("mkg" "hlml_snn" "snn_lora" "alade_snn" "cls_er" "ewc")
DEVICE="cuda:0"

echo "Starting Table 1 Evaluation: OWCIL + OOD Isolation"

for SEED in "${SEEDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "-------------------------------------------------------------------"
        echo "Running $MODEL | Seed: $SEED | Dataset: Split-ImageNet-100 + ImageNet-O"
        echo "-------------------------------------------------------------------"
        
        python main.py \
            --config $CONFIG \
            --model $MODEL \
            --seed $SEED \
            --device $DEVICE \
            --log_dir logs/table1_owcil/${MODEL}_seed${SEED}
            
        if [ $? -ne 0 ]; then
            echo "Error running $MODEL with seed $SEED. Exiting."
            exit 1
        fi
    done
done

echo "Table 1 evaluation complete. Results aggregated in logs/table1_owcil/"