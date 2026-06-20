#!/bin/bash
# ==============================================================================
# Reproduce Table 3: Robustness to Corruptions & Topological Stability
# Datasets: CIFAR-10-C, ImageNet-C (Severities 1-5)
# ==============================================================================

CONFIG="configs/experiments/table3_corruptions.yaml"
DATASETS=("cifar10_c" "imagenet_c")
SEVERITIES=(1 2 3 4 5)
MODELS=("mkg" "der_pp" "hlml_snn" "hlop")
SEED=42
DEVICE="cuda:0"

echo "Starting Table 3 Evaluation: Corruptions & Betti Stability"

for DATASET in "${DATASETS[@]}"; do
    for SEV in "${SEVERITIES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo "-------------------------------------------------------------------"
            echo "Running $MODEL | Dataset: $DATASET | Severity: $SEV"
            echo "-------------------------------------------------------------------"
            
            python main.py \
                --config $CONFIG \
                --dataset $DATASET \
                --model $MODEL \
                --seed $SEED \
                --override dataset.eval_severity=$SEV \
                --device $DEVICE \
                --log_dir logs/table3_corruptions/${DATASET}_sev${SEV}/${MODEL}
        done
    done
done

echo "Table 3 evaluation complete."