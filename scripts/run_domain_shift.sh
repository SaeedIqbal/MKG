#!/bin/bash
# ==============================================================================
# Reproduce Table 2: Domain-Incremental Generalization
# Datasets: Mini-DomainNet, Office-Home, VisDA-2017
# ==============================================================================

CONFIG_BASE="configs/experiments/table2_domain_shift.yaml"
DATASETS=("mini_domainnet" "office_home" "visda_2017")
MODELS=("mkg" "hlml_snn" "snn_lora" "hlop" "ch_hnn")
SEED=42
DEVICE="cuda:0"

echo "Starting Table 2 Evaluation: Domain-Incremental Generalization"

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "-------------------------------------------------------------------"
        echo "Running $MODEL | Dataset: $DATASET"
        echo "-------------------------------------------------------------------"
        
        python main.py \
            --config $CONFIG_BASE \
            --dataset $DATASET \
            --model $MODEL \
            --seed $SEED \
            --device $DEVICE \
            --log_dir logs/table2_domain_shift/${DATASET}/${MODEL}
    done
done

echo "Table 2 evaluation complete."