#!/bin/bash
# ==============================================================================
# Reproduce Table 4: Cross-Modal Continual Learning (Text Streams)
# Sequence: AG News -> Amazon Reviews -> Yelp
# ==============================================================================

CONFIG="configs/experiments/table4_text_streams.yaml"
MODELS=("mkg" "ewc" "lora_cl" "hlml_snn")
SEED=42
DEVICE="cuda:0"

echo "Starting Table 4 Evaluation: Cross-Modal Text Streams"

for MODEL in "${MODELS[@]}"; do
    echo "-------------------------------------------------------------------"
    echo "Running $MODEL | Dataset: Text Streams (AG -> Amazon -> Yelp)"
    echo "-------------------------------------------------------------------"
    
    python main.py \
        --config $CONFIG \
        --model $MODEL \
        --seed $SEED \
        --device $DEVICE \
        --log_dir logs/table4_text_streams/${MODEL}
done

echo "Table 4 evaluation complete."