#!/bin/bash
# ==============================================================================
# Reproduce Table 7: Ablation - Topological Routing Threshold (tau) Sensitivity
# Sweep: tau ∈ {0.1, 0.3, 0.4, 0.7}
# ==============================================================================

CONFIG="configs/experiments/table7_topo_threshold.yaml"
THRESHOLDS=(0.1 0.3 0.4 0.7)
SEED=42
DEVICE="cuda:0"

echo "Starting Table 7 Evaluation: Topological Threshold Sensitivity"

# Run Baselines (Threshold concept does not apply, run once)
echo "-------------------------------------------------------------------"
echo "Running Baselines: HLML-SNN & Euclidean Routing"
echo "-------------------------------------------------------------------"
python main.py --config $CONFIG --model hlml_snn --seed $SEED --device $DEVICE --log_dir logs/table7_ablation_tau/hlml_snn
python main.py --config $CONFIG --model mkg --override model.topo_routing_type=euclidean --seed $SEED --device $DEVICE --log_dir logs/table7_ablation_tau/euclidean_routing

# Run MKG with different thresholds
for TAU in "${THRESHOLDS[@]}"; do
    echo "-------------------------------------------------------------------"
    echo "Running MKG | Threshold (tau): $TAU"
    echo "-------------------------------------------------------------------"
    
    python main.py \
        --config $CONFIG \
        --model mkg \
        --seed $SEED \
        --override model.topo_threshold=$TAU \
        --device $DEVICE \
        --log_dir logs/table7_ablation_tau/mkg_tau${TAU}
done

echo "Table 7 evaluation complete."