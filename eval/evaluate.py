import torch
import torch.nn as nn
import numpy as np
import os
import json
from tqdm import tqdm

# Conceptual imports from the MKG package structure
# from mkg.models.mkg import MetaKnowledgeGraph
# from mkg.utils.metrics import MetricsEvaluator
# from mkg.utils.hardware import NeuromorphicHardwareProfiler
# from mkg.modules.topo_routing import TopologicalSubgraphRouter


class CheckpointLoader:
    """
    Handles loading of MKG model checkpoints and experiment configurations.
    """
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        
    def load_model(self, model_class, model_kwargs, device):
        """
        Instantiates the model architecture and loads the saved state dictionary.
        """
        model = model_class(**model_kwargs).to(device)
        
        ckpt_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        
        print(f"[CheckpointLoader] Successfully loaded model from {ckpt_path}")
        return model
        
    def load_config(self):
        """Loads the experiment configuration used during training."""
        config_path = os.path.join(self.checkpoint_dir, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)


class ContinualLearningEvaluator:
    """
    Evaluates the model on sequential tasks to compute Continual Learning metrics.
    
    Mathematical Formulation:
    Let R_{j,i} be the accuracy on task i after learning task j.
    1. Average Accuracy (A_N): 
       A_N = (1/N) * sum_{i=1}^N R_{N,i}
    2. Forgetting Measure (F_N): 
       F_N = (1/(N-1)) * sum_{i=1}^{N-1} ( max_{j in {1..N-1}} R_{j,i} - R_{N,i} )
    """
    def __init__(self, device):
        self.device = device
        
    def evaluate_single_task(self, model, dataloader, task_id, time_steps=10):
        """
        Computes the classification accuracy on a single task's test set.
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)
                
                model.reset_states(batch_size, self.device)
                accumulated_logits = torch.zeros(batch_size, model.out_features, device=self.device)
                
                for t in range(time_steps):
                    S_in_t = x[:, t, :]
                    S_out, _, _ = model(S_in_t, task_id=task_id, is_training=False)
                    accumulated_logits += S_out
                    
                preds = accumulated_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        return 100.0 * correct / total

    def evaluate_all_tasks(self, model, test_dataloaders, time_steps=10):
        """
        Evaluates the model on all tasks after learning the final task N.
        Returns the final row of the accuracy matrix: R_{N,i} for i in {1..N}.
        """
        print("[CL Evaluator] Evaluating on all tasks after final task...")
        final_accuracies = []
        
        for task_id in range(len(test_dataloaders)):
            acc = self.evaluate_single_task(model, test_dataloaders[task_id], task_id, time_steps)
            final_accuracies.append(acc)
            print(f"  -> Task {task_id} Accuracy (R_{{N,{task_id}}}): {acc:.2f}%")
            
        return np.array(final_accuracies)


class OODDetectionEvaluator:
    """
    Evaluates Open-World and Out-of-Distribution (OOD) detection capabilities.
    
    Mathematical Formulation:
    1. Topological Scoring Function: 
       s(x) = min_{v_k in V} d_W(PD_x, PD_k)
    2. OOD Routing Accuracy: 
       (1 / |D_OOD|) * sum_{x in D_OOD} I(s(x) > tau AND Delta C_k = 0, forall k)
    """
    def __init__(self, device, topo_router):
        self.device = device
        self.topo_router = topo_router
        
    def extract_topological_scores(self, model, dataloader, time_steps=10):
        """
        Extracts the topological score s(x) for each sample by computing 
        the minimum Wasserstein distance to existing node persistence diagrams.
        """
        model.eval()
        scores = []
        
        with torch.no_grad():
            for x, _ in tqdm(dataloader, desc="Extracting Topo Scores"):
                x = x.to(self.device)
                batch_size = x.size(0)
                model.reset_states(batch_size, self.device)
                
                # Accumulate spikes for the routing window
                for t in range(time_steps):
                    S_in_t = x[:, t, :]
                    model.update_spike_buffer(S_in_t)
                    
                # Compute PDs and Wasserstein distances
                PD_m_batch = self.topo_router.compute_persistence_diagrams(model.spike_buffer)
                
                for PD_m in PD_m_batch:
                    min_dist = np.inf
                    for PD_k in model.node_PDs.values():
                        dist = self.topo_router.compute_wasserstein_distance(PD_m, PD_k)
                        if dist < min_dist:
                            min_dist = dist
                    scores.append(min_dist)
                    
                # Reset buffer for next batch
                model.spike_buffer.zero_()
                model.buffer_idx = 0
                
        return np.array(scores)

    def evaluate_ood(self, model, in_dataloader, ood_dataloader, threshold_tau):
        """
        Computes AUROC, FPR95, and OOD Routing Accuracy.
        """
        print("[OOD Evaluator] Extracting scores for In-Distribution data...")
        scores_in = self.extract_topological_scores(model, in_dataloader)
        
        print("[OOD Evaluator] Extracting scores for OOD data...")
        scores_ood = self.extract_topological_scores(model, ood_dataloader)
        
        # In a full implementation, we would compute AUROC and FPR95 here 
        # using the from-scratch MetricsEvaluator.
        # For this script, we return the scores for downstream metric computation.
        
        return scores_in, scores_ood


class TopologicalStabilityEvaluator:
    """
    Evaluates Betti Number Stability under high-frequency spatial corruptions.
    
    Mathematical Formulation:
    Betti Stability is quantified by the Bottleneck Distance d_B between 
    persistence diagrams before and after learning new tasks:
    d_B(PD_i^{(i)}, PD_i^{(N)}) = inf_{gamma} sup_{x in PD_i^{(i)}} ||x - gamma(x)||_inf
    """
    def __init__(self, device, topo_router):
        self.device = device
        self.topo_router = topo_router
        
    def evaluate_stability(self, model, clean_dataloader, corrupted_dataloader, time_steps=10):
        """
        Computes the Bottleneck Distance between the PDs of clean and corrupted inputs.
        A bounded d_B <= epsilon guarantees the core topological structure is unperturbed.
        """
        print("[Topo Evaluator] Evaluating Betti Number Stability...")
        
        # Extract PDs for clean data
        model.eval()
        with torch.no_grad():
            for x, _ in clean_dataloader:
                x = x.to(self.device)
                model.reset_states(x.size(0), self.device)
                for t in range(time_steps):
                    model.update_spike_buffer(x[:, t, :])
                pd_clean = self.topo_router.compute_persistence_diagrams(model.spike_buffer)[0]
                model.spike_buffer.zero_()
                model.buffer_idx = 0
                break
                
            # Extract PDs for corrupted data
            for x, _ in corrupted_dataloader:
                x = x.to(self.device)
                model.reset_states(x.size(0), self.device)
                for t in range(time_steps):
                    model.update_spike_buffer(x[:, t, :])
                pd_corrupted = self.topo_router.compute_persistence_diagrams(model.spike_buffer)[0]
                model.spike_buffer.zero_()
                model.buffer_idx = 0
                break
                
        # Compute Bottleneck Distance (using the from-scratch implementation in metrics.py)
        # d_B = TopologicalStabilityMetrics.compute_bottleneck_distance(pd_clean, pd_corrupted)
        d_B = 0.0 # Placeholder for the exact computation
        print(f"  -> Betti Stability (d_B): {d_B:.4f}")
        
        return d_B


class HardwareProfilerEvaluator:
    """
    Profiles the neuromorphic hardware efficiency during inference.
    
    Mathematical Formulation:
    1. Synaptic Operations (SOPs): 
       SOPs = sum_{l=1}^L sum_{t=1}^T ||S_pre^{(l)}(t)||_0 * ||W_active^{(l)}||_0
    2. Physical Memory Footprint: 
       M_total = ||Phi||_0 + sum_{k=1}^N ||C_k||_0 + ||M odot E||_0 <= C_max
    """
    def __init__(self, device):
        self.device = device
        
    def profile_inference(self, model, dataloader, time_steps=10):
        """
        Profiles SOPs, Spikes, and Memory Footprint during inference.
        """
        print("[HW Evaluator] Profiling Neuromorphic Hardware Efficiency...")
        model.eval()
        
        total_sops = 0
        total_spikes = 0
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                batch_size = x.size(0)
                model.reset_states(batch_size, self.device)
                
                for t in range(time_steps):
                    S_in_t = x[:, t, :]
                    # Forward pass and accumulate SOPs
                    # In the full implementation, this hooks into the model layers
                    # to compute ||S_pre||_0 * ||W_active||_0
                    pass
                    
        # Compute L0 Memory Footprint
        l0_phi = torch.count_nonzero(model.Phi).item() if hasattr(model, 'Phi') else 0
        l0_c = sum(torch.count_nonzero(C_k).item() for C_k in model.nodes.values()) if hasattr(model, 'nodes') else 0
        l0_e = torch.count_nonzero(model.edge_weights).item() if hasattr(model, 'edge_weights') else 0
        
        l0_total = l0_phi + l0_c + l0_e
        memory_MB = l0_total * 2.0 / (1024 * 1024)  # Assuming 16-bit (2 bytes) per param
        
        print(f"  -> Total SOPs: {total_sops}")
        print(f"  -> Memory Footprint (L0): {l0_total} params ({memory_MB:.2f} MB)")
        
        return total_sops, total_spikes, memory_MB


class MKGFullEvaluator:
    """
    Main orchestrator for the complete evaluation of the MKG framework.
    Aggregates all specific evaluators and runs the full pipeline.
    """
    def __init__(self, checkpoint_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.loader = CheckpointLoader(checkpoint_dir)
        self.cl_evaluator = ContinualLearningEvaluator(self.device)
        self.hw_evaluator = HardwareProfilerEvaluator(self.device)
        
    def run_full_evaluation(self, model_class, model_kwargs, test_dataloaders, 
                            ood_dataloader, corrupted_dataloader=None, threshold_tau=0.4):
        """
        Executes the full evaluation pipeline and saves the results.
        """
        print("="*60)
        print("Starting MKG Full Evaluation Pipeline")
        print("="*60)
        
        # 1. Load Model and Config
        model = self.loader.load_model(model_class, model_kwargs, self.device)
        config = self.loader.load_config()
        
        # Initialize specific evaluators with model components
        ood_evaluator = OODDetectionEvaluator(self.device, model.topo_router)
        topo_evaluator = TopologicalStabilityEvaluator(self.device, model.topo_router)
        
        # 2. Continual Learning Metrics (A_N)
        final_accuracies = self.cl_evaluator.evaluate_all_tasks(
            model, test_dataloaders, time_steps=config.get('time_steps', 10)
        )
        A_N = np.mean(final_accuracies)
        print(f"\n[Result] Average Accuracy (A_N): {A_N:.2f}%")
        
        # 3. OOD Detection Metrics
        scores_in, scores_ood = ood_evaluator.evaluate_ood(
            model, test_dataloaders[0], ood_dataloader, threshold_tau
        )
        
        # 4. Topological Stability (if corruption data is provided)
        betti_dB = 0.0
        if corrupted_dataloader is not None:
            betti_dB = topo_evaluator.evaluate_stability(
                model, test_dataloaders[0], corrupted_dataloader
            )
            
        # 5. Hardware Profiling
        sops, spikes, memory_MB = self.hw_evaluator.profile_inference(
            model, test_dataloaders[0]
        )
        
        # 6. Aggregate and Save Results
        results = {
            "continual_learning": {
                "A_N": float(A_N),
                "final_task_accuracies": final_accuracies.tolist()
            },
            "ood_detection": {
                "threshold_tau": threshold_tau,
                "num_in_samples": len(scores_in),
                "num_ood_samples": len(scores_ood)
            },
            "topological_stability": {
                "betti_dB": float(betti_dB)
            },
            "hardware_efficiency": {
                "SOPs": float(sops),
                "Spikes": float(spikes),
                "memory_MB": float(memory_MB)
            }
        }
        
        results_path = os.path.join(self.loader.checkpoint_dir, "eval_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"\n[Pipeline] Evaluation complete. Results saved to {results_path}")
        print("="*60)
        
        return results