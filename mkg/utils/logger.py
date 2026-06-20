import os
import json
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime

class BaseLogger(ABC):
    """
    Abstract base class for experiment logging.
    Defines the interface that all concrete loggers must implement.
    """
    
    @abstractmethod
    def init_experiment(self, experiment_name, config):
        """Initializes the logging backend with experiment metadata."""
        pass
    
    @abstractmethod
    def log_scalar(self, key, value, step):
        """Logs a single scalar metric."""
        pass
    
    @abstractmethod
    def log_scalars(self, metric_dict, step):
        """Logs a dictionary of scalar metrics."""
        pass
    
    @abstractmethod
    def log_matrix(self, matrix, name, step):
        """Logs a 2D matrix (e.g., accuracy matrix for A_N / F_N computation)."""
        pass
    
    @abstractmethod
    def log_graph_topology(self, num_nodes, num_edges, memory_MB, step):
        """Logs the MKG graph structure and memory footprint."""
        pass
    
    @abstractmethod
    def log_sops_energy(self, sops, spikes, energy_mJ, step):
        """Logs neuromorphic hardware efficiency metrics."""
        pass
    
    @abstractmethod
    def log_topological_stability(self, betti_dB, dataset_name, step):
        """Logs Betti Number Stability (Bottleneck Distance)."""
        pass
    
    @abstractmethod
    def close(self):
        """Finalizes and closes the logging session."""
        pass


class ConsoleLogger(BaseLogger):
    """
    Lightweight console-based logger for environments where WandB/TensorBoard 
    are not available. Writes structured JSON logs to disk.
    """
    
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        self.experiment_name = None
        self.log_entries = []
        self.config = {}
        
    def init_experiment(self, experiment_name, config):
        """Initializes the console logger and creates the log directory."""
        self.experiment_name = experiment_name
        self.config = config
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.run_dir, "config.json")
        serializable_config = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v)
                
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
            
        print(f"[ConsoleLogger] Initialized experiment: {experiment_name}")
        print(f"[ConsoleLogger] Log directory: {self.run_dir}")
        
    def log_scalar(self, key, value, step):
        """Logs a single scalar metric to console and buffer."""
        entry = {"type": "scalar", "key": key, "value": value, "step": step}
        self.log_entries.append(entry)
        print(f"[Step {step}] {key}: {value:.6f}")
        
    def log_scalars(self, metric_dict, step):
        """Logs a dictionary of scalar metrics."""
        for key, value in metric_dict.items():
            self.log_scalar(key, value, step)
            
    def log_matrix(self, matrix, name, step):
        """Logs a 2D matrix (e.g., task accuracy matrix) to disk."""
        matrix_path = os.path.join(self.run_dir, f"{name}_step{step}.npy")
        np.save(matrix_path, matrix)
        
        entry = {"type": "matrix", "key": name, "shape": list(matrix.shape), "step": step}
        self.log_entries.append(entry)
        print(f"[Step {step}] Matrix '{name}' saved: shape {matrix.shape}")
        
    def log_graph_topology(self, num_nodes, num_edges, memory_MB, step):
        """Logs the MKG graph structure: |V|, |E|, and ||Theta_total||_0."""
        metrics = {
            "graph/num_nodes_V": num_nodes,
            "graph/num_edges_E": num_edges,
            "graph/memory_footprint_MB": memory_MB
        }
        self.log_scalars(metrics, step)
        
    def log_sops_energy(self, sops, spikes, energy_mJ, step):
        """Logs neuromorphic hardware efficiency metrics."""
        metrics = {
            "hardware/SOPs": sops,
            "hardware/spikes": spikes,
            "hardware/energy_mJ": energy_mJ
        }
        self.log_scalars(metrics, step)
        
    def log_topological_stability(self, betti_dB, dataset_name, step):
        """Logs Betti Number Stability d_B."""
        self.log_scalar(f"topology/betti_dB_{dataset_name}", betti_dB, step)
        
    def close(self):
        """Finalizes the logging session and writes all entries to disk."""
        log_path = os.path.join(self.run_dir, "full_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
        print(f"[ConsoleLogger] Full log saved to: {log_path}")


class WandBLogger(BaseLogger):
    """
    Weights & Biases (WandB) logger for cloud-based experiment tracking.
    
    Tracks all MKG-specific metrics including:
    - Continual Learning: A_N, F_N
    - OOD Detection: AUROC, FPR95, OOD Routing Accuracy
    - Topological: Betti Stability d_B
    - Hardware: Memory Footprint (L0), SOPs, Energy (mJ)
    """
    
    def __init__(self, project_name="MKG-SNN-Continual-Learning", entity=None):
        self.project_name = project_name
        self.entity = entity
        self.run = None
        
    def init_experiment(self, experiment_name, config):
        """Initializes a WandB run."""
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "WandB is not installed. Install it via: pip install wandb"
            )
            
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=experiment_name,
            config=config,
            reinit=True
        )
        print(f"[WandBLogger] Initialized run: {self.run.name} (ID: {self.run.id})")
        
    def log_scalar(self, key, value, step):
        """Logs a single scalar metric to WandB."""
        if self.run is not None:
            self.run.log({key: value}, step=step)
            
    def log_scalars(self, metric_dict, step):
        """Logs a dictionary of scalar metrics to WandB."""
        if self.run is not None:
            self.run.log(metric_dict, step=step)
            
    def log_matrix(self, matrix, name, step):
        """Logs a 2D matrix as a WandB Table and Heatmap."""
        try:
            import wandb
        except ImportError:
            return
            
        if self.run is not None:
            # Log as a WandB Image (heatmap)
            fig_data = [[matrix[i, j] for j in range(matrix.shape[1])] 
                        for i in range(matrix.shape[0])]
            
            table = wandb.Table(
                columns=[f"Task_{j}" for j in range(matrix.shape[1])],
                data=fig_data
            )
            self.run.log({f"{name}_table": table}, step=step)
            
    def log_graph_topology(self, num_nodes, num_edges, memory_MB, step):
        """Logs the MKG graph structure to WandB."""
        metrics = {
            "graph/num_nodes_V": num_nodes,
            "graph/num_edges_E": num_edges,
            "graph/memory_footprint_MB": memory_MB,
            "graph/L0_pseudo_norm": num_nodes + num_edges
        }
        self.log_scalars(metrics, step)
        
    def log_sops_energy(self, sops, spikes, energy_mJ, step):
        """Logs neuromorphic hardware efficiency metrics to WandB."""
        metrics = {
            "hardware/SOPs": sops,
            "hardware/total_spikes": spikes,
            "hardware/energy_mJ": energy_mJ,
            "hardware/SOPs_per_spike": sops / max(spikes, 1)
        }
        self.log_scalars(metrics, step)
        
    def log_topological_stability(self, betti_dB, dataset_name, step):
        """Logs Betti Number Stability d_B to WandB."""
        self.log_scalar(f"topology/betti_dB_{dataset_name}", betti_dB, step)
        
    def close(self):
        """Finishes the WandB run."""
        if self.run is not None:
            self.run.finish()
            print("[WandBLogger] Run finished.")


class TensorBoardLogger(BaseLogger):
    """
    TensorBoard logger for local experiment visualization.
    
    Writes scalar summaries, histograms, and text summaries to a 
    TensorBoard-compatible event file using torch.utils.tensorboard.
    """
    
    def __init__(self, log_dir="./runs"):
        self.log_dir = log_dir
        self.writer = None
        
    def init_experiment(self, experiment_name, config):
        """Initializes the TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard is not installed. Install via: pip install tensorboard"
            )
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.log_dir, f"{experiment_name}_{timestamp}")
        self.writer = SummaryWriter(log_dir=run_dir)
        
        # Log hyperparameters as text
        config_text = "\n".join([f"{k}: {v}" for k, v in config.items()])
        self.writer.add_text("hyperparameters", config_text, global_step=0)
        
        print(f"[TensorBoardLogger] Initialized: {run_dir}")
        
    def log_scalar(self, key, value, step):
        """Logs a single scalar metric to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar(key, value, global_step=step)
            
    def log_scalars(self, metric_dict, step):
        """Logs a dictionary of scalar metrics to TensorBoard."""
        if self.writer is not None:
            for key, value in metric_dict.items():
                self.writer.add_scalar(key, value, global_step=step)
                
    def log_matrix(self, matrix, name, step):
        """Logs a 2D matrix as a flattened histogram to TensorBoard."""
        if self.writer is not None:
            self.writer.add_histogram(f"{name}_values", matrix.flatten(), global_step=step)
            
    def log_graph_topology(self, num_nodes, num_edges, memory_MB, step):
        """Logs the MKG graph structure to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar("graph/num_nodes_V", num_nodes, global_step=step)
            self.writer.add_scalar("graph/num_edges_E", num_edges, global_step=step)
            self.writer.add_scalar("graph/memory_footprint_MB", memory_MB, global_step=step)
            
    def log_sops_energy(self, sops, spikes, energy_mJ, step):
        """Logs neuromorphic hardware efficiency metrics to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar("hardware/SOPs", sops, global_step=step)
            self.writer.add_scalar("hardware/total_spikes", spikes, global_step=step)
            self.writer.add_scalar("hardware/energy_mJ", energy_mJ, global_step=step)
            
    def log_topological_stability(self, betti_dB, dataset_name, step):
        """Logs Betti Number Stability d_B to TensorBoard."""
        self.log_scalar(f"topology/betti_dB_{dataset_name}", betti_dB, step)
        
    def close(self):
        """Flushes and closes the TensorBoard writer."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            print("[TensorBoardLogger] Writer closed.")


class MKGExperimentLogger:
    """
    Unified experiment logger for the Meta-Knowledge Graph (MKG) framework.
    
    Aggregates one or more backend loggers (Console, WandB, TensorBoard) and 
    provides high-level methods that map directly to the manuscript's metrics:
    - A_N, F_N (Continual Learning)
    - AUROC, FPR95, OOD Routing Accuracy (Open-World Detection)
    - Betti Stability d_B (Topological Robustness)
    - Memory Footprint, SOPs, Energy (Neuromorphic Efficiency)
    """
    
    def __init__(self):
        self.backends = []
        
    def add_backend(self, logger_instance):
        """Registers a logging backend."""
        if isinstance(logger_instance, BaseLogger):
            self.backends.append(logger_instance)
        else:
            raise TypeError(f"Logger must be an instance of BaseLogger, got {type(logger_instance)}")
            
    def init_experiment(self, experiment_name, config):
        """Initializes all registered backends."""
        for backend in self.backends:
            backend.init_experiment(experiment_name, config)
            
    def log_continual_learning(self, accuracy_matrix, task_step):
        """
        Computes and logs A_N and F_N from the task accuracy matrix.
        
        Mathematical Formulation:
        A_N = (1/N) * sum_{i=1}^N R_{N,i}
        F_N = (1/(N-1)) * sum_{i=1}^{N-1} ( max_{j} R_{j,i} - R_{N,i} )
        """
        N = accuracy_matrix.shape[0]
        
        # Average Accuracy
        A_N = float(np.mean([accuracy_matrix[N-1, i] for i in range(N)]))
        
        # Forgetting Measure
        F_N = 0.0
        if N > 1:
            for i in range(N - 1):
                max_prev = float(np.max([accuracy_matrix[j, i] for j in range(i, N-1)]))
                F_N += (max_prev - float(accuracy_matrix[N-1, i]))
            F_N /= (N - 1)
            
        metrics = {
            "continual/A_N": A_N,
            "continual/F_N": F_N,
            "continual/num_tasks": N
        }
        
        for backend in self.backends:
            backend.log_scalars(metrics, step=task_step)
            backend.log_matrix(accuracy_matrix, "accuracy_matrix", step=task_step)
            
    def log_ood_detection(self, auroc, fpr95, ood_routing_acc, step):
        """
        Logs Open-World / OOD detection metrics.
        """
        metrics = {
            "ood/AUROC": auroc,
            "ood/FPR95": fpr95,
            "ood/routing_accuracy": ood_routing_acc
        }
        for backend in self.backends:
            backend.log_scalars(metrics, step=step)
            
    def log_topological_stability(self, betti_dB, dataset_name, step):
        """
        Logs Betti Number Stability.
        A bounded d_B <= epsilon guarantees core topological structure is unperturbed.
        """
        for backend in self.backends:
            backend.log_topological_stability(betti_dB, dataset_name, step)
            
    def log_hardware_efficiency(self, model, sops, spikes, energy_mJ, step):
        """
        Logs the complete neuromorphic hardware profile:
        ||Theta_total||_0 = ||Phi||_0 + sum ||C_k||_0 + ||E||_0
        """
        # Compute L0 memory footprint
        l0_phi = 0
        l0_c = 0
        l0_e = 0
        
        import torch
        if hasattr(model, 'Phi'):
            l0_phi = torch.count_nonzero(model.Phi).item()
        if hasattr(model, 'nodes'):
            for node_id, C_k in model.nodes.items():
                l0_c += torch.count_nonzero(C_k).item()
        if hasattr(model, 'edge_weights'):
            l0_e = torch.count_nonzero(model.edge_weights).item()
            
        l0_total = l0_phi + l0_c + l0_e
        memory_MB = l0_total * 2.0 / (1024 * 1024)  # Assuming 16-bit (2 bytes) per param
        
        for backend in self.backends:
            backend.log_graph_topology(
                num_nodes=len(model.nodes) if hasattr(model, 'nodes') else 0,
                num_edges=l0_e,
                memory_MB=memory_MB,
                step=step
            )
            backend.log_sops_energy(sops, spikes, energy_mJ, step)
            
        return l0_total, memory_MB
        
    def log_hyperparameter_sweep(self, param_name, param_value, metric_name, metric_value, step):
        """
        Logs hyperparameter sensitivity results (e.g., SpaLRD rank r, threshold tau).
        """
        metrics = {
            f"sweep/{param_name}": param_value,
            f"sweep/{metric_name}": metric_value
        }
        for backend in self.backends:
            backend.log_scalars(metrics, step=step)
            
    def close(self):
        """Closes all registered logging backends."""
        for backend in self.backends:
            backend.close()
        print("[MKGExperimentLogger] All backends closed.")