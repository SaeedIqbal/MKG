import torch
import torch.nn as nn
import numpy as np

class HardwareConstants:
    """
    Stores standard neuromorphic hardware constants for energy and memory estimation.
    Based on Intel Loihi 2 specifications as referenced in the manuscript.
    """
    def __init__(self, E_SOP_pJ=4.6, E_spike_pJ=20.0, bits_per_param=16):
        # Energy per Synaptic Operation (Joules)
        self.E_SOP = E_SOP_pJ * 1e-12  
        # Energy per spike routing/communication (Joules)
        self.E_spike = E_spike_pJ * 1e-12  
        self.bits_per_param = bits_per_param
        self.bytes_per_param = bits_per_param / 8.0

class MemoryFootprintCalculator:
    """
    Calculates the Physical Memory Footprint of the MKG framework.
    
    Mathematical Formulation:
    M_total = ||Phi||_0 + sum_{k=1}^N ||C_k||_0 + ||M odot E||_0 <= C_max
    where ||.||_0 denotes the L0 pseudo-norm (number of non-zero elements).
    """
    def __init__(self, hardware_constants):
        self.hw_constants = hardware_constants
        
    def compute_l0_norm(self, tensor):
        """Computes the L0 pseudo-norm (number of non-zero elements)."""
        return torch.count_nonzero(tensor).item()
        
    def compute_model_memory(self, model):
        """
        Computes the total physical memory footprint of the MKG model.
        Assumes the model has attributes: Phi, nodes (dict of C_k), and edge_weights.
        """
        total_non_zero_params = 0
        
        # 1. Shared Spiking Basis (Phi)
        if hasattr(model, 'Phi'):
            total_non_zero_params += self.compute_l0_norm(model.Phi)
            
        # 2. Task-specific Coefficients (C_k)
        if hasattr(model, 'nodes'):
            for node_id, C_k in model.nodes.items():
                total_non_zero_params += self.compute_l0_norm(C_k)
                
        # 3. Graph Edges (E) - effectively M odot E after active forgetting
        if hasattr(model, 'edge_weights'):
            total_non_zero_params += self.compute_l0_norm(model.edge_weights)
            
        # Convert to bytes and Megabytes
        memory_bytes = total_non_zero_params * self.hw_constants.bytes_per_param
        memory_MB = memory_bytes / (1024 * 1024)
        
        return {
            "non_zero_params": total_non_zero_params,
            "memory_bytes": memory_bytes,
            "memory_MB": memory_MB
        }

class SOPCalculator:
    """
    Calculates the Synaptic Operations (SOPs) during inference.
    
    Mathematical Formulation:
    SOPs = sum_{l=1}^L sum_{t=1}^T ||S_pre^{(l)}(t)||_0 * ||W_active^{(l)}||_0
    """
    def __init__(self):
        self.total_sops = 0.0
        self.total_spikes = 0.0
        
    def reset(self):
        """Resets the accumulators for a new evaluation sequence."""
        self.total_sops = 0.0
        self.total_spikes = 0.0
        
    def compute_layer_sops(self, S_pre, W_active):
        """
        Computes SOPs for a single layer at a single time step.
        
        Args:
            S_pre (torch.Tensor): Presynaptic spike train. Shape: (batch, in_features)
            W_active (torch.Tensor): Active weight matrix. Shape: (in_features, out_features)
        Returns:
            tuple: (sops_per_sample, spikes_per_sample)
        """
        # ||S_pre||_0: number of active presynaptic spikes per sample
        spikes_per_sample = torch.count_nonzero(S_pre, dim=1).float() 
        avg_spikes = spikes_per_sample.mean().item()
        
        # ||W_active||_0: number of non-zero active weights
        active_weights = torch.count_nonzero(W_active).item()
        
        # SOPs for this time step and layer (averaged over the batch)
        sops_per_sample = avg_spikes * active_weights
        
        return sops_per_sample, avg_spikes

    def accumulate_sops(self, S_pre, W_active):
        """Accumulates SOPs and spike counts across time steps and layers."""
        sops, spikes = self.compute_layer_sops(S_pre, W_active)
        self.total_sops += sops
        self.total_spikes += spikes
        
    def get_total_sops(self):
        return self.total_sops
        
    def get_total_spikes(self):
        return self.total_spikes

class EnergyEstimator:
    """
    Estimates the total energy consumption of the SNN inference.
    
    Mathematical Formulation:
    E_total = N_SOP * E_SOP + N_spike * E_spike
    """
    def __init__(self, hardware_constants):
        self.hw_constants = hardware_constants
        
    def compute_energy_joules(self, N_SOP, N_spike):
        """Computes total energy in Joules."""
        E_SOP_total = N_SOP * self.hw_constants.E_SOP
        E_spike_total = N_spike * self.hw_constants.E_spike
        return E_SOP_total + E_spike_total
        
    def compute_energy_mJ(self, N_SOP, N_spike):
        """Computes total energy in milliJoules (mJ)."""
        return self.compute_energy_joules(N_SOP, N_spike) * 1000.0

class NeuromorphicHardwareProfiler:
    """
    Main orchestrator for profiling the MKG framework's neuromorphic hardware efficiency.
    Integrates memory footprint, SOPs, and energy consumption calculations.
    """
    def __init__(self, E_SOP_pJ=4.6, E_spike_pJ=20.0, bits_per_param=16):
        self.hw_constants = HardwareConstants(E_SOP_pJ, E_spike_pJ, bits_per_param)
        self.memory_calc = MemoryFootprintCalculator(self.hw_constants)
        self.sop_calc = SOPCalculator()
        self.energy_est = EnergyEstimator(self.hw_constants)
        
    def profile_memory(self, model):
        """Profiles the physical memory footprint of the model."""
        return self.memory_calc.compute_model_memory(model)
        
    def profile_layer_inference(self, S_pre, W_active):
        """Profiles SOPs and spikes for a single layer forward pass at time t."""
        self.sop_calc.accumulate_sops(S_pre, W_active)
        
    def get_inference_profile(self):
        """Returns the complete inference profile (SOPs, Spikes, Energy)."""
        N_SOP = self.sop_calc.get_total_sops()
        N_spike = self.sop_calc.get_total_spikes()
        
        energy_mJ = self.energy_est.compute_energy_mJ(N_SOP, N_spike)
        
        return {
            "SOPs": N_SOP,
            "Spikes": N_spike,
            "Energy_mJ": energy_mJ
        }
        
    def reset_inference_profile(self):
        """Resets the SOP and spike counters for a new evaluation."""
        self.sop_calc.reset()