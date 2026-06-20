import torch
import torch.nn as nn
import numpy as np

class HebbianUpdater:
    """
    Inner-loop local Hebbian updater for the Meta-Knowledge Graph (MKG).
    
    Performs event-driven, local synaptic plasticity updates without 
    Backpropagation Through Time (BPTT), ensuring biological plausibility 
    and neuromorphic hardware efficiency.
    
    Mathematical Formulation:
    1. Eligibility Traces (Discrete equivalent of STDP kernel convolution):
       x_pre(t) = alpha_+ * x_pre(t-1) + S_pre(t)
       x_post(t) = alpha_- * x_post(t-1) + S_post(t)
       where alpha_+ = exp(-1 / tau_+), alpha_- = exp(-1 / tau_-)
       
    2. Local Weight Update:
       Delta W = eta_h(k) * [ A_+ * S_post(t) @ x_pre(t)^T - A_- * S_pre(t) @ x_post(t)^T ]
       W(t) = W(t-1) + Delta W
       
    3. Exponentially Decaying Learning Rate:
       eta_h(k) = eta_0 * gamma^k
    """
    
    def __init__(self, eta_0=0.01, gamma=0.95, 
                 A_plus=0.01, tau_plus=20.0, 
                 A_minus=0.012, tau_minus=20.0):
        """
        Initializes the Hebbian updater with STDP kernel parameters and learning rate schedule.
        
        Args:
            eta_0 (float): Initial learning rate.
            gamma (float): Decay factor for the learning rate.
            A_plus (float): Amplitude of the potentiation window.
            tau_plus (float): Time constant of the potentiation window.
            A_minus (float): Amplitude of the depression window.
            tau_minus (float): Time constant of the depression window.
        """
        self.eta_0 = eta_0
        self.gamma = gamma
        
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        
        # Precompute decay factors for the eligibility traces
        self.alpha_plus = np.exp(-1.0 / tau_plus)
        self.alpha_minus = np.exp(-1.0 / tau_minus)
        
        # Eligibility traces (initialized lazily on first update)
        self.trace_pre = None
        self.trace_post = None
        
    def reset_traces(self, batch_size, pre_dim, post_dim, device):
        """Initializes or resets the eligibility traces to zero."""
        self.trace_pre = torch.zeros(batch_size, pre_dim, device=device, dtype=torch.float32)
        self.trace_post = torch.zeros(batch_size, post_dim, device=device, dtype=torch.float32)
        
    def get_current_learning_rate(self, step_k):
        """
        Computes the exponentially decaying learning rate at inner-loop step k.
        eta_h(k) = eta_0 * gamma^k
        """
        return self.eta_0 * (self.gamma ** step_k)
        
    def compute_delta_W(self, S_pre, S_post):
        """
        Computes the local Hebbian weight update Delta W using eligibility traces.
        
        Args:
            S_pre (torch.Tensor): Presynaptic spikes. Shape: (batch, pre_dim)
            S_post (torch.Tensor): Postsynaptic spikes. Shape: (batch, post_dim)
            
        Returns:
            delta_W (torch.Tensor): The computed weight update. Shape: (pre_dim, post_dim)
        """
        # 1. Update eligibility traces
        self.trace_pre = self.alpha_plus * self.trace_pre + S_pre
        self.trace_post = self.alpha_minus * self.trace_post + S_post
        
        # 2. Potentiation term: A_+ * S_post @ x_pre^T
        # S_post: (B, post, 1), trace_pre: (B, 1, pre) -> (B, post, pre)
        delta_W_plus = self.A_plus * torch.bmm(S_post.unsqueeze(2), self.trace_pre.unsqueeze(1))
        
        # 3. Depression term: A_- * S_pre @ x_post^T
        # S_pre: (B, pre, 1), trace_post: (B, 1, post) -> (B, pre, post)
        delta_W_minus = self.A_minus * torch.bmm(S_pre.unsqueeze(2), self.trace_post.unsqueeze(1))
        
        # We want the final shape to be (pre_dim, post_dim).
        # delta_W_plus is (B, post, pre) -> transpose to (B, pre, post)
        delta_W_plus = delta_W_plus.transpose(1, 2)
        # delta_W_minus is already (B, pre, post)
        
        # 4. Combine and average over the batch
        delta_W = (delta_W_plus - delta_W_minus).mean(dim=0)
        
        return delta_W

    def update_spalrd_coefficients(self, C_m_param, z, S_m, step_k):
        """
        Applies the local Hebbian update to the SpaLRD coefficient matrix C_m.
        This strictly bypasses global backpropagation (no BPTT).
        
        Mathematical Formulation:
        Delta C_m = eta_h(k) * int_0^T int_0^T K(tau-tau') z(tau) S_m(tau')^T d tau d tau'
        Implemented via discrete eligibility traces.
        
        Args:
            C_m_param (nn.Parameter): The coefficient matrix C_m to be updated in-place.
            z (torch.Tensor): Projected presynaptic activity (Phi^T S_in). Shape: (batch, rank)
            S_m (torch.Tensor): Postsynaptic spike train. Shape: (batch, out_features)
            step_k (int): Current inner-loop step for the decaying learning rate.
        """
        # Ensure traces are initialized
        if self.trace_pre is None or self.trace_pre.shape != (z.size(0), z.size(1)):
            self.reset_traces(z.size(0), z.size(1), S_m.size(1), z.device)
            
        # Compute the local update (detached from the computational graph)
        with torch.no_grad():
            delta_C = self.compute_delta_W(S_pre=z, S_post=S_m)
            eta_h = self.get_current_learning_rate(step_k)
            
            # In-place update: C_m <- C_m + eta_h * Delta C_m
            C_m_param.data.add_(eta_h * delta_C)

    def update_edge_weights(self, W_E_buffer, S_i, S_j, step_k):
        """
        Applies the local Hebbian update to the MKG edge weights w_ij^E.
        This strictly bypasses global backpropagation (no BPTT).
        
        Mathematical Formulation:
        w_ij^E(t) = w_ij^E(t-1) + eta_E * sum_{t_i} sum_{t_j} K(t_i - t_j)
        
        Args:
            W_E_buffer (torch.Tensor): The edge weights buffer to be updated in-place. 
                                       Shape: (num_nodes, num_nodes)
            S_i (torch.Tensor): Presynaptic node activations. Shape: (batch, num_nodes)
            S_j (torch.Tensor): Postsynaptic node activations. Shape: (batch, num_nodes)
            step_k (int): Current inner-loop step for the decaying learning rate.
        """
        # Ensure traces are initialized
        if self.trace_pre is None or self.trace_pre.shape != (S_i.size(0), S_i.size(1)):
            self.reset_traces(S_i.size(0), S_i.size(1), S_j.size(1), S_i.device)
            
        # Compute the local update (detached from the computational graph)
        with torch.no_grad():
            delta_W = self.compute_delta_W(S_pre=S_i, S_post=S_j)
            eta_h = self.get_current_learning_rate(step_k)
            
            # In-place update: W_E <- W_E + eta_h * Delta W
            W_E_buffer.add_(eta_h * delta_W)