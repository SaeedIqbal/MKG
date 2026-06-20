import torch
import torch.nn as nn
import numpy as np

class MetaPlasticSTDP(nn.Module):
    """
    Implements fine-grained, differentiable Spike-Timing-Dependent Plasticity (STDP) 
    with Meta-Plasticity for the Meta-Knowledge Graph (MKG) edges.
    
    This module computes the edge weight updates based on the precise temporal 
    differences between pre- and post-synaptic spikes, replacing rate-based 
    approximations with millisecond-precise temporal dynamics.
    
    The continuous weight dynamics are defined by the convolution of spike trains 
    with an asymmetric STDP kernel W(Delta t):
    
    dw_ij/dt = eta * [ S_i(t) * integral(W_-(Delta t) S_j(t - Delta t)) 
                     + S_j(t) * integral(W_+(Delta t) S_i(t - Delta t)) ]
                     
    where:
    W_+(Delta t) = A_+ * exp(-Delta t / tau_+)  for Delta t > 0 (Potentiation)
    W_-(Delta t) = -A_- * exp(Delta t / tau_-)  for Delta t < 0 (Depression)
    
    In discrete time, this is efficiently computed using exponential eligibility traces:
    x_pre(t) = exp(-1/tau_+) * x_pre(t-1) + S_pre(t)
    x_post(t) = exp(-1/tau_-) * x_post(t-1) + S_post(t)
    
    Delta W(t) = A_+ * S_post(t) * x_pre(t)^T - A_- * S_pre(t) * x_post(t)^T
    W(t) = W(t-1) + eta * Delta W(t)
    
    Crucially, the STDP meta-parameters phi_ij = {A_+, tau_+, A_-, tau_-} are 
    implemented as learnable nn.Parameters (using log-space to ensure strict positivity).
    This allows the meta-learner to optimize the plasticity rules via gradient 
    descent on the query loss, utilizing the chain rule through the unrolled 
    Hebbian dynamics, exactly as formulated in the manuscript:
    
    phi_ij <- phi_ij - eta_meta * sum_k ( dL_m/dW_ij(k) * integral( grad_phi W_phi(Delta t) S_i S_j ) )
    """
    def __init__(self, in_features, out_features, A_plus=0.01, tau_plus=20.0, 
                 A_minus=0.012, tau_minus=20.0, eta=0.01, learnable_params=True):
        super(MetaPlasticSTDP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eta = eta
        
        # STDP Meta-parameters (phi_ij)
        # Initialized in log-space to ensure strict positivity during optimization
        if learnable_params:
            self.log_A_plus = nn.Parameter(torch.tensor(np.log(A_plus)))
            self.log_tau_plus = nn.Parameter(torch.tensor(np.log(tau_plus)))
            self.log_A_minus = nn.Parameter(torch.tensor(np.log(A_minus)))
            self.log_tau_minus = nn.Parameter(torch.tensor(np.log(tau_minus)))
        else:
            self.register_buffer('log_A_plus', torch.tensor(np.log(A_plus)))
            self.register_buffer('log_tau_plus', torch.tensor(np.log(tau_plus)))
            self.register_buffer('log_A_minus', torch.tensor(np.log(A_minus)))
            self.register_buffer('log_tau_minus', torch.tensor(np.log(tau_minus)))
            
        self.trace_pre = None
        self.trace_post = None

    def get_params(self):
        """Returns the actual STDP parameters (ensuring positivity via exp)."""
        A_plus = torch.exp(self.log_A_plus)
        tau_plus = torch.exp(self.log_tau_plus)
        A_minus = torch.exp(self.log_A_minus)
        tau_minus = torch.exp(self.log_tau_minus)
        return A_plus, tau_plus, A_minus, tau_minus

    def reset_states(self, batch_size, device):
        """Initializes or resets the eligibility traces."""
        self.trace_pre = torch.zeros(batch_size, self.in_features, device=device, dtype=torch.float32)
        self.trace_post = torch.zeros(batch_size, self.out_features, device=device, dtype=torch.float32)

    def forward(self, S_pre, S_post, W):
        """
        Computes one step of STDP weight update.
        
        Args:
            S_pre (torch.Tensor): Presynaptic spikes (S_j). Shape: (batch, in_features)
            S_post (torch.Tensor): Postsynaptic spikes (S_i). Shape: (batch, out_features)
            W (torch.Tensor): Current edge weights (w_ij). Shape: (in_features, out_features)
            
        Returns:
            W_new (torch.Tensor): Updated edge weights. Shape: (in_features, out_features)
        """
        batch_size = S_pre.size(0)
        if self.trace_pre is None or self.trace_pre.size(0) != batch_size:
            self.reset_states(batch_size, S_pre.device)
            
        A_plus, tau_plus, A_minus, tau_minus = self.get_params()
        
        # Decay factors for the eligibility traces
        # alpha_+ = exp(-1 / tau_+)
        # alpha_- = exp(-1 / tau_-)
        alpha_plus = torch.exp(-1.0 / tau_plus)
        alpha_minus = torch.exp(-1.0 / tau_minus)
        
        # Update eligibility traces
        # x_pre(t) = alpha_+ * x_pre(t-1) + S_pre(t)
        self.trace_pre = alpha_plus * self.trace_pre + S_pre
        # x_post(t) = alpha_- * x_post(t-1) + S_post(t)
        self.trace_post = alpha_minus * self.trace_post + S_post
        
        # Compute weight update Delta W
        # Delta W_ij = A_+ * S_post_i * x_pre_j - A_- * S_pre_j * x_post_i
        # Using batch matrix multiplication for the outer product:
        # S_post: (B, out, 1), trace_pre: (B, 1, in) -> (B, out, in)
        delta_W_plus = A_plus * torch.bmm(S_post.unsqueeze(2), self.trace_pre.unsqueeze(1))
        delta_W_minus = A_minus * torch.bmm(self.trace_post.unsqueeze(2), S_pre.unsqueeze(1))
        
        # The manuscript defines W as (in_features, out_features).
        # The outer product S_post (out) x trace_pre (in) gives (out, in).
        # We transpose it to match W's shape (in, out).
        delta_W_plus = delta_W_plus.transpose(1, 2)
        delta_W_minus = delta_W_minus.transpose(1, 2)
        
        delta_W = delta_W_plus - delta_W_minus
        
        # Average over batch to get the weight update
        delta_W_mean = delta_W.mean(dim=0)
        
        # Update weights: W(t) = W(t-1) + eta * Delta W(t)
        W_new = W + self.eta * delta_W_mean
        
        return W_new