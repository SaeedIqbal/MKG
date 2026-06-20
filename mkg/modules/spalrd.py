import torch
import torch.nn as nn
import numpy as np

class SpaLRDLayer(nn.Module):
    """
    Spiking Low-Rank Dynamics (SpaLRD) Layer.
    
    Factorizes synaptic weights as W_m = Phi @ C_m, where:
    - Phi in R^{d_in x r} is a frozen, shared spiking basis (orthonormal: Phi^T Phi = I_r)
    - C_m in R^{r x d_out} is a task-specific coefficient matrix updated via local Hebbian rules.
    
    Mathematical Formulation (as defined in the manuscript):
    1. Basis Consolidation: Phi is derived as top-r left singular vectors of 
       Sigma_spike = E[S_in(t) S_in(t)^T], ensuring strict orthonormality.
    2. Projection: z(t) = Phi^T @ S_in(t)
    3. Hebbian Update (Discrete equivalent of the continuous integral):
       Delta C_m = eta_h(k) * int_0^T int_0^T K(tau-tau') z(tau) S_m(tau')^T d tau d tau'
       Implemented efficiently via recursive eligibility traces.
    4. Learning Rate Schedule: eta_h(k) = eta_0 * gamma^k
    5. Interference Bound: ||W_i - W_j||_F^2 = ||C_i - C_j||_F^2 (exact equality due to Phi^T Phi = I_r)
    """
    
    def __init__(self, in_features, out_features, rank,
                 A_plus=0.01, tau_plus=20.0, A_minus=0.012, tau_minus=20.0,
                 eta_0=0.01, gamma=0.95, dt=1.0):
        super(SpaLRDLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # 1. Shared Spiking Basis (Phi)
        # Initialized randomly, will be consolidated via input covariance SVD
        self.Phi = nn.Parameter(torch.randn(in_features, rank) / np.sqrt(in_features))
        self.Phi.requires_grad = False  # Strictly frozen: Delta Phi = 0
        
        # 2. Task-Specific Coefficient Matrix (C_m)
        self.C = nn.Parameter(torch.randn(rank, out_features) / np.sqrt(rank))
        
        # 3. STDP Kernel Parameters for Hebbian Update
        self.A_plus = A_plus
        self.A_minus = A_minus
        # Decay factors: alpha = exp(-dt / tau)
        self.alpha_plus = np.exp(-dt / tau_plus)
        self.alpha_minus = np.exp(-dt / tau_minus)
        
        # 4. Learning Rate Schedule Parameters: eta_h(k) = eta_0 * gamma^k
        self.eta_0 = eta_0
        self.gamma = gamma
        self.step_k = 0
        
        # Eligibility Traces (initialized lazily)
        self.trace_pre = None
        self.trace_post = None
        
    def consolidate_basis(self, S_in_data, top_r=None):
        """
        Derives Phi as top-r eigenvectors of the aggregated input spike covariance matrix.
        Ensures strict orthonormality: Phi^T @ Phi = I_r
        
        Args:
            S_in_data (torch.Tensor): Input spike data shape (batch, time, in_features)
            top_r (int): Target rank for the basis (defaults to self.rank)
        """
        if top_r is None:
            top_r = self.rank
            
        # Flatten to compute covariance: (batch*time, in_features)
        S_flat = S_in_data.reshape(-1, self.in_features)
        cov = torch.matmul(S_flat.T, S_flat) / S_flat.size(0)
        
        # Eigen-decomposition of symmetric covariance matrix
        # eigvecs are sorted by ascending eigenvalues
        _, eigvecs = torch.linalg.eigh(cov)
        
        # Assign top-r principal components to Phi
        self.Phi.data = eigvecs[:, -top_r:]
        self.Phi.requires_grad = False  # Enforce freezing
        
        # Verify orthonormality
        ortho_error = torch.norm(self.Phi.T @ self.Phi - torch.eye(top_r, device=self.Phi.device))
        print(f"[SpaLRD] Basis Phi consolidated. Shape: {self.Phi.shape} | "
              f"Orthonormality Error (should be ~0): {ortho_error:.4e}")
        
    def reset_traces(self, batch_size, device):
        """Reset eligibility traces for a new sequence or task."""
        self.trace_pre = torch.zeros(batch_size, self.rank, device=device, dtype=torch.float32)
        self.trace_post = torch.zeros(batch_size, self.out_features, device=device, dtype=torch.float32)
        
    def reset_learning_step(self):
        """Reset inner-loop Hebbian step counter for a new task."""
        self.step_k = 0
        
    def forward(self, S_in):
        """
        Forward pass: Projects input spikes onto the shared basis and applies coefficients.
        
        Mathematical Flow:
        1. z(t) = Phi^T @ S_in(t)
        2. I_proj(t) = C_m^T @ z(t)
        
        Args:
            S_in (torch.Tensor): Input spike train. Shape: (batch_size, in_features)
        Returns:
            I_proj (torch.Tensor): Projected synaptic drive. Shape: (batch_size, out_features)
            z (torch.Tensor): Projected presynaptic activity. Shape: (batch_size, rank)
        """
        batch_size = S_in.size(0)
        if self.trace_pre is None or self.trace_pre.size(0) != batch_size:
            self.reset_traces(batch_size, S_in.device)
            
        # Projection onto shared orthonormal basis
        z = torch.matmul(S_in, self.Phi)  # z(t) = Phi^T @ S_in(t)
        
        # Linear projection via task-specific coefficients
        I_proj = torch.matmul(z, self.C)  # I_proj(t) = C_m^T @ z(t)
        
        return I_proj, z
        
    def hebbian_update(self, z, S_m):
        """
        Applies local, event-driven Hebbian update to C_m.
        
        Discrete recursive equivalent of the manuscript's double integral:
        x_pre(t) = alpha_+ * x_pre(t-1) + z(t)
        x_post(t) = alpha_- * x_post(t-1) + S_m(t)
        Delta C_m = eta_h(k) * [ A_+ * S_m(t) @ x_pre(t)^T - A_- * z(t) @ x_post(t)^T ]
        
        Args:
            z (torch.Tensor): Projected presynaptic activity. Shape: (batch_size, rank)
            S_m (torch.Tensor): Postsynaptic spike train. Shape: (batch_size, out_features)
        Returns:
            eta_h (float): Current learning rate at step k
            delta_C (torch.Tensor): Computed coefficient update. Shape: (rank, out_features)
        """
        # Update eligibility traces
        self.trace_pre = self.alpha_plus * self.trace_pre + z
        self.trace_post = self.alpha_minus * self.trace_post + S_m
        
        # Compute current learning rate: eta_h(k) = eta_0 * gamma^k
        eta_h = self.eta_0 * (self.gamma ** self.step_k)
        
        # Potentiation term: A_+ * S_m(t) @ x_pre(t)^T
        delta_C_plus = self.A_plus * torch.bmm(S_m.unsqueeze(2), self.trace_pre.unsqueeze(1))
        
        # Depression term: A_- * z(t) @ x_post(t)^T
        delta_C_minus = self.A_minus * torch.bmm(z.unsqueeze(2), self.trace_post.unsqueeze(1))
        
        # Combine terms: delta_C shape -> (batch, out, rank) -> transpose to (batch, rank, out)
        delta_C = (delta_C_plus - delta_C_minus).transpose(1, 2)
        
        # Average over batch for stable update
        delta_C_mean = delta_C.mean(dim=0)
        
        # Apply update locally (bypasses backpropagation graph as intended)
        with torch.no_grad():
            self.C.data += eta_h * delta_C_mean
            
        self.step_k += 1
        return eta_h, delta_C_mean
        
    def compute_interference_bound(self, C_other):
        """
        Computes the exact Frobenius norm bound between two tasks.
        Since Phi^T @ Phi = I_r, ||W_i - W_j||_F^2 = ||C_i - C_j||_F^2 exactly.
        
        Args:
            C_other (torch.Tensor): Coefficient matrix of another task. Shape: (rank, out_features)
        Returns:
            bound (torch.Tensor): Interference bound value.
        """
        return torch.norm(self.C - C_other, p='fro')**2
        
    def get_effective_weight(self):
        """
        Reconstructs the full effective synaptic weight matrix for analysis: W_m = Phi @ C_m
        """
        return torch.matmul(self.Phi, self.C)
        
    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, frozen_basis=True')