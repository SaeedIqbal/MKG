import torch
import torch.nn as nn
import numpy as np

class SurrogateHeaviside(torch.autograd.Function):
    """
    Surrogate gradient for the Heaviside step function.
    While the MKG framework primarily relies on local Hebbian updates, 
    this surrogate gradient is included to allow standard gradient-based 
    pre-training or benchmarking if required.
    """
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Fast sigmoid surrogate gradient: 1 / (1 + 10|x|)^2
        surrogate = 1.0 / (1.0 + 10.0 * torch.abs(input_tensor))**2
        return grad_input * surrogate


class AlphaSynapticFilter(nn.Module):
    """
    Implements the exponential post-synaptic potential (PSP) filtering 
    using an alpha-function kernel: alpha(t) = (t / tau_s) * exp(1 - t / tau_s).
    
    In discrete time, this is efficiently computed using two cascaded 
    first-order low-pass filters (exponential decays), avoiding the need 
    for explicit convolution over time.
    """
    def __init__(self, tau_s=5.0, dt=1.0):
        super(AlphaSynapticFilter, self).__init__()
        self.tau_s = tau_s
        self.dt = dt
        # Decay factor for the cascaded exponentials
        self.alpha = np.exp(-dt / tau_s)
        self.x = None
        self.I_syn = None

    def reset_states(self, shape, device):
        """Initializes or resets the synaptic hidden states."""
        self.x = torch.zeros(shape, device=device, dtype=torch.float32)
        self.I_syn = torch.zeros(shape, device=device, dtype=torch.float32)

    def forward(self, S_in):
        """
        Args:
            S_in (torch.Tensor): Presynaptic spike train at time t.
        Returns:
            I_syn (torch.Tensor): Filtered synaptic current.
        """
        if self.x is None or self.x.shape != S_in.shape:
            self.reset_states(S_in.shape, S_in.device)
            
        # First cascaded filter
        self.x = self.alpha * self.x + S_in
        # Second cascaded filter (yields the alpha-function temporal shape)
        self.I_syn = self.alpha * self.I_syn + self.x
        
        return self.I_syn


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron dynamics.
    
    Discrete-time membrane potential evolution:
    u(t) = beta * u(t-1) + R_m * I_syn(t)
    
    Spike generation and soft reset:
    S(t) = Theta(u(t) - V_th)
    u(t) = u(t) - V_th * S(t)
    """
    def __init__(self, tau_m=20.0, v_th=1.0, v_rest=0.0, r_m=1.0, dt=1.0, surrogate_gradient=True):
        super(LIFNeuron, self).__init__()
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_rest = v_rest
        self.r_m = r_m
        self.dt = dt
        
        # Membrane decay factor (beta = exp(-dt / tau_m))
        self.beta = np.exp(-dt / tau_m)
        
        self.surrogate_gradient = surrogate_gradient
        if surrogate_gradient:
            self.spike_fn = SurrogateHeaviside.apply
        else:
            self.spike_fn = lambda x: torch.heaviside(x, torch.zeros_like(x))
            
        self.u = None

    def reset_states(self, shape, device):
        """Initializes or resets the membrane potential."""
        self.u = torch.full(shape, self.v_rest, device=device, dtype=torch.float32)

    def forward(self, I_syn):
        """
        Args:
            I_syn (torch.Tensor): Synaptic current at time t.
        Returns:
            S_out (torch.Tensor): Output spike train at time t.
        """
        if self.u is None or self.u.shape != I_syn.shape:
            self.reset_states(I_syn.shape, I_syn.device)
            
        # 1. Membrane Potential Integration
        self.u = self.beta * self.u + self.r_m * I_syn
        
        # 2. Spike Generation (Heaviside step function)
        S_out = self.spike_fn(self.u - self.v_th)
        
        # 3. Soft Reset (u(t) = u(t) - V_th * S(t))
        self.u = self.u - self.v_th * S_out
        
        return S_out


class SpikingDenseLayer(nn.Module):
    """
    Complete Spiking Dense Layer integrating Synaptic Filtering, 
    Linear Projection (supporting SpaLRD factorization W = Phi @ C), 
    and LIF Neuron Dynamics.
    
    This directly implements the manuscript's formulation:
    1. I_syn(t) = alpha * I_syn(t-1) + S_in(t)  [Alpha Filtering]
    2. z(t) = Phi^T I_syn(t)                    [SpaLRD Projection]
    3. I_proj(t) = C^T z(t)                     [Coefficient Adaptation]
    4. u(t) = beta u(t-1) + R_m I_proj(t)       [Membrane Integration]
    5. S_out(t) = Theta(u(t) - V_th)            [Spike Generation]
    """
    def __init__(self, in_features, out_features, tau_s=5.0, tau_m=20.0, 
                 v_th=1.0, r_m=1.0, rank=None, surrogate_gradient=True):
        super(SpikingDenseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # 1. Synaptic Filter
        self.syn_filter = AlphaSynapticFilter(tau_s=tau_s)
        
        # 2. Linear Projection (Weights)
        if rank is not None and rank < min(in_features, out_features):
            # Spiking Low-Rank Dynamics (SpaLRD) factorization: W = Phi @ C
            # Phi is the frozen shared spiking basis (in_features x rank)
            # C is the task-specific coefficient matrix (rank x out_features)
            self.use_spalrd = True
            self.Phi = nn.Parameter(torch.randn(in_features, rank) / np.sqrt(in_features))
            self.C = nn.Parameter(torch.randn(rank, out_features) / np.sqrt(rank))
            
            # CRITICAL: Freeze Phi after initialization (as per manuscript: Delta Phi = 0)
            self.Phi.requires_grad = False 
        else:
            # Standard dense weight matrix (if SpaLRD is not used)
            self.use_spalrd = False
            self.weight = nn.Parameter(torch.randn(in_features, out_features) / np.sqrt(in_features))
            
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # 3. LIF Neuron
        self.neuron = LIFNeuron(tau_m=tau_m, v_th=v_th, r_m=r_m, surrogate_gradient=surrogate_gradient)
        
    def reset_states(self, batch_size, device):
        """Resets all internal states (synaptic traces and membrane potentials)."""
        shape = (batch_size, self.out_features)
        self.syn_filter.reset_states((batch_size, self.in_features), device)
        self.neuron.reset_states(shape, device)
        
    def forward(self, S_in):
        """
        Args:
            S_in (torch.Tensor): Input spike train. Shape: (batch_size, in_features)
        Returns:
            S_out (torch.Tensor): Output spike train. Shape: (batch_size, out_features)
        """
        batch_size = S_in.size(0)
        
        # Initialize states on the first forward pass
        if self.syn_filter.x is None:
            self.reset_states(batch_size, S_in.device)
            
        # 1. Synaptic Filtering (Alpha-function kernel)
        I_syn = self.syn_filter(S_in)
        
        # 2. Linear Projection
        if self.use_spalrd:
            # W = Phi @ C  =>  W^T = C^T @ Phi^T
            # z(t) = Phi^T @ I_syn(t)
            z = torch.matmul(I_syn, self.Phi) 
            # I_proj(t) = C^T @ z(t)
            I_proj = torch.matmul(z, self.C)  
        else:
            I_proj = torch.matmul(I_syn, self.weight)
            
        I_proj = I_proj + self.bias
        
        # 3. LIF Dynamics & Spike Generation
        S_out = self.neuron(I_proj)
        
        return S_out

    def extra_repr(self):
        spalrd_str = f", rank={self.rank} (SpaLRD)" if self.use_spalrd else ""
        return f'in_features={self.in_features}, out_features={self.out_features}{spalrd_str}'