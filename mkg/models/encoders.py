import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# Shared Surrogate Gradient
# ==========================================
class SurrogateHeaviside(torch.autograd.Function):
    """
    Surrogate gradient for the Heaviside step function Theta(x).
    Required for pre-training the encoders via Backpropagation Through Time (BPTT)
    before freezing them for the local Hebbian MKG training phase.
    """
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Fast sigmoid surrogate: 1 / (1 + 10|x|)^2
        surrogate = 1.0 / (1.0 + 10.0 * torch.abs(input_tensor))**2
        return grad_input * surrogate


# ==========================================
# 1. Spiking CNN for Vision Tasks
# ==========================================
class SpikingConvLayer(nn.Module):
    """
    Single Spiking Convolutional Layer with Alpha-Synaptic and LIF Membrane Dynamics.
    
    Mathematical Formulation:
    1. Synaptic Current: I_syn(t) = alpha * I_syn(t-1) + W_conv * S_in(t)
    2. Membrane Potential: u(t) = beta * u(t-1) + I_syn(t)
    3. Spike Generation: S_out(t) = Theta(u(t) - V_th)
    4. Soft Reset: u(t) = u(t) - V_th * S_out(t)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 tau_syn=5.0, tau_mem=20.0, v_th=1.0, dt=1.0):
        super(SpikingConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.tau_syn = tau_syn
        self.tau_mem = tau_mem
        self.v_th = v_th
        
        # Decay factors
        self.alpha = np.exp(-dt / tau_syn)
        self.beta = np.exp(-dt / tau_mem)
        
        # State variables (initialized lazily)
        self.I_syn = None
        self.u = None

    def reset_states(self, batch_size, H_out, W_out, device):
        """Initializes synaptic and membrane states."""
        self.I_syn = torch.zeros(batch_size, self.conv.out_channels, H_out, W_out, device=device)
        self.u = torch.zeros(batch_size, self.conv.out_channels, H_out, W_out, device=device)

    def forward(self, S_in):
        """
        Args:
            S_in (torch.Tensor): Input spike train. Shape: (batch, in_channels, H, W)
        Returns:
            S_out (torch.Tensor): Output spike train. Shape: (batch, out_channels, H_out, W_out)
        """
        batch_size = S_in.shape[0]
        device = S_in.device
        
        # Calculate output spatial dimensions
        H_in, W_in = S_in.shape[2], S_in.shape[3]
        H_out = (H_in + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) // self.conv.stride[0] + 1
        W_out = (W_in + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
        
        # Initialize states if needed
        if self.I_syn is None or self.I_syn.shape[0] != batch_size:
            self.reset_states(batch_size, H_out, W_out, device)

        # 1. Convolution & Synaptic Filtering
        self.I_syn = self.alpha * self.I_syn + self.conv(S_in)
        
        # 2. Membrane Integration
        self.u = self.beta * self.u + self.I_syn
        
        # 3. Spike Generation
        S_out = SurrogateHeaviside.apply(self.u - self.v_th)
        
        # 4. Soft Reset
        self.u = self.u - self.v_th * S_out
        
        return S_out


class SpikingCNN(nn.Module):
    """
    Spiking Convolutional Neural Network Encoder for Vision Tasks.
    Converts static images into a sequence of flattened spikes S_enc(t) in {0, 1}^{d_in}.
    
    Mathematical Formulation:
    S_enc(t) = Flatten( Conv_L( ... Conv_1( Poisson(x) ) ) )
    """
    def __init__(self, in_channels=3, img_size=32, base_channels=32, num_layers=3,
                 tau_syn=5.0, tau_mem=20.0, v_th=1.0):
        super(SpikingCNN, self).__init__()
        
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            layers.append(SpikingConvLayer(current_channels, out_channels, kernel_size=3, 
                                           stride=2, padding=1, tau_syn=tau_syn, 
                                           tau_mem=tau_mem, v_th=v_th))
            current_channels = out_channels
            
        self.layers = nn.ModuleList(layers)
        
        # Calculate final flattened dimension d_in for the MKG core
        self.d_in = self._get_flattened_dim(img_size, in_channels)
        
    def _get_flattened_dim(self, img_size, in_channels):
        """Helper to compute the flattened output dimension d_in."""
        dummy = torch.zeros(1, in_channels, img_size, img_size)
        for layer in self.layers:
            dummy = layer(dummy)
        return dummy.view(1, -1).shape[1]

    def reset_states(self):
        """Clears internal states for a new sequence."""
        for layer in self.layers:
            layer.I_syn = None
            layer.u = None

    def forward(self, x, time_steps=10):
        """
        Args:
            x (torch.Tensor): Static input image. Shape: (batch, C, H, W)
            time_steps (int): Number of time steps to simulate.
        Returns:
            S_enc_seq (torch.Tensor): Sequence of flattened spikes. 
                                      Shape: (batch, time_steps, d_in)
        """
        self.reset_states()
        batch_size = x.shape[0]
        
        # Normalize x to [0, 1] for Poisson firing rates
        x_norm = torch.clamp(x, 0.0, 1.0)
        
        S_enc_seq = []
        
        for t in range(time_steps):
            # Generate Poisson spikes based on pixel intensity
            S_in = (torch.rand_like(x) < x_norm).float()
            
            # Pass through conv layers
            S_out = S_in
            for layer in self.layers:
                S_out = layer(S_out)
                
            # Flatten spatial dimensions to create the input spike train for MKG
            S_flat = S_out.view(batch_size, -1)
            S_enc_seq.append(S_flat)
            
        return torch.stack(S_enc_seq, dim=1)


# ==========================================
# 2. Neuromorphic Text Encoder
# ==========================================
class NeuromorphicTextEncoder(nn.Module):
    """
    Lightweight, fixed Neuromorphic Text Encoder.
    Converts sequences of word indices into spike trains S_text(t) in {0, 1}^{d_in}.
    
    Mathematical Formulation:
    1. Embedding Lookup: E_t = Embed(word_t)
    2. Membrane Potential: u(t) = beta * u(t-1) + E_t
    3. Spike Generation: S_text(t) = Theta(u(t) - V_th)
    
    The embedding matrix is strictly frozen (Delta W_emb = 0) to ensure 
    stable, cross-task canonical text representations, acting as the 
    text-equivalent of the frozen spiking basis Phi in vision tasks.
    """
    def __init__(self, vocab_size, embed_dim, tau_mem=20.0, v_th=0.5, dt=1.0):
        super(NeuromorphicTextEncoder, self).__init__()
        
        # Fixed Embedding Matrix (Strictly Frozen)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.requires_grad = False 
        
        # Initialize embeddings with a fixed distribution
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        
        self.tau_mem = tau_mem
        self.v_th = v_th
        self.beta = np.exp(-dt / tau_mem)
        self.d_in = embed_dim
        
        self.u = None

    def reset_states(self, batch_size, device):
        """Initializes membrane potential to resting state (0)."""
        self.u = torch.zeros(batch_size, self.d_in, device=device)

    def forward(self, word_indices):
        """
        Args:
            word_indices (torch.Tensor): Sequence of word indices. 
                                         Shape: (batch, seq_len)
        Returns:
            S_text_seq (torch.Tensor): Sequence of text spikes. 
                                       Shape: (batch, seq_len, d_in)
        """
        batch_size, seq_len = word_indices.shape
        device = word_indices.device
        
        self.reset_states(batch_size, device)
        
        S_text_seq = []
        
        for t in range(seq_len):
            # 1. Embedding Lookup
            word_t = word_indices[:, t]
            E_t = self.embedding(word_t) # Shape: (batch, embed_dim)
            
            # 2. Membrane Integration
            self.u = self.beta * self.u + E_t
            
            # 3. Spike Generation
            S_t = SurrogateHeaviside.apply(self.u - self.v_th)
            
            # 4. Soft Reset
            self.u = self.u - self.v_th * S_t
            
            S_text_seq.append(S_t)
            
        return torch.stack(S_text_seq, dim=1)