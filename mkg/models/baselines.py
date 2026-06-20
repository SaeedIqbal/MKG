import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ==========================================
# Minimal LIF Implementation for Baselines
# ==========================================
class SurrogateHeaviside(torch.autograd.Function):
    """Surrogate gradient for backpropagation through time (BPTT) in baselines."""
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        surrogate = 1.0 / (1.0 + 10.0 * torch.abs(input_tensor))**2
        return grad_input * surrogate

class LIFLayer(nn.Module):
    """Standard LIF layer for baseline SNNs."""
    def __init__(self, tau_m=20.0, v_th=1.0, dt=1.0):
        super().__init__()
        self.tau_m = tau_m
        self.v_th = v_th
        self.beta = np.exp(-dt / tau_m)
        self.u = None

    def reset(self):
        self.u = None

    def forward(self, x):
        if self.u is None:
            self.u = torch.zeros_like(x)
        self.u = self.beta * self.u + x
        spike = SurrogateHeaviside.apply(self.u - self.v_th)
        self.u = self.u - self.v_th * spike
        return spike


# ==========================================
# 1. HLML-SNN (Primary Target Baseline)
# ==========================================
class HLML_SNN(nn.Module):
    """
    HLML-SNN: Hebbian Learning-driven Meta-Learning SNN.
    
    Manuscript Critique Alignment:
    - Relies on a frozen pre-trained backbone (decoupling feature extraction from spikes).
    - Uses a static global meta-parameter initialization (theta_meta).
    - Employs rate-based Hebbian updates: Delta W = eta * mean(S_pre) * mean(S_post)^T.
    - Fails under domain shifts and OOD streams due to temporal manifold collapse.
    """
    def __init__(self, in_features, hidden_features, out_features, tau_m=20.0, v_th=1.0):
        super(HLML_SNN, self).__init__()
        
        # 1. Frozen Pre-trained Backbone (Simulating ResNet feature extractor)
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        for param in self.backbone.parameters():
            param.requires_grad = False  # Strictly frozen
            
        # 2. SNN Head
        self.lif = LIFLayer(tau_m=tau_m, v_th=v_th)
        self.fc = nn.Linear(hidden_features, out_features, bias=False)
        
        # 3. Static Meta-Parameter Initialization (theta_meta)
        # The manuscript notes HLML-SNN treats meta-learning as a static initializer.
        self.register_buffer('theta_meta', self.fc.weight.data.clone())
        
    def reset(self):
        self.lif.reset()
        
    def forward(self, x, time_steps=10):
        """Forward pass over multiple time steps."""
        self.reset()
        spikes_out = []
        
        # Extract static features (Bottleneck)
        features = self.backbone(x) 
        
        for t in range(time_steps):
            # Inject static features as constant input current
            membrane_drive = self.fc(features)
            spikes = self.lif(membrane_drive)
            spikes_out.append(spikes)
            
        return torch.stack(spikes_out, dim=1) # Shape: (batch, time, out_features)

    def rate_based_hebbian_update(self, S_pre_features, S_post_spikes, eta=0.01):
        """
        Implements the flawed rate-based Hebbian update defined in the manuscript:
        Delta W = eta * (1/T * sum S_pre) * (1/T * sum S_post)^T
        """
        # Compute mean firing rates over time (Temporal Collapse)
        rate_pre = S_pre_features.mean(dim=1)   # (batch, hidden)
        rate_post = S_post_spikes.mean(dim=1)   # (batch, out)
        
        # Batch outer product and mean
        # Delta W_ij = eta * mean_i(rate_post_i * rate_pre_j)
        delta_W = eta * torch.bmm(rate_post.unsqueeze(2), rate_pre.unsqueeze(1)).mean(dim=0)
        
        # Update weights (No gradient tracking, purely local Hebbian)
        with torch.no_grad():
            self.fc.weight.data += delta_W


# ==========================================
# 2. SNN-LoRA (Parameter-Efficient Baseline)
# ==========================================
class SNN_LoRA(nn.Module):
    """
    SNN-LoRA: Low-Rank Adaptation for Spiking Neural Networks.
    
    Manuscript Critique Alignment:
    - Decomposes static weight matrices: W_m = W_0 + B_m A_m.
    - Ignores the temporal spike manifold, operating on static spatial weights.
    - Requires global Backpropagation Through Time (BPTT) for temporal credit assignment,
      violating biological plausibility and incurring high computational costs.
    """
    def __init__(self, in_features, out_features, rank=8, tau_m=20.0, v_th=1.0):
        super(SNN_LoRA, self).__init__()
        
        # 1. Frozen Base Weight Matrix (W_0)
        self.W0 = nn.Linear(in_features, out_features, bias=False)
        self.W0.weight.requires_grad = False 
        
        # 2. Low-Rank Matrices (B_m and A_m)
        # W_m = W_0 + B_m @ A_m
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize B to zero so initial behavior matches W_0
        nn.init.zeros_(self.B.weight)
        
        # SNN Dynamics
        self.lif = LIFLayer(tau_m=tau_m, v_th=v_th)
        
    def reset(self):
        self.lif.reset()
        
    def forward(self, x, time_steps=10):
        """Forward pass requiring global BPTT through the SNN layer."""
        self.reset()
        spikes_out = []
        
        for t in range(time_steps):
            # Effective weight projection: W_m @ x = W_0 @ x + B @ A @ x
            current = self.W0(x) + self.B(self.A(x))
            spikes = self.lif(current)
            spikes_out.append(spikes)
            
        return torch.stack(spikes_out, dim=1)


# ==========================================
# 3. ALADE-SNN (Dynamic Expansion Baseline)
# ==========================================
class ALADE_SNN(nn.Module):
    """
    ALADE-SNN: Adaptive Logit Alignment in Dynamically Expandable SNNs.
    
    Manuscript Critique Alignment:
    - Mitigates interference by physically expanding the network topology per task.
    - Allocates fresh neuronal capacity (new parameters) for every new task.
    - Memory footprint scales linearly: ||Theta^(N)||_0 = sum ||Theta_k||_0 = O(N).
    - Directly violates the strict neuromorphic hardware constraint ||Theta||_0 <= C_max.
    """
    def __init__(self, in_features, shared_hidden_features, expansion_per_task=20):
        super(ALADE_SNN, self).__init__()
        
        # Shared feature extractor
        self.shared_backbone = nn.Sequential(
            nn.Linear(in_features, shared_hidden_features),
            nn.ReLU()
        )
        
        # Task-specific heads (Physically expanded topology)
        self.task_heads = nn.ModuleList()
        self.expansion_per_task = expansion_per_task
        self.num_tasks = 0
        
    def add_new_task(self, num_classes):
        """Physically expands the network by allocating fresh neurons for a new task."""
        # Allocate new physical parameters
        new_head = nn.Linear(self.shared_backbone[-1].out_features, num_classes)
        self.task_heads.append(new_head)
        self.num_tasks += 1
        print(f"[ALADE-SNN] Expanded topology. Total tasks: {self.num_tasks}. "
              f"Memory bloat: O({self.num_tasks})")
        
    def forward(self, x, task_id):
        """Routes input to the specific physical sub-network allocated for task_id."""
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} not allocated. Call add_new_task() first.")
            
        features = self.shared_backbone(x)
        return self.task_heads[task_id](features)


# ==========================================
# 4. CLS-ER (Replay-Based Baseline)
# ==========================================
class EpisodicMemoryBuffer:
    """
    Fixed-size episodic memory buffer using Reservoir Sampling.
    """
    def __init__(self, max_size, x_shape, device):
        self.max_size = max_size
        self.device = device
        self.x_buffer = torch.zeros(max_size, *x_shape, device=device)
        self.y_buffer = torch.zeros(max_size, device=device, dtype=torch.long)
        self.current_size = 0
        self.num_added = 0
        
    def add(self, x, y):
        """Adds samples to the buffer using reservoir sampling."""
        for i in range(x.size(0)):
            if self.current_size < self.max_size:
                self.x_buffer[self.current_size] = x[i]
                self.y_buffer[self.current_size] = y[i]
                self.current_size += 1
            else:
                # Reservoir sampling probability
                j = random.randint(0, self.num_added)
                if j < self.max_size:
                    self.x_buffer[j] = x[i]
                    self.y_buffer[j] = y[i]
            self.num_added += 1
            
    def sample(self, batch_size):
        if self.current_size == 0:
            return None, None
        k = min(batch_size, self.current_size)
        indices = random.sample(range(self.current_size), k)
        return self.x_buffer[indices], self.y_buffer[indices]

class CLS_ER(nn.Module):
    """
    CLS-ER: Continual Learning with Episodic Replay.
    
    Manuscript Critique Alignment:
    - Maintains a fixed-size episodic memory buffer M_buffer to rehearse past data.
    - Objective function: L_total = L_current + lambda * E_{(x,y) ~ M_buffer} [l(f(x), y)].
    - Incurs significant memory overhead and requires dense training cycles,
      diminishing the efficiency of sparse spiking computation.
    """
    def __init__(self, in_features, out_features, hidden_features=256, 
                 buffer_size=500, lambda_replay=0.5, tau_m=20.0, v_th=1.0):
        super(CLS_ER, self).__init__()
        
        # Standard SNN Classifier
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            LIFLayer(tau_m=tau_m, v_th=v_th),
            nn.Linear(hidden_features, out_features)
        )
        
        # Episodic Memory Buffer
        self.buffer_size = buffer_size
        self.lambda_replay = lambda_replay
        self.buffer = None # Initialized lazily on first batch
        
    def forward(self, x, time_steps=5):
        """Forward pass with temporal averaging for classification."""
        # Reset LIF states
        for module in self.net.modules():
            if isinstance(module, LIFLayer):
                module.reset()
                
        logits = 0
        for t in range(time_steps):
            current = self.net[0](x)
            spikes = self.net[1](current)
            logits += self.net[2](spikes)
            
        return logits / time_steps

    def compute_loss(self, x_current, y_current):
        """
        Computes the joint loss: L_total = L_current + lambda * L_replay.
        """
        # 1. Current Task Loss
        logits_current = self.forward(x_current)
        loss_current = F.cross_entropy(logits_current, y_current)
        
        # 2. Initialize buffer if needed
        if self.buffer is None:
            self.buffer = EpisodicMemoryBuffer(
                max_size=self.buffer_size, 
                x_shape=x_current.shape[1:], 
                device=x_current.device
            )
            
        # 3. Add current batch to episodic buffer
        self.buffer.add(x_current.detach(), y_current.detach())
        
        # 4. Replay Loss
        loss_total = loss_current
        if self.buffer.current_size > 0:
            x_replay, y_replay = self.buffer.sample(batch_size=x_current.size(0))
            if x_replay is not None:
                logits_replay = self.forward(x_replay)
                loss_replay = F.cross_entropy(logits_replay, y_replay)
                loss_total = loss_current + self.lambda_replay * loss_replay
                
        return loss_total