import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HardQuantizationOperator(nn.Module):
    """
    Implements the hard quantization operator H_{epsilon_q} as defined in the manuscript:
    H_{epsilon_q}(x) = x * I(|x| > epsilon_q)
    where I is the indicator function. This induces structural sparsity by resetting 
    negligible trace values to exactly zero.
    """
    def __init__(self, epsilon_q=1e-4):
        super(HardQuantizationOperator, self).__init__()
        self.epsilon_q = epsilon_q

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Quantized tensor where values with absolute magnitude 
                          <= epsilon_q are set to zero.
        """
        mask = (torch.abs(x) > self.epsilon_q).float()
        return x * mask


class CompressedEligibilityTrace(nn.Module):
    """
    Models the compressed eligibility trace E_{ij}(t) as a leaky integrator 
    with a hard quantization operator, replacing continuous spike-trace storage.
    
    Mathematical Formulation:
    E_{ij}(t) = H_{epsilon_q}( lambda * E_{ij}(t-1) + S_i(t) * S_j(t) )
    where lambda < 1 is the decay factor.
    """
    def __init__(self, num_edges, lambda_decay=0.95, epsilon_q=1e-4):
        super(CompressedEligibilityTrace, self).__init__()
        self.num_edges = num_edges
        self.lambda_decay = lambda_decay
        self.quantizer = HardQuantizationOperator(epsilon_q=epsilon_q)
        
        # Initialize eligibility traces (lazily allocated on first update)
        self.E = None

    def reset(self, device):
        """Initializes or resets the eligibility traces to zero."""
        self.E = torch.zeros(self.num_edges, device=device, dtype=torch.float32)

    def update(self, S_i, S_j):
        """
        Updates the eligibility trace for a given time step.
        
        Args:
            S_i (torch.Tensor): Presynaptic spike train/activation. Shape: (num_edges,)
            S_j (torch.Tensor): Postsynaptic spike train/activation. Shape: (num_edges,)
        Returns:
            torch.Tensor: Updated and quantized eligibility trace.
        """
        if self.E is None:
            self.reset(S_i.device)
            
        # Leaky integration: E(t) = lambda * E(t-1) + S_i(t) * S_j(t)
        self.E = self.lambda_decay * self.E + (S_i * S_j)
        
        # Hard quantization: H_{epsilon_q}(E(t))
        self.E = self.quantizer(self.E)
        
        return self.E


class FisherInformationUtility(nn.Module):
    """
    Computes the edge utility score U_{ij} approximated via the diagonal of the 
    Fisher Information Matrix (FIM) over previously learned tasks D_old.
    
    Mathematical Formulation:
    U_{ij} = E_{(x,y) ~ D_old} [ ( d log p(y|x, G) / d w_{ij}^E )^2 ]
    """
    def __init__(self):
        super(FisherInformationUtility, self).__init__()

    def compute_utility(self, model, dataloader, edge_weights_param):
        """
        Computes the FIM diagonal approximation for the given edge weights.
        
        Args:
            model (nn.Module): The Meta-Knowledge Graph model.
            dataloader (DataLoader): DataLoader for the previously learned tasks D_old.
            edge_weights_param (nn.Parameter): The edge weights w_{ij}^E to compute utility for.
            
        Returns:
            torch.Tensor: Utility scores U_{ij} for each edge. Shape: matches edge_weights_param
        """
        device = edge_weights_param.device
        fim_diag = torch.zeros_like(edge_weights_param)
        num_samples = 0
        
        model.eval()
        # Disable gradient tracking for the model parameters, except for the edge weights
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            num_samples += batch_size
            
            # Forward pass
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            
            # Extract log probability of the true class
            target_log_probs = log_probs.gather(1, y.unsqueeze(1)).squeeze()
            
            # Compute gradients of the log-likelihood w.r.t the edge weights
            grads = torch.autograd.grad(
                outputs=target_log_probs.sum(), 
                inputs=edge_weights_param, 
                retain_graph=False,
                create_graph=False
            )[0]
            
            # Accumulate squared gradients (Element-wise)
            fim_diag += grads ** 2
            
        # Compute expectation (mean over the dataset)
        fim_diag /= num_samples
        
        return fim_diag


class ActiveForgettingGate(nn.Module):
    """
    Implements the bio-inspired forgetting gate mechanism formulated as a 
    constrained optimization problem to prune redundant transfer pathways.
    
    Mathematical Formulation:
    min_M sum_{e_ij in E} (1 - M_ij) U_ij
    subject to: ||M||_0 <= rho_max * |E|
    
    This is solved efficiently by keeping the top (rho_max * |E|) edges 
    with the highest utility U_ij.
    """
    def __init__(self, rho_max=0.2):
        super(ActiveForgettingGate, self).__init__()
        self.rho_max = rho_max

    def compute_pruning_mask(self, utility_scores):
        """
        Computes the binary pruning mask M by solving the constrained optimization.
        
        Args:
            utility_scores (torch.Tensor): Edge utility scores U_ij.
            
        Returns:
            torch.Tensor: Binary pruning mask M.
        """
        num_edges = utility_scores.numel()
        num_to_keep = int(self.rho_max * num_edges)
        
        if num_to_keep >= num_edges:
            return torch.ones_like(utility_scores)
            
        if num_to_keep <= 0:
            return torch.zeros_like(utility_scores)
            
        # To keep the top `num_to_keep` elements, we find the threshold value.
        # torch.kthvalue finds the k-th smallest element.
        # We want the (num_edges - num_to_keep)-th smallest element to be the threshold.
        k = num_edges - num_to_keep
        
        # Flatten to ensure 1D tensor for kthvalue
        flat_scores = utility_scores.flatten()
        threshold, _ = torch.kthvalue(flat_scores, k)
        
        # Create binary mask: 1 if utility >= threshold, else 0
        mask = (flat_scores >= threshold).float()
        
        # Reshape back to original shape if it was multi-dimensional
        return mask.reshape(utility_scores.shape)

    def apply_pruning(self, edge_weights, utility_scores):
        """
        Applies the pruning mask to the edge weights, effectively forgetting 
        the least useful connections.
        
        Args:
            edge_weights (torch.Tensor): Current edge weights.
            utility_scores (torch.Tensor): Edge utility scores.
            
        Returns:
            torch.Tensor: Pruned edge weights.
            torch.Tensor: The applied binary mask.
        """
        mask = self.compute_pruning_mask(utility_scores)
        pruned_weights = edge_weights * mask
        return pruned_weights, mask


class LifelongMemoryManager(nn.Module):
    """
    Orchestrates the active forgetting and eligibility trace compression for the 
    Meta-Knowledge Graph edges, ensuring the physical memory footprint remains 
    strictly bounded during lifelong deployment.
    """
    def __init__(self, num_edges, lambda_decay=0.95, epsilon_q=1e-4, rho_max=0.2):
        super(LifelongMemoryManager, self).__init__()
        self.num_edges = num_edges
        
        # Compressed Eligibility Traces
        self.eligibility_traces = CompressedEligibilityTrace(
            num_edges=num_edges, 
            lambda_decay=lambda_decay, 
            epsilon_q=epsilon_q
        )
        
        # Fisher Information Utility Calculator
        self.fim_utility = FisherInformationUtility()
        
        # Active Forgetting Gate
        self.forgetting_gate = ActiveForgettingGate(rho_max=rho_max)
        
    def update_traces(self, S_pre, S_post):
        """
        Updates the compressed eligibility traces for the current time step.
        
        Args:
            S_pre (torch.Tensor): Presynaptic spikes. Shape: (num_edges,)
            S_post (torch.Tensor): Postsynaptic spikes. Shape: (num_edges,)
        Returns:
            torch.Tensor: Updated eligibility traces.
        """
        return self.eligibility_traces.update(S_pre, S_post)
        
    def perform_active_forgetting(self, model, old_task_dataloader, edge_weights_param):
        """
        Executes the active forgetting mechanism at the end of a task or periodically.
        Computes FIM utility and prunes the graph to maintain the memory bound.
        
        Args:
            model (nn.Module): The MKG model.
            old_task_dataloader (DataLoader): DataLoader for D_old.
            edge_weights_param (nn.Parameter): The edge weights to be pruned.
            
        Returns:
            torch.Tensor: Pruned edge weights.
            torch.Tensor: The applied pruning mask.
            torch.Tensor: The computed utility scores.
        """
        # 1. Compute edge utility scores via FIM diagonal approximation
        utility_scores = self.fim_utility.compute_utility(
            model, old_task_dataloader, edge_weights_param
        )
        
        # 2. Apply active forgetting gate to prune low-utility edges
        pruned_weights, mask = self.forgetting_gate.apply_pruning(
            edge_weights_param.data, utility_scores
        )
        
        # 3. Update the parameter data in-place to enforce structural sparsity
        edge_weights_param.data.copy_(pruned_weights)
        
        return pruned_weights, mask, utility_scores

    def get_memory_footprint(self, edge_weights):
        """
        Calculates the current physical memory footprint of the edges (L0 pseudo-norm).
        
        Args:
            edge_weights (torch.Tensor): Current edge weights.
        Returns:
            int: Number of non-zero edges (active parameters).
        """
        return torch.count_nonzero(edge_weights).item()