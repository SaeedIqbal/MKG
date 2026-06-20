import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaLearner:
    """
    Outer-loop Meta-Learner for Edge Plasticity.
    
    Optimizes the parameters of the synaptic plasticity rules (STDP meta-parameters phi_ij)
    rather than the synaptic weights themselves.
    
    Mathematical Formulation:
    Inner-loop Hebbian dynamics (unrolled for K steps):
    w_ij^E(k) = w_ij^E(k-1) + int_0^T W_{phi_ij}(Delta t) S_i^{(k)}(t) S_j^{(k)}(t-Delta t) dDelta t
    
    Outer-loop meta-gradient update:
    phi_ij <- phi_ij - eta_meta * sum_{k=1}^K ( dL_m/dw_ij^E(k) * int_0^T nabla_{phi_ij} W_{phi_ij}(Delta t) S_i^{(k)}(t) S_j^{(k)}(t-Delta t) dDelta t )
    
    Note: By unrolling the inner loop in the computational graph using differentiable 
    operations, standard backpropagation (query_loss.backward()) automatically computes 
    this exact chain rule gradient without requiring manual derivative calculations.
    """
    
    def __init__(self, edge_stdp_module, meta_optimizer, K_inner_steps=5):
        """
        Args:
            edge_stdp_module (nn.Module): The MetaPlasticSTDP module containing learnable phi_ij.
            meta_optimizer (torch.optim.Optimizer): Optimizer for the meta-parameters (e.g., Adam).
            K_inner_steps (int): Number of inner-loop Hebbian steps to unroll (K).
        """
        self.edge_stdp = edge_stdp_module
        self.meta_optimizer = meta_optimizer
        self.K_inner_steps = K_inner_steps
        
    def unroll_inner_loop(self, S_i_seq, S_j_seq, initial_edge_weights):
        """
        Unrolls the inner-loop Hebbian dynamics for K steps.
        
        Args:
            S_i_seq (list of torch.Tensor): Presynaptic node activations for each step. 
                                            Shape of each: (batch, num_nodes)
            S_j_seq (list of torch.Tensor): Postsynaptic node activations for each step. 
                                            Shape of each: (batch, num_nodes)
            initial_edge_weights (torch.Tensor): Initial edge weights w_ij^E(0). 
                                                 Shape: (num_nodes, num_nodes)
            
        Returns:
            final_edge_weights (torch.Tensor): Edge weights after K steps.
            w_seq (list of torch.Tensor): Sequence of edge weights for gradient tracking.
        """
        w_k = initial_edge_weights
        
        # Extract differentiable STDP meta-parameters (phi_ij)
        # Using log-space ensures strict positivity during gradient descent
        A_plus = torch.exp(self.edge_stdp.log_A_plus)
        tau_plus = torch.exp(self.edge_stdp.log_tau_plus)
        A_minus = torch.exp(self.edge_stdp.log_A_minus)
        tau_minus = torch.exp(self.edge_stdp.log_tau_minus)
        
        # Decay factors for eligibility traces
        alpha_plus = torch.exp(-1.0 / tau_plus)
        alpha_minus = torch.exp(-1.0 / tau_minus)
        
        batch_size, num_nodes = S_i_seq[0].shape
        device = S_i_seq[0].device
        
        # Initialize eligibility traces
        trace_pre = torch.zeros(batch_size, num_nodes, device=device)
        trace_post = torch.zeros(batch_size, num_nodes, device=device)
        
        w_seq = [w_k]
        
        for k in range(self.K_inner_steps):
            S_pre = S_i_seq[k]  # Shape: (batch, num_nodes)
            S_post = S_j_seq[k] # Shape: (batch, num_nodes)
            
            # Update eligibility traces (Differentiable w.r.t phi_ij via alpha)
            trace_pre = alpha_plus * trace_pre + S_pre
            trace_post = alpha_minus * trace_post + S_post
            
            # Compute delta_W using batch matrix multiplication
            # S_post: (B, N, 1), trace_pre: (B, 1, N) -> (B, N, N)
            delta_W_plus = A_plus * torch.bmm(S_post.unsqueeze(2), trace_pre.unsqueeze(1))
            delta_W_minus = A_minus * torch.bmm(trace_post.unsqueeze(2), S_pre.unsqueeze(1))
            
            # Transpose to match edge_weights shape (N, N) and average over batch
            delta_W_plus = delta_W_plus.transpose(1, 2)
            delta_W_minus = delta_W_minus.transpose(1, 2)
            delta_W = (delta_W_plus - delta_W_minus).mean(dim=0)
            
            # Update weights: w(k) = w(k-1) + eta * delta_W
            # This step connects w(k) to phi_ij in the computational graph
            w_k = w_k + self.edge_stdp.eta * delta_W
            
            w_seq.append(w_k)
            
        return w_k, w_seq

    def compute_query_loss(self, query_logits, query_targets):
        """
        Computes the query loss L_m for the meta-update.
        """
        return F.cross_entropy(query_logits, query_targets)

    def meta_step(self, S_i_seq, S_j_seq, initial_edge_weights, query_logits, query_targets):
        """
        Performs one complete meta-learning step.
        1. Unrolls the inner-loop Hebbian dynamics.
        2. Computes the query loss using the final edge weights.
        3. Backpropagates the query loss to update the meta-parameters phi_ij.
        
        Args:
            S_i_seq (list): Presynaptic node activations.
            S_j_seq (list): Postsynaptic node activations.
            initial_edge_weights (torch.Tensor): Initial edge weights.
            query_logits (torch.Tensor): Logits computed on the query set.
            query_targets (torch.Tensor): Ground truth labels for the query set.
            
        Returns:
            float: The query loss value.
        """
        # 1. Unroll inner loop
        final_edge_weights, w_seq = self.unroll_inner_loop(S_i_seq, S_j_seq, initial_edge_weights)
        
        # 2. Compute query loss
        loss = self.compute_query_loss(query_logits, query_targets)
        
        # 3. Meta-update
        # Because we unrolled the inner loop using differentiable operations (torch.exp, torch.bmm),
        # calling loss.backward() automatically computes the exact chain rule gradient defined 
        # in the manuscript: sum_{k=1}^K ( dL_m/dw_ij^E(k) * nabla_{phi} delta_W(k) )
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
        return loss.item()