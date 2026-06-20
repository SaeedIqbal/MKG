import torch
import torch.nn as nn
import numpy as np
from modules.neurons import AlphaSynapticFilter, LIFNeuron
from modules.topo_routing import TopologicalSubgraphRouter
from modules.stdp import MetaPlasticSTDP
from modules.memory import LifelongMemoryManager

class MetaKnowledgeGraph(nn.Module):
    """
    Meta-Knowledge Graph (MKG) Orchestrator.
    
    Manages the dynamic evolution of the graph G = (V, E), integrating:
    - Shared Spiking Basis (Phi)
    - Task-specific Coefficients (C_k) for nodes V
    - Topological Subgraph Routing for OOD detection and node instantiation
    - Meta-Plastic STDP for edge weight updates
    - Active Forgetting for memory bounding
    
    Mathematical Formulation:
    1. Node Parameterization: W_k = Phi @ C_k
    2. Routing: min_{v_k} d_W(PD_m, PD_k) > tau => instantiate v_m
    3. Edge Update: dw_ij^E/dt = eta [ S_i(t) * integral(W_- S_j) + S_j(t) * integral(W_+ S_i) ]
    4. Memory Bound: ||Theta_total||_0 = ||Phi||_0 + sum ||C_k||_0 + ||E||_0 <= C_max
    """
    
    def __init__(self, in_features, out_features, rank, 
                 tau_s=5.0, tau_m=20.0, v_th=1.0, 
                 topo_threshold=0.4, rho_max=0.2, 
                 A_plus=0.01, tau_plus=20.0, A_minus=0.012, tau_minus=20.0,
                 routing_window=50):
        super(MetaKnowledgeGraph, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.routing_window = routing_window
        
        # 1. Shared Spiking Basis (Phi) - Frozen
        self.Phi = nn.Parameter(torch.randn(in_features, rank) / np.sqrt(in_features))
        self.Phi.requires_grad = False 
        
        # 2. Graph Components: Nodes (V) and Edges (E)
        self.nodes = nn.ParameterDict() # Stores C_k for each node k
        self.node_activations = {}     # Tracks historical activation magnitude A_k
        self.node_PDs = {}            # Stores Persistence Diagrams for each node
        self.num_nodes = 0
        
        # Edge weights E (K x K)
        self.register_buffer('edge_weights', torch.zeros(0, 0))
        
        # 3. Synaptic Filtering (Alpha-function)
        self.syn_filter = AlphaSynapticFilter(tau_s=tau_s)
        
        # 4. LIF Neuron Dynamics
        self.lif_neuron = LIFNeuron(tau_m=tau_m, v_th=v_th)
        
        # 5. Topological Subgraph Router
        self.topo_router = TopologicalSubgraphRouter(threshold_tau=topo_threshold)
        
        # 6. Meta-Plastic STDP for Edges
        self.edge_stdp = MetaPlasticSTDP(
            in_features=1, out_features=1, 
            A_plus=A_plus, tau_plus=tau_plus, 
            A_minus=A_minus, tau_minus=tau_minus,
            learnable_params=True
        )
        
        # 7. Lifelong Memory Manager
        self.memory_manager = LifelongMemoryManager(num_edges=0, rho_max=rho_max)
        
        # Buffer for accumulating spikes for topological routing
        self.register_buffer('spike_buffer', torch.zeros(0, 0, 0))
        self.buffer_idx = 0
        
    def consolidate_basis(self, S_in_data):
        """
        Derives Phi as top-r eigenvectors of the aggregated input spike covariance matrix.
        Ensures strict orthonormality: Phi^T @ Phi = I_r
        """
        S_flat = S_in_data.reshape(-1, self.in_features)
        cov = torch.matmul(S_flat.T, S_flat) / S_flat.size(0)
        _, eigvecs = torch.linalg.eigh(cov)
        self.Phi.data = eigvecs[:, -self.rank:]
        self.Phi.requires_grad = False
        
    def add_node(self, node_id, PD_m):
        """
        Instantiates a new virtual node v_m with coefficient matrix C_m.
        """
        C_m = nn.Parameter(torch.randn(self.rank, self.out_features) / np.sqrt(self.rank))
        self.nodes[str(node_id)] = C_m
        self.node_activations[str(node_id)] = 0.0
        self.node_PDs[str(node_id)] = PD_m
        
        # Resize edge weights matrix to (K+1 x K+1)
        K = self.num_nodes
        new_edges = torch.zeros(K + 1, K + 1, device=self.edge_weights.device)
        if K > 0:
            new_edges[:K, :K] = self.edge_weights
        self.edge_weights = new_edges
        self.num_nodes += 1
        
        # Update memory manager edge count
        self.memory_manager.num_edges = self.num_nodes * self.num_nodes
        
    def extract_features(self, S_in_t):
        """
        Projects input spikes onto the shared basis: z(t) = Phi^T S_in(t).
        Applies synaptic filtering first.
        """
        I_syn = self.syn_filter(S_in_t)
        z = torch.matmul(I_syn, self.Phi)
        return z
        
    def update_spike_buffer(self, S_in_t):
        """Accumulates spikes over time for topological routing."""
        batch_size, num_neurons = S_in_t.shape
        device = S_in_t.device
        
        if self.spike_buffer.numel() == 0:
            self.spike_buffer = torch.zeros(batch_size, num_neurons, self.routing_window, device=device)
            
        self.spike_buffer[:, :, self.buffer_idx] = S_in_t
        self.buffer_idx = (self.buffer_idx + 1) % self.routing_window
        
    def route_task(self, S_in_seq):
        """
        Evaluates topological distance and routes to a virtual node if d_W > tau.
        """
        existing_PDs = [self.node_PDs[str(k)] for k in range(self.num_nodes)]
        is_novel, closest_idx, min_dist = self.topo_router.route_task(S_in_seq, existing_PDs)
        return is_novel, closest_idx, min_dist

    def forward(self, S_in_t, task_id=None, is_training=True):
        """
        Forward pass for a single time step t.
        
        Args:
            S_in_t (torch.Tensor): Input spikes at time t. Shape: (batch, in_features)
            task_id (int, optional): Ground truth task ID for supervised routing.
            is_training (bool): Whether the model is in training mode.
            
        Returns:
            S_out (torch.Tensor): Output spikes. Shape: (batch, out_features)
            target_node_id (int): The ID of the activated node.
            is_novel (bool): Whether a new node was instantiated.
        """
        batch_size = S_in_t.size(0)
        device = S_in_t.device
        
        # Initialize LIF and Synaptic states if needed
        if self.syn_filter.x is None:
            self.syn_filter.reset_states((batch_size, self.in_features), device)
            self.lif_neuron.reset_states((batch_size, self.out_features), device)
            
        # 1. Accumulate spikes for routing
        self.update_spike_buffer(S_in_t)
        
        # 2. Topological Routing
        target_node_id = task_id
        is_novel = False
        PD_m = None
        
        if is_training or task_id is None:
            # Compute PDs for the accumulated spike sequence
            PD_m_batch = self.topo_router.compute_persistence_diagrams(self.spike_buffer)
            # Average PD across batch for node representation
            PD_m = np.mean(PD_m_batch, axis=0) 
            
            is_novel, closest_idx, min_dist = self.route_task(self.spike_buffer)
            
            if is_novel:
                new_id = self.num_nodes
                self.add_node(new_id, PD_m)
                target_node_id = new_id
            else:
                target_node_id = closest_idx
                # Update existing node's PD with the new observation
                if PD_m is not None and len(PD_m) > 0:
                    self.node_PDs[str(target_node_id)] = PD_m 
                    
        # 3. Extract features using shared basis
        z = self.extract_features(S_in_t) # (batch, rank)
        
        # 4. Compute node output
        if target_node_id is not None and str(target_node_id) in self.nodes:
            C_k = self.nodes[str(target_node_id)]
            # I_proj = C_k^T z
            I_proj = torch.matmul(z, C_k)
            
            # LIF dynamics
            S_out = self.lif_neuron(I_proj)
            
            # Update historical activation magnitude A_k
            self.node_activations[str(target_node_id)] += S_out.mean().item()
            
            return S_out, target_node_id, is_novel
        else:
            # Fallback if no node is active
            return torch.zeros(batch_size, self.out_features, device=device), -1, False

    def update_edges_hebbian(self, S_out_nodes):
        """
        Updates edge weights w_ij^E based on node co-activations using Meta-Plastic STDP.
        
        Args:
            S_out_nodes (torch.Tensor): Binary activation of nodes. Shape: (batch, num_nodes)
        """
        if self.num_nodes < 2:
            return
            
        A_plus, tau_plus, A_minus, tau_minus = self.edge_stdp.get_params()
        alpha_plus = torch.exp(-1.0 / tau_plus)
        alpha_minus = torch.exp(-1.0 / tau_minus)
        
        if not hasattr(self, 'node_trace_pre'):
            self.node_trace_pre = torch.zeros(self.num_nodes, device=S_out_nodes.device)
            self.node_trace_post = torch.zeros(self.num_nodes, device=S_out_nodes.device)
            
        # Average over batch
        S_mean = S_out_nodes.mean(dim=0)
        
        self.node_trace_pre = alpha_plus * self.node_trace_pre + S_mean
        self.node_trace_post = alpha_minus * self.node_trace_post + S_mean
        
        delta_W_plus = A_plus * torch.outer(self.node_trace_post, self.node_trace_pre)
        delta_W_minus = A_minus * torch.outer(S_mean, self.node_trace_post)
        
        delta_W = delta_W_plus - delta_W_minus
        
        self.edge_weights += self.edge_stdp.eta * delta_W
        
    def consolidate_memory(self, model, old_task_dataloader):
        """
        Executes the active forgetting mechanism to prune the graph.
        """
        if self.num_nodes == 0:
            return
            
        # Flatten edge weights for the memory manager
        flat_edges = self.edge_weights.flatten()
        
        # Create a temporary parameter for FIM computation
        temp_edge_param = nn.Parameter(flat_edges.clone())
        
        # Compute utility and prune
        pruned_flat, mask, utility = self.memory_manager.perform_active_forgetting(
            model, old_task_dataloader, temp_edge_param
        )
        
        # Reshape and apply
        self.edge_weights = pruned_flat.reshape(self.num_nodes, self.num_nodes)
        
    def get_memory_footprint(self):
        """
        Calculates the current physical memory footprint:
        ||Theta_total||_0 = ||Phi||_0 + sum ||C_k||_0 + ||E||_0
        """
        phi_norm = torch.count_nonzero(self.Phi).item()
        c_norm = sum(torch.count_nonzero(self.nodes[str(k)]).item() for k in range(self.num_nodes))
        e_norm = torch.count_nonzero(self.edge_weights).item()
        
        return phi_norm + c_norm + e_norm

    def reset_states(self, batch_size, device):
        """Resets all internal states for a new sequence."""
        self.syn_filter.reset_states((batch_size, self.in_features), device)
        self.lif_neuron.reset_states((batch_size, self.out_features), device)
        self.spike_buffer.zero_()
        self.buffer_idx = 0
        if hasattr(self, 'node_trace_pre'):
            self.node_trace_pre.zero_()
            self.node_trace_post.zero_()