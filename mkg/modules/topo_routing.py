import torch
import numpy as np
from gtda.homology import VietorisRipsPersistence
import ot

class TopologicalSubgraphRouter:
    """
    Implements Topological Subgraph Routing for the Meta-Knowledge Graph (MKG).
    
    This module dynamically allocates functional capacity based on the intrinsic 
    geometric structure of incoming spike trains, replacing raw Euclidean feature 
    similarity with rigorous topological invariants.
    
    Mathematical Formulation:
    1. Point Cloud Representation: 
       P_m = {(t_i^{(f)}, i) | i in {1..d_in}, f in N}
    2. Vietoris-Rips Filtration: 
       F_m = {VR(P_m, epsilon)}_{epsilon >= 0}
    3. Persistence Diagram Extraction: 
       PD_m^{(p)} = {(b_j, d_j)}_{j=1}^{N_p}
    4. p-Wasserstein Distance: 
       d_W(PD_m, PD_k) = inf_{gamma} ( sum_{(x,y) in gamma} ||x - y||_inf^p + ... )^{1/p}
    5. Routing Logic: 
       If min_{v_k} d_W(PD_m, PD_k) > tau, instantiate new virtual node v_m.
    """
    
    def __init__(self, threshold_tau=0.4, homology_dimensions=[0, 1], max_edge_length=np.inf):
        self.tau = threshold_tau
        self.homology_dimensions = homology_dimensions
        
        # Initialize Giotto-TDA for persistent homology computation
        self.vr = VietorisRipsPersistence(
            homology_dimensions=homology_dimensions, 
            max_edge_length=max_edge_length
        )
        
    def spikes_to_point_cloud(self, S):
        """
        Converts a binary spike tensor S into a list of 2D point clouds.
        The 2D space captures both temporal (spike time) and spatial (neuron index) topology.
        
        Args:
            S (torch.Tensor): Binary spike train. Shape: (batch_size, num_neurons, time_steps)
            
        Returns:
            point_clouds (list): List of numpy arrays, each of shape (num_spikes, 2).
        """
        point_clouds = []
        batch_size, num_neurons, time_steps = S.shape
        
        for b in range(batch_size):
            # Find indices of spikes: shape (num_spikes, 2) -> [neuron_idx, time_idx]
            indices = torch.nonzero(S[b], as_tuple=False)
            
            if indices.shape[0] < 2:
                # Vietoris-Rips requires at least 2 points to form a 1-simplex
                pc = np.array([[0.0, 0.0], [1.0, 1.0]])
            else:
                pc = indices.cpu().numpy().astype(float)
                # Normalize time dimension to match neuron index scale
                # This prevents time from dominating the distance metric in VR filtration
                if time_steps > 1:
                    pc[:, 1] = pc[:, 1] / (time_steps - 1) * num_neurons
                    
            point_clouds.append(pc)
            
        return point_clouds

    def compute_persistence_diagrams(self, S):
        """
        Computes the Persistence Diagrams for a batch of spike trains.
        
        Args:
            S (torch.Tensor): Binary spike train. Shape: (batch_size, num_neurons, time_steps)
            
        Returns:
            batch_PDs (list): List of numpy arrays, each of shape (n_features, 2) -> [birth, death].
        """
        point_clouds = self.spikes_to_point_cloud(S)
        
        # Compute persistent homology using Vietoris-Rips filtration
        # diagrams shape: (batch_size, n_features, 3) -> [dimension, birth, death]
        diagrams = self.vr.fit_transform(point_clouds)
        
        batch_PDs = []
        for b in range(diagrams.shape[0]):
            diagram = diagrams[b]
            # Filter valid topological features (birth < death, and death != infinity)
            valid = (diagram[:, 2] > diagram[:, 1]) & (diagram[:, 2] != np.inf)
            pd = diagram[valid, 1:]  # Extract [birth, death]
            batch_PDs.append(pd)
            
        return batch_PDs

    def compute_wasserstein_distance(self, pd1, pd2, p=2):
        """
        Computes the p-Wasserstein distance between two persistence diagrams 
        using diagonal augmentation and the POT (Python Optimal Transport) library.
        
        Args:
            pd1 (np.ndarray): Persistence diagram 1. Shape: (n, 2) -> [birth, death]
            pd2 (np.ndarray): Persistence diagram 2. Shape: (m, 2) -> [birth, death]
            p (int): Order of the Wasserstein distance (default: 2).
            
        Returns:
            dist (float): The p-Wasserstein distance.
        """
        n, m = len(pd1), len(pd2)
        
        # Handle edge cases where one or both diagrams are empty
        if n == 0 and m == 0:
            return 0.0
        if n == 0:
            return np.sum((np.abs(pd2[:, 0] - pd2[:, 1]) / 2.0)**p)**(1.0 / p)
        if m == 0:
            return np.sum((np.abs(pd1[:, 0] - pd1[:, 1]) / 2.0)**p)**(1.0 / p)
            
        # Compute projections of points onto the diagonal (x, x)
        # The projection of (b, d) is ((b+d)/2, (b+d)/2)
        proj1 = (pd1[:, 0] + pd1[:, 1]) / 2.0
        proj2 = (pd2[:, 0] + pd2[:, 1]) / 2.0
        
        # Augment pd1 with projections of pd2 (size n+m)
        aug1 = np.vstack([pd1, np.column_stack([proj2, proj2])])
        # Augment pd2 with projections of pd1 (size m+n)
        aug2 = np.vstack([pd2, np.column_stack([proj1, proj1])])
        
        # Compute L_infinity cost matrix: C[i, j] = ||aug1[i] - aug2[j]||_inf^p
        diff_0 = np.abs(aug1[:, 0:1] - aug2[:, 0:1].T)
        diff_1 = np.abs(aug1[:, 1:2] - aug2[:, 1:2].T)
        C = np.maximum(diff_0, diff_1)**p
        
        # Uniform probability distributions for EMD
        a = np.ones(n + m) / (n + m)
        b = np.ones(n + m) / (n + m)
        
        # Solve Earth Mover's Distance (Optimal Transport)
        G = ot.emd(a, b, C)
        
        # Compute total transport cost
        dist = np.sum(G * C)**(1.0 / p)
        return dist

    def route_task(self, S_m, existing_nodes_PDs):
        """
        Evaluates topological distance and routes to a virtual node if d_W > tau.
        
        Args:
            S_m (torch.Tensor): Spike train of the new task. Shape: (batch, neurons, time)
            existing_nodes_PDs (list): List of Persistence Diagrams for existing nodes.
            
        Returns:
            is_novel (bool): True if a new virtual node should be instantiated.
            target_node_idx (int): Index of the closest existing node (-1 if novel).
            min_distance (float): The minimum Wasserstein distance found.
        """
        # Compute PDs for the incoming batch
        batch_PDs = self.compute_persistence_diagrams(S_m)
        
        if not existing_nodes_PDs:
            return True, -1, np.inf
            
        min_dist = np.inf
        closest_node = -1
        
        # Compute distance to each existing node's PD
        for idx, PD_k in enumerate(existing_nodes_PDs):
            # Average Wasserstein distance across the batch
            batch_dist = 0.0
            for PD_m in batch_PDs:
                dist = self.compute_wasserstein_distance(PD_m, PD_k)
                batch_dist += dist
            avg_dist = batch_dist / len(batch_PDs)
            
            if avg_dist < min_dist:
                min_dist = avg_dist
                closest_node = idx
                
        # Routing decision based on threshold tau
        if min_dist > self.tau:
            # Topologically distinct: Instantiate new virtual node v_m
            return True, -1, min_dist
        else:
            # Route to existing node v_k
            return False, closest_node, min_dist