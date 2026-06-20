import numpy as np
from scipy.spatial.distance import cdist

class ContinualLearningMetrics:
    """
    Computes standard Continual Learning metrics as defined in the manuscript.
    
    Mathematical Formulation:
    1. Average Accuracy (A_N): 
       A_N = (1/N) * sum_{i=1}^N R_{N,i}
    2. Forgetting Measure (F_N): 
       F_N = (1/(N-1)) * sum_{i=1}^{N-1} ( max_{j in {1..N-1}} R_{j,i} - R_{N,i} )
       
    where R_{j,i} is the classification accuracy on the test set of task i 
    after the model has sequentially learned up to task j.
    """
    
    @staticmethod
    def compute_average_accuracy(accuracy_matrix):
        """
        Computes the Average Accuracy (A_N) after learning N tasks.
        
        Args:
            accuracy_matrix (np.ndarray): Matrix of shape (N, N) where entry [j, i] 
                                          is the accuracy on task i after learning task j.
        Returns:
            float: The Average Accuracy A_N.
        """
        N = accuracy_matrix.shape[0]
        # R_{N,i} is the last row of the matrix
        return np.mean([accuracy_matrix[N-1, i] for i in range(N)])

    @staticmethod
    def compute_forgetting_measure(accuracy_matrix):
        """
        Computes the Forgetting Measure (F_N) after learning N tasks.
        
        Args:
            accuracy_matrix (np.ndarray): Matrix of shape (N, N).
        Returns:
            float: The Forgetting Measure F_N.
        """
        N = accuracy_matrix.shape[0]
        if N <= 1:
            return 0.0
            
        forgetting = 0.0
        for i in range(N - 1):
            # max_{j in {1..N-1}} R_{j,i}
            max_prev_acc = np.max([accuracy_matrix[j, i] for j in range(i, N-1)])
            forgetting += (max_prev_acc - accuracy_matrix[N-1, i])
            
        return forgetting / (N - 1)


class OODDetectionMetrics:
    """
    Computes Open-World and Out-of-Distribution (OOD) detection metrics.
    
    Mathematical Formulation:
    1. Topological Scoring Function: 
       s(x) = min_{v_k in V} d_W(PD_x, PD_k)
    2. OOD Routing Accuracy: 
       (1 / |D_OOD|) * sum_{x in D_OOD} I(s(x) > tau AND Delta C_k = 0, forall k)
    """
    
    @staticmethod
    def compute_auroc(scores_in, scores_ood):
        """
        Computes the Area Under the Receiver Operating Characteristic curve (AUROC) 
        from scratch using the trapezoidal rule.
        
        Args:
            scores_in (np.ndarray): Topological scores s(x) for in-distribution samples.
            scores_ood (np.ndarray): Topological scores s(x) for OOD samples.
        Returns:
            float: The AUROC score.
        """
        scores = np.concatenate([scores_in, scores_ood])
        labels = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_ood)])
        
        # Sort by scores in descending order
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        tpr_list = []
        fpr_list = []
        
        tp = 0
        fp = 0
        total_pos = np.sum(labels == 1)
        total_neg = np.sum(labels == 0)
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / total_pos)
            fpr_list.append(fp / total_neg)
            
        # Compute area using the trapezoidal rule
        return np.trapz(tpr_list, fpr_list)

    @staticmethod
    def compute_fpr95(scores_in, scores_ood):
        """
        Computes the False Positive Rate at 95% True Positive Rate (FPR95).
        
        Args:
            scores_in (np.ndarray): Topological scores for in-distribution samples.
            scores_ood (np.ndarray): Topological scores for OOD samples.
        Returns:
            float: The FPR95 score.
        """
        scores = np.concatenate([scores_in, scores_ood])
        labels = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_ood)])
        
        sorted_indices = np.argsort(-scores)
        sorted_labels = labels[sorted_indices]
        
        tp = 0
        fp = 0
        total_pos = np.sum(labels == 1)
        total_neg = np.sum(labels == 0)
        
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
                
            tpr = tp / total_pos
            if tpr >= 0.95:
                return fp / total_neg
                
        return 1.0

    @staticmethod
    def compute_ood_routing_accuracy(scores_ood, threshold_tau, coefficients_frozen_mask):
        """
        Computes the OOD Routing Accuracy.
        
        Mathematical Formulation:
        (1 / |D_OOD|) * sum_{x in D_OOD} I(s(x) > tau AND Delta C_k = 0, forall k)
        
        Args:
            scores_ood (np.ndarray): Topological scores s(x) for OOD samples.
            threshold_tau (float): The topological routing threshold tau.
            coefficients_frozen_mask (np.ndarray): Boolean array of shape (num_ood_samples, num_nodes).
                                                   True if Delta C_k = 0 for that node.
        Returns:
            float: The OOD Routing Accuracy.
        """
        # Check if ALL nodes were frozen (Delta C_k = 0, forall k)
        all_frozen = np.all(coefficients_frozen_mask, axis=1)
        
        # Check both conditions: s(x) > tau AND Delta C_k = 0
        correct_isolations = np.sum((scores_ood > threshold_tau) & all_frozen)
        
        return correct_isolations / len(scores_ood)


class TopologicalStabilityMetrics:
    """
    Computes the Betti Number Stability using the Bottleneck Distance (d_B) 
    between Persistence Diagrams.
    
    Mathematical Formulation:
    d_B(PD_i^{(i)}, PD_i^{(N)}) = inf_{gamma} sup_{x in PD_i^{(i)}} ||x - gamma(x)||_inf
    
    This is computed exactly using a binary search over the threshold epsilon 
    and a maximum bipartite matching algorithm (implemented from scratch via DFS).
    """
    
    @staticmethod
    def _max_bipartite_matching(adj):
        """
        Computes the maximum bipartite matching using Depth-First Search (DFS) 
        augmenting paths (Hopcroft-Karp style).
        
        Args:
            adj (np.ndarray): Boolean adjacency matrix of shape (N, M).
        Returns:
            int: The size of the maximum matching.
        """
        n, m = adj.shape
        match_u = [-1] * n
        match_v = [-1] * m
        
        def dfs(u, visited):
            for v in range(m):
                if adj[u, v] and not visited[v]:
                    visited[v] = True
                    if match_v[v] == -1 or dfs(match_v[v], visited):
                        match_u[u] = v
                        match_v[v] = u
                        return True
            return False

        result = 0
        for u in range(n):
            visited = [False] * m
            if dfs(u, visited):
                result += 1
        return result

    @staticmethod
    def compute_bottleneck_distance(P, Q):
        """
        Computes the exact Bottleneck Distance between two Persistence Diagrams P and Q.
        
        Args:
            P (np.ndarray): Persistence diagram 1. Shape: (n, 2) -> [birth, death]
            Q (np.ndarray): Persistence diagram 2. Shape: (m, 2) -> [birth, death]
        Returns:
            float: The Bottleneck Distance d_B.
        """
        n = len(P)
        m = len(Q)
        
        if n == 0 and m == 0:
            return 0.0
            
        # 1. Compute pairwise L_inf distances between P and Q
        if n > 0 and m > 0:
            dist_PQ = cdist(P, Q, metric='chebyshev') # L_inf norm
        else:
            dist_PQ = np.empty((n, m))
            
        # 2. Compute distances to the diagonal: ||(b, d) - Delta||_inf = |b - d| / 2
        dist_P_diag = np.abs(P[:, 0] - P[:, 1]) / 2.0 if n > 0 else np.array([])
        dist_Q_diag = np.abs(Q[:, 0] - Q[:, 1]) / 2.0 if m > 0 else np.array([])
        
        # 3. Collect all unique possible distances for binary search
        all_dists = []
        if dist_PQ.size > 0:
            all_dists.extend(dist_PQ.flatten())
        if dist_P_diag.size > 0:
            all_dists.extend(dist_P_diag)
        if dist_Q_diag.size > 0:
            all_dists.extend(dist_Q_diag)
            
        all_dists = np.unique(all_dists)
        all_dists.sort()
        
        if len(all_dists) == 0:
            return 0.0
            
        # 4. Binary search for the minimum epsilon that allows a perfect matching
        low, high = 0, len(all_dists) - 1
        ans = all_dists[-1]
        
        while low <= high:
            mid = (low + high) // 2
            eps = all_dists[mid]
            
            # Build bipartite graph for threshold eps
            # Left nodes: P U Q_diag (size n+m)
            # Right nodes: Q U P_diag (size m+n)
            adj = np.zeros((n + m, n + m), dtype=bool)
            
            # Edges from P to Q
            if n > 0 and m > 0:
                adj[:n, :m] = dist_PQ <= eps
                
            # Edges from P to P_diag (Right indices: m to m+n)
            if n > 0:
                for i in range(n):
                    adj[i, m + i] = dist_P_diag[i] <= eps
                    
            # Edges from Q_diag (Left indices: n to n+m) to Q
            if m > 0:
                for j in range(m):
                    adj[n + j, j] = dist_Q_diag[j] <= eps
                    
            # Edges from Q_diag to P_diag (Distance is 0, so always <= eps)
            if n > 0 and m > 0:
                for j in range(m):
                    for i in range(n):
                        adj[n + j, m + i] = True
                        
            # Check if a perfect matching of size (n+m) exists
            max_match = TopologicalStabilityMetrics._max_bipartite_matching(adj)
            
            if max_match == n + m:
                ans = eps
                high = mid - 1  # Try to find a smaller epsilon
            else:
                low = mid + 1   # Need a larger epsilon
                
        return ans


class MetricsEvaluator:
    """
    Main orchestrator for evaluating the Meta-Knowledge Graph (MKG) framework.
    Aggregates all specific metric classes into a unified evaluation interface.
    """
    
    def __init__(self):
        self.cl_metrics = ContinualLearningMetrics()
        self.ood_metrics = OODDetectionMetrics()
        self.topo_metrics = TopologicalStabilityMetrics()
        
    def evaluate_continual_learning(self, accuracy_matrix):
        """Evaluates A_N and F_N."""
        return {
            "A_N": self.cl_metrics.compute_average_accuracy(accuracy_matrix),
            "F_N": self.cl_metrics.compute_forgetting_measure(accuracy_matrix)
        }
        
    def evaluate_ood_detection(self, scores_in, scores_ood, threshold_tau=None, coeff_frozen_mask=None):
        """Evaluates AUROC, FPR95, and OOD Routing Accuracy."""
        res = {
            "AUROC": self.ood_metrics.compute_auroc(scores_in, scores_ood),
            "FPR95": self.ood_metrics.compute_fpr95(scores_in, scores_ood)
        }
        
        if threshold_tau is not None and coeff_frozen_mask is not None:
            res["OOD_Routing_Acc"] = self.ood_metrics.compute_ood_routing_accuracy(
                scores_ood, threshold_tau, coeff_frozen_mask
            )
        return res
        
    def evaluate_topological_stability(self, pd_before, pd_after):
        """Evaluates Betti Number Stability (Bottleneck Distance)."""
        return {
            "Betti_Stability_dB": self.topo_metrics.compute_bottleneck_distance(pd_before, pd_after)
        }