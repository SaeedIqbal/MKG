import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class OWCILTrainer:
    """
    Main Open-World Class-Incremental Learning (OWCIL) Trainer for the Meta-Knowledge Graph (MKG).
    
    Manages the sequential learning process across tasks T = {T_1, ..., T_N}, integrating:
    1. Basis Consolidation (Phi)
    2. Topological Routing & Node Instantiation
    3. Local Hebbian Updates (SpaLRD) for C_m
    4. Meta-Plastic STDP for Edges E
    5. Active Forgetting for Memory Bounding
    
    Mathematical Formulation:
    - SpaLRD Update: Delta C_m = eta_h(k) * int_0^T int_0^T K(tau-tau') z(tau) S_m(tau')^T d tau d tau'
    - Edge Update: dw_ij^E/dt = eta [ S_i(t) * integral(W_- S_j) + S_j(t) * integral(W_+ S_i) ]
    - Meta-Plasticity: phi_ij <- phi_ij - eta_meta * nabla_{phi_ij} L_m
    - Memory Bound: ||Theta_total||_0 = ||Phi||_0 + sum ||C_k||_0 + ||E||_0 <= C_max
    """
    
    def __init__(self, model, device, eta_meta=0.001):
        self.model = model
        self.device = device
        self.eta_meta = eta_meta
        
        # Optimizer for Meta-Plasticity (only optimizes edge STDP parameters phi_ij)
        # We extract the learnable parameters from the edge_stdp module
        self.meta_optimizer = optim.Adam(self.model.edge_stdp.parameters(), lr=eta_meta)
        
        self.current_task_id = 0
        self.task_accuracies = None
        
    def consolidate_basis(self, dataloader):
        """
        Phase 1: Consolidate the shared spiking basis Phi using the first task's data.
        Phi is derived as top-r eigenvectors of the aggregated input spike covariance matrix.
        Sigma_spike = E[S_in(t) S_in(t)^T]
        """
        print(f"[Trainer] Consolidating shared spiking basis Phi...")
        self.model.train()
        
        all_spikes = []
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx > 10: # Use a subset for efficiency
                break
            x = x.to(self.device)
            all_spikes.append(x)
            
        S_in_data = torch.cat(all_spikes, dim=0)
        # Reshape to (batch * time, in_features)
        S_flat = S_in_data.reshape(-1, self.model.in_features)
        
        self.model.consolidate_basis(S_flat)
        
    def train_task(self, dataloader, task_id):
        """
        Phase 2: Train the MKG on a specific task T_k.
        Involves topological routing, local Hebbian updates, and meta-plasticity.
        """
        print(f"\n[Trainer] Training on Task {task_id}...")
        self.current_task_id = task_id
        self.model.train()
        
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"Task {task_id}")):
            x = x.to(self.device) # Shape: (batch, time, in_features)
            y = y.to(self.device)
            
            batch_size, time_steps, in_features = x.shape
            
            # Reset internal states for the new sequence
            self.model.reset_states(batch_size, self.device)
            
            # Accumulate node activations for edge Hebbian update
            node_activations = torch.zeros(batch_size, self.model.num_nodes, device=self.device)
            
            loss_ce = 0.0
            
            # Store z and S_out for Hebbian update
            z_seq = []
            S_out_seq = []
            
            for t in range(time_steps):
                S_in_t = x[:, t, :] # (batch, in_features)
                
                # Extract features using shared basis: z(t) = Phi^T S_in(t)
                I_syn = self.model.syn_filter(S_in_t)
                z = torch.matmul(I_syn, self.model.Phi)
                
                # Forward pass with topological routing
                S_out, target_node_id, is_novel = self.model(S_in_t, task_id=task_id, is_training=True)
                
                z_seq.append(z)
                S_out_seq.append(S_out)
                
                # Track node activation for edge updates
                if target_node_id != -1:
                    node_activations[:, target_node_id] = S_out.mean(dim=1) 
                    
                # Compute Cross-Entropy Loss for Meta-Plasticity outer loop
                logits = S_out 
                loss_ce += nn.functional.cross_entropy(logits, y) / time_steps
                
            # 1. Local Hebbian Update (SpaLRD) for the active node's C_m
            # Delta C_m = eta_h(k) * int int K(tau-tau') z(tau) S_m(tau')^T
            if target_node_id != -1 and str(target_node_id) in self.model.nodes:
                self._apply_spalrd_hebbian(target_node_id, torch.stack(z_seq, dim=1), torch.stack(S_out_seq, dim=1))
                
            # 2. Edge Hebbian Update (Meta-Plastic STDP)
            # dw_ij^E/dt = eta [ S_i(t) * integral(W_- S_j) + S_j(t) * integral(W_+ S_i) ]
            self.model.update_edges_hebbian(node_activations)
            
            # 3. Meta-Plasticity Outer Loop (Gradient Descent on phi_ij)
            # phi_ij <- phi_ij - eta_meta * nabla_{phi_ij} L_m
            self.meta_optimizer.zero_grad()
            loss_ce.backward()
            self.meta_optimizer.step()
            
            epoch_loss += loss_ce.item()
            
        print(f"[Trainer] Task {task_id} finished. Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    def _apply_spalrd_hebbian(self, node_id, z_seq, S_m_seq):
        """
        Applies the local Hebbian update to the coefficient matrix C_m of the specified node.
        """
        # Get the SpaLRD layer for this node (Assuming the model has a method to retrieve it)
        spalrd_layer = self.model.get_spalrd_layer(node_id) 
        if spalrd_layer is None:
            return
            
        batch_size, time_steps, rank = z_seq.shape
        
        # Reset traces for this layer
        spalrd_layer.reset_traces(batch_size, self.device)
        spalrd_layer.reset_learning_step()
        
        # Apply discrete recursive equivalent of the double integral
        for t in range(time_steps):
            z_t = z_seq[:, t, :]
            S_m_t = S_m_seq[:, t, :]
            spalrd_layer.hebbian_update(z_t, S_m_t)

    def evaluate_task(self, dataloader, task_id):
        """
        Evaluates the model on a specific task's test set.
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size, time_steps, _ = x.shape
                
                self.model.reset_states(batch_size, self.device)
                
                accumulated_logits = torch.zeros(batch_size, self.model.out_features, device=self.device)
                
                for t in range(time_steps):
                    S_in_t = x[:, t, :]
                    S_out, _, _ = self.model(S_in_t, task_id=task_id, is_training=False)
                    accumulated_logits += S_out
                    
                preds = accumulated_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        accuracy = 100.0 * correct / total
        return accuracy

    def perform_active_forgetting(self, old_task_dataloader):
        """
        Phase 3: Execute active forgetting at the end of the task sequence 
        to prune the graph and bound memory.
        min_M sum (1 - M_ij) U_ij subject to ||M||_0 <= rho_max |E|
        """
        print(f"[Trainer] Performing Active Forgetting to bound memory...")
        self.model.consolidate_memory(self.model, old_task_dataloader)
        footprint = self.model.get_memory_footprint()
        print(f"[Trainer] Memory Footprint after pruning: {footprint} parameters")

    def run_continual_learning(self, task_dataloaders, test_dataloaders, num_tasks):
        """
        Main orchestration loop for Open-World Class-Incremental Learning.
        """
        self.task_accuracies = np.zeros((num_tasks, num_tasks))
        
        # 1. Consolidate Basis on the first task
        self.consolidate_basis(task_dataloaders[0]['train'])
        
        for t in range(num_tasks):
            print(f"\n{'='*20} TASK {t+1}/{num_tasks} {'='*20}")
            
            # 2. Train on current task
            self.train_task(task_dataloaders[t]['train'], task_id=t)
            
            # 3. Evaluate on all seen tasks
            for eval_t in range(t + 1):
                acc = self.evaluate_task(test_dataloaders[eval_t], task_id=eval_t)
                self.task_accuracies[t, eval_t] = acc
                print(f"  -> Accuracy on Task {eval_t}: {acc:.2f}%")
                
            # 4. Active Forgetting
            if t > 0:
                self.perform_active_forgetting(task_dataloaders[t-1]['train'])
                
        # Compute final metrics A_N and F_N
        # A_N = 1/N sum_{i=1}^N R_{N,i}
        A_N = np.mean([self.task_accuracies[num_tasks-1, i] for i in range(num_tasks)])
        
        # F_N = 1/(N-1) sum_{i=1}^{N-1} (max_{j} R_{j,i} - R_{N,i})
        F_N = 0.0
        for i in range(num_tasks - 1):
            max_prev = np.max([self.task_accuracies[j, i] for j in range(i, num_tasks-1)])
            F_N += (max_prev - self.task_accuracies[num_tasks-1, i])
        F_N /= (num_tasks - 1)
        
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        print(f"Average Accuracy (A_N): {A_N:.2f}%")
        print(f"Forgetting Measure (F_N): {F_N:.2f}%")
        print(f"Final Memory Footprint: {self.model.get_memory_footprint()} parameters")
        
        return A_N, F_N