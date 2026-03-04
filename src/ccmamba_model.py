"""
CCMamba Model — Combinatorial Complex Mamba
============================================
Higher-Order Graph State Space Model for crop yield prediction.

Based on: Ma et al. (2026) "CCMamba: Combinatorial Complex Mamba 
for Learning on Higher-Order Graph Structures"

Architecture:
    Input Features → Graph Embedding
    → Local CCMamba Block (neighborhood-level Mamba)
    → Global CCMamba Block (full-graph Mamba)  
    → Higher-Order Feature Fusion
    → MLP Regression Head → Yield Prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_block import MambaBlock


class GraphConvLayer(nn.Module):
    """Simple graph convolution: H' = σ(A·H·W + b)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        # x: (N, F), adj: (N, N)
        h = torch.mm(adj, x)   # aggregate neighbors
        h = self.linear(h)
        return F.relu(h)


class LocalCCMambaBlock(nn.Module):
    """
    Local CCMamba Block: Processes neighborhood-level features.
    
    1. Graph convolution to aggregate local neighborhood
    2. Mamba SSM on sorted local sequences
    """
    def __init__(self, d_model, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        self.gcn = GraphConvLayer(d_model, d_model)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand=2, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, adj):
        """
        x: (N, d_model) node features
        adj: (N, N) normalized adjacency
        Returns: (N, d_model)
        """
        # Graph convolution for local aggregation
        h = self.gcn(x, adj)
        h = self.norm(h)
        
        # Mamba SSM on the sequence (treat nodes as sequence)
        h = h.unsqueeze(0)  # (1, N, d_model) — batch dim
        h = self.mamba(h)
        h = h.squeeze(0)  # (N, d_model)
        
        return h


class GlobalCCMambaBlock(nn.Module):
    """
    Global CCMamba Block: Processes the full graph via Mamba SSM.
    
    1. Sort nodes by a learned priority score
    2. Apply Mamba on the sorted global sequence
    """
    def __init__(self, d_model, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        self.score_proj = nn.Linear(d_model, 1)
        self.mamba = MambaBlock(d_model, d_state, d_conv, expand=2, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: (N, d_model)
        Returns: (N, d_model) 
        """
        # Compute priority scores for ordering
        scores = self.score_proj(x).squeeze(-1)  # (N,)
        sorted_idx = torch.argsort(scores)
        
        # Sort nodes
        x_sorted = x[sorted_idx]
        x_sorted = self.norm(x_sorted)
        
        # Apply Mamba on sorted sequence
        x_sorted = x_sorted.unsqueeze(0)  # (1, N, d_model)
        x_sorted = self.mamba(x_sorted)
        x_sorted = x_sorted.squeeze(0)  # (N, d_model)
        
        # Unsort back to original order
        unsorted_idx = torch.argsort(sorted_idx)
        x_out = x_sorted[unsorted_idx]
        
        return x_out


class HigherOrderFusion(nn.Module):
    """
    Fuse rank-0 (node) and rank-1 (higher-order) embeddings.
    Uses the higher-order adjacency to propagate information.
    """
    def __init__(self, d_model):
        super().__init__()
        self.linear_node = nn.Linear(d_model, d_model)
        self.linear_ho = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, h_node, adj_ho):
        """
        h_node: (N, d_model) — node embeddings
        adj_ho: (N, N) — higher-order adjacency
        """
        # Higher-order aggregation
        h_ho = torch.mm(adj_ho, h_node)
        
        h_n = self.linear_node(h_node)
        h_h = self.linear_ho(h_ho)
        
        # Gated fusion
        gate_input = torch.cat([h_n, h_h], dim=-1)
        gate_val = torch.sigmoid(self.gate(gate_input))
        
        fused = gate_val * h_n + (1 - gate_val) * h_h
        return self.norm(fused)


class CCMambaEncoder(nn.Module):
    """
    Full CCMamba Encoder for crop yield prediction.
    
    Pipeline:
        Features → Embedding → Local CCMamba → Global CCMamba 
        → Higher-Order Fusion → MLP → Yield
    """
    def __init__(self, input_dim, hidden_dim=64, d_state=16, d_conv=4, 
                 n_layers=2, dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Local CCMamba blocks
        self.local_blocks = nn.ModuleList([
            LocalCCMambaBlock(hidden_dim, d_state, d_conv, dropout)
            for _ in range(n_layers)
        ])
        
        # Global CCMamba blocks
        self.global_blocks = nn.ModuleList([
            GlobalCCMambaBlock(hidden_dim, d_state, d_conv, dropout)
            for _ in range(n_layers)
        ])
        
        # Higher-order fusion
        self.ho_fusion = HigherOrderFusion(hidden_dim)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x, adj_norm, adj_ho_norm):
        """
        x: (N, input_dim) — features
        adj_norm: (N, N) — kNN adjacency (normalized)
        adj_ho_norm: (N, N) — higher-order adjacency (normalized) 
        Returns: (N,) — predicted yields
        """
        # Embed
        h = self.input_proj(x)
        
        # Local CCMamba blocks
        for block in self.local_blocks:
            h = h + block(h, adj_norm)
        
        # Global CCMamba blocks
        for block in self.global_blocks:
            h = h + block(h)
        
        # Higher-order fusion
        h = self.ho_fusion(h, adj_ho_norm)
        
        # Predict
        out = self.head(h).squeeze(-1)
        return out


class CCMambaModel:
    """
    Wrapper class with fit/predict interface for the CCMamba model.
    """
    def __init__(self, input_dim, hidden_dim=64, d_state=16, d_conv=4,
                 n_layers=2, dropout=0.1, lr=0.001, epochs=100, 
                 batch_size=2048, device='cpu'):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = CCMambaEncoder(
            input_dim, hidden_dim, d_state, d_conv, n_layers, dropout
        ).to(device)
        
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
    
    def _make_batched_adj(self, adj_full, idx):
        """Extract submatrix of adjacency for a batch of indices."""
        return adj_full[idx][:, idx]
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            adj_norm=None, adj_ho_norm=None, verbose=True):
        """Train the model."""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        n = len(X_train)
        
        # Build adjacency for training set
        if adj_norm is None:
            adj_norm = torch.eye(n).to(self.device)
        if adj_ho_norm is None:
            adj_ho_norm = torch.eye(n).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Mini-batch training
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                
                X_batch = X_t[idx]
                y_batch = y_t[idx]
                
                # Get batch adjacency
                adj_batch = self._make_batched_adj(adj_norm, idx)
                adj_ho_batch = self._make_batched_adj(adj_ho_norm, idx)
                
                self.optimizer.zero_grad()
                pred = self.model(X_batch, adj_batch, adj_ho_batch)
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            self.scheduler.step()
            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            
            # Validation
            if X_val is not None:
                val_loss = self._evaluate_loss(X_val, y_val, adj_norm, adj_ho_norm)
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break
                
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{self.epochs} — Train Loss: {avg_loss:.2f}, Val Loss: {val_loss:.2f}")
            else:
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.2f}")
        
        # Restore best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
    
    def _evaluate_loss(self, X, y, adj_norm, adj_ho_norm):
        """Compute MSE loss on a dataset."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).to(self.device)
        n = len(X)
        
        with torch.no_grad():
            # Use identity adjacency for validation to keep it simple
            adj = torch.eye(n).to(self.device)
            pred = self.model(X_t, adj, adj)
            loss = F.mse_loss(pred, y_t)
        return loss.item()
    
    def predict(self, X):
        """Predict yields."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        n = len(X)
        
        with torch.no_grad():
            adj = torch.eye(n).to(self.device)
            pred = self.model(X_t, adj, adj)
        
        return pred.cpu().numpy()
