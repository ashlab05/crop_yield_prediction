"""
Baseline Models for Comparison
================================
Provides consistent fit/predict interface for:
1. Random Forest
2. XGBoost  
3. MLP (PyTorch)
4. Standard GCN (PyTorch)
5. Graph-Mamba (Mamba on graph, without higher-order — ablation)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from .mamba_block import MambaBlock

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# =============================================================================
# 1. Random Forest Wrapper
# =============================================================================
class RFModel:
    def __init__(self, n_estimators=300, max_depth=20, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1
        )
        self.name = "Random Forest"
    
    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)


# =============================================================================
# 2. XGBoost Wrapper
# =============================================================================
class XGBModel:
    def __init__(self, n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
            tree_method='hist', verbosity=0
        )
        self.name = "XGBoost"
    
    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train, verbose=False)
    
    def predict(self, X):
        return self.model.predict(X)


# =============================================================================
# 3. MLP Model (PyTorch)
# =============================================================================
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPModel:
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2, 
                 lr=0.001, epochs=150, batch_size=512, device='cpu'):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = MLPNet(input_dim, hidden_dim, dropout).to(device)
        self.lr = lr
        self.name = "MLP"
        self.train_losses = []
    
    def fit(self, X_train, y_train, **kwargs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        n = len(X_train)
        
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0
            n_b = 0
            
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                
                optimizer.zero_grad()
                pred = self.model(X_t[idx])
                loss = F.mse_loss(pred, y_t[idx])
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_b += 1
            
            scheduler.step()
            self.train_losses.append(epoch_loss / n_b)
    
    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()


# =============================================================================
# 4. Standard GCN Model
# =============================================================================
class GCNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        h = torch.mm(adj, x)
        h = F.relu(self.w1(h))
        h = self.dropout(h)
        h = torch.mm(adj, h)
        h = F.relu(self.w2(h))
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


class GCNModel:
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2,
                 lr=0.001, epochs=150, batch_size=2048, device='cpu'):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = GCNNet(input_dim, hidden_dim, dropout).to(device)
        self.lr = lr
        self.name = "GCN"
        self.train_losses = []
    
    def fit(self, X_train, y_train, **kwargs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        n = len(X_train)
        adj = torch.eye(n).to(self.device)  # identity for batched — will augment in pipeline
        
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0
            n_b = 0
            
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                
                batch_adj = adj[idx][:, idx]
                
                optimizer.zero_grad()
                pred = self.model(X_t[idx], batch_adj)
                loss = F.mse_loss(pred, y_t[idx])
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_b += 1
            
            self.train_losses.append(epoch_loss / n_b)
    
    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        n = len(X)
        adj = torch.eye(n).to(self.device)
        with torch.no_grad():
            return self.model(X_t, adj).cpu().numpy()


# =============================================================================
# 5. Graph-Mamba (without higher-order — ablation baseline)
# =============================================================================
class GraphMambaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, d_state=16, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, d_conv=4, expand=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x, adj):
        h = self.input_proj(x)
        h = F.relu(self.gcn(torch.mm(adj, h)))
        
        h = h.unsqueeze(0)  # (1, N, D)
        for block in self.mamba_blocks:
            h = block(h)
        h = h.squeeze(0)
        
        return self.head(h).squeeze(-1)


class GraphMambaModel:
    def __init__(self, input_dim, hidden_dim=64, d_state=16, n_layers=2,
                 dropout=0.1, lr=0.001, epochs=100, batch_size=2048, device='cpu'):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = GraphMambaNet(input_dim, hidden_dim, d_state, n_layers, dropout).to(device)
        self.lr = lr
        self.name = "Graph-Mamba"
        self.train_losses = []
    
    def fit(self, X_train, y_train, **kwargs):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        n = len(X_train)
        adj = torch.eye(n).to(self.device)
        
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0
            n_b = 0
            
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                
                batch_adj = adj[idx][:, idx]
                
                optimizer.zero_grad()
                pred = self.model(X_t[idx], batch_adj)
                loss = F.mse_loss(pred, y_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_b += 1
            
            scheduler.step()
            self.train_losses.append(epoch_loss / n_b)
    
    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        n = len(X)
        adj = torch.eye(n).to(self.device)
        with torch.no_grad():
            return self.model(X_t, adj).cpu().numpy()
