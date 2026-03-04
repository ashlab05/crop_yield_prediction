"""
Mamba Block — Selective State Space Model
==========================================
Pure PyTorch implementation of the Mamba selective SSM block.
Compatible with M4 Air (CPU/MPS, no CUDA kernels needed).

Based on: Gu & Dao (2024) "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MambaBlock(nn.Module):
    """
    Selective State Space Model (Mamba) block.
    
    Components:
    1. Input projection (expand)
    2. 1D depthwise convolution
    3. Selective SSM with data-dependent Δ, B, C
    4. Output projection
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection: x -> (z, x_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D conv on the x branch
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        
        # SSM parameters projection: x -> (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Δ (delta) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # A parameter (initialized as log of a structured matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def selective_scan(self, x, delta, A, B, C, D):
        """
        Selective scan (sequential SSM step).
        
        x: (batch, seq_len, d_inner)
        delta: (batch, seq_len, d_inner) 
        A: (d_inner, d_state)
        B: (batch, seq_len, d_state)
        C: (batch, seq_len, d_state)
        D: (d_inner,)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B using delta
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)
        
        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, D)
        y = y + x * D.unsqueeze(0).unsqueeze(0)
        
        return y
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        
        # Project and split
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)
        
        # Conv branch
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :x.shape[1]]  # trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_branch = F.silu(x_conv)
        
        # SSM parameters from x
        ssm_params = self.x_proj(x_branch)
        delta_raw = ssm_params[:, :, :1]  # (B, L, 1)
        B = ssm_params[:, :, 1:1 + self.d_state]  # (B, L, d_state)
        C = ssm_params[:, :, 1 + self.d_state:]  # (B, L, d_state)
        
        # Delta softplus
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, d_inner)
        
        # Get A 
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Selective scan
        y = self.selective_scan(x_branch, delta, A, B, C, self.D)
        
        # Gate with z
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual
