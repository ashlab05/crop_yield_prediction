#!/usr/bin/env python3
"""
Script 02: Climate Graph Construction
Builds climate-similarity adjacency matrix for GNN.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Step 2: Climate Graph Construction")
print("=" * 60)

# Load country climate vectors
print("\n📊 Loading country climate vectors...")
country_climate_vectors = np.load('scripts/country_climate_vectors.npy')
num_countries = len(country_climate_vectors)
print(f"✓ Loaded vectors for {num_countries} countries")

# Normalize vectors
print("\n🔧 Normalizing climate vectors...")
scaler_clim = StandardScaler()
country_climate_vectors_scaled = scaler_clim.fit_transform(country_climate_vectors)

# Compute Cosine Similarity
print("🔧 Computing cosine similarity...")
sim_matrix = cosine_similarity(country_climate_vectors_scaled)

# Threshold to create graph (countries with similar climates)
threshold = 0.95
adj_climate = (sim_matrix > threshold).astype(float)
np.fill_diagonal(adj_climate, 1.0)  # Self-loops

# Geographic adjacency (Identity matrix as baseline)
adj_geo = np.eye(num_countries)

def normalize_adj(adj):
    """Normalize adjacency matrix (D^-0.5 A D^-0.5 - GCN standard)."""
    d = np.diag(np.sum(adj, axis=1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return d_inv_sqrt.dot(adj).dot(d_inv_sqrt)

adj_climate_norm = normalize_adj(adj_climate)
adj_geo_norm = normalize_adj(adj_geo)

print(f"\n📈 Climate Graph Statistics:")
print(f"  • Total edges: {int(np.sum(adj_climate))}")
print(f"  • Nodes: {num_countries}")
print(f"  • Density: {np.sum(adj_climate) / (num_countries**2):.4f}")

# Save adjacency matrices
print("\n💾 Saving adjacency matrices...")
np.save('scripts/adj_climate_norm.npy', adj_climate_norm)
np.save('scripts/adj_geo_norm.npy', adj_geo_norm)
print("✓ Saved adj_climate_norm.npy")
print("✓ Saved adj_geo_norm.npy")

print("\n✅ Climate graph construction complete!")
