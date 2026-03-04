"""
Higher-Order Graph Construction
================================
Builds a combinatorial complex graph for the crop yield dataset.

Rank-0: Data point nodes
Rank-1: Pairwise edges (k-NN by feature similarity)
Rank-2: Higher-order cells (triangles from feature interaction groups)
"""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import KBinsDiscretizer


def build_knn_graph(X, k=10):
    """
    Build k-nearest neighbor adjacency from feature matrix.
    Returns edge_index (2, num_edges) as numpy array.
    """
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    src = []
    dst = []
    for i in range(len(X)):
        for j in indices[i][1:]:  # skip self
            src.append(i)
            dst.append(j)
            # make undirected
            src.append(j)
            dst.append(i)
    
    edge_index = np.array([src, dst])
    # Remove duplicates
    edge_set = set()
    unique_src, unique_dst = [], []
    for s, d in zip(src, dst):
        if (s, d) not in edge_set:
            edge_set.add((s, d))
            unique_src.append(s)
            unique_dst.append(d)
    
    return np.array([unique_src, unique_dst])


def build_higher_order_cells(X_raw, n_bins=5):
    """
    Build higher-order (rank-2) cells by grouping nodes that share
    discretized feature interaction patterns.
    
    Groups are formed by: (rainfall_bin, temp_bin, pesticide_bin)
    Nodes in the same group form a higher-order cell.
    
    Returns:
        higher_order_cells: list of lists (each inner list = node indices in a cell)
        cell_features: numpy array of cell-level features (mean of member features)
    """
    # Discretize continuous features for grouping
    # X_raw columns: [country_id, crop_id, Year, rainfall, pesticides, temp]
    continuous_cols = X_raw[:, 3:]  # rainfall, pesticides, temp
    
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    binned = discretizer.fit_transform(continuous_cols).astype(int)
    
    # Group by (rainfall_bin, temp_bin) — higher-order interaction cells
    groups = {}
    for i in range(len(binned)):
        key = (int(binned[i, 0]), int(binned[i, 2]))  # rainfall_bin, temp_bin
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    
    # Filter: only keep groups with 3+ members (meaningful higher-order cells)
    higher_order_cells = [nodes for nodes in groups.values() if len(nodes) >= 3]
    
    # Compute cell-level features (mean of member features)
    cell_features = []
    for cell_nodes in higher_order_cells:
        cell_feat = X_raw[cell_nodes].mean(axis=0)
        cell_features.append(cell_feat)
    
    cell_features = np.array(cell_features) if cell_features else np.zeros((0, X_raw.shape[1]))
    
    return higher_order_cells, cell_features


def build_boundary_matrices(edge_index, higher_order_cells, num_nodes):
    """
    Build boundary operator matrices for the combinatorial complex.
    
    B1: (num_nodes, num_edges) — maps edges to their boundary nodes
    B2: (num_edges, num_cells) — maps higher-order cells to their boundary edges
    
    For efficiency, we return sparse-like representations as dense tensors,
    but clip to manageable sizes.
    """
    num_edges = edge_index.shape[1]
    
    # B1: node-edge incidence (simplified: each edge connects 2 nodes)
    # For message passing, we just use the adjacency directly
    # Build adjacency matrix from edge_index
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_edges):
        adj[edge_index[0, i], edge_index[1, i]] = 1.0
    
    # Degree normalization (symmetric)
    deg = adj.sum(dim=1).clamp(min=1)
    deg_inv_sqrt = deg.pow(-0.5)
    adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    
    # Higher-order adjacency: nodes in same cell are connected
    adj_ho = torch.zeros(num_nodes, num_nodes)
    for cell_nodes in higher_order_cells:
        for i in cell_nodes:
            for j in cell_nodes:
                if i != j:
                    adj_ho[i, j] = 1.0
    
    # Normalize higher-order adjacency
    deg_ho = adj_ho.sum(dim=1).clamp(min=1)
    deg_ho_inv_sqrt = deg_ho.pow(-0.5)
    adj_ho_norm = deg_ho_inv_sqrt.unsqueeze(1) * adj_ho * deg_ho_inv_sqrt.unsqueeze(0)
    
    return adj_norm, adj_ho_norm


def build_combinatorial_complex(X, k=10, n_bins=5, max_nodes=5000, device='cpu'):
    """
    Full pipeline: build the combinatorial complex graph.
    
    For large datasets, we subsample for the adjacency computation
    and use the full dataset for features.
    
    Returns dict with adjacency matrices and higher-order info.
    """
    num_nodes = len(X)
    
    print(f"  Building k-NN graph (k={k})...")
    
    # For large datasets, build adjacency on a subsample then extend
    if num_nodes > max_nodes:
        # Use feature-based adjacency: build on unique feature patterns
        # Sample indices for adjacency construction
        sample_idx = np.random.choice(num_nodes, max_nodes, replace=False)
        X_sample = X[sample_idx]
        
        edge_index_sample = build_knn_graph(X_sample, k)
        
        # Map back to original indices
        edge_index = np.array([
            sample_idx[edge_index_sample[0]],
            sample_idx[edge_index_sample[1]]
        ])
        
        print(f"  Subsampled {max_nodes}/{num_nodes} nodes for adjacency")
    else:
        edge_index = build_knn_graph(X, k)
    
    num_edges = edge_index.shape[1]
    print(f"  Edges: {num_edges}")
    
    print(f"  Building higher-order cells...")
    higher_order_cells, cell_features = build_higher_order_cells(X, n_bins)
    print(f"  Higher-order cells: {len(higher_order_cells)}")
    
    # For very large graphs, limit higher-order cell size
    max_cell_size = 200
    higher_order_cells = [
        cell[:max_cell_size] if len(cell) > max_cell_size else cell
        for cell in higher_order_cells
    ]
    
    print(f"  Building boundary matrices...")
    adj_norm, adj_ho_norm = build_boundary_matrices(
        edge_index, higher_order_cells, num_nodes
    )
    
    return {
        'adj_norm': adj_norm.to(device),
        'adj_ho_norm': adj_ho_norm.to(device),
        'edge_index': torch.tensor(edge_index, dtype=torch.long).to(device),
        'higher_order_cells': higher_order_cells,
        'cell_features': torch.tensor(cell_features, dtype=torch.float32).to(device),
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_cells': len(higher_order_cells),
    }
