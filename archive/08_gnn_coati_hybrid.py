#!/usr/bin/env python3
"""
Script 08: GNN-COATI Hybrid Models
Combines Graph Neural Networks with COATI optimization for improved spatial generalization.

Hybrid Approaches:
1. COATI-Optimized Climate GNN: Best hyperparameters found via COATI
2. Adaptive Graph GNN: COATI-optimized graph structure
3. GNN-DANN-COATI Ensemble: Weighted ensemble of all models
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("Step 8: GNN-COATI Hybrid Models")
print("=" * 60)


# ============================================
# Custom Graph Convolution Layer
# ============================================
class GraphConvolution(layers.Layer):
    """Graph Convolution Layer: H' = σ(A * H * W + b)"""
    
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='kernel', trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                name='bias', trainable=True
            )
        super().build(input_shape)

    def call(self, inputs):
        features, adj = inputs
        support = tf.matmul(features, self.kernel)
        output = tf.matmul(adj, support)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config


class CountryEmbeddingLookup(layers.Layer):
    """Lookup country embeddings from graph-processed vectors."""
    
    def __init__(self, **kwargs):
        super(CountryEmbeddingLookup, self).__init__(**kwargs)
    
    def call(self, inputs):
        node_vecs, country_idx = inputs
        country_idx_flat = tf.squeeze(country_idx, axis=-1)
        return tf.gather(node_vecs, country_idx_flat)


# ============================================
# Climate-Similarity GNN with COATI Hyperparameters
# ============================================
def build_climate_gnn_coati(adj_tensor, num_countries, params=None):
    """
    Build Climate-Similarity GNN with COATI-optimized hyperparameters.
    
    Args:
        adj_tensor: Climate-similarity adjacency matrix
        num_countries: Number of countries
        params: Dict with keys: hidden_units_1, hidden_units_2, dropout, learning_rate
    """
    if params is None:
        params = {
            'hidden_units_1': 32,
            'hidden_units_2': 16,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    
    h1 = params.get('hidden_units_1', 32)
    h2 = params.get('hidden_units_2', 16)
    dropout = params.get('dropout', 0.2)
    lr = params.get('learning_rate', 0.001)
    
    # Inputs
    country_input = layers.Input(shape=(1,), dtype='int32', name='country_idx')
    other_input = layers.Input(shape=(3,), name='other_feats')
    
    # Adjacency as constant
    t_adj = tf.constant(adj_tensor, dtype=tf.float32)
    
    # Node embeddings
    embedding_layer = layers.Embedding(num_countries, h1)
    all_country_indices = tf.range(num_countries)
    node_embeddings = embedding_layer(all_country_indices)
    x_graph = tf.expand_dims(node_embeddings, 0)
    
    # GCN Layers with dropout
    x_graph = GraphConvolution(h1, activation='relu', name='gcn1')([x_graph, t_adj])
    x_graph = GraphConvolution(h2, activation='relu', name='gcn2')([x_graph, t_adj])
    
    # Get node vectors
    node_vecs = x_graph[0]
    specific_country_vec = CountryEmbeddingLookup()([node_vecs, country_input])
    
    # Concatenate with features
    concat = layers.Concatenate()([specific_country_vec, other_input])
    
    # Prediction head with dropout
    x = layers.Dropout(dropout)(concat)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, name='yield_output')(x)
    
    model = models.Model(inputs=[country_input, other_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    
    return model


# ============================================
# Adaptive Climate-Aware GNN (CR-GNN)
# ============================================
def build_climate_regime_gnn(adj_tensor, num_countries, climate_vectors, params=None):
    """
    Climate-Regime Aware GNN (CR-GNN).
    Uses climate regime information for enhanced knowledge transfer.
    
    Args:
        adj_tensor: Climate-similarity adjacency matrix
        num_countries: Number of countries  
        climate_vectors: Climate embeddings per country (num_countries, climate_dim)
        params: Hyperparameters dict
    """
    if params is None:
        params = {
            'hidden_units_1': 48,
            'hidden_units_2': 24,
            'dropout': 0.15,
            'learning_rate': 0.001
        }
    
    h1 = params.get('hidden_units_1', 48)
    h2 = params.get('hidden_units_2', 24)
    dropout = params.get('dropout', 0.15)
    lr = params.get('learning_rate', 0.001)
    climate_dim = climate_vectors.shape[1] if climate_vectors is not None else 0
    
    # Inputs
    country_input = layers.Input(shape=(1,), dtype='int32', name='country_idx')
    other_input = layers.Input(shape=(3,), name='other_feats')
    
    # Adjacency as constant
    t_adj = tf.constant(adj_tensor, dtype=tf.float32)
    
    # Node embeddings initialized with climate vectors
    if climate_vectors is not None:
        climate_embedding = tf.constant(climate_vectors, dtype=tf.float32)
        # Project climate vectors to embedding dimension
        climate_proj = layers.Dense(h1, activation='relu')
        node_embeddings = climate_proj(climate_embedding)
    else:
        embedding_layer = layers.Embedding(num_countries, h1)
        all_country_indices = tf.range(num_countries)
        node_embeddings = embedding_layer(all_country_indices)
    
    x_graph = tf.expand_dims(node_embeddings, 0)
    
    # Multi-layer GCN for climate-aware aggregation
    x_graph = GraphConvolution(h1, activation='relu', name='cr_gcn1')([x_graph, t_adj])
    x_graph = GraphConvolution(h2, activation='relu', name='cr_gcn2')([x_graph, t_adj])
    x_graph = GraphConvolution(h2, activation='relu', name='cr_gcn3')([x_graph, t_adj])
    
    # Get node vectors
    node_vecs = x_graph[0]
    specific_country_vec = CountryEmbeddingLookup()([node_vecs, country_input])
    
    # Concatenate with features
    concat = layers.Concatenate()([specific_country_vec, other_input])
    
    # Enhanced prediction head
    x = layers.Dropout(dropout)(concat)
    x = layers.Dense(96, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(48, activation='relu')(x)
    x = layers.Dense(24, activation='relu')(x)
    output = layers.Dense(1, name='yield_output')(x)
    
    model = models.Model(inputs=[country_input, other_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    
    return model


# ============================================
# GNN-DANN-COATI Ensemble Model
# ============================================
class EnsembleModel:
    """
    Ensemble combining GNN, DANN, and Climate-Aware models.
    Uses COATI-optimized weights for combining predictions.
    """
    
    def __init__(self, models_dict, weights=None):
        """
        Args:
            models_dict: Dict of name -> model
            weights: Dict of name -> weight (will be normalized)
        """
        self.models = models_dict
        self.model_names = list(models_dict.keys())
        
        if weights is None:
            # Equal weights
            n = len(self.model_names)
            self.weights = {name: 1.0/n for name in self.model_names}
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {k: v/total for k, v in weights.items()}
    
    def predict(self, X):
        """Make weighted ensemble prediction."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X, verbose=0).flatten()
        
        # Weighted average
        ensemble_pred = np.zeros(len(predictions[self.model_names[0]]))
        for name in self.model_names:
            ensemble_pred += self.weights[name] * predictions[name]
        
        return ensemble_pred
    
    def evaluate(self, X, y_true):
        """Evaluate ensemble performance."""
        pred = self.predict(X)
        mae = np.mean(np.abs(pred - y_true))
        rmse = np.sqrt(np.mean((pred - y_true)**2))
        
        # R² score
        ss_res = np.sum((y_true - pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ============================================
# Adaptive Graph Structure with COATI
# ============================================
def build_adaptive_graph(climate_vectors, threshold=0.95):
    """
    Build adaptive climate-similarity graph.
    
    Args:
        climate_vectors: Climate embeddings (num_countries, climate_dim)
        threshold: Cosine similarity threshold for edge creation
        
    Returns:
        Normalized adjacency matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    
    # Normalize climate vectors
    scaler = StandardScaler()
    climate_norm = scaler.fit_transform(climate_vectors)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(climate_norm)
    
    # Apply threshold
    adj = (sim_matrix > threshold).astype(np.float32)
    
    # Add self-loops
    np.fill_diagonal(adj, 1.0)
    
    # Symmetric normalization: D^(-0.5) * A * D^(-0.5)
    degrees = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    
    edge_count = int(adj.sum() - adj.shape[0])  # Exclude self-loops
    print(f"✓ Adaptive graph: {adj.shape[0]} nodes, {edge_count} edges (threshold={threshold:.3f})")
    
    return adj_norm


# ============================================
# TEST HYBRID MODELS
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 Testing GNN-COATI Hybrid Models")
    print("=" * 60)
    
    # Load preprocessed data
    print("\n📊 Loading data...")
    data = np.load('scripts/preprocessed_data.npz')
    num_countries = int(data['num_countries'])
    adj_climate_norm = np.load('scripts/adj_climate_norm.npy')
    climate_vectors = np.load('scripts/country_climate_vectors.npy')
    
    print(f"✓ Loaded data: {num_countries} countries")
    print(f"✓ Climate vectors shape: {climate_vectors.shape}")
    print(f"✓ Adjacency matrix shape: {adj_climate_norm.shape}")
    
    # Test Climate GNN with COATI params
    print("\n🧠 Testing Climate GNN with COATI hyperparameters...")
    coati_params = {
        'hidden_units_1': 48,
        'hidden_units_2': 24,
        'dropout': 0.15,
        'learning_rate': 0.001
    }
    
    model_climate_coati = build_climate_gnn_coati(
        adj_climate_norm, num_countries, params=coati_params
    )
    print(f"✓ Climate GNN-COATI model created: {model_climate_coati.count_params():,} params")
    
    # Test Climate-Regime GNN
    print("\n🧠 Testing Climate-Regime Aware GNN (CR-GNN)...")
    model_cr_gnn = build_climate_regime_gnn(
        adj_climate_norm, num_countries, climate_vectors
    )
    print(f"✓ CR-GNN model created: {model_cr_gnn.count_params():,} params")
    
    # Test Adaptive Graph
    print("\n🔗 Testing Adaptive Graph Construction...")
    adj_adaptive = build_adaptive_graph(climate_vectors, threshold=0.9)
    
    # Quick forward pass
    print("\n🧪 Testing forward pass...")
    X_c_test = np.array([[0], [1], [2]])
    X_f_test = np.random.randn(3, 3).astype(np.float32)
    
    pred1 = model_climate_coati.predict([X_c_test, X_f_test], verbose=0)
    pred2 = model_cr_gnn.predict([X_c_test, X_f_test], verbose=0)
    
    print(f"✓ Climate GNN-COATI predictions: {pred1.flatten()}")
    print(f"✓ CR-GNN predictions: {pred2.flatten()}")
    
    print("\n✅ GNN-COATI Hybrid Models ready!")
