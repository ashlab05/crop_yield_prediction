#!/usr/bin/env python3
"""
Script 03: GNN Model Definition
Defines the GraphConvolution layer and GNN model.
Fixed for Keras 3.x compatibility.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Step 3: GNN Model Definition")
print("=" * 60)

# Define Custom Graph Convolution Layer
class GraphConvolution(layers.Layer):
    """
    Custom Graph Convolution Layer for GNN.
    Implements: H' = sigma(A * H * W + b)
    """
    
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
            name='kernel', 
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                name='bias', 
                trainable=True
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
    """Custom layer to lookup country embeddings from graph-processed vectors."""
    
    def __init__(self, **kwargs):
        super(CountryEmbeddingLookup, self).__init__(**kwargs)
    
    def call(self, inputs):
        node_vecs, country_idx = inputs
        # node_vecs: (Nodes, Features)
        # country_idx: (Batch, 1) - indices
        country_idx_flat = tf.squeeze(country_idx, axis=-1)  # (Batch,)
        result = tf.gather(node_vecs, country_idx_flat)  # (Batch, Features)
        return result


print("✓ GraphConvolution layer defined")
print("✓ CountryEmbeddingLookup layer defined")

def build_gnn_model(adj_tensor, num_countries):
    """
    Build a GNN model for spatial generalization.
    Uses Keras 3.x compatible custom layers.
    """
    # Inputs
    country_input = layers.Input(shape=(1,), dtype='int32', name='country_idx')
    other_input = layers.Input(shape=(3,), name='other_feats')
    
    # Adjacency as constant
    t_adj = tf.constant(adj_tensor, dtype=tf.float32)
    
    # Initialize node embeddings for all countries
    embedding_layer = layers.Embedding(num_countries, 32)
    all_country_indices = tf.range(num_countries)
    node_embeddings = embedding_layer(all_country_indices)  # (Nodes, 32)
    x_graph = tf.expand_dims(node_embeddings, 0)  # (1, Nodes, 32)
    
    # GCN Layers - aggregate from neighbors
    x_graph = GraphConvolution(32, activation='relu', name='gcn1')([x_graph, t_adj])
    x_graph = GraphConvolution(16, activation='relu', name='gcn2')([x_graph, t_adj])
    
    # Get the node vectors (squeeze batch dim)
    node_vecs = x_graph[0]  # (Nodes, 16)
    
    # Use custom layer for country lookup (Keras 3.x compatible)
    specific_country_vec = CountryEmbeddingLookup()([node_vecs, country_input])
    
    # Concatenate country embedding with other features
    concat = layers.Concatenate()([specific_country_vec, other_input])
    
    # Prediction head
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, name='yield_output')(x)
    
    model = models.Model(inputs=[country_input, other_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Test model creation
print("\n🧪 Testing GNN model creation...")
data = np.load('scripts/preprocessed_data.npz')
num_countries = int(data['num_countries'])
adj_climate_norm = np.load('scripts/adj_climate_norm.npy')

test_model = build_gnn_model(adj_climate_norm, num_countries)
print(f"✓ GNN model created successfully")
print(f"  • Input shapes: {[i.shape for i in test_model.inputs]}")
print(f"  • Output shape: {test_model.output.shape}")
print(f"  • Total params: {test_model.count_params():,}")

# Quick forward pass test
X_c_test_sample = np.array([[0], [1], [2]])
X_f_test_sample = np.random.randn(3, 3).astype(np.float32)
pred = test_model.predict([X_c_test_sample, X_f_test_sample], verbose=0)
print(f"  • Test prediction shape: {pred.shape}")

print("\n✅ GNN model definition complete!")
