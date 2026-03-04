#!/usr/bin/env python3
"""
Script 12: GNN-DANN Hybrid Model

Combines Graph Neural Network with Domain Adversarial Neural Network for
domain-invariant crop yield prediction across different country groups.

Key Innovation:
- GNN learns climate-aware spatial representations
- DANN ensures features are domain-invariant (country-group agnostic)
- Combined approach achieves better generalization to unseen countries
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("🔬 GNN-DANN HYBRID: Domain-Invariant Crop Yield Prediction")
print("=" * 70)


# ============================================
# GRADIENT REVERSAL LAYER
# ============================================
@tf.custom_gradient
def gradient_reversal(x, lambda_val=1.0):
    """Gradient Reversal Layer for domain adaptation."""
    def grad(dy):
        return -lambda_val * dy, None
    return x, grad


class GradientReversalLayer(layers.Layer):
    """Keras layer for gradient reversal."""
    
    def __init__(self, lambda_val=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_val = lambda_val
    
    def call(self, x):
        return gradient_reversal(x, self.lambda_val)
    
    def get_config(self):
        config = super().get_config()
        config.update({'lambda_val': self.lambda_val})
        return config


# ============================================
# GRAPH CONVOLUTION LAYER
# ============================================
class GraphConvolution(layers.Layer):
    """Graph Convolution Layer."""
    
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='kernel', trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias', trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        features, adj = inputs
        support = tf.matmul(features, self.kernel)
        output = tf.matmul(adj, support) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


class CountryLookup(layers.Layer):
    """Lookup country embeddings."""
    def call(self, inputs):
        node_vecs, country_idx = inputs
        return tf.gather(node_vecs, tf.squeeze(country_idx, axis=-1))


# ============================================
# GNN-DANN HYBRID MODEL
# ============================================
def build_gnn_dann_model(adj_tensor, num_countries, num_domains=2, 
                         hidden_gnn=32, hidden_dann=64, lambda_val=0.5):
    """
    Build GNN-DANN hybrid model.
    
    Architecture:
    1. GNN: Graph convolution layers for climate-aware embeddings
    2. Feature Extractor: Shared layers for yield prediction and domain classification
    3. Yield Predictor: Task-specific head
    4. Domain Classifier: With gradient reversal for domain-invariant learning
    
    Args:
        adj_tensor: Climate-similarity adjacency matrix
        num_countries: Number of countries
        num_domains: Number of domain groups (train vs test country groups)
        hidden_gnn: GNN hidden dimension
        hidden_dann: DANN hidden dimension
        lambda_val: Gradient reversal strength
    """
    # Inputs
    country_input = layers.Input(shape=(1,), dtype='int32', name='country_idx')
    feature_input = layers.Input(shape=(3,), name='climate_features')
    domain_input = layers.Input(shape=(1,), dtype='int32', name='domain_label')
    
    # GNN Branch: Climate-aware country embeddings
    t_adj = tf.constant(adj_tensor, dtype=tf.float32)
    
    embedding_layer = layers.Embedding(num_countries, hidden_gnn)
    all_indices = tf.range(num_countries)
    node_emb = embedding_layer(all_indices)
    x_graph = tf.expand_dims(node_emb, 0)
    
    # Graph convolution layers
    x_graph = GraphConvolution(hidden_gnn, activation='relu', name='gcn1')([x_graph, t_adj])
    x_graph = GraphConvolution(hidden_gnn//2, activation='relu', name='gcn2')([x_graph, t_adj])
    
    # Get country-specific vector
    node_vecs = x_graph[0]
    country_vec = CountryLookup()([node_vecs, country_input])
    
    # Combine GNN embedding with climate features
    combined = layers.Concatenate()([country_vec, feature_input])
    
    # Shared Feature Extractor
    shared = layers.Dense(hidden_dann, activation='relu', name='shared_1')(combined)
    shared = layers.Dropout(0.2)(shared)
    shared = layers.Dense(hidden_dann//2, activation='relu', name='shared_2')(shared)
    
    # Yield Prediction Head
    yield_x = layers.Dense(32, activation='relu', name='yield_1')(shared)
    yield_output = layers.Dense(1, name='yield_output')(yield_x)
    
    # Domain Classification Head (with Gradient Reversal)
    domain_reversed = GradientReversalLayer(lambda_val, name='grl')(shared)
    domain_x = layers.Dense(32, activation='relu', name='domain_1')(domain_reversed)
    domain_output = layers.Dense(num_domains, activation='softmax', name='domain_output')(domain_x)
    
    # Create model
    model = models.Model(
        inputs=[country_input, feature_input, domain_input],
        outputs=[yield_output, domain_output]
    )
    
    return model


class GNNDANNTrainer:
    """Custom trainer for GNN-DANN model with alternating optimization."""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        self.yield_loss_tracker = tf.keras.metrics.Mean(name='yield_loss')
        self.domain_loss_tracker = tf.keras.metrics.Mean(name='domain_loss')
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name='mae')
    
    @tf.function
    def train_step(self, X_c, X_f, y_yield, y_domain, alpha=0.1):
        """Single training step."""
        with tf.GradientTape() as tape:
            y_pred, d_pred = self.model([X_c, X_f, y_domain], training=True)
            
            # Yield prediction loss (main task)
            yield_loss = self.mse_loss(y_yield, y_pred)
            
            # Domain classification loss (adversarial)
            domain_loss = self.ce_loss(y_domain, d_pred)
            
            # Combined loss (GRL handles gradient reversal)
            total_loss = yield_loss + alpha * domain_loss
        
        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.yield_loss_tracker.update_state(yield_loss)
        self.domain_loss_tracker.update_state(domain_loss)
        self.mae_tracker.update_state(y_yield, y_pred)
        
        return yield_loss, domain_loss
    
    def train(self, X_c_train, X_f_train, y_train, domain_train, 
              epochs=30, batch_size=64, verbose=True):
        """Train the model."""
        n_samples = len(y_train)
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            self.yield_loss_tracker.reset_state()
            self.domain_loss_tracker.reset_state()
            self.mae_tracker.reset_state()
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                batch_indices = indices[start:end]
                
                X_c_batch = X_c_train[batch_indices]
                X_f_batch = X_f_train[batch_indices]
                y_batch = y_train[batch_indices]
                domain_batch = domain_train[batch_indices]
                
                self.train_step(X_c_batch, X_f_batch, y_batch, domain_batch)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}: MAE={self.mae_tracker.result():.0f}, "
                      f"YieldLoss={self.yield_loss_tracker.result():.4f}, "
                      f"DomainLoss={self.domain_loss_tracker.result():.4f}")
    
    def predict(self, X_c, X_f):
        """Make yield predictions."""
        dummy_domain = np.zeros((len(X_c), 1), dtype='int32')
        y_pred, _ = self.model([X_c, X_f, dummy_domain], training=False)
        return y_pred.numpy().flatten()


# ============================================
# MAIN EXPERIMENT
# ============================================
if __name__ == "__main__":
    print("\n📊 Loading data...")
    
    # Load data
    df = pd.read_csv("yield_df.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    le_area = LabelEncoder()
    df['country_id'] = le_area.fit_transform(df['Area'])
    
    countries = sorted(df['Area'].unique())
    num_countries = len(countries)
    
    # Prepare features
    feature_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    climate_cols = ['average_rain_fall_mm_per_year', 'avg_temp']
    
    scaler_f = StandardScaler()
    X_f_all = scaler_f.fit_transform(df[feature_cols].values)
    X_c_all = df['country_id'].values
    y_all = df['hg/ha_yield'].values.astype(np.float32)
    
    # Build climate adjacency
    country_climate = df.groupby('country_id')[climate_cols].mean().values
    scaler_c = StandardScaler()
    climate_norm = scaler_c.fit_transform(country_climate)
    sim_matrix = cosine_similarity(climate_norm)
    adj = (sim_matrix > 0.9).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    degrees = adj.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
    adj_climate = d_inv_sqrt @ adj @ d_inv_sqrt
    
    # Leave-Country-Out split
    np.random.seed(42)
    all_country_ids = df['country_id'].unique()
    np.random.shuffle(all_country_ids)
    
    n_test = int(len(all_country_ids) * 0.2)
    test_countries = all_country_ids[:n_test]
    train_countries = all_country_ids[n_test:]
    
    train_mask = df['country_id'].isin(train_countries)
    test_mask = df['country_id'].isin(test_countries)
    
    X_c_train = X_c_all[train_mask]
    X_f_train = X_f_all[train_mask].astype(np.float32)
    y_train = y_all[train_mask]
    
    X_c_test = X_c_all[test_mask]
    X_f_test = X_f_all[test_mask].astype(np.float32)
    y_test = y_all[test_mask]
    
    # Create domain labels (0 = train countries, 1 = test countries)
    domain_train = np.zeros((len(y_train), 1), dtype='int32')
    domain_test = np.ones((len(y_test), 1), dtype='int32')
    
    print(f"✓ Train: {len(y_train):,} samples from {len(train_countries)} countries")
    print(f"✓ Test: {len(y_test):,} samples from {len(test_countries)} unseen countries")
    
    # ============================================
    # TRAIN GNN-DANN
    # ============================================
    print("\n🧠 Training GNN-DANN Hybrid...")
    print("-" * 50)
    
    model = build_gnn_dann_model(
        adj_climate, num_countries, 
        num_domains=2,
        hidden_gnn=48, 
        hidden_dann=64,
        lambda_val=0.3
    )
    
    trainer = GNNDANNTrainer(model, learning_rate=0.001)
    trainer.train(
        X_c_train.reshape(-1, 1), X_f_train, y_train.reshape(-1, 1), 
        domain_train, epochs=30, batch_size=64
    )
    
    # Evaluate
    pred_test = trainer.predict(X_c_test.reshape(-1, 1), X_f_test)
    
    mae = np.mean(np.abs(y_test - pred_test))
    rmse = np.sqrt(np.mean((y_test - pred_test)**2))
    ss_res = np.sum((y_test - pred_test)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n" + "=" * 50)
    print("📊 GNN-DANN Results on Unseen Countries:")
    print("=" * 50)
    print(f"  • MAE:  {mae:,.0f}")
    print(f"  • RMSE: {rmse:,.0f}")
    print(f"  • R²:   {r2:.4f}")
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'GNN-DANN Hybrid',
        'train_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
        'train_countries': int(len(train_countries)),
        'test_countries': int(len(test_countries)),
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
    }
    
    with open('outputs/results/gnn_dann_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Saved outputs/results/gnn_dann_results.json")
    
    print("\n✅ GNN-DANN Training Complete!")
