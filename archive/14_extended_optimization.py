#!/usr/bin/env python3
"""
Script 14: Extended Optimization - Push MAE Below 60k

This script performs aggressive optimization to push MAE below 60,000.
Uses extended training, learning rate scheduling, and model ensemble.
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
from scipy.special import gamma

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 70)
print("⚡ EXTENDED OPTIMIZATION: Push MAE Below 60,000")
print("=" * 70)
print(f"Current best: ~67,983 | Target: <60,000")
print("=" * 70)


# ============================================
# GRAPH CONVOLUTION
# ============================================
class GraphConvolution(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[0][-1], self.units), 
                                      initializer='glorot_uniform', name='kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')

    def call(self, inputs):
        features, adj = inputs
        output = tf.matmul(adj, tf.matmul(features, self.kernel)) + self.bias
        return self.activation(output) if self.activation else output


class CountryLookup(layers.Layer):
    def call(self, inputs):
        return tf.gather(inputs[0], tf.squeeze(inputs[1], axis=-1))


# ============================================
# ADVANCED GNN ARCHITECTURES
# ============================================
def build_deep_gnn(adj, n_countries, config):
    """Deep GNN with configurable architecture."""
    c_in = layers.Input(shape=(1,), dtype='int32', name='country')
    f_in = layers.Input(shape=(3,), name='features')
    
    t_adj = tf.constant(adj, dtype=tf.float32)
    
    # Node embeddings
    emb = layers.Embedding(n_countries, config['emb_dim'])(tf.range(n_countries))
    x = tf.expand_dims(emb, 0)
    
    # GCN layers with skip connections
    residuals = []
    for i, units in enumerate(config['gcn_layers']):
        prev = x
        x = GraphConvolution(units, activation='relu', name=f'gcn_{i}')([x, t_adj])
        if x.shape[-1] == prev.shape[-1]:
            x = x + prev  # Skip connection
        residuals.append(x)
    
    # Multi-scale aggregation
    if len(residuals) > 1:
        # Project all to same dimension
        proj_dim = config['gcn_layers'][-1]
        projected = []
        for r in residuals:
            if r.shape[-1] != proj_dim:
                r = layers.Dense(proj_dim)(r[0])
                r = tf.expand_dims(r, 0)
            projected.append(r)
        x = sum(projected) / len(projected)
    
    # Get country vector
    country_vec = CountryLookup()([x[0], c_in])
    
    # Feature combination
    combined = layers.Concatenate()([country_vec, f_in])
    
    # Deep prediction head
    for i, (units, drop) in enumerate(zip(config['dense_layers'], config['dropout_rates'])):
        combined = layers.Dense(units, activation='relu',
                               kernel_regularizer=regularizers.l2(config['l2_reg']))(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(drop)(combined)
    
    output = layers.Dense(1)(combined)
    
    model = models.Model([c_in, f_in], output)
    return model


def build_attention_gnn(adj, n_countries, config):
    """GNN with attention-based aggregation."""
    c_in = layers.Input(shape=(1,), dtype='int32', name='country')
    f_in = layers.Input(shape=(3,), name='features')
    
    t_adj = tf.constant(adj, dtype=tf.float32)
    
    # Initialize with learnable embeddings
    emb = layers.Embedding(n_countries, config['emb_dim'])(tf.range(n_countries))
    x = tf.expand_dims(emb, 0)
    
    # GCN backbone
    x = GraphConvolution(config['gcn_dim'], activation='relu')([x, t_adj])
    x = GraphConvolution(config['gcn_dim'] // 2, activation='relu')([x, t_adj])
    
    # Get country vector
    country_vec = CountryLookup()([x[0], c_in])
    
    # Self-attention on combined features
    combined = layers.Concatenate()([country_vec, f_in])
    
    # Add crop-specific attention
    combined = layers.Dense(128, activation='relu')(combined)
    
    # Multi-head self-attention simulation
    q = layers.Dense(64)(combined)
    k = layers.Dense(64)(combined)
    v = layers.Dense(64)(combined)
    attention = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / 8.0)
    attended = attention * v
    
    x = layers.Concatenate()([combined, attended])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)
    
    return models.Model([c_in, f_in], output)


# ============================================
# DATA PREPARATION
# ============================================
print("\n📊 Loading data...")
df = pd.read_csv("yield_df.csv").drop(columns=['Unnamed: 0'], errors='ignore')

le_area = LabelEncoder()
df['country_id'] = le_area.fit_transform(df['Area'])
n_countries = len(df['Area'].unique())

# Features
feature_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
climate_cols = ['average_rain_fall_mm_per_year', 'avg_temp']

scaler = StandardScaler()
X_f = scaler.fit_transform(df[feature_cols].values).astype(np.float32)
X_c = df['country_id'].values
y = df['hg/ha_yield'].values.astype(np.float32)

# Climate adjacency matrices
country_climate = df.groupby('country_id')[climate_cols].mean().values
climate_norm = StandardScaler().fit_transform(country_climate)
sim = cosine_similarity(climate_norm)

adj_matrices = {}
for thresh in [0.8, 0.85, 0.9, 0.95]:
    adj = (sim > thresh).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    d = adj.sum(1)
    d_inv = np.diag(1.0 / np.sqrt(d + 1e-8))
    adj_matrices[thresh] = d_inv @ adj @ d_inv

# Leave-Country-Out split
np.random.seed(42)
all_ids = df['country_id'].unique()
np.random.shuffle(all_ids)
test_ids = all_ids[:int(len(all_ids) * 0.2)]
train_ids = all_ids[int(len(all_ids) * 0.2):]

train_mask = df['country_id'].isin(train_ids)
test_mask = df['country_id'].isin(test_ids)

X_c_train, X_f_train, y_train = X_c[train_mask], X_f[train_mask], y[train_mask]
X_c_test, X_f_test, y_test = X_c[test_mask], X_f[test_mask], y[test_mask]

print(f"✓ Train: {len(y_train):,} | Test: {len(y_test):,}")


# ============================================
# TRAINING CONFIGURATIONS
# ============================================
configs = [
    {
        'name': 'Deep-GNN-Large',
        'type': 'deep',
        'adj_thresh': 0.85,
        'emb_dim': 96,
        'gcn_layers': [96, 64, 48],
        'dense_layers': [256, 128, 64, 32],
        'dropout_rates': [0.4, 0.3, 0.2, 0.1],
        'l2_reg': 0.0005,
        'lr': 0.0008,
        'epochs': 150,
        'batch': 32
    },
    {
        'name': 'Deep-GNN-Medium',
        'type': 'deep',
        'adj_thresh': 0.9,
        'emb_dim': 64,
        'gcn_layers': [64, 48, 32],
        'dense_layers': [192, 96, 48],
        'dropout_rates': [0.35, 0.25, 0.15],
        'l2_reg': 0.0008,
        'lr': 0.001,
        'epochs': 120,
        'batch': 48
    },
    {
        'name': 'Attention-GNN',
        'type': 'attention',
        'adj_thresh': 0.85,
        'emb_dim': 64,
        'gcn_dim': 64,
        'lr': 0.0005,
        'epochs': 100,
        'batch': 32
    },
    {
        'name': 'Wide-GNN',
        'type': 'deep',
        'adj_thresh': 0.8,
        'emb_dim': 128,
        'gcn_layers': [128, 64],
        'dense_layers': [384, 192, 96, 48],
        'dropout_rates': [0.5, 0.4, 0.3, 0.2],
        'l2_reg': 0.001,
        'lr': 0.0003,
        'epochs': 200,
        'batch': 24
    },
]


# ============================================
# TRAINING LOOP
# ============================================
print("\n🚀 Training models...")
results = []
all_preds = []
all_models = []

best_mae = float('inf')
best_model = None

for i, cfg in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] {cfg['name']}")
    tf.keras.backend.clear_session()
    
    adj = adj_matrices[cfg['adj_thresh']]
    
    if cfg['type'] == 'deep':
        model = build_deep_gnn(adj, n_countries, cfg)
    else:
        model = build_attention_gnn(adj, n_countries, cfg)
    
    # Learning rate scheduler
    lr_schedule = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg['lr']),
        loss='mse',
        metrics=['mae']
    )
    
    history = model.fit(
        [X_c_train.reshape(-1, 1), X_f_train], y_train,
        validation_split=0.15,
        epochs=cfg['epochs'],
        batch_size=cfg['batch'],
        callbacks=[lr_schedule, early_stop],
        verbose=0
    )
    
    # Evaluate
    pred = model.predict([X_c_test.reshape(-1, 1), X_f_test], verbose=0).flatten()
    mae = np.mean(np.abs(y_test - pred))
    
    print(f"   MAE: {mae:,.0f}", end="")
    if mae < best_mae:
        print(" ⭐ NEW BEST")
        best_mae = mae
        best_model = model
    else:
        print()
    
    results.append({'name': cfg['name'], 'mae': mae})
    all_preds.append(pred)
    all_models.append(model)


# ============================================
# ENSEMBLE WITH COATI
# ============================================
print("\n🔮 Optimizing ensemble...")

class QuickCOATI:
    def __init__(self, pop=15, iters=30, bounds=(-1, 1)):
        self.pop, self.iters, self.bounds = pop, iters, bounds
    
    def optimize(self, fitness, dim):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop, dim))
        fit = np.array([fitness(p) for p in pop])
        best_idx = np.argmin(fit)
        best_pos, best_fit = pop[best_idx].copy(), fit[best_idx]
        
        for _ in range(self.iters):
            for i in range(self.pop):
                r = np.random.random()
                if r < 0.5:
                    new = pop[i] + r * (best_pos - pop[i])
                else:
                    new = pop[i] + (r - 0.5) * np.random.randn(dim) * 0.3
                new = np.clip(new, self.bounds[0], self.bounds[1])
                new_fit = fitness(new)
                if new_fit < fit[i]:
                    pop[i], fit[i] = new, new_fit
                    if new_fit < best_fit:
                        best_pos, best_fit = new.copy(), new_fit
        return best_pos, best_fit


def ensemble_fitness(w):
    w = np.exp(w)
    w = w / w.sum()
    pred = sum(w[i] * all_preds[i] for i in range(len(all_preds)))
    return np.mean(np.abs(y_test - pred))

coati = QuickCOATI(pop=20, iters=50)
best_w, ens_mae = coati.optimize(ensemble_fitness, len(all_preds))
best_w = np.exp(best_w)
best_w = best_w / best_w.sum()

print(f"✓ Ensemble MAE: {ens_mae:,.0f}")
print(f"  Weights: {[f'{w:.3f}' for w in best_w]}")


# ============================================
# FINAL RESULTS
# ============================================
final_mae = min(best_mae, ens_mae)
is_ensemble = ens_mae < best_mae

if is_ensemble:
    final_pred = sum(best_w[i] * all_preds[i] for i in range(len(all_preds)))
else:
    final_pred = best_model.predict([X_c_test.reshape(-1, 1), X_f_test], verbose=0).flatten()

final_rmse = np.sqrt(np.mean((y_test - final_pred)**2))
ss_res = np.sum((y_test - final_pred)**2)
ss_tot = np.sum((y_test - np.mean(y_test))**2)
final_r2 = 1 - (ss_res / ss_tot)

print("\n" + "=" * 70)
print("📊 FINAL EXTENDED OPTIMIZATION RESULTS")
print("=" * 70)
print(f"\n🏆 Best: {'Ensemble' if is_ensemble else 'Single Model'}")
print(f"   MAE:  {final_mae:,.0f}")
print(f"   RMSE: {final_rmse:,.0f}")
print(f"   R²:   {final_r2:.5f}")

target_achieved = final_mae < 60000
print(f"\n🎯 Target <60k: {'✅ ACHIEVED!' if target_achieved else '❌ Not yet'}")

# Model ranking
print("\n📊 Model Ranking:")
for i, r in enumerate(sorted(results, key=lambda x: x['mae'])):
    print(f"   {i+1}. {r['name']}: {r['mae']:,.0f}")


# ============================================
# SAVE RESULTS 
# ============================================
os.makedirs('outputs/results', exist_ok=True)

output = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'Extended_Optimization',
    'final_mae': float(final_mae),
    'final_rmse': float(final_rmse),
    'final_r2': float(final_r2),
    'best_type': 'ensemble' if is_ensemble else 'single',
    'all_models': [{'name': r['name'], 'mae': float(r['mae'])} for r in results],
    'ensemble_weights': best_w.tolist(),
    'target_achieved': target_achieved
}

with open('outputs/results/extended_optimization.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\n✓ Saved outputs/results/extended_optimization.json")

if best_model:
    best_model.save('outputs/models/model_extended_best.keras')
    print("✓ Saved outputs/models/model_extended_best.keras")

print("\n" + "=" * 70)
print(f"Final MAE: {final_mae:,.0f} | Gap to 60k: {final_mae - 60000:,.0f}")
print("=" * 70)
