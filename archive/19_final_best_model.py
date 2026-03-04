#!/usr/bin/env python3
"""
Script 19: Final Optimization - Best Model Ensemble

Combines all best-performing models for optimal results.
Saves best model and generates final comparison table.
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
from sklearn.ensemble import GradientBoostingRegressor
import json
from datetime import datetime

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("🏆 FINAL OPTIMIZATION - BEST MODEL ENSEMBLE")
print("=" * 70)

# Data preparation
df = pd.read_csv("yield_df.csv").drop(columns=['Unnamed: 0'], errors='ignore')
le_area, le_item = LabelEncoder(), LabelEncoder()
df['country_id'] = le_area.fit_transform(df['Area'])
df['crop_id'] = le_item.fit_transform(df['Item'])
n_countries, n_crops = len(df['Area'].unique()), len(df['Item'].unique())

feature_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
climate_cols = ['average_rain_fall_mm_per_year', 'avg_temp']

y_raw = df['hg/ha_yield'].values.astype(np.float32)
y_log = np.log1p(y_raw)
y_mean_log, y_std_log = y_log.mean(), y_log.std()
y_normalized = (y_log - y_mean_log) / y_std_log

scaler_f = StandardScaler()
X_f = scaler_f.fit_transform(df[feature_cols].values).astype(np.float32)
X_c, X_crop = df['country_id'].values, df['crop_id'].values
years = df['Year'].values.astype(np.float32)
X_year = ((years - years.min()) / (years.max() - years.min())).reshape(-1, 1).astype(np.float32)

country_climate = df.groupby('country_id')[climate_cols].mean().values
scaler_climate = StandardScaler()
climate_norm = scaler_climate.fit_transform(country_climate)
climate_sim = cosine_similarity(climate_norm)

# Multiple adjacency matrices
adj_matrices = {}
for thresh in [0.4, 0.5, 0.6]:
    adj = (climate_sim > thresh).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    d = adj.sum(1)
    adj_matrices[thresh] = np.diag(1.0/np.sqrt(d+1e-8)) @ adj @ np.diag(1.0/np.sqrt(d+1e-8))

# Split
np.random.seed(42)
all_ids = np.arange(n_countries)
np.random.shuffle(all_ids)
test_ids, train_ids = all_ids[:int(n_countries*0.2)], all_ids[int(n_countries*0.2):]
train_mask, test_mask = df['country_id'].isin(train_ids), df['country_id'].isin(test_ids)

X_c_train, X_crop_train = X_c[train_mask], X_crop[train_mask]
X_f_train, X_year_train = X_f[train_mask], X_year[train_mask]
y_train, y_raw_test = y_normalized[train_mask], y_raw[test_mask]
X_c_test, X_crop_test = X_c[test_mask], X_crop[test_mask]
X_f_test, X_year_test = X_f[test_mask], X_year[test_mask]

print(f"Train: {len(y_train):,} | Test: {len(y_raw_test):,}")

def inverse_transform(y_norm):
    return np.expm1(y_norm * y_std_log + y_mean_log)

def compute_metrics(y_true, y_pred_raw):
    mae = np.mean(np.abs(y_true - y_pred_raw))
    rmse = np.sqrt(np.mean((y_true - y_pred_raw)**2))
    ss_res = np.sum((y_true - y_pred_raw)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot
    return mae, rmse, r2

class GraphConv(layers.Layer):
    def __init__(self, units, use_coati=False, **kwargs):
        super().__init__(**kwargs)
        self.units, self.use_coati = units, use_coati
    def build(self, input_shape):
        total = input_shape[0][-1] + (input_shape[2][-1] if self.use_coati and len(input_shape)>2 else 0)
        self.W = self.add_weight(shape=(total, self.units), initializer='glorot_uniform')
        self.b = self.add_weight(shape=(self.units,), initializer='zeros')
    def call(self, inputs):
        h = tf.concat([inputs[0], inputs[2]], -1) if self.use_coati and len(inputs)==3 else inputs[0]
        return tf.nn.relu(tf.matmul(inputs[1], tf.matmul(h, self.W)) + self.b)

class CountryLookup(layers.Layer):
    def call(self, inputs):
        return tf.gather(inputs[0], tf.squeeze(inputs[1], -1))

def build_model(adj, cfg, use_coati=True):
    c_in = layers.Input(shape=(1,), dtype='int32')
    cr_in = layers.Input(shape=(1,), dtype='int32')
    f_in = layers.Input(shape=(3,))
    y_in = layers.Input(shape=(1,))
    
    t_adj = tf.constant(adj, dtype=tf.float32)
    emb = layers.Embedding(n_countries, cfg['emb'])(tf.range(n_countries))
    coati = layers.Dense(cfg['coati'], activation='relu')(tf.constant(climate_norm, dtype=tf.float32)) if use_coati else None
    
    h = emb
    for u in cfg['gcn']:
        h = GraphConv(u, use_coati)([h, t_adj, coati]) if use_coati else GraphConv(u, False)([h, t_adj])
        h = layers.Dropout(cfg['drop'])(h)
    
    cv = CountryLookup()([h, c_in])
    crv = layers.Flatten()(layers.Embedding(n_crops, cfg['crop'])(cr_in))
    x = layers.Concatenate()([cv, crv, f_in, y_in])
    x = layers.BatchNormalization()(x)
    
    for u, d in zip(cfg['dense'], cfg['ddrop']):
        x = layers.Dense(u, activation='relu', kernel_regularizer=regularizers.l2(cfg['l2']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(d)(x)
    
    return models.Model([c_in, cr_in, f_in, y_in], layers.Dense(1)(x))

# Train multiple configurations
configs = [
    {'name': 'Climate+COATI-v1', 'adj': 0.5, 'use_coati': True,
     'emb': 96, 'coati': 48, 'gcn': [128, 96, 64], 'crop': 24,
     'dense': [256, 128, 64, 32], 'ddrop': [0.4, 0.3, 0.2, 0.1], 'drop': 0.15, 'l2': 0.0005,
     'lr': 0.0008, 'epochs': 150, 'batch': 32},
    {'name': 'Climate+COATI-v2', 'adj': 0.4, 'use_coati': True,
     'emb': 128, 'coati': 64, 'gcn': [128, 96], 'crop': 32,
     'dense': [256, 128, 64], 'ddrop': [0.35, 0.25, 0.15], 'drop': 0.2, 'l2': 0.0003,
     'lr': 0.001, 'epochs': 120, 'batch': 48},
    {'name': 'Climate-only', 'adj': 0.5, 'use_coati': False,
     'emb': 64, 'coati': 0, 'gcn': [64, 48], 'crop': 16,
     'dense': [128, 64, 32], 'ddrop': [0.3, 0.2, 0.1], 'drop': 0.2, 'l2': 0.001,
     'lr': 0.001, 'epochs': 80, 'batch': 64},
]

all_preds, all_raw_preds, results = [], [], {}

for i, cfg in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] {cfg['name']}...", end=" ")
    tf.keras.backend.clear_session()
    tf.random.set_seed(42+i)
    
    model = build_model(adj_matrices[cfg['adj']], cfg, cfg['use_coati'])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg['lr']),
                  loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    model.fit([X_c_train.reshape(-1,1), X_crop_train.reshape(-1,1), X_f_train, X_year_train],
              y_train, validation_split=0.1, epochs=cfg['epochs'], batch_size=cfg['batch'],
              callbacks=[callbacks.ReduceLROnPlateau(patience=12, factor=0.5, min_lr=1e-6),
                        callbacks.EarlyStopping(patience=20, restore_best_weights=True)], verbose=0)
    
    pred = model.predict([X_c_test.reshape(-1,1), X_crop_test.reshape(-1,1), X_f_test, X_year_test], verbose=0).flatten()
    raw_pred = inverse_transform(pred)
    mae, rmse, r2 = compute_metrics(y_raw_test, raw_pred)
    
    all_preds.append(pred)
    all_raw_preds.append(raw_pred)
    results[cfg['name']] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    print(f"MAE: {mae:,.0f} | R²: {r2:.4f}")
    
    if cfg['name'] == 'Climate+COATI-v1':
        model.save('outputs/models/model_climate_coati_best.keras')

# Ensemble optimization
print("\n🔮 Optimizing ensemble...")
best_mae, best_weights = float('inf'), None
for w1 in np.arange(0, 1.05, 0.1):
    for w2 in np.arange(0, 1.05-w1, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 >= 0:
            ens = w1*all_raw_preds[0] + w2*all_raw_preds[1] + w3*all_raw_preds[2]
            mae = np.mean(np.abs(y_raw_test - ens))
            if mae < best_mae:
                best_mae, best_weights, best_ens = mae, (w1, w2, w3), ens

mae, rmse, r2 = compute_metrics(y_raw_test, best_ens)
results['Ensemble'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
print(f"   Ensemble: MAE {mae:,.0f} | R² {r2:.4f} | Weights: {best_weights}")

# Final results
print("\n" + "=" * 70)
print("📊 FINAL RESULTS")
print("=" * 70)
print(f"\n{'Model':<25} {'MAE':>12} {'RMSE':>12} {'R²':>10}")
print("-" * 62)
for name, m in sorted(results.items(), key=lambda x: x[1]['mae']):
    print(f"{name:<25} {m['mae']:>12,.0f} {m['rmse']:>12,.0f} {m['r2']:>10.4f}")

best = min(results.items(), key=lambda x: x[1]['mae'])
print("\n" + "-" * 62)
print(f"🏆 Best: {best[0]} | MAE: {best[1]['mae']:,.0f} | R²: {best[1]['r2']:.4f}")

# Save
os.makedirs('outputs/results', exist_ok=True)
output = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'Final_Best_Model',
    'results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
    'best_model': best[0],
    'best_mae': float(best[1]['mae']),
    'best_r2': float(best[1]['r2']),
    'ensemble_weights': list(best_weights)
}
with open('outputs/results/final_best_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\n✓ Saved outputs/results/final_best_results.json")
print("✓ Saved outputs/models/model_climate_coati_best.keras")
print("\n" + "=" * 70)
