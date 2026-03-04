#!/usr/bin/env python3
"""
Script 04: GNN Training and Evaluation
Trains Climate-GNN and compares with baselines.
Saves figures and results to outputs/ folder.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import json
from datetime import datetime

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf.random.set_seed(42)
np.random.seed(42)

# Create output directories
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 60)
print("Step 4: GNN Training and Evaluation")
print("=" * 60)

# Import model components from script 03
exec(open('scripts/03_gnn_model.py').read().split('# Test model creation')[0])

# Load data
print("\n📊 Loading preprocessed data...")
data = np.load('scripts/preprocessed_data.npz')
X_c_train = data['X_c_train']
X_f_train = data['X_f_train']
y_train = data['y_train']
X_c_test = data['X_c_test']
X_f_test = data['X_f_test']
y_test = data['y_test']
num_countries = int(data['num_countries'])

adj_climate_norm = np.load('scripts/adj_climate_norm.npy')
adj_geo_norm = np.load('scripts/adj_geo_norm.npy')

print(f"✓ Train samples: {len(y_train)}")
print(f"✓ Test samples (unseen countries): {len(y_test)}")

# Train Climate-Similarity GNN
print("\n" + "=" * 60)
print("🧠 Training Climate-Similarity GNN...")
print("=" * 60)
model_climate = build_gnn_model(adj_climate_norm, num_countries)
history_climate = model_climate.fit(
    [X_c_train, X_f_train], y_train,
    validation_data=([X_c_test, X_f_test], y_test),
    epochs=15, batch_size=64, verbose=1
)

# Train Geographic GNN (baseline)
print("\n" + "=" * 60)
print("🧠 Training Geographic GNN (Baseline)...")
print("=" * 60)
model_geo = build_gnn_model(adj_geo_norm, num_countries)
history_geo = model_geo.fit(
    [X_c_train, X_f_train], y_train,
    validation_data=([X_c_test, X_f_test], y_test),
    epochs=15, batch_size=64, verbose=0
)
print("✓ Training complete")

# Train MLP (country-blind baseline)
print("\n" + "=" * 60)
print("🧠 Training MLP Baseline (Country-Blind)...")
print("=" * 60)
input_mlp = layers.Input(shape=(3,))
x = layers.Dense(128, activation='relu')(input_mlp)
x = layers.Dense(64, activation='relu')(x)
out = layers.Dense(1)(x)
model_mlp = models.Model(input_mlp, out)
model_mlp.compile(loss='mse', metrics=['mae'], optimizer='adam')
history_mlp = model_mlp.fit(X_f_train, y_train, validation_data=(X_f_test, y_test), epochs=15, verbose=0)
print("✓ Training complete")

# Evaluate all models
print("\n" + "=" * 60)
print("📊 GNN SPATIAL GENERALIZATION RESULTS")
print("=" * 60)

mae_climate = model_climate.evaluate([X_c_test, X_f_test], y_test, verbose=0)[1]
mae_geo = model_geo.evaluate([X_c_test, X_f_test], y_test, verbose=0)[1]
mae_mlp = model_mlp.evaluate(X_f_test, y_test, verbose=0)[1]

print(f"\n{'Model':<35} {'MAE on Unseen Countries':<25}")
print("-" * 60)
print(f"{'1. MLP (No Country Info)':<35} {mae_mlp:>20,.0f}")
print(f"{'2. Geographic GNN (Identity)':<35} {mae_geo:>20,.0f}")
print(f"{'3. Climate-Similarity GNN (Ours)':<35} {mae_climate:>20,.0f}")

# Save results to JSON
results = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'GNN_Spatial_Generalization',
    'train_samples': int(len(y_train)),
    'test_samples': int(len(y_test)),
    'results': {
        'MLP_baseline': {'mae': float(mae_mlp)},
        'Geographic_GNN': {'mae': float(mae_geo)},
        'Climate_GNN': {'mae': float(mae_climate)}
    }
}
with open('outputs/results/gnn_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved results to outputs/results/gnn_results.json")

# Create and save figures
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model Comparison Bar Chart
ax1 = axes[0]
models_names = ['MLP\n(Country-Blind)', 'Geographic GNN\n(Baseline)', 'Climate GNN\n(Ours)']
maes = [mae_mlp, mae_geo, mae_climate]
colors = ['#e74c3c', '#3498db', '#2ecc71']
bars = ax1.bar(models_names, maes, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('MAE on Unseen Countries', fontsize=12)
ax1.set_title('Spatial Generalization Performance\n(Leave-Countries-Out)', fontsize=14, fontweight='bold')
for bar, mae in zip(bars, maes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
             f'{mae:,.0f}', ha='center', fontsize=11, fontweight='bold')

# Plot 2: Training History
ax2 = axes[1]
ax2.plot(history_climate.history['val_mae'], label='Climate GNN (Val)', linewidth=2, color='#2ecc71')
ax2.plot(history_geo.history['val_mae'], label='Geographic GNN (Val)', linewidth=2, color='#3498db', linestyle='--')
ax2.plot(history_mlp.history['val_mae'], label='MLP Baseline (Val)', linewidth=2, color='#e74c3c', linestyle=':')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation MAE', fontsize=12)
ax2.set_title('Training Progress', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('outputs/figures/gnn_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved figure to outputs/figures/gnn_comparison.png")

# Save models
model_climate.save('outputs/models/model_climate_gnn.keras')
model_geo.save('outputs/models/model_geo_gnn.keras')
model_mlp.save('outputs/models/model_mlp_baseline.keras')
print("✓ Saved models to outputs/models/")

print("\n" + "-" * 60)
if mae_climate < mae_geo:
    improvement = (mae_geo - mae_climate) / mae_geo * 100
    print(f"✅ Climate GNN outperforms Geographic baseline by {improvement:.1f}%")
else:
    print("📈 Climate GNN comparable to baseline - may benefit from tuning")

print("\n✅ GNN training and evaluation complete!")
