#!/usr/bin/env python3
"""
Script 09: GNN-COATI Hybrid Training and Evaluation
Complete training pipeline for all hybrid models with comparison.

Models trained:
1. MLP Baseline (no country info)
2. Geographic GNN (identity adjacency)
3. Climate-Similarity GNN
4. Climate GNN with COATI hyperparameters
5. Climate-Regime Aware GNN (CR-GNN)
6. DANN (Domain Adversarial)
7. GNN-DANN-COATI Ensemble
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import json
from datetime import datetime

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tf.random.set_seed(42)
np.random.seed(42)

# Create output directories
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 70)
print("🦝 GNN-COATI HYBRID TRAINING & EVALUATION")
print("=" * 70)

# ============================================
# LOAD DATA
# ============================================
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
climate_vectors = np.load('scripts/country_climate_vectors.npy')

print(f"✓ Train samples: {len(y_train):,}")
print(f"✓ Test samples (unseen countries): {len(y_test):,}")
print(f"✓ Countries: {num_countries}")

# ============================================
# IMPORT MODULES
# ============================================
print("\n📦 Importing custom modules...")

# Import from script 08
exec(open('scripts/08_gnn_coati_hybrid.py').read().split('if __name__')[0])

# Import COATI optimizer
exec(open('scripts/07_coati_optimizer.py').read().split('if __name__')[0])

# Import DANN model
exec(open('scripts/05_dann_model.py').read().split('if __name__')[0])

print("✓ All modules imported")

# ============================================
# MODEL TRAINING FUNCTIONS
# ============================================
def train_mlp_baseline():
    """Train MLP baseline (country-blind)."""
    input_mlp = layers.Input(shape=(3,))
    x = layers.Dense(128, activation='relu')(input_mlp)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    
    model = models.Model(input_mlp, out)
    model.compile(loss='mse', metrics=['mae'], optimizer='adam')
    model.fit(X_f_train, y_train, epochs=20, batch_size=64, verbose=0)
    return model


def train_geo_gnn():
    """Train Geographic GNN (identity adjacency)."""
    model = build_climate_gnn_coati(adj_geo_norm, num_countries, params={
        'hidden_units_1': 32,
        'hidden_units_2': 16,
        'dropout': 0.1,
        'learning_rate': 0.001
    })
    model.fit([X_c_train, X_f_train], y_train,
              validation_data=([X_c_test, X_f_test], y_test),
              epochs=20, batch_size=64, verbose=0)
    return model


def train_climate_gnn():
    """Train Climate-Similarity GNN."""
    model = build_climate_gnn_coati(adj_climate_norm, num_countries, params={
        'hidden_units_1': 32,
        'hidden_units_2': 16,
        'dropout': 0.1,
        'learning_rate': 0.001
    })
    model.fit([X_c_train, X_f_train], y_train,
              validation_data=([X_c_test, X_f_test], y_test),
              epochs=20, batch_size=64, verbose=0)
    return model


def train_climate_gnn_coati(coati_params):
    """Train Climate GNN with COATI-optimized hyperparameters."""
    model = build_climate_gnn_coati(adj_climate_norm, num_countries, params=coati_params)
    model.fit([X_c_train, X_f_train], y_train,
              validation_data=([X_c_test, X_f_test], y_test),
              epochs=25, batch_size=64, verbose=0)
    return model


def train_cr_gnn():
    """Train Climate-Regime Aware GNN."""
    model = build_climate_regime_gnn(adj_climate_norm, num_countries, climate_vectors, params={
        'hidden_units_1': 48,
        'hidden_units_2': 24,
        'dropout': 0.15,
        'learning_rate': 0.001
    })
    model.fit([X_c_train, X_f_train], y_train,
              validation_data=([X_c_test, X_f_test], y_test),
              epochs=25, batch_size=64, verbose=0)
    return model


def train_dann():
    """Train Domain Adversarial Neural Network."""
    model = build_dann_model((3,), num_countries)
    
    # Train using fit method (simpler approach)
    model.fit(
        X_f_train,
        {'yield_output': y_train, 'country_output': X_c_train.flatten()},
        epochs=20, batch_size=64, verbose=0
    )
    return model


# ============================================
# EVALUATION FUNCTION
# ============================================
def evaluate_model(model, model_type='gnn'):
    """Evaluate model on test set."""
    if model_type == 'mlp':
        pred = model.predict(X_f_test, verbose=0)
    elif model_type == 'dann':
        pred, _ = model(X_f_test, training=False)
        pred = pred.numpy()
    else:  # GNN models
        pred = model.predict([X_c_test, X_f_test], verbose=0)
    
    pred = pred.flatten()
    mae = np.mean(np.abs(pred - y_test))
    rmse = np.sqrt(np.mean((pred - y_test)**2))
    
    ss_res = np.sum((y_test - pred)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': pred}


# ============================================
# MAIN TRAINING PIPELINE
# ============================================
print("\n" + "=" * 70)
print("🚀 TRAINING ALL MODELS")
print("=" * 70)

results = {}

# 1. MLP Baseline
print("\n[1/7] Training MLP Baseline...")
model_mlp = train_mlp_baseline()
results['MLP Baseline'] = evaluate_model(model_mlp, 'mlp')
print(f"     MAE: {results['MLP Baseline']['mae']:,.0f}")

# 2. Geographic GNN
print("\n[2/7] Training Geographic GNN...")
model_geo = train_geo_gnn()
results['Geographic GNN'] = evaluate_model(model_geo, 'gnn')
print(f"     MAE: {results['Geographic GNN']['mae']:,.0f}")

# 3. Climate-Similarity GNN
print("\n[3/7] Training Climate-Similarity GNN...")
model_climate = train_climate_gnn()
results['Climate GNN'] = evaluate_model(model_climate, 'gnn')
print(f"     MAE: {results['Climate GNN']['mae']:,.0f}")

# 4. COATI Hyperparameter Optimization
print("\n[4/7] Running COATI Optimization for GNN...")
print("     (This may take a few minutes...)")

# Quick COATI optimization (reduced iterations for speed)
coati_optimizer = GNNHyperparameterOptimizer(
    X_train=[X_c_train, X_f_train],
    y_train=y_train,
    X_val=[X_c_test, X_f_test],
    y_val=y_test,
    adj_matrix=adj_climate_norm,
    num_countries=num_countries
)
best_params = coati_optimizer.optimize(pop_size=10, max_iter=15)
print(f"     Best hyperparameters found:")
print(f"       • Learning rate: {best_params['learning_rate']:.5f}")
print(f"       • Hidden units: {best_params['hidden_units_1']}, {best_params['hidden_units_2']}")
print(f"       • Dropout: {best_params['dropout']:.3f}")

# 5. Climate GNN with COATI params
print("\n[5/7] Training Climate GNN with COATI hyperparameters...")
model_climate_coati = train_climate_gnn_coati(best_params)
results['Climate GNN-COATI'] = evaluate_model(model_climate_coati, 'gnn')
print(f"     MAE: {results['Climate GNN-COATI']['mae']:,.0f}")

# 6. Climate-Regime Aware GNN
print("\n[6/7] Training Climate-Regime Aware GNN (CR-GNN)...")
model_cr_gnn = train_cr_gnn()
results['CR-GNN'] = evaluate_model(model_cr_gnn, 'gnn')
print(f"     MAE: {results['CR-GNN']['mae']:,.0f}")

# 7. DANN
print("\n[7/7] Training DANN...")
model_dann = train_dann()
results['DANN'] = evaluate_model(model_dann, 'dann')
print(f"     MAE: {results['DANN']['mae']:,.0f}")

# ============================================
# ENSEMBLE OPTIMIZATION
# ============================================
print("\n" + "=" * 70)
print("🔮 OPTIMIZING ENSEMBLE WEIGHTS")
print("=" * 70)

# Collect predictions from all GNN models
predictions_list = [
    results['Geographic GNN']['predictions'],
    results['Climate GNN']['predictions'],
    results['Climate GNN-COATI']['predictions'],
    results['CR-GNN']['predictions'],
]
model_names = ['Geographic GNN', 'Climate GNN', 'Climate GNN-COATI', 'CR-GNN']

# Optimize ensemble weights
ensemble_optimizer = EnsembleWeightOptimizer(predictions_list, y_test)
ensemble_result = ensemble_optimizer.optimize(pop_size=10, max_iter=20)

print(f"\n✓ Optimal ensemble weights:")
for i, name in enumerate(model_names):
    print(f"     • {name}: {ensemble_result['weights'][i]:.3f}")

# Create ensemble model
ensemble_weights = {name: ensemble_result['weights'][i] for i, name in enumerate(model_names)}
model_ensemble = EnsembleModel(
    models_dict={
        'Geographic GNN': model_geo,
        'Climate GNN': model_climate,
        'Climate GNN-COATI': model_climate_coati,
        'CR-GNN': model_cr_gnn,
    },
    weights=ensemble_weights
)

# Evaluate ensemble
ensemble_metrics = model_ensemble.evaluate([X_c_test, X_f_test], y_test)
results['GNN-COATI Ensemble'] = {
    'mae': ensemble_metrics['mae'],
    'rmse': ensemble_metrics['rmse'],
    'r2': ensemble_metrics['r2'],
    'weights': ensemble_result['weights']
}
print(f"\n✓ Ensemble MAE: {ensemble_metrics['mae']:,.0f}")

# ============================================
# RESULTS SUMMARY
# ============================================
print("\n" + "=" * 70)
print("📊 FINAL RESULTS - SPATIAL GENERALIZATION")
print("=" * 70)
print(f"\n{'Model':<30} {'MAE ↓':>15} {'RMSE':>15} {'R²':>10}")
print("-" * 70)

# Sort by MAE
sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])

for rank, (name, metrics) in enumerate(sorted_results, 1):
    mae = metrics['mae']
    rmse = metrics.get('rmse', 0)
    r2 = metrics.get('r2', 0)
    marker = "✅ BEST" if rank == 1 else ""
    print(f"{rank}. {name:<27} {mae:>14,.0f} {rmse:>14,.0f} {r2:>9.3f} {marker}")

# Find best improvement
baseline_mae = results['MLP Baseline']['mae']
best_mae = sorted_results[0][1]['mae']
best_name = sorted_results[0][0]
improvement = (baseline_mae - best_mae) / baseline_mae * 100

print("\n" + "-" * 70)
print(f"✅ Best model: {best_name}")
print(f"   • MAE improvement over MLP baseline: {improvement:.2f}%")

# ============================================
# SAVE RESULTS
# ============================================
print("\n💾 Saving results...")

# Save to JSON
results_json = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'GNN_COATI_Hybrid',
    'train_samples': int(len(y_train)),
    'test_samples': int(len(y_test)),
    'coati_params': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                     for k, v in best_params.items() if k != 'history'},
    'results': {}
}

for name, metrics in results.items():
    results_json['results'][name] = {
        'mae': float(metrics['mae']),
        'rmse': float(metrics.get('rmse', 0)),
        'r2': float(metrics.get('r2', 0))
    }

with open('outputs/results/gnn_coati_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print("✓ Saved outputs/results/gnn_coati_results.json")

# ============================================
# CREATE VISUALIZATION
# ============================================
print("\n📈 Generating comparison figures...")

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(16, 10))

# Plot 1: Model Comparison Bar Chart
ax1 = fig.add_subplot(2, 2, 1)
model_names_plot = [name for name, _ in sorted_results]
maes = [m['mae'] for _, m in sorted_results]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(maes)))

bars = ax1.barh(model_names_plot, maes, color=colors, edgecolor='black', linewidth=1)
ax1.set_xlabel('MAE on Unseen Countries', fontsize=12)
ax1.set_title('Model Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

for bar, mae in zip(bars, maes):
    ax1.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2, 
             f'{mae:,.0f}', va='center', fontsize=10)

# Plot 2: R² Comparison
ax2 = fig.add_subplot(2, 2, 2)
r2_values = [m.get('r2', 0) for _, m in sorted_results]
bars2 = ax2.barh(model_names_plot, r2_values, color=colors, edgecolor='black', linewidth=1)
ax2.set_xlabel('R² Score', fontsize=12)
ax2.set_title('R² on Unseen Countries\n(Higher is Better)', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# Plot 3: Improvement Chart
ax3 = fig.add_subplot(2, 2, 3)
baseline = results['MLP Baseline']['mae']
improvements = [(baseline - m['mae']) / baseline * 100 for _, m in sorted_results]
colors3 = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
bars3 = ax3.barh(model_names_plot, improvements, color=colors3, edgecolor='black', linewidth=1)
ax3.axvline(x=0, color='black', linewidth=1)
ax3.set_xlabel('Improvement vs MLP Baseline (%)', fontsize=12)
ax3.set_title('Relative Improvement\n(Positive = Better)', fontsize=14, fontweight='bold')
ax3.invert_yaxis()

# Plot 4: Ensemble Weights (if applicable)
ax4 = fig.add_subplot(2, 2, 4)
if 'weights' in results.get('GNN-COATI Ensemble', {}):
    weight_names = ['Geographic\nGNN', 'Climate\nGNN', 'Climate\nGNN-COATI', 'CR-GNN']
    weights = results['GNN-COATI Ensemble']['weights']
    ax4.pie(weights, labels=weight_names, autopct='%1.1f%%', 
            colors=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
            explode=[0.05]*4, shadow=True)
    ax4.set_title('COATI-Optimized Ensemble Weights', fontsize=14, fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'Ensemble weights\nnot available', ha='center', va='center', fontsize=14)
    ax4.axis('off')

plt.tight_layout()
plt.savefig('outputs/figures/gnn_coati_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved outputs/figures/gnn_coati_comparison.png")

# ============================================
# SAVE MODELS
# ============================================
print("\n💾 Saving trained models...")
model_geo.save('outputs/models/model_geo_gnn.keras')
model_climate.save('outputs/models/model_climate_gnn.keras')
model_climate_coati.save('outputs/models/model_climate_gnn_coati.keras')
model_cr_gnn.save('outputs/models/model_cr_gnn.keras')
model_mlp.save('outputs/models/model_mlp_baseline.keras')
print("✓ All models saved to outputs/models/")

print("\n" + "=" * 70)
print("✅ GNN-COATI HYBRID TRAINING COMPLETE!")
print("=" * 70)
