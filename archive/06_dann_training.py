#!/usr/bin/env python3
"""
Script 06: DANN Training and Evaluation
Trains Domain Adversarial Network for country-invariant features.
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
print("Step 6: DANN Training and Evaluation")
print("=" * 60)

# Import model components from script 05
exec(open('scripts/05_dann_model.py').read().split('# Test model creation')[0])

# Load data
print("\n📊 Loading preprocessed data...")
data = np.load('scripts/preprocessed_data.npz')
X_f_train = data['X_f_train']
X_c_train = data['X_c_train']
y_train = data['y_train']
X_f_test = data['X_f_test']
X_c_test = data['X_c_test']
y_test = data['y_test']
num_countries = int(data['num_countries'])

print(f"✓ Train samples: {len(y_train)}")
print(f"✓ Test samples (unseen countries): {len(y_test)}")

# Build and train DANN
print("\n" + "=" * 60)
print("🧠 Training Domain Adversarial Network...")
print("=" * 60)
print("Goal: Learn features predictive of yield but NOT predictive of country")

dann_model = build_dann_model((3,), num_countries)

history = dann_model.fit(
    X_f_train,
    {'yield_output': y_train, 'country_output': X_c_train},
    validation_data=(
        X_f_test,
        {'yield_output': y_test, 'country_output': X_c_test}
    ),
    epochs=15,
    batch_size=64,
    verbose=1
)

# Evaluate
print("\n" + "=" * 60)
print("📊 DANN EVALUATION RESULTS")
print("=" * 60)

results_eval = dann_model.evaluate(
    X_f_test,
    {'yield_output': y_test, 'country_output': X_c_test},
    verbose=0
)

# Results order: total_loss, yield_loss, country_loss, yield_mae, country_accuracy
total_loss, yield_mse, country_loss, yield_mae, country_acc = results_eval

print(f"\n{'Metric':<30} {'Value':<20}")
print("-" * 50)
print(f"{'Yield MSE':<30} {yield_mse:>15,.0f}")
print(f"{'Yield MAE':<30} {yield_mae:>15,.0f}")
print(f"{'Yield RMSE':<30} {np.sqrt(yield_mse):>15,.0f}")
print(f"{'Country Classification Loss':<30} {country_loss:>15.4f}")
print(f"{'Country Classification Acc':<30} {country_acc:>15.2%}")

# Save results to JSON
results = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'DANN_Domain_Adversarial',
    'train_samples': int(len(y_train)),
    'test_samples': int(len(y_test)),
    'num_countries': int(num_countries),
    'results': {
        'yield_mse': float(yield_mse),
        'yield_mae': float(yield_mae),
        'yield_rmse': float(np.sqrt(yield_mse)),
        'country_classification_loss': float(country_loss),
        'country_classification_accuracy': float(country_acc),
        'random_baseline_accuracy': float(1.0 / num_countries)
    }
}
with open('outputs/results/dann_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved results to outputs/results/dann_results.json")

# Create and save figures
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Yield Loss History
ax1 = axes[0]
ax1.plot(history.history['yield_output_loss'], label='Train', linewidth=2, color='#3498db')
ax1.plot(history.history['val_yield_output_loss'], label='Validation', linewidth=2, color='#e74c3c')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Yield MSE Loss', fontsize=12)
ax1.set_title('Yield Prediction Loss', fontsize=14, fontweight='bold')
ax1.legend()

# Plot 2: Country Classification Loss
ax2 = axes[1]
ax2.plot(history.history['country_output_loss'], label='Train', linewidth=2, color='#3498db')
ax2.plot(history.history['val_country_output_loss'], label='Validation', linewidth=2, color='#e74c3c')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Country Classification Loss', fontsize=12)
ax2.set_title('Country Classifier Loss\n(Higher = more invariant)', fontsize=14, fontweight='bold')
ax2.legend()

# Plot 3: Summary Bar Chart
ax3 = axes[2]
random_acc = 1.0 / num_countries
x_pos = [0, 1]
heights = [random_acc, country_acc]
colors = ['#95a5a6', '#2ecc71' if country_acc < random_acc * 2 else '#e74c3c']
labels = ['Random\nBaseline', 'DANN\nCountry Acc']
bars = ax3.bar(x_pos, heights, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('Country Classification Comparison\n(Lower = better invariance)', fontsize=14, fontweight='bold')
for bar, h in zip(bars, heights):
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.01, 
             f'{h:.2%}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/dann_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved figure to outputs/figures/dann_training.png")

# Save model
dann_model.save('outputs/models/model_dann.keras')
print("✓ Saved model to outputs/models/model_dann.keras")

print("\n" + "-" * 50)
print("💡 INTERPRETATION:")
print("   • Lower country accuracy = more country-invariant features")
print("   • Goal: Features should predict yield well but NOT identify country")

random_acc = 1.0 / num_countries
print(f"\n   Random guessing accuracy: {random_acc:.2%}")
if country_acc < random_acc * 2:
    print("   ✅ Country classifier near random = features are country-invariant!")
else:
    print("   📈 Country classifier still learning = consider increasing adversarial weight")

print("\n✅ DANN training and evaluation complete!")
