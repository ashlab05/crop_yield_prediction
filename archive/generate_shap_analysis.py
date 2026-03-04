#!/usr/bin/env python3
"""
Generate SHAP Summary Plot for Model Explainability
====================================================
Creates SHAP (SHapley Additive exPlanations) visualizations to understand
which features contribute most to crop yield predictions.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap

import tensorflow as tf
from tensorflow.keras import models

print("=" * 80)
print("🔍 SHAP EXPLAINABILITY ANALYSIS")
print("=" * 80)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv("yield_df.csv")
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Prepare features
feature_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
feature_names = ['Rainfall (mm/year)', 'Pesticides (tonnes)', 'Temperature (°C)']

X = df[feature_cols].values
y = df['hg/ha_yield'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Features: {feature_names}")
print(f"   Samples: {len(X):,}")

# Load the best model (MLP for SHAP compatibility)
print("\n🧠 Loading MLP Baseline model for SHAP analysis...")
model_path = 'outputs/models/model_mlp_baseline.keras'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("   Please run run_complete_pipeline.py first.")
    exit(1)

model = models.load_model(model_path)
print(f"   ✓ Model loaded: {model_path}")

# Sample data for SHAP (use smaller sample for speed)
print("\n⚙️  Preparing SHAP explainer...")
np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled), size=min(500, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_indices]

# Use KernelExplainer (model-agnostic, more robust)
print("   Computing SHAP values (this may take 1-2 minutes)...")

# Create prediction function
def model_predict(data):
    return model.predict(data, verbose=0).flatten()

# Use a smaller background dataset for speed
background_data = X_scaled[np.random.choice(X_scaled.shape[0], 50, replace=False)]
explainer = shap.KernelExplainer(model_predict, background_data)

# Compute SHAP values with smaller sample
X_explain = X_sample[:200]  # Use 200 samples for explanation
shap_values = explainer.shap_values(X_explain, nsamples=100)

print(f"   ✓ SHAP values computed: {shap_values.shape}")

# Create visualizations
print("\n📊 Generating SHAP visualizations...")

# Figure 1: Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_explain, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot - Feature Impact on Crop Yield Predictions', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/figures/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: outputs/figures/shap_summary_plot.png")

# Figure 2: Bar Plot (Mean Absolute SHAP values)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_explain, feature_names=feature_names, 
                  plot_type="bar", show=False)
plt.title('Feature Importance - Mean |SHAP| Values', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/figures/shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: outputs/figures/shap_feature_importance.png")

# Calculate and display feature importance
mean_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP|': mean_shap
}).sort_values('Mean |SHAP|', ascending=False)

print("\n" + "=" * 80)
print("📊 FEATURE IMPORTANCE RANKING")
print("=" * 80)
print(importance_df.to_string(index=False))

# Save importance to JSON
import json
importance_dict = {
    'feature_importance': {
        feature_names[i]: float(mean_shap[i]) 
        for i in range(len(feature_names))
    },
    'ranking': [
        {'rank': i+1, 'feature': row['Feature'], 'importance': float(row['Mean |SHAP|'])}
        for i, (_, row) in enumerate(importance_df.iterrows())
    ]
}

with open('outputs/results/shap_feature_importance.json', 'w') as f:
    json.dump(importance_dict, f, indent=2)

print("\n   ✓ Saved: outputs/results/shap_feature_importance.json")

# Organize outputs folder
print("\n" + "=" * 80)
print("📁 ORGANIZING OUTPUTS FOLDER")
print("=" * 80)

# Create subdirectories if they don't exist
os.makedirs('outputs/figures/gnn_dann', exist_ok=True)
os.makedirs('outputs/figures/shap', exist_ok=True)
os.makedirs('outputs/figures/legacy', exist_ok=True)

# Move files to organized structure
import shutil

# Move SHAP files
shap_files = ['shap_summary_plot.png', 'shap_feature_importance.png']
for f in shap_files:
    src = f'outputs/figures/{f}'
    dst = f'outputs/figures/shap/{f}'
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"   Moved: {f} → shap/")

# Move GNN/DANN files
gnn_dann_files = ['complete_comparison.png', 'results_table.png']
for f in gnn_dann_files:
    src = f'outputs/figures/{f}'
    dst = f'outputs/figures/gnn_dann/{f}'
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"   Moved: {f} → gnn_dann/")

# Move legacy files
legacy_files = ['dann_training.png', 'gnn_comparison.png']
for f in legacy_files:
    src = f'outputs/figures/{f}'
    dst = f'outputs/figures/legacy/{f}'
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"   Moved: {f} → legacy/")

print("\n✅ SHAP Analysis Complete!")
print("=" * 80)
print("""
📁 Organized Outputs Structure:
   outputs/
   ├── figures/
   │   ├── gnn_dann/          # Main comparison plots
   │   ├── shap/              # Explainability plots ✨
   │   └── legacy/            # Previous experiments
   ├── models/                # Trained models
   └── results/               # JSON metrics
""")
