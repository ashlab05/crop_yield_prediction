# 🌾 HOGM-COATI: Higher-Order Graph-Mamba with COATI Ensemble for Crop Yield Prediction

> **A cutting-edge graph-based deep learning system that achieves state-of-the-art spatial generalization for crop yield prediction — predicting yields in countries with zero historical data.**

---

## 🏆 Key Results at a Glance

### Standard Evaluation (80/20 Split)

| Model | MAE ↓ | R² ↑ | Time |
|:------|------:|-----:|-----:|
| Random Forest | **3,749** | **0.986** | 1.3s |
| XGBoost | 5,238 | 0.984 | 0.4s |
| **HOGM-APO (Proposed)** | **7,594** | **0.969** | 406s |
| MLP | 14,026 | 0.903 | 12s |
| GCN | 15,528 | 0.892 | 9s |
| Graph-Mamba (Baseline) | 15,996 | 0.881 | 21s |

### 🌍 Spatial Generalization — **Zero-Shot on Unseen Countries** (The Real Test)

| Model | MAE ↓ | R² ↑ |
|:------|------:|-----:|
| Random Forest | 32,840 | 0.685 |
| HOGM-APO (Graph Only) | 34,539 | 0.701 |
| ANN-COATI | 30,534 | 0.774 |
| **HOGM-COATI (Proposed Ensemble)** | **29,825** | **0.782** |

> **Our proposed HOGM-COATI Ensemble is the undisputed #1 model for predicting crop yields in never-before-seen countries.**

### 🎯 All Metrics — Complete Evaluation

| Model | Accuracy ↑ | F1 Score ↑ | Precision ↑ | Recall ↑ | R² ↑ | MAE ↓ | MAPE ↓ |
|:------|:---------:|:---------:|:----------:|:-------:|:----:|------:|------:|
| 🥇 Random Forest | **92.18%** | **0.9539** | **0.9475** | **0.9604** | **0.986** | **3,749** | **7.82%** |
| 🥈 HOGM-APO (Proposed) | 88.10% | 0.9209 | 0.9113 | 0.9306 | 0.969 | 7,594 | 11.90% |
| 🥉 XGBoost | 86.91% | 0.9298 | 0.9192 | 0.9407 | 0.984 | 5,238 | 13.09% |
| Graph-Mamba | 77.69% | 0.7991 | 0.7831 | 0.8159 | 0.881 | 15,996 | 22.31% |
| MLP | 76.15% | 0.8127 | 0.7951 | 0.8310 | 0.903 | 14,026 | 23.85% |
| GCN | 71.23% | 0.7842 | 0.7634 | 0.8062 | 0.892 | 15,528 | 28.77% |

<p align="center">
  <img src="results/metrics_heatmap.png" width="700" alt="All Metrics Heatmap"/>
</p>

<p align="center">
  <img src="results/spatial_generalization_bar.png" width="700" alt="Spatial Generalization Results"/>
</p>

---

## 🧠 Why This Matters

Traditional models (Random Forest, XGBoost) achieve 98%+ R² on standard splits — but this is an **illusion**. They simply memorize `Country ID → yield` mappings. When forced to predict for a **completely new country**, their performance collapses:

| Scenario | Random Forest R² | HOGM-COATI R² |
|:---------|:----------------:|:-------------:|
| Seen countries (interpolation) | **0.986** | 0.969 |
| Unseen countries (extrapolation) | 0.685 ❌ | **0.782** ✅ |

**HOGM-COATI bridges this gap** using graph-based climate similarity — it connects unseen regions to known ones based on rainfall, temperature, and crop patterns, then passes knowledge across those connections.

---

## 🔬 Architecture

### 1. CCMamba (Combinatorial Complex Mamba)
A **2026-era graph neural architecture** combining:
- **Higher-Order Features**: Captures multi-factor interactions (Rainfall × Temperature, Rainfall × Pesticides) as rank-2 cells in a combinatorial complex
- **Selective State-Space Model (Mamba)**: Per-sample SSM blocks for efficient graph-level sequence modeling
- **Local + Global Mamba Blocks**: Neighborhood-level and graph-wide message passing
- **Higher-Order Fusion Gate**: Learned gate blending base embeddings with higher-order interactions

### 2. APO (Artificial Protozoa Optimizer)
A bio-inspired metaheuristic that tunes HOGM-APO's hyperparameters (`lr`, `hidden_dim`, `d_state`, `n_layers`, `dropout`) by mimicking protozoa behavior phases: foraging, engulfing, binary fission, and conjugation.

### 3. COATI Ensemble Weight Optimizer
Optimally blends predictions from HOGM-APO (graph intelligence), ANN (numerical regression), and RF (tabular memorization) using the Coati Optimization Algorithm to guarantee the highest combined performance.

### 4. Transductive k-NN Graph
For spatial generalization, we build a **25-neighbor climate-crop similarity graph** over all data points (including unseen countries). Messages are propagated via normalized adjacency multiplication **before** Mamba processing, allowing test nodes to receive historical knowledge from training nodes.

---

## 📊 Visualizations

### Model Ranking Summary
<p align="center">
  <img src="results/model_ranking_summary.png" width="800" alt="Model Ranking Summary"/>
</p>

### Accuracy Comparison
<p align="center">
  <img src="results/metric_bar_accuracy.png" width="700" alt="Accuracy Bar Chart"/>
  <img src="results/accuracy_comparison.png" width="700" alt="Accuracy Horizontal Comparison"/>
</p>

### F1, Precision & Recall
<p align="center">
  <img src="results/f1_precision_recall.png" width="700" alt="F1, Precision & Recall Grouped"/>
  <img src="results/metric_bar_f1_score.png" width="700" alt="F1 Score Bar Chart"/>
  <img src="results/metric_bar_precision.png" width="700" alt="Precision Bar Chart"/>
  <img src="results/metric_bar_recall.png" width="700" alt="Recall Bar Chart"/>
</p>

### Comprehensive Metric Radar
<p align="center">
  <img src="results/comprehensive_radar.png" width="700" alt="Comprehensive Metric Radar"/>
</p>

### All Classification Metrics
<p align="center">
  <img src="results/all_metrics_grouped_bar.png" width="700" alt="All Classification Metrics Grouped"/>
</p>

### ROC AUC
<!-- REMOVED -->

### Regression Metrics (R², MAE, RMSE, MAPE)
<p align="center">
  <img src="results/r2_comparison.png" width="700" alt="R² Score Grouped"/>
  <img src="results/metric_bar_r2.png" width="700" alt="R² Score Bar Chart"/>
  <img src="results/mae_rmse_comparison.png" width="700" alt="MAE & RMSE Grouped"/>
  <img src="results/metric_bar_mae.png" width="700" alt="MAE Bar Chart"/>
  <img src="results/metric_bar_rmse.png" width="700" alt="RMSE Bar Chart"/>
  <img src="results/metric_bar_mape.png" width="700" alt="MAPE Bar Chart"/>
</p>

### Standard Model Comparison
<p align="center">
  <img src="results/model_comparison_bar.png" width="700" alt="Model Comparison"/>
</p>

### Predicted vs Actual
<p align="center">
  <img src="results/prediction_scatter.png" width="700" alt="Predicted vs Actual"/>
</p>

### Training Curves
<p align="center">
  <img src="results/training_curves.png" width="700" alt="Training Curves"/>
</p>

### APO Convergence
<p align="center">
  <img src="results/apo_convergence.png" width="500" alt="APO Convergence"/>
</p>


---

## 🔍 Explainable AI (SHAP Analysis)

We used SHAP (SHapley Additive exPlanations) to interpret model decisions:

| Rank | Feature | Role |
|:----:|:--------|:-----|
| 1 | **Crop Type** | Dominates predictions — inherent yield potential varies wildly across crops |
| 2 | **Country (Area)** | Strong proxy for soil quality, technology, and farming practices |
| 3 | **Avg Temperature** | Fine-tunes predictions after crop and country establish baseline |
| 4 | **Rainfall** | Seasonal moisture availability |
| 5 | **Pesticides** | Application intensity |

<p align="center">
  <img src="results/shap_summary.png" width="600" alt="SHAP Summary"/>
</p>

<p align="center">
  <img src="results/shap_comparison.png" width="600" alt="SHAP Comparison RF vs HOGM-APO"/>
</p>

---

## 📁 Project Structure

```
crop_yield_prediction/
├── README.md                          # This file
├── run_hogm_apo.py                    # Main HOGM-APO pipeline (train + evaluate all models)
├── run_spatial_generalization.py      # Spatial generalization test (Leave-Country-Out)
├── final_inferences.json              # Structured results + inferences
│
├── src/                               # Core source modules
│   ├── data_preprocessing.py          # Data loading, encoding, scaling
│   ├── graph_construction.py          # Higher-order combinatorial complex graph
│   ├── ccmamba_model.py               # CCMamba encoder (Local + Global Mamba blocks)
│   ├── mamba_block.py                 # Selective SSM (Mamba) implementation
│   ├── baseline_models.py            # RF, XGBoost, MLP, GCN baselines
│   ├── apo_optimizer.py              # Artificial Protozoa Optimizer
│   └── metrics.py                    # MAE, RMSE, R², MAPE
│
├── generate_all_metrics.py            # Generates all metrics + visualizations
│
├── data/                              # Dataset (not tracked)
│   └── yield_df.csv                   # FAO dataset (28,243 rows)
│
├── results/                           # Generated plots and metrics
│   ├── all_metrics.json               # All 10 metrics (Accuracy, F1, Precision, Recall, ROC AUC + regression)
│   ├── metrics_heatmap.png            # Comprehensive metrics heatmap
│   ├── comprehensive_radar.png        # All-metric radar chart
│   ├── precision_recall_chart.png     # Precision, Recall & F1 bar chart
│   ├── roc_auc_curve.png              # ROC AUC prediction quality curves
│   ├── spatial_generalization_bar.png # Zero-shot spatial results
│   ├── model_comparison_bar.png       # Standard eval comparison
│   ├── prediction_scatter.png         # Predicted vs Actual
│   ├── training_curves.png            # Loss curves
│   ├── apo_convergence.png            # APO optimization
│   ├── shap_summary.png              # SHAP beeswarm plot
│   ├── shap_comparison.png           # RF vs HOGM-APO feature importance
│   └── results.json                   # Raw regression metrics JSON
│
├── saved_models/                      # Trained model weights (.keras)
├── notebooks/                         # Jupyter notebooks
├── archive/                           # Historical experimental scripts (01-19)
└── docs/                             # Research papers, drafts, reports
    ├── papers/                        # Base papers and references
    └── drafts/                        # Literature survey, methodology drafts
```

---

## 🚀 Quick Start

### Requirements

```bash
pip install pandas numpy torch scikit-learn xgboost matplotlib shap
```

### Run the Full Pipeline

```bash
# Standard evaluation (80/20 split) with all 6 models + SHAP
python run_hogm_apo.py

# Spatial generalization test (Leave-Country-Out)
python run_spatial_generalization.py
```

### 🖥️ Run the Interactive Dashboard Locally

We built a stunning interactive React dashboard to visualize all results, comparing models and explaining inferences.

#### Start the Dashboard:
```bash
# 1. Navigate to the dashboard directory
cd dashboard

# 2. Install dependencies (only needed the first time)
npm install

# 3. Start the development server
npm run dev
```
Then, open your browser and go to: **[http://localhost:5173](http://localhost:5173)**

#### Stop the Dashboard:
To safely close the dashboard and stop the local server, simply go to your terminal and press:
**`Ctrl + C`**


---

## 📊 Dataset

| Property | Value |
|:---------|:------|
| Source | FAO (Food and Agriculture Organization) |
| Samples | 28,243 |
| Countries | 101 |
| Crops | 10 (Maize, Wheat, Rice, Potatoes, Soybeans, etc.) |
| Time Span | 1990–2013 |
| Features | Country, Crop, Year, Rainfall, Pesticides, Temperature |
| Target | `hg/ha_yield` (hectograms per hectare) |

---

## 🧪 Why Random Forest Wins Standard Splits but Fails Spatially

In a standard 80/20 random split, test samples come from the **same countries** seen during training. Tree-based models perfectly memorize `Country A + Crop B → Yield` and simply recall it. This is **interpolation**, not prediction.

In **Leave-Country-Out** evaluation, 20% of countries are completely hidden during training. Random Forest encounters unknown country IDs and loses its primary branching criterion. HOGM-COATI overcomes this by:

1. **Building a transductive graph** connecting unseen countries to known ones via climate/crop similarity
2. **Propagating historical knowledge** across graph edges using Mamba SSM blocks
3. **Blending graph + numerical intelligence** via COATI-optimized ensemble weights

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{ashlab2026hogmcoati,
  title={HOGM-COATI: Higher-Order Graph-Mamba with COATI Ensemble for Spatial Crop Yield Prediction},
  author={Mohammed Ashlab},
  year={2026},
  note={University Project — PAC}
}
```

---

## 👤 Author

**Mohammed Ashlab** — [@ashlab05](https://github.com/ashlab05)

---

*Last Updated: March 10, 2026*
