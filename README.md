# 🌾 Crop Yield Prediction: A Generalization-First Approach

> **Using only annual country-level climate and pesticide data, this study reframes crop yield forecasting as a generalization problem rather than an interpolation task, introducing climate-regime abstraction, causality-aware temporal modeling, and uncertainty-aware explainability to address real-world deployment challenges.**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Research Gaps Addressed](#-research-gaps-addressed)
- [Methodology](#-methodology)
- [Models Implemented](#-models-implemented)
- [Current Results](#-current-results)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Future Work](#-future-work)
- [Contributors](#-contributors)

---

## 🎯 Project Overview

This project develops a **robust, generalizable, and interpretable** crop yield prediction system using FAO country-level agricultural data. Unlike traditional approaches that achieve high accuracy through country-specific memorization, our approach focuses on:

1. **Generalization to unseen countries** (not just unseen years)
2. **Climate-regime-based reasoning** instead of raw numeric values
3. **Causality-aware temporal modeling** with proper train/test splits
4. **Uncertainty quantification** for policy-relevant predictions
5. **Explainable AI (XAI)** for trustworthy decision-making

### Why This Matters

Country-level yield predictions inform:
- Agricultural policy decisions
- Food security planning
- Climate adaptation strategies
- International trade forecasting

Current models fail in real deployment because they **memorize country baselines** instead of learning **climate-yield relationships**.

---

## 📊 Dataset Description

### Source
FAO (Food and Agriculture Organization) country-level agricultural statistics.

### Features Available

| Feature | Description | Type |
|---------|-------------|------|
| `Area` | Country name | Categorical (101 unique) |
| `Item` | Crop type (Maize, Potatoes, Rice, Sorghum, Soybeans, Wheat, etc.) | Categorical (10 unique) |
| `Year` | Year of observation (1990-2013) | Numeric |
| `average_rain_fall_mm_per_year` | Mean annual rainfall | Numeric (51-3240 mm) |
| `avg_temp` | Mean annual temperature | Numeric |
| `pesticides_tonnes` | Pesticide application | Numeric (0.04-367,778 tonnes) |
| `hg/ha_yield` | **Target**: Yield in hectograms/hectare | Numeric (50-501,412) |

### Dataset Statistics
- **Total samples**: 28,243
- **Countries**: 101 (filtered to those with ≥100 records)
- **Crops**: 10 types
- **Time span**: 24 years (1990-2013)

### What We DON'T Have
❌ Soil data  
❌ Satellite imagery  
❌ Farm-level management practices  
❌ Spatial grid data  

This constraint makes the problem **harder but more realistic** for many developing regions.

---

## 🔍 Research Gaps Addressed

### Gap 1: Country Leakage Inflates Accuracy
**Problem**: Models encode country as a categorical ID, learning country-specific baselines instead of generalizable patterns.

**Our Solution**: 
- Spatial hold-out evaluation (leave-countries-out)
- Country de-identification experiments
- Climate-regime-based features that transfer across regions

### Gap 2: Raw Climate Variables Lack Transferability
**Problem**: 800mm rainfall means "good" in one climate zone but "drought" in another.

**Our Solution**:
- Climate regime abstraction (Cool-Wet, Warm-Dry, etc.)
- Relative climate encoding instead of absolute values
- Cross-country climate similarity analysis

### Gap 3: Temporal Causality Ignored
**Problem**: Random train/test splits allow future data to leak into training.

**Our Solution**:
- Chronological splits (train ≤2007, test >2007)
- Sequence modeling with past-to-future constraints
- Proper time series cross-validation

### Gap 4: Pesticide Effects Oversimplified
**Problem**: Linear treatment of pesticide ignores diminishing returns and crop-specific effects.

**Our Solution**:
- Log-transformed pesticide features
- Non-linear dose-response analysis
- SHAP-based pesticide effect interpretation

### Gap 5: No Uncertainty or Trust
**Problem**: Point predictions without confidence intervals are inadequate for policy decisions.

**Our Solution**:
- Monte Carlo Dropout for uncertainty estimation
- Calibrated prediction intervals
- Coverage and calibration error reporting

---

## 🔬 Methodology

### A. Climate-Regime Abstraction (Key Novelty)

Transform raw rainfall and temperature into interpretable climate regimes:

```
Climate Regimes:
├── Cool-Wet
├── Cool-Dry
├── Warm-Wet
├── Warm-Dry
├── Hot-Wet
├── Hot-Dry
└── High-Variability
```

This removes scale dependency and makes climate comparable across countries.

### B. Causality-Aware Temporal Modeling

Model sequences with strict temporal ordering:
```
(Country, Crop):
[Climate_2000, Climate_2001, ..., Climate_2012] → Yield_2013
```

No future leakage. No random year shuffling.

### C. Country De-Identification

Remove country ID as a direct predictor. Use instead:
- Climate regime frequencies
- Crop-specific patterns
- Temporal trends

Country used only for grouping, not prediction.

### D. Evaluation Protocol

| Split Type | Purpose |
|------------|---------|
| Random (80/20) | Baseline comparison |
| Chronological | Temporal generalization |
| Spatial (leave-countries-out) | Geographic generalization |

---

## 🤖 Models Implemented

### Traditional ML Models (with 5-Fold Cross-Validation)

| Model | Test R² | Notes |
|-------|---------|-------|
| Bagging Regressor | 98.59% | Best traditional |
| Random Forest | 98.56% | Top performer |
| Decision Tree | 97.62% | - |
| XGBoost | 97.43% | - |
| Gradient Boosting | 83.11% | - |
| KNN | 28.94% | Poor performance |
| Linear Regression | 7.37% | Baseline |

### Deep Learning Models

| Model | Test R² | Notes |
|-------|---------|-------|
| **LSTM** 🥇 | ~96% | Best DL model |
| Vanilla Transformer 🥈 | ~94% | Fixed positional encoding |
| CNN-LSTM 🥉 | ~94% | Hybrid architecture |
| ANN | ~93% | Baseline DL |
| 1D-CNN | ~92% | Fast and efficient |

> **Note**: Deep learning results may vary slightly between runs. Models are saved after training for reproducibility.

### Key Findings

1. **LSTM performs best** among deep learning models for this tabular-as-sequence formulation
2. **Transformer works well** after fixing positional encoding and hyperparameters
3. **No overfitting detected** (train-test gap monitored for all models)
4. **Bagging/Random Forest remain highly competitive** with deep learning approaches
5. **Features treated as sequences** allow LSTM/CNN-LSTM to capture feature interactions

---

## 🧪 Generalization Experiments (Novel Contribution)

### Split Strategy Comparison

| Evaluation Scenario | Random Forest R² | LSTM R² | Key Insight |
|---------------------|------------------|---------|-------------|
| Random Split (Baseline) | ~98.5% | ~96% | High due to country memorization |
| Temporal Split (≤2007→>2007) | ~97% | ~95% | Maintains good performance |
| Spatial Split (Leave countries out) | ~96% | ~93% | Country ID helps but unrealistic |
| Spatial + No Country | ~85% | N/A | **True generalization challenge** |
| Spatial + Climate Regimes | ~87% | N/A | **Climate regimes help transfer** |

### Key Research Findings

#### 🔴 Country Leakage Problem
- With country feature: ~98% R²
- Without country feature: ~85% R²
- **~13% performance drop reveals country-specific memorization**

#### 🟢 Climate Regimes Help Generalization
- Raw climate (no country): ~85% R²
- Climate regimes (no country): ~87% R²
- **Regimes provide transferable representations**

#### 🔵 Temporal Generalization Works
- Training on 1990-2007, testing on 2008-2013
- Models maintain ~95%+ R²
- **Causal modeling approach validated**

## 🧠 Novel Approaches: GNN, DANN & COATI for Spatial Generalization

### Problem: Standard Models Can't Generalize to New Countries

Traditional models achieve 98%+ accuracy but fail when tested on **completely unseen countries** because they memorize country-specific baselines rather than learning transferable patterns.

### Our Solutions

#### 1. Climate-Similarity Graph Neural Network (Climate GNN)

**Hypothesis**: Countries with similar climates should have similar agricultural patterns.

- Connects countries based on **climate similarity** (cosine similarity > 0.95)
- Uses Graph Convolution to aggregate information from climatically similar regions
- Enables knowledge transfer to unseen countries

#### 2. Domain Adversarial Neural Network (DANN)

**Goal**: Learn features that predict yield well but **cannot identify which country** the sample came from.

- Uses **Gradient Reversal Layer** to penalize country-identifying features
- Forces the model to learn country-invariant representations

#### 3. 🦝 COATI Optimization Algorithm (NEW)

**Goal**: Optimize GNN hyperparameters using bio-inspired metaheuristic optimization.

- Mimics coati hunting (**exploration**) and escape (**exploitation**) behaviors
- Optimizes: learning rate, hidden units, dropout, graph threshold
- Finds optimal ensemble weights for model combination

#### 4. Climate-Regime Aware GNN (CR-GNN)

**Enhancement**: Initialize node embeddings with climate regime information.

- Uses climate vectors (rainfall, temperature) as node features
- 3-layer GCN for deeper aggregation
- Enhanced prediction head with 96→48→24 units

### 📊 Spatial Generalization Results (Leave-Countries-Out)

Evaluation on **20 completely unseen countries** (no training data from these regions):

| Rank | Model | MAE ↓ | R² | Improvement |
|------|-------|-------|-----|-------------|
| 🥇 | **Climate GNN** | **33,317** | **0.6759** | **+45% vs Ridge** |
| 🥈 | Climate+COATI | 36,654 | 0.5933 | +39% vs Ridge |
| 🥉 | Geographic GNN | 43,736 | 0.4487 | +28% vs Ridge |
| 4 | Gradient Boosting | 59,493 | -0.0148 | +2% vs Ridge |
| 5 | Ridge Baseline | 60,533 | -0.0821 | — |

### Key Scientific Finding: COATI Trade-off

> **COATI improves in-distribution prediction but degrades out-of-country generalization.**

| Setting | Best Model | Reason |
|---------|------------|--------|
| Random Split | Climate + COATI | COATI adds rich semantic priors |
| Leave-Country-Out | Climate GNN | Universal physics transfers better |

**Root Cause**: COATI implicitly encodes country identity, causing overfitting to training countries.

### Key Findings

✅ **Climate GNN achieves MAE 33,317, R² 0.6759** on completely unseen countries  
✅ **Log normalization `log(y+1)` was critical** for proper gradient flow  
✅ **Climate-similarity edges** enable universal knowledge transfer  
✅ **45% improvement** over non-graph baselines (Ridge, GB)  
✅ **COATI trade-off** reveals important semantic-vs-transfer balance

### Run the Complete Pipeline

```bash
# Run all hybrid models with COATI optimization
python scripts/09_hybrid_training.py

# Or run the original pipeline
python run_complete_pipeline.py
```

For detailed methodology, see:
- [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) - Full project report
- [docs/COATI_METHODOLOGY.md](docs/COATI_METHODOLOGY.md) - COATI algorithm details

---

## 📁 Project Structure

```
crop_yield_prediction/
├── README.md                              # This documentation file
├── crop-yield-prediction-99.ipynb         # Main Jupyter notebook (120+ cells)
├── yield_df.csv                           # FAO dataset (28,243 rows)
├── run_complete_pipeline.py               # Complete GNN/DANN pipeline script
│
├── docs/
│   ├── PROJECT_REPORT.md                  # Comprehensive project report
│   ├── DISCUSSION.md                      # Model comparison analysis
│   ├── SHAP_ANALYSIS.md                   # Feature importance analysis
│   └── COATI_METHODOLOGY.md               # COATI algorithm documentation
│
├── outputs/                               # Generated outputs
│   ├── figures/
│   │   ├── gnn_coati_comparison.png       # GNN-COATI results (NEW)
│   │   ├── complete_comparison.png        # GNN/DANN comparison chart
│   │   └── results_table.png              # Results summary table
│   ├── models/
│   │   ├── model_mlp_baseline.keras
│   │   ├── model_geo_gnn.keras
│   │   ├── model_climate_gnn.keras
│   │   ├── model_climate_gnn_coati.keras  # COATI-optimized (NEW)
│   │   ├── model_cr_gnn.keras             # Climate-Regime GNN (NEW)
│   │   └── model_dann.keras
│   └── results/
│       ├── gnn_coati_results.json         # COATI results (NEW)
│       └── complete_results.json          # Metrics JSON
│
├── scripts/                               # Modular Python scripts
│   ├── 01_data_preprocessing.py
│   ├── 02_climate_graph.py
│   ├── 03_gnn_model.py
│   ├── 04_gnn_training.py
│   ├── 05_dann_model.py
│   ├── 06_dann_training.py
│   ├── 07_coati_optimizer.py              # COATI algorithm (NEW)
│   ├── 08_gnn_coati_hybrid.py             # Hybrid models (NEW)
│   └── 09_hybrid_training.py              # Hybrid training (NEW)
│
└── saved_models/                          # Trained models (auto-loaded on re-run)
    ├── ann_model.keras
    ├── lstm_model.keras
    ├── transformer_model.keras
    └── ...
```

### Notebook Structure

| Section | Description |
|---------|-------------|
| Step 1-3 | Data loading, EDA, preprocessing |
| Step 4 | Visualization and analysis |
| Step 5 | Traditional ML models (RF, XGBoost, etc.) |
| Step 6 | Deep Learning models (ANN, LSTM, Transformer, CNN) |
| **Step 7** | **Novel Research Components** |
| 7.1 | Climate regime abstraction |
| 7.2 | Random vs Temporal vs Spatial splits |
| 7.3 | Country de-identification experiments |
| 7.4 | Pesticide dose-response analysis |
| 7.5 | Uncertainty quantification (MC Dropout) |
| 7.6 | SHAP explainability |
| 7.7 | LSTM with generalization splits |
| Step 8 | Final results and conclusions |

### Model Save/Load System

Models are automatically saved after training and loaded on subsequent runs:
- ✅ Saves training time (skip retraining)
- ✅ Preserves best weights
- ✅ Includes scalers for proper inference

---

## 🚀 Installation & Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow xgboost geopandas plotly
```

### Quick Start

```bash
# Open the notebook
jupyter notebook crop-yield-prediction-99.ipynb

# Or run all cells - models will auto-load if previously trained
```

### Model Loading Example

```python
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model('saved_models/lstm_model.keras')

# Load scalers
with open('saved_models/lstm_model_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

# Predict
X_scaled = scaler_X.transform(X_new)
X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
y_pred_scaled = model.predict(X_lstm)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
```

---

## 🔮 Completed Work & Future Extensions

### ✅ Phase 1: Climate Regime Implementation (COMPLETED)
- [x] Implement climate regime clustering (9 regimes: Cool/Warm/Hot × Dry/Moderate/Wet)
- [x] Create regime-based features
- [x] Evaluate regime transferability

### ✅ Phase 2: Spatial Generalization (COMPLETED)
- [x] Leave-countries-out experiments (20% countries held out)
- [x] Country de-identification experiments
- [x] Cross-region transfer analysis

### ✅ Phase 3: Temporal Causality (COMPLETED)
- [x] Chronological split evaluation (train ≤2007, test >2007)
- [x] Tested both RF and LSTM on temporal splits
- [x] Validated causal generalization

### ✅ Phase 4: Uncertainty & XAI (COMPLETED)
- [x] Monte Carlo Dropout implementation (50 forward passes)
- [x] SHAP analysis for feature importance
- [x] Calibrated prediction intervals with coverage metrics

### 🔄 Phase 5: Paper Writing (IN PROGRESS)
- [x] Full experimental results documented
- [x] Ablation studies completed (random vs temporal vs spatial)
- [ ] Formal paper draft
- [ ] Comparison with SOTA literature
- [ ] Policy implications discussion

### 🔮 Future Extensions
- [ ] Graph Neural Network for country similarity
- [ ] Ensemble methods combining best models
- [ ] Web deployment for real-time predictions

---

## 📈 How This Differs From Existing Work

| Aspect | Typical FAO Papers | **This Work** |
|--------|-------------------|---------------|
| Country usage | Direct feature | Grouping only |
| Climate | Raw numeric | Regime-based |
| Evaluation | Random split | Spatial + temporal |
| Pesticide | Linear | Non-linear + explained |
| Output | Point estimate | Mean + uncertainty |
| Goal | Accuracy | **Deployability** |

---

## 📚 References

- FAO Statistical Database (FAOSTAT)
- TensorFlow/Keras for deep learning models
- Scikit-learn for traditional ML
- XGBoost for gradient boosting
- SHAP for model explainability

---

## 👥 Contributors

- Mohammed Ashlab

---

## 📄 License

This project is for academic/research purposes.

---

## 🎯 Target Venues

Potential publication venues:
- **Computers and Electronics in Agriculture**
- **Agricultural Systems**
- **Environmental Modelling & Software**
- **IJCAI/AAAI AI for Social Good tracks**

---

*Last Updated: January 18, 2026*
