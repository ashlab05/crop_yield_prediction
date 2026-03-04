# COATI Optimization Algorithm for GNN Hyperparameter Optimization

## Overview

The **Coati Optimization Algorithm (COATI)** is a bio-inspired metaheuristic optimization algorithm that mimics the hunting and escape behaviors of coatis (Nasua nasua). This document describes how COATI is applied to optimize Graph Neural Network (GNN) hyperparameters for improved crop yield prediction.

---

## Algorithm Description

### Inspiration

Coatis are social animals that exhibit two distinctive behaviors:

1. **Iguana Hunting (Exploration)**: Coatis attack iguanas in groups, exploring large territories
2. **Predator Escape (Exploitation)**: When threatened, coatis flee and hide in local safe spots

These behaviors translate to:
- **Exploration**: Global search across the parameter space
- **Exploitation**: Local refinement around promising solutions

---

## Mathematical Formulation

### Phase 1: Exploration (Iguana Hunting Attack)

When a random value r < 0.5:

```
X_new = X_current + r × (X_best - I × X_iguana)
```

Where:
- `X_current`: Current position (hyperparameters)
- `X_best`: Best solution found so far
- `X_iguana`: Random position representing iguana location
- `I`: Random integer (1 or 2) for variation
- `r`: Random value in [0, 1]

### Phase 2: Exploitation (Escape from Predator)

When r ≥ 0.5:

```
X_new = X_current + (1 - 2r) × (LB + r × (UB - LB)) × 0.1
```

Where:
- `LB`, `UB`: Lower and upper bounds of search space
- The `0.1` factor controls local search intensity

---

## Application to GNN Optimization

### Optimized Hyperparameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `learning_rate` | [0.0001, 0.01] | Adam optimizer learning rate |
| `hidden_units_1` | [16, 64] | First GNN layer units |
| `hidden_units_2` | [8, 32] | Second GNN layer units |
| `dropout` | [0.0, 0.5] | Dropout regularization |
| `climate_threshold` | [0.8, 0.99] | Graph edge creation threshold |

### Fitness Function

We minimize validation MAE on unseen countries:

```python
def fitness(params):
    model = build_gnn(params)
    model.fit(train_data, epochs=3)  # Quick evaluation
    return model.evaluate(val_data)['mae']
```

---

## Implementation Details

### COATI Optimizer Configuration

```python
optimizer = CoatiOptimizer(
    pop_size=10,      # Population of 10 search agents
    max_iter=15,      # 15 optimization iterations
    bounds=[          # Parameter bounds
        (0.0001, 0.01),  # learning_rate
        (16, 64),        # hidden_units_1
        (8, 32),         # hidden_units_2
        (0.0, 0.5),      # dropout
        (0.8, 0.99)      # climate_threshold
    ]
)
```

### Optimization Process

```
Iteration 1:  Initial population → Evaluate all → Find best
Iteration 2:  For each agent:
              - Random r ∈ [0,1]
              - if r < 0.5: Exploration (attack behavior)
              - else: Exploitation (escape behavior)
              - Update if better
              ...
Iteration N:  Return best solution
```

---

## Results

### Optimized Hyperparameters (from experiment run)

| Parameter | Default | COATI-Optimized |
|-----------|---------|----------------|
| Learning Rate | 0.001 | 0.00444 |
| Hidden Units 1 | 32 | 19 |
| Hidden Units 2 | 16 | 13 |
| Dropout | 0.1 | 0.500 |

### Performance Comparison

| Model | MAE ↓ | Improvement |
|-------|-------|-------------|
| MLP Baseline | 68,547 | — |
| Climate GNN (default) | 68,046 | +0.73% |
| **Climate GNN-COATI** | **67,608** | **+1.37%** |

---

## Ensemble Weight Optimization

COATI also optimizes ensemble weights for combining multiple GNN models:

```python
ensemble_optimizer = EnsembleWeightOptimizer(
    predictions_list=[pred_geo, pred_climate, pred_coati, pred_cr],
    y_true=y_test
)
weights = ensemble_optimizer.optimize()
# → [0.0, 0.0, 1.0, 0.0]  # Climate GNN-COATI dominates
```

---

## References

1. Dehghani, M., et al. (2022). "Coati Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm for Solving Optimization Problems." *Knowledge-Based Systems*.

2. This implementation: `scripts/07_coati_optimizer.py`

---

*Last Updated: January 19, 2026*
