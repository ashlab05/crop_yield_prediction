"""
Evaluation Metrics Module
==========================
Standard regression metrics for model comparison.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    """
    Compute all standard regression metrics.
    
    Returns dict with: MAE, RMSE, R², MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape,
    }


def print_comparison_table(results_dict):
    """
    Print a formatted comparison table.
    
    Args:
        results_dict: {model_name: metrics_dict}
    """
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'MAE':>12} {'RMSE':>12} {'R²':>10} {'MAPE (%)':>10}")
    print("-" * 80)
    
    # Sort by MAE
    sorted_models = sorted(results_dict.items(), key=lambda x: x[1]['MAE'])
    
    for i, (name, metrics) in enumerate(sorted_models):
        marker = " ★" if i == 0 else ""
        print(f"{name:<20} {metrics['MAE']:>12,.0f} {metrics['RMSE']:>12,.0f} "
              f"{metrics['R²']:>10.4f} {metrics['MAPE (%)']:>10.2f}{marker}")
    
    print("=" * 80)
    print("★ = Best model (lowest MAE)")
