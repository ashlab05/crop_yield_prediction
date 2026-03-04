#!/usr/bin/env python3
"""
Script 07: Coati Optimization Algorithm (COATI)
Bio-inspired metaheuristic optimizer for GNN hyperparameter optimization.

The Coati Optimization Algorithm mimics the hunting and escape behaviors of coatis:
1. Exploration Phase (Iguana Hunting): Group attack behavior for global search
2. Exploitation Phase (Escape): Predator avoidance for local refinement

Reference: Dehghani et al. (2022) - Coati Optimization Algorithm
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Step 7: Coati Optimization Algorithm (COATI)")
print("=" * 60)


class CoatiOptimizer:
    """
    Coati Optimization Algorithm for hyperparameter optimization.
    
    Mimics two natural behaviors of coatis:
    1. Iguana hunting (exploration): Coatis attack iguanas in groups
    2. Predator escape (exploitation): Coatis flee from predators
    """
    
    def __init__(self, pop_size=20, max_iter=50, bounds=None, verbose=True):
        """
        Initialize COATI optimizer.
        
        Args:
            pop_size: Number of search agents (coati population)
            max_iter: Maximum iterations
            bounds: List of (min, max) tuples for each dimension
            verbose: Print progress
        """
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.bounds = bounds
        self.verbose = verbose
        
    def _initialize_population(self, dim):
        """Initialize random population within bounds."""
        population = np.zeros((self.pop_size, dim))
        for i in range(dim):
            lb, ub = self.bounds[i]
            population[:, i] = np.random.uniform(lb, ub, self.pop_size)
        return population
    
    def _clip_to_bounds(self, position, dim):
        """Ensure positions stay within bounds."""
        for i in range(dim):
            lb, ub = self.bounds[i]
            position[i] = np.clip(position[i], lb, ub)
        return position
    
    def optimize(self, fitness_func, dim):
        """
        Run COATI optimization.
        
        Args:
            fitness_func: Function to minimize (takes array, returns scalar)
            dim: Number of dimensions to optimize
            
        Returns:
            best_position: Optimal parameter values
            best_fitness: Best fitness achieved
            history: Convergence history
        """
        # Initialize population
        population = self._initialize_population(dim)
        
        # Evaluate initial fitness
        fitness = np.array([fitness_func(ind) for ind in population])
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = [best_fitness]
        
        if self.verbose:
            print(f"\n🦝 Starting COATI Optimization")
            print(f"  • Population size: {self.pop_size}")
            print(f"  • Max iterations: {self.max_iter}")
            print(f"  • Initial best fitness: {best_fitness:.4f}")
        
        # Main optimization loop
        for iter_num in range(self.max_iter):
            for i in range(self.pop_size):
                r = np.random.random()
                
                # ============================================
                # PHASE 1: EXPLORATION (Iguana Hunting Attack)
                # ============================================
                if r < 0.5:
                    # Iguana position (random target in search space)
                    iguana_pos = np.zeros(dim)
                    for d in range(dim):
                        lb, ub = self.bounds[d]
                        iguana_pos[d] = np.random.uniform(lb, ub)
                    
                    # Coati attacks iguana - move towards best and iguana
                    I = np.random.randint(1, 3)  # Randomly 1 or 2
                    new_pos = population[i] + r * (best_position - I * iguana_pos)
                    
                # ============================================
                # PHASE 2: EXPLOITATION (Escape from Predator)
                # ============================================
                else:
                    # Local search around current position
                    new_pos = np.zeros(dim)
                    for d in range(dim):
                        lb, ub = self.bounds[d]
                        # Escape in random direction within local neighborhood
                        new_pos[d] = population[i, d] + (1 - 2*r) * (lb + r * (ub - lb)) * 0.1
                
                # Ensure within bounds
                new_pos = self._clip_to_bounds(new_pos, dim)
                
                # Evaluate new position
                new_fitness = fitness_func(new_pos)
                
                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_position = new_pos.copy()
            
            history.append(best_fitness)
            
            if self.verbose and (iter_num + 1) % 10 == 0:
                print(f"  Iteration {iter_num + 1}/{self.max_iter}: Best Fitness = {best_fitness:.4f}")
        
        if self.verbose:
            print(f"\n✅ COATI Optimization Complete!")
            print(f"  • Final best fitness: {best_fitness:.4f}")
        
        return best_position, best_fitness, history


class GNNHyperparameterOptimizer:
    """
    COATI-based optimizer for GNN hyperparameters.
    Optimizes: learning_rate, hidden_units, dropout, climate_threshold
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, adj_matrix, num_countries):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.adj_matrix = adj_matrix
        self.num_countries = num_countries
        
        # Define parameter bounds
        # [learning_rate, hidden_units_1, hidden_units_2, dropout, climate_threshold]
        self.bounds = [
            (0.0001, 0.01),    # learning_rate
            (16, 64),          # hidden_units_1
            (8, 32),           # hidden_units_2
            (0.0, 0.5),        # dropout
            (0.8, 0.99),       # climate_threshold for graph construction
        ]
        self.dim = len(self.bounds)
        
    def fitness_function(self, params):
        """
        Evaluate GNN with given hyperparameters.
        Returns validation MAE (to minimize).
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Suppress TF logs
        tf.get_logger().setLevel('ERROR')
        
        lr, h1, h2, dropout, threshold = params
        h1, h2 = int(h1), int(h2)
        
        try:
            # Build simple model for fast evaluation
            # Using simplified architecture for speed
            X_c_train, X_f_train = self.X_train
            X_c_val, X_f_val = self.X_val
            
            # Simple feedforward for hyperparameter search
            input_c = layers.Input(shape=(1,), dtype='int32')
            input_f = layers.Input(shape=(3,))
            
            emb = layers.Embedding(self.num_countries, h1)(input_c)
            emb = layers.Flatten()(emb)
            emb = layers.Dropout(dropout)(emb)
            
            concat = layers.Concatenate()([emb, input_f])
            x = layers.Dense(h1, activation='relu')(concat)
            x = layers.Dropout(dropout)(x)
            x = layers.Dense(h2, activation='relu')(x)
            output = layers.Dense(1)(x)
            
            model = models.Model([input_c, input_f], output)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
            
            # Quick training (3 epochs for speed)
            model.fit([X_c_train, X_f_train], self.y_train, 
                     epochs=3, batch_size=64, verbose=0)
            
            # Evaluate
            _, mae = model.evaluate([X_c_val, X_f_val], self.y_val, verbose=0)
            
            # Clean up
            tf.keras.backend.clear_session()
            
            return mae
            
        except Exception as e:
            return float('inf')  # Return bad fitness on error
    
    def optimize(self, pop_size=15, max_iter=30):
        """Run COATI optimization for GNN hyperparameters."""
        optimizer = CoatiOptimizer(
            pop_size=pop_size,
            max_iter=max_iter,
            bounds=self.bounds,
            verbose=True
        )
        
        best_params, best_fitness, history = optimizer.optimize(
            self.fitness_function,
            self.dim
        )
        
        return {
            'learning_rate': best_params[0],
            'hidden_units_1': int(best_params[1]),
            'hidden_units_2': int(best_params[2]),
            'dropout': best_params[3],
            'climate_threshold': best_params[4],
            'best_mae': best_fitness,
            'history': history
        }


class EnsembleWeightOptimizer:
    """
    COATI-based optimizer for ensemble model weights.
    Finds optimal combination of multiple model predictions.
    """
    
    def __init__(self, predictions_list, y_true):
        """
        Args:
            predictions_list: List of prediction arrays from different models
            y_true: Ground truth values
        """
        self.predictions = np.array(predictions_list)  # (n_models, n_samples)
        self.y_true = y_true
        self.n_models = len(predictions_list)
        
        # Bounds: weights for each model (will be normalized)
        self.bounds = [(0.0, 1.0) for _ in range(self.n_models)]
    
    def fitness_function(self, weights):
        """Calculate MAE for weighted ensemble."""
        # Normalize weights to sum to 1
        weights = np.array(weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-8)
        
        # Weighted average prediction
        ensemble_pred = np.zeros(len(self.y_true), dtype=np.float64)
        for i, w in enumerate(weights):
            ensemble_pred += w * self.predictions[i].astype(np.float64)
        
        # Calculate MAE
        mae = np.mean(np.abs(ensemble_pred - self.y_true))
        return mae
    
    def optimize(self, pop_size=15, max_iter=30):
        """Run COATI optimization for ensemble weights."""
        optimizer = CoatiOptimizer(
            pop_size=pop_size,
            max_iter=max_iter,
            bounds=self.bounds,
            verbose=True
        )
        
        best_weights, best_fitness, history = optimizer.optimize(
            self.fitness_function,
            self.n_models
        )
        
        # Normalize final weights
        best_weights = np.array(best_weights)
        best_weights = best_weights / best_weights.sum()
        
        return {
            'weights': best_weights.tolist(),
            'best_mae': best_fitness,
            'history': history
        }


# ============================================
# TEST COATI OPTIMIZER
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 Testing COATI Optimizer")
    print("=" * 60)
    
    # Test on Sphere function (simple benchmark)
    def sphere(x):
        return np.sum(x**2)
    
    # Optimize in 5D with bounds [-5, 5]
    test_bounds = [(-5, 5) for _ in range(5)]
    
    optimizer = CoatiOptimizer(
        pop_size=20,
        max_iter=50,
        bounds=test_bounds,
        verbose=True
    )
    
    best_pos, best_fit, history = optimizer.optimize(sphere, dim=5)
    
    print(f"\n📊 Results:")
    print(f"  • Optimal position: {best_pos}")
    print(f"  • Optimal fitness: {best_fit:.6f}")
    print(f"  • True optimum: 0.0 at [0, 0, 0, 0, 0]")
    
    if best_fit < 0.1:
        print("✅ COATI optimizer working correctly!")
    else:
        print("⚠️ COATI may need more iterations for better convergence")
    
    print("\n✅ COATI Optimizer module ready!")
