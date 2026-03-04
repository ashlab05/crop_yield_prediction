"""
Artificial Protozoa Optimizer (APO) — 2025
==========================================
Population-based metaheuristic inspired by protozoa behavior.

Based on: Shehab et al. (2025) "Artificial Protozoa Optimizer: 
A Novel Bio-inspired Metaheuristic Algorithm"

Phases:
1. Foraging — explore search space
2. Engulfing — exploit promising regions  
3. Binary fission — reproduce best solutions
4. Conjugation — exchange information between solutions
"""
import math
import numpy as np
from copy import deepcopy


class APO:
    """
    Artificial Protozoa Optimizer for hyperparameter search.
    
    Each protozoa represents a hyperparameter configuration.
    """
    def __init__(self, objective_fn, search_space, pop_size=10, max_iter=10, 
                 random_state=42, verbose=True):
        """
        Args:
            objective_fn: callable(config_dict) -> float (lower is better)
            search_space: dict of {param_name: (low, high, type)}
                         type is 'float', 'int', or 'log_float'
            pop_size: population size
            max_iter: max iterations
        """
        self.objective_fn = objective_fn
        self.search_space = search_space
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
        
        self.param_names = list(search_space.keys())
        self.n_params = len(self.param_names)
        
        # Bounds
        self.lb = np.array([search_space[p][0] for p in self.param_names])
        self.ub = np.array([search_space[p][1] for p in self.param_names])
        self.types = [search_space[p][2] for p in self.param_names]
        
        self.best_config = None
        self.best_score = float('inf')
        self.history = []
    
    def _encode(self, config):
        """Config dict -> continuous vector."""
        vec = np.zeros(self.n_params)
        for i, name in enumerate(self.param_names):
            val = config[name]
            if self.types[i] == 'log_float':
                vec[i] = np.log(val)
            else:
                vec[i] = val
        return vec
    
    def _decode(self, vec):
        """Continuous vector -> config dict."""
        config = {}
        for i, name in enumerate(self.param_names):
            val = np.clip(vec[i], self.lb[i], self.ub[i])
            if self.types[i] == 'int':
                config[name] = int(round(val))
            elif self.types[i] == 'log_float':
                config[name] = float(np.exp(np.clip(val, self.lb[i], self.ub[i])))
            else:
                config[name] = float(val)
        return config
    
    def _random_solution(self):
        """Generate a random solution vector."""
        vec = np.zeros(self.n_params)
        for i in range(self.n_params):
            vec[i] = self.rng.uniform(self.lb[i], self.ub[i])
        return vec
    
    def _foraging(self, position, best_position, iteration, max_iter):
        """Foraging phase: explore search space using Lévy flights."""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        
        u = self.rng.randn(self.n_params) * sigma
        v = self.rng.randn(self.n_params)
        levy = u / (np.abs(v) ** (1 / beta))
        
        # Decreasing step size
        alpha = 1.0 * (1 - iteration / max_iter)
        
        new_pos = position + alpha * levy * (best_position - position)
        return np.clip(new_pos, self.lb, self.ub)
    
    def _engulfing(self, position, best_position, prey_position):
        """Engulfing phase: exploit promising regions."""
        r = self.rng.rand()
        if r < 0.5:
            new_pos = best_position + self.rng.randn(self.n_params) * 0.1 * (best_position - position)
        else:
            new_pos = prey_position + self.rng.randn(self.n_params) * 0.1 * (prey_position - position)
        return np.clip(new_pos, self.lb, self.ub)
    
    def _binary_fission(self, position):
        """Binary fission: create offspring near parent."""
        mutation = self.rng.randn(self.n_params) * 0.05 * (self.ub - self.lb)
        child = position + mutation
        return np.clip(child, self.lb, self.ub)
    
    def _conjugation(self, pos1, pos2):
        """Conjugation: crossover between two solutions."""
        mask = self.rng.rand(self.n_params) > 0.5
        child = np.where(mask, pos1, pos2)
        return np.clip(child, self.lb, self.ub)
    
    def optimize(self):
        """Run the APO optimization loop."""
        # Initialize population
        population = [self._random_solution() for _ in range(self.pop_size)]
        fitness = np.array([float('inf')] * self.pop_size)
        
        # Evaluate initial population
        if self.verbose:
            print(f"  APO: Evaluating initial population ({self.pop_size} configs)...")
        
        for i in range(self.pop_size):
            config = self._decode(population[i])
            try:
                score = self.objective_fn(config)
                fitness[i] = score
                if score < self.best_score:
                    self.best_score = score
                    self.best_config = deepcopy(config)
                if self.verbose:
                    print(f"    Init {i+1}/{self.pop_size}: score={score:.2f} | {config}")
            except Exception as e:
                if self.verbose:
                    print(f"    Init {i+1}/{self.pop_size}: FAILED ({e})")
                fitness[i] = float('inf')
        
        self.history.append(self.best_score)
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            if self.verbose:
                print(f"\n  APO Iteration {iteration+1}/{self.max_iter} (best={self.best_score:.2f})")
            
            best_idx = np.argmin(fitness)
            best_pos = population[best_idx].copy()
            
            for i in range(self.pop_size):
                r = self.rng.rand()
                
                if r < 0.4:
                    # Foraging phase
                    new_pos = self._foraging(population[i], best_pos, iteration, self.max_iter)
                elif r < 0.7:
                    # Engulfing phase
                    prey_idx = self.rng.randint(self.pop_size)
                    new_pos = self._engulfing(population[i], best_pos, population[prey_idx])
                elif r < 0.85:
                    # Binary fission
                    new_pos = self._binary_fission(population[i])
                else:
                    # Conjugation
                    partner_idx = self.rng.randint(self.pop_size)
                    new_pos = self._conjugation(population[i], population[partner_idx])
                
                # Evaluate new position
                config = self._decode(new_pos)
                try:
                    score = self.objective_fn(config)
                    
                    # Greedy selection
                    if score < fitness[i]:
                        population[i] = new_pos
                        fitness[i] = score
                        
                        if score < self.best_score:
                            self.best_score = score
                            self.best_config = deepcopy(config)
                            if self.verbose:
                                print(f"    ★ New best: {score:.2f} | {config}")
                except Exception as e:
                    pass  # Keep old position
            
            self.history.append(self.best_score)
        
        if self.verbose:
            print(f"\n  APO complete. Best score: {self.best_score:.2f}")
            print(f"  Best config: {self.best_config}")
        
        return self.best_config, self.best_score
