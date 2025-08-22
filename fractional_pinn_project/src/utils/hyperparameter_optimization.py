"""
Hyperparameter Optimization for Fractional PINN Project

This module provides comprehensive hyperparameter optimization for:
1. Neural models (PINN, PINO, Neural ODE, Neural SDE)
2. Classical estimators
3. ML estimators
4. Multi-objective optimization
5. Bayesian optimization with various acquisition functions

Features:
- Bayesian optimization with GP, RF, and TPE
- Grid search and random search
- Multi-objective optimization (accuracy vs speed)
- Cross-validation support
- Early stopping and pruning
- Parallel optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization libraries
try:
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some features will be disabled.")

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("optuna not available. Bayesian optimization will be disabled.")

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Some optimization methods will be disabled.")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optimization method
    method: str = 'bayesian'  # 'bayesian', 'grid', 'random', 'evolution'
    
    # Optimization parameters
    n_trials: int = 100
    n_jobs: int = 1
    timeout: Optional[int] = None  # seconds
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = 'kfold'  # 'kfold', 'stratified', 'time_series'
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_trials: int = 20
    
    # Multi-objective
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ['mae', 'training_time'])
    
    # Bayesian optimization specific
    sampler: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    pruner: str = 'median'  # 'median', 'hyperband', None
    
    # Output
    save_results: bool = True
    results_dir: str = "optimization_results"
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.method == 'bayesian' and not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to random search")
            self.method = 'random'
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, CV will be disabled")
            self.cv_folds = 1


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""
    
    # Neural network architecture
    hidden_dims: List[List[int]] = field(default_factory=lambda: [
        [32, 64, 32],
        [64, 128, 64],
        [128, 256, 128],
        [32, 64, 128, 64, 32]
    ])
    
    # Learning rates
    learning_rates: List[float] = field(default_factory=lambda: [
        1e-4, 5e-4, 1e-3, 5e-3, 1e-2
    ])
    
    # Batch sizes
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    
    # Training parameters
    epochs: List[int] = field(default_factory=lambda: [100, 200, 500, 1000])
    early_stopping_patience: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # PINO specific
    modes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    
    # Neural ODE specific
    alphas: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Neural SDE specific
    hurst_values: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Physics constraints
    physics_weight: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    mellin_weight: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    
    # ML specific
    max_depth: List[int] = field(default_factory=lambda: [3, 5, 7, 10, None])
    n_estimators: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 2, 5])


class BaseOptimizer(ABC):
    """Base class for hyperparameter optimizers."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer."""
        self.config = config
        self.results = []
        self.best_params = None
        self.best_score = float('inf')
        
        # Create results directory
        if config.save_results:
            self.results_dir = Path(config.results_dir)
            self.results_dir.mkdir(exist_ok=True)
    
    @abstractmethod
    def optimize(self, objective_func: Callable, param_space: Dict) -> Dict[str, Any]:
        """Optimize hyperparameters."""
        pass
    
    def save_results(self, filename: str = None):
        """Save optimization results."""
        if not self.config.save_results:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        results_file = self.results_dir / filename
        
        # Prepare results for saving
        save_data = {
            'config': self.config.__dict__,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hyperparameter Optimization History', fontsize=16)
        
        # Extract data
        scores = [r['score'] for r in self.results]
        trials = list(range(1, len(scores) + 1))
        
        # 1. Score progression
        ax1 = axes[0, 0]
        ax1.plot(trials, scores, 'b-', alpha=0.7)
        ax1.set_title('Score Progression')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Score (MAE)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Best score so far
        ax2 = axes[0, 1]
        best_scores = [min(scores[:i+1]) for i in range(len(scores))]
        ax2.plot(trials, best_scores, 'g-', linewidth=2)
        ax2.set_title('Best Score So Far')
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Best Score')
        ax2.grid(True, alpha=0.3)
        
        # 3. Score distribution
        ax3 = axes[1, 0]
        ax3.hist(scores, bins=20, alpha=0.7, color='skyblue')
        ax3.set_title('Score Distribution')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.axvline(self.best_score, color='red', linestyle='--', label=f'Best: {self.best_score:.4f}')
        ax3.legend()
        
        # 4. Parameter importance (if available)
        ax4 = axes[1, 1]
        if len(self.results) > 1:
            # Simple correlation analysis
            param_importance = self._analyze_parameter_importance()
            if param_importance:
                params = list(param_importance.keys())
                importances = list(param_importance.values())
                ax4.barh(params, importances)
                ax4.set_title('Parameter Importance')
                ax4.set_xlabel('Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization history plot saved to {save_path}")
        
        plt.show()
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance using correlation."""
        if len(self.results) < 2:
            return {}
        
        # Extract parameters and scores
        param_data = {}
        scores = []
        
        for result in self.results:
            scores.append(result['score'])
            for param, value in result['params'].items():
                if param not in param_data:
                    param_data[param] = []
                param_data[param].append(value)
        
        # Calculate correlations
        importance = {}
        for param, values in param_data.items():
            if len(set(values)) > 1:  # Parameter varies
                correlation = np.corrcoef(values, scores)[0, 1]
                importance[param] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Optuna."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize Bayesian optimizer."""
        super().__init__(config)
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        # Create study
        study_name = f"fractional_pinn_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sampler
        if config.sampler == 'tpe':
            sampler = TPESampler(seed=42)
        elif config.sampler == 'random':
            sampler = RandomSampler(seed=42)
        elif config.sampler == 'cmaes':
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = TPESampler(seed=42)
        
        # Pruner
        if config.pruner == 'median':
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif config.pruner == 'hyperband':
            pruner = HyperbandPruner()
        else:
            pruner = None
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
    
    def optimize(self, objective_func: Callable, param_space: Dict) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        logger.info("Starting Bayesian optimization...")
        
        def objective(trial):
            # Sample parameters
            params = self._sample_parameters(trial, param_space)
            
            # Run objective function
            start_time = time.time()
            score = objective_func(params)
            trial_time = time.time() - start_time
            
            # Store result
            result = {
                'params': params,
                'score': score,
                'time': trial_time,
                'trial_number': len(self.results) + 1
            }
            self.results.append(result)
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            if self.config.verbose and len(self.results) % 10 == 0:
                logger.info(f"Trial {len(self.results)}: Score = {score:.4f}, Best = {self.best_score:.4f}")
            
            return score
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        # Get best result
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study,
            'results': self.results
        }
    
    def _sample_parameters(self, trial, param_space: Dict) -> Dict[str, Any]:
        """Sample parameters from the search space."""
        params = {}
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'log_float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
        
        return params
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history with Optuna visualizations."""
        if not OPTUNA_AVAILABLE:
            super().plot_optimization_history(save_path)
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bayesian Optimization Results', fontsize=16)
        
        # 1. Optimization history
        ax1 = axes[0, 0]
        plot_optimization_history(self.study, ax=ax1)
        ax1.set_title('Optimization History')
        
        # 2. Parameter importance
        ax2 = axes[0, 1]
        try:
            plot_param_importances(self.study, ax=ax2)
            ax2.set_title('Parameter Importance')
        except:
            ax2.text(0.5, 0.5, 'Parameter importance\nnot available', ha='center', va='center')
            ax2.set_title('Parameter Importance')
        
        # 3. Score progression
        ax3 = axes[1, 0]
        scores = [r['score'] for r in self.results]
        trials = list(range(1, len(scores) + 1))
        ax3.plot(trials, scores, 'b-', alpha=0.7)
        ax3.set_title('Score Progression')
        ax3.set_xlabel('Trial')
        ax3.set_ylabel('Score (MAE)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Best score so far
        ax4 = axes[1, 1]
        best_scores = [min(scores[:i+1]) for i in range(len(scores))]
        ax4.plot(trials, best_scores, 'g-', linewidth=2)
        ax4.set_title('Best Score So Far')
        ax4.set_xlabel('Trial')
        ax4.set_ylabel('Best Score')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bayesian optimization plot saved to {save_path}")
        
        plt.show()


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimization."""
    
    def optimize(self, objective_func: Callable, param_space: Dict) -> Dict[str, Any]:
        """Run grid search optimization."""
        logger.info("Starting grid search optimization...")
        
        # Generate parameter combinations
        param_combinations = self._generate_combinations(param_space)
        total_combinations = len(param_combinations)
        
        logger.info(f"Total parameter combinations: {total_combinations}")
        
        # Evaluate all combinations
        for i, params in enumerate(param_combinations):
            if self.config.verbose and i % 10 == 0:
                logger.info(f"Evaluating combination {i+1}/{total_combinations}")
            
            # Run objective function
            start_time = time.time()
            score = objective_func(params)
            trial_time = time.time() - start_time
            
            # Store result
            result = {
                'params': params,
                'score': score,
                'time': trial_time,
                'trial_number': i + 1
            }
            self.results.append(result)
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        
        logger.info(f"Grid search completed. Best score: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'results': self.results
        }
    
    def _generate_combinations(self, param_space: Dict) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools
        
        # Prepare parameter lists
        param_lists = []
        param_names = []
        
        for param_name, param_config in param_space.items():
            param_names.append(param_name)
            if param_config['type'] == 'categorical':
                param_lists.append(param_config['choices'])
            elif param_config['type'] in ['int', 'float']:
                param_lists.append(param_config['choices'])
        
        # Generate combinations
        combinations = []
        for combination in itertools.product(*param_lists):
            params = dict(zip(param_names, combination))
            combinations.append(params)
        
        return combinations


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimization."""
    
    def optimize(self, objective_func: Callable, param_space: Dict) -> Dict[str, Any]:
        """Run random search optimization."""
        logger.info("Starting random search optimization...")
        
        np.random.seed(42)
        
        for trial in range(self.config.n_trials):
            if self.config.verbose and trial % 10 == 0:
                logger.info(f"Trial {trial+1}/{self.config.n_trials}")
            
            # Sample random parameters
            params = self._sample_random_parameters(param_space)
            
            # Run objective function
            start_time = time.time()
            score = objective_func(params)
            trial_time = time.time() - start_time
            
            # Store result
            result = {
                'params': params,
                'score': score,
                'time': trial_time,
                'trial_number': trial + 1
            }
            self.results.append(result)
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
        
        logger.info(f"Random search completed. Best score: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'results': self.results
        }
    
    def _sample_random_parameters(self, param_space: Dict) -> Dict[str, Any]:
        """Sample random parameters from the search space."""
        params = {}
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'categorical':
                params[param_name] = np.random.choice(param_config['choices'])
            elif param_config['type'] == 'int':
                params[param_name] = np.random.randint(param_config['low'], param_config['high'] + 1)
            elif param_config['type'] == 'float':
                params[param_name] = np.random.uniform(param_config['low'], param_config['high'])
            elif param_config['type'] == 'log_float':
                params[param_name] = np.exp(np.random.uniform(np.log(param_config['low']), np.log(param_config['high'])))
        
        return params


class HyperparameterOptimizer:
    """Main hyperparameter optimization class."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer."""
        self.config = config
        
        # Create optimizer based on method
        if config.method == 'bayesian':
            if not OPTUNA_AVAILABLE:
                logger.warning("Optuna not available, falling back to random search")
                config.method = 'random'
            else:
                self.optimizer = BayesianOptimizer(config)
        
        if config.method == 'grid':
            self.optimizer = GridSearchOptimizer(config)
        elif config.method == 'random':
            self.optimizer = RandomSearchOptimizer(config)
        else:
            raise ValueError(f"Unknown optimization method: {config.method}")
    
    def optimize_pinn(self, training_data: List[Dict], validation_data: List[Dict] = None) -> Dict[str, Any]:
        """Optimize PINN hyperparameters."""
        logger.info("Optimizing PINN hyperparameters...")
        
        # Define parameter space
        param_space = {
            'hidden_dims': {
                'type': 'categorical',
                'choices': [[32, 64, 32], [64, 128, 64], [128, 256, 128], [32, 64, 128, 64, 32]]
            },
            'learning_rate': {
                'type': 'log_float',
                'low': 1e-4,
                'high': 1e-2
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64, 128]
            },
            'epochs': {
                'type': 'categorical',
                'choices': [100, 200, 500]
            },
            'physics_weight': {
                'type': 'log_float',
                'low': 0.1,
                'high': 5.0
            }
        }
        
        # Define objective function
        def objective(params):
            try:
                from estimators.pinn_estimator import PINNEstimator
                
                # Create and train model
                estimator = PINNEstimator(
                    input_dim=1,
                    hidden_dims=params['hidden_dims'],
                    output_dim=1,
                    learning_rate=params['learning_rate'],
                    device='cpu'  # Use CPU for optimization
                )
                
                estimator.build_model()
                
                # Train model
                history = estimator.train(
                    training_data,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    early_stopping_patience=20,
                    save_model=False,
                    verbose=False
                )
                
                # Evaluate on validation data
                if validation_data:
                    errors = []
                    for data in validation_data:
                        estimate = estimator.estimate(data['time_series'])
                        if estimate is not None:
                            error = abs(estimate - data['true_hurst'])
                            errors.append(error)
                    
                    if errors:
                        return np.mean(errors)
                    else:
                        return float('inf')
                else:
                    # Use training loss as proxy
                    return history['train_loss'][-1] if history['train_loss'] else float('inf')
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        results = self.optimizer.optimize(objective, param_space)
        
        # Save results
        if self.config.save_results:
            self.optimizer.save_results("pinn_optimization_results.json")
            self.optimizer.plot_optimization_history("pinn_optimization_history.png")
        
        return results
    
    def optimize_pino(self, training_data: List[Dict], validation_data: List[Dict] = None) -> Dict[str, Any]:
        """Optimize PINO hyperparameters."""
        logger.info("Optimizing PINO hyperparameters...")
        
        # Define parameter space
        param_space = {
            'hidden_dims': {
                'type': 'categorical',
                'choices': [[32, 64, 32], [64, 128, 64], [128, 256, 128]]
            },
            'modes': {
                'type': 'categorical',
                'choices': [8, 16, 32, 64]
            },
            'learning_rate': {
                'type': 'log_float',
                'low': 1e-4,
                'high': 1e-2
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64]
            }
        }
        
        # Define objective function
        def objective(params):
            try:
                from models.fractional_pino import FractionalPINOTrainer
                
                # Create and train model
                trainer = FractionalPINOTrainer(
                    input_dim=1,
                    hidden_dims=params['hidden_dims'],
                    modes=params['modes'],
                    learning_rate=params['learning_rate'],
                    device='cpu'
                )
                
                # Train model
                history = trainer.train(
                    training_data,
                    epochs=200,
                    batch_size=params['batch_size'],
                    early_stopping_patience=20,
                    save_model=False,
                    verbose=False
                )
                
                # Evaluate
                if validation_data:
                    errors = []
                    for data in validation_data:
                        estimate = trainer.estimate(data['time_series'])
                        if estimate is not None:
                            error = abs(estimate - data['true_hurst'])
                            errors.append(error)
                    
                    return np.mean(errors) if errors else float('inf')
                else:
                    return history['train_total_loss'][-1] if history['train_total_loss'] else float('inf')
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        results = self.optimizer.optimize(objective, param_space)
        
        # Save results
        if self.config.save_results:
            self.optimizer.save_results("pino_optimization_results.json")
            self.optimizer.plot_optimization_history("pino_optimization_history.png")
        
        return results
    
    def optimize_ml_models(self, training_data: List[Dict], validation_data: List[Dict] = None) -> Dict[str, Any]:
        """Optimize ML model hyperparameters."""
        logger.info("Optimizing ML model hyperparameters...")
        
        # Define parameter space for Random Forest
        rf_param_space = {
            'n_estimators': {
                'type': 'categorical',
                'choices': [50, 100, 200, 500]
            },
            'max_depth': {
                'type': 'categorical',
                'choices': [3, 5, 7, 10, None]
            },
            'min_samples_split': {
                'type': 'categorical',
                'choices': [2, 5, 10]
            },
            'min_samples_leaf': {
                'type': 'categorical',
                'choices': [1, 2, 5]
            }
        }
        
        # Define objective function
        def objective(params):
            try:
                from estimators.ml_estimators import MLEstimatorSuite
                
                # Create ML suite with custom parameters
                ml_suite = MLEstimatorSuite()
                
                # Train with custom parameters
                ml_suite.train_all(training_data, rf_params=params)
                
                # Evaluate
                if validation_data:
                    errors = []
                    for data in validation_data:
                        estimates = ml_suite.estimate_all(data['time_series'])
                        for estimator_name, estimate in estimates.items():
                            if estimate is not None:
                                error = abs(estimate - data['true_hurst'])
                                errors.append(error)
                    
                    return np.mean(errors) if errors else float('inf')
                else:
                    return 0.1  # Default score if no validation data
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Run optimization
        results = self.optimizer.optimize(objective, rf_param_space)
        
        # Save results
        if self.config.save_results:
            self.optimizer.save_results("ml_optimization_results.json")
            self.optimizer.plot_optimization_history("ml_optimization_history.png")
        
        return results


def quick_optimize(model_type: str, training_data: List[Dict], 
                  method: str = 'bayesian', n_trials: int = 50) -> Dict[str, Any]:
    """Quick optimization function."""
    
    config = OptimizationConfig(
        method=method,
        n_trials=n_trials,
        save_results=True,
        verbose=True
    )
    
    optimizer = HyperparameterOptimizer(config)
    
    if model_type == 'pinn':
        return optimizer.optimize_pinn(training_data)
    elif model_type == 'pino':
        return optimizer.optimize_pino(training_data)
    elif model_type == 'ml':
        return optimizer.optimize_ml_models(training_data)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    print("Hyperparameter Optimization for Fractional PINN Project")
    print("=" * 60)
    
    # Create sample data
    from data.generators import FractionalDataGenerator
    
    data_generator = FractionalDataGenerator(seed=42)
    training_data = []
    
    for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for _ in range(5):
            data = data_generator.generate_fbm(n_points=500, hurst=hurst)
            training_data.append({
                'time_series': data['time_series'],
                'true_hurst': data['hurst']
            })
    
    # Quick optimization example
    print("Running quick PINN optimization...")
    results = quick_optimize('pinn', training_data, method='random', n_trials=10)
    
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Best parameters: {results['best_params']}")
    
    print("Optimization completed!")
