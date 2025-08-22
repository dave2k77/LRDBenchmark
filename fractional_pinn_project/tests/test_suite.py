"""
Comprehensive Test Suite for Fractional PINN Project

This module provides comprehensive testing for:
1. Unit tests for all components
2. Integration tests for complete pipelines
3. Performance tests and benchmarks
4. Edge case testing
5. Error handling validation

Usage:
    python -m pytest tests/test_suite.py -v
    python tests/test_suite.py  # Run specific test
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import unittest
import warnings
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.generators import FractionalDataGenerator
from estimators.classical_estimators import ClassicalEstimatorSuite, DFAEstimator, RSEstimator
from estimators.ml_estimators import MLEstimatorSuite, FeatureExtractor
from estimators.pinn_estimator import PINNEstimator
from models.fractional_pino import FractionalPINOTrainer
from models.neural_fractional_ode import NeuralFractionalODETrainer
from models.neural_fractional_sde import NeuralFractionalSDETrainer
from models.model_persistence import ModelPersistenceManager, quick_save_model, quick_load_model
from models.model_comparison import ModelConfig, ModelComparisonFramework
from utils.visualization import FractionalVisualizer
from utils.benchmarking import BenchmarkMetrics, StatisticalTesting, AutomatedBenchmark

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TestDataGenerators(unittest.TestCase):
    """Test data generation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_generator = FractionalDataGenerator(seed=42)
        self.n_points = 1000
        
    def test_fbm_generation(self):
        """Test Fractional Brownian Motion generation."""
        # Test different Hurst values
        for hurst in [0.1, 0.5, 0.9]:
            data = self.data_generator.generate_fbm(n_points=self.n_points, hurst=hurst)
            
            self.assertIn('time_series', data)
            self.assertEqual(len(data['time_series']), self.n_points)
            self.assertEqual(data['hurst'], hurst)
            self.assertIsInstance(data['time_series'], np.ndarray)
            
            # Check basic properties
            self.assertFalse(np.any(np.isnan(data['time_series'])))
            self.assertFalse(np.any(np.isinf(data['time_series'])))
    
    def test_fgn_generation(self):
        """Test Fractional Gaussian Noise generation."""
        for hurst in [0.1, 0.5, 0.9]:
            data = self.data_generator.generate_fgn(n_points=self.n_points, hurst=hurst)
            
            self.assertIn('time_series', data)
            self.assertEqual(len(data['time_series']), self.n_points)
            self.assertEqual(data['hurst'], hurst)
            
            # Check stationarity (mean should be roughly constant)
            mean_1 = np.mean(data['time_series'][:self.n_points//2])
            mean_2 = np.mean(data['time_series'][self.n_points//2:])
            self.assertLess(abs(mean_1 - mean_2), 0.5)
    
    def test_arfima_generation(self):
        """Test ARFIMA model generation."""
        for d in [-0.4, 0.0, 0.4]:
            data = self.data_generator.generate_arfima(n_points=self.n_points, d=d)
            
            self.assertIn('time_series', data)
            self.assertEqual(len(data['time_series']), self.n_points)
            self.assertEqual(data['d'], d)
    
    def test_mrw_generation(self):
        """Test Multifractal Random Walk generation."""
        for hurst in [0.1, 0.5, 0.9]:
            data = self.data_generator.generate_mrw(n_points=self.n_points, hurst=hurst)
            
            self.assertIn('time_series', data)
            self.assertEqual(len(data['time_series']), self.n_points)
            self.assertEqual(data['hurst'], hurst)
    
    def test_contamination(self):
        """Test data contamination methods."""
        base_data = self.data_generator.generate_fbm(n_points=self.n_points, hurst=0.7)
        
        # Test different contamination types
        contamination_types = ['noise', 'outliers', 'trends', 'seasonality']
        
        for contam_type in contamination_types:
            contaminated_data = self.data_generator.apply_contamination(
                base_data, contamination_type=contam_type
            )
            
            self.assertIn('time_series', contaminated_data)
            self.assertEqual(len(contaminated_data['time_series']), self.n_points)
            self.assertIn('contamination_info', contaminated_data)
    
    def test_edge_cases(self):
        """Test edge cases in data generation."""
        # Very small n_points
        data = self.data_generator.generate_fbm(n_points=10, hurst=0.5)
        self.assertEqual(len(data['time_series']), 10)
        
        # Extreme Hurst values
        data = self.data_generator.generate_fbm(n_points=100, hurst=0.01)
        self.assertEqual(data['hurst'], 0.01)
        
        data = self.data_generator.generate_fbm(n_points=100, hurst=0.99)
        self.assertEqual(data['hurst'], 0.99)


class TestClassicalEstimators(unittest.TestCase):
    """Test classical estimation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_generator = FractionalDataGenerator(seed=42)
        self.classical_suite = ClassicalEstimatorSuite()
        
        # Generate test data
        self.test_data = self.data_generator.generate_fbm(n_points=1000, hurst=0.7)
        self.time_series = self.test_data['time_series']
        self.true_hurst = self.test_data['hurst']
    
    def test_dfa_estimator(self):
        """Test DFA estimator."""
        estimator = DFAEstimator()
        estimated_hurst = estimator.estimate(self.time_series)
        
        self.assertIsNotNone(estimated_hurst)
        self.assertIsInstance(estimated_hurst, float)
        self.assertGreater(estimated_hurst, 0)
        self.assertLess(estimated_hurst, 1)
        
        # Check reasonable accuracy
        error = abs(estimated_hurst - self.true_hurst)
        self.assertLess(error, 0.3)  # Should be within 0.3 of true value
    
    def test_rs_estimator(self):
        """Test R/S estimator."""
        estimator = RSEstimator()
        estimated_hurst = estimator.estimate(self.time_series)
        
        self.assertIsNotNone(estimated_hurst)
        self.assertIsInstance(estimated_hurst, float)
        self.assertGreater(estimated_hurst, 0)
        self.assertLess(estimated_hurst, 1)
    
    def test_all_classical_estimators(self):
        """Test all classical estimators."""
        estimates = self.classical_suite.estimate_all(self.time_series)
        
        self.assertIsInstance(estimates, dict)
        self.assertGreater(len(estimates), 0)
        
        for estimator_name, estimate in estimates.items():
            if estimate is not None:
                self.assertIsInstance(estimate, float)
                self.assertGreater(estimate, 0)
                self.assertLess(estimate, 1)
    
    def test_estimator_robustness(self):
        """Test estimator robustness to noise."""
        base_data = self.data_generator.generate_fbm(n_points=1000, hurst=0.7)
        
        # Add noise
        noisy_data = self.data_generator.apply_contamination(
            base_data, contamination_type='noise', noise_level=0.1
        )
        
        estimates_clean = self.classical_suite.estimate_all(base_data['time_series'])
        estimates_noisy = self.classical_suite.estimate_all(noisy_data['time_series'])
        
        # Check that estimates are still reasonable
        for estimator_name in estimates_noisy:
            if estimates_noisy[estimator_name] is not None:
                self.assertGreater(estimates_noisy[estimator_name], 0)
                self.assertLess(estimates_noisy[estimator_name], 1)


class TestMLEstimators(unittest.TestCase):
    """Test machine learning estimators."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_generator = FractionalDataGenerator(seed=42)
        self.ml_suite = MLEstimatorSuite()
        self.feature_extractor = FeatureExtractor()
        
        # Generate training data
        self.training_data = []
        for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(5):
                data = self.data_generator.generate_fbm(n_points=1000, hurst=hurst)
                self.training_data.append({
                    'time_series': data['time_series'],
                    'true_hurst': data['hurst']
                })
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        time_series = self.training_data[0]['time_series']
        features = self.feature_extractor.extract_features(time_series)
        
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check that features are numeric
        for feature_name, feature_value in features.items():
            self.assertIsInstance(feature_value, (int, float, np.number))
            self.assertFalse(np.isnan(feature_value))
            self.assertFalse(np.isinf(feature_value))
    
    def test_ml_training(self):
        """Test ML model training."""
        # Train all ML models
        self.ml_suite.train_all(self.training_data)
        
        # Test estimation on new data
        test_data = self.data_generator.generate_fbm(n_points=1000, hurst=0.6)
        estimates = self.ml_suite.estimate_all(test_data['time_series'])
        
        self.assertIsInstance(estimates, dict)
        self.assertGreater(len(estimates), 0)
        
        for estimator_name, estimate in estimates.items():
            if estimate is not None:
                self.assertIsInstance(estimate, float)
                self.assertGreater(estimate, 0)
                self.assertLess(estimate, 1)
    
    def test_individual_ml_estimators(self):
        """Test individual ML estimators."""
        # Train models
        self.ml_suite.train_all(self.training_data)
        
        test_data = self.data_generator.generate_fbm(n_points=1000, hurst=0.6)
        
        # Test each estimator individually
        estimators = [
            self.ml_suite.random_forest_estimator,
            self.ml_suite.gradient_boosting_estimator,
            self.ml_suite.svr_estimator,
            self.ml_suite.linear_regression_estimator
        ]
        
        for estimator in estimators:
            estimate = estimator.estimate(test_data['time_series'])
            if estimate is not None:
                self.assertIsInstance(estimate, float)
                self.assertGreater(estimate, 0)
                self.assertLess(estimate, 1)


class TestPINNEstimator(unittest.TestCase):
    """Test PINN estimator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'  # Use CPU for testing
        self.data_generator = FractionalDataGenerator(seed=42)
        
        # Generate training data
        self.training_data = []
        for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(3):
                data = self.data_generator.generate_fbm(n_points=500, hurst=hurst)
                self.training_data.append({
                    'time_series': data['time_series'],
                    'true_hurst': data['hurst']
                })
    
    def test_pinn_initialization(self):
        """Test PINN estimator initialization."""
        estimator = PINNEstimator(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            output_dim=1,
            learning_rate=0.001,
            device=self.device
        )
        
        self.assertIsNotNone(estimator)
        self.assertEqual(estimator.input_dim, 1)
        self.assertEqual(estimator.output_dim, 1)
        self.assertEqual(estimator.device, self.device)
    
    def test_pinn_model_building(self):
        """Test PINN model building."""
        estimator = PINNEstimator(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            output_dim=1,
            learning_rate=0.001,
            device=self.device
        )
        
        estimator.build_model()
        
        self.assertIsNotNone(estimator.pinn)
        self.assertIsInstance(estimator.pinn, torch.nn.Module)
    
    def test_pinn_training(self):
        """Test PINN training."""
        estimator = PINNEstimator(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            output_dim=1,
            learning_rate=0.001,
            device=self.device
        )
        
        estimator.build_model()
        
        # Train for a few epochs
        history = estimator.train(
            self.training_data,
            epochs=10,  # Small number for testing
            early_stopping_patience=5,
            save_model=False,
            verbose=False
        )
        
        self.assertIsInstance(history, dict)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        
        # Check that loss is decreasing
        if len(history['train_loss']) > 1:
            self.assertLessEqual(history['train_loss'][-1], history['train_loss'][0])
    
    def test_pinn_estimation(self):
        """Test PINN estimation."""
        estimator = PINNEstimator(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            output_dim=1,
            learning_rate=0.001,
            device=self.device
        )
        
        estimator.build_model()
        estimator.train(
            self.training_data,
            epochs=10,
            early_stopping_patience=5,
            save_model=False,
            verbose=False
        )
        
        # Test estimation
        test_data = self.data_generator.generate_fbm(n_points=500, hurst=0.6)
        estimated_hurst = estimator.estimate(test_data['time_series'])
        
        self.assertIsNotNone(estimated_hurst)
        self.assertIsInstance(estimated_hurst, float)
        self.assertGreater(estimated_hurst, 0)
        self.assertLess(estimated_hurst, 1)


class TestNeuralModels(unittest.TestCase):
    """Test neural models (PINO, Neural ODE, Neural SDE)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'  # Use CPU for testing
        self.data_generator = FractionalDataGenerator(seed=42)
        
        # Generate training data
        self.training_data = []
        for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(3):
                data = self.data_generator.generate_fbm(n_points=500, hurst=hurst)
                self.training_data.append({
                    'time_series': data['time_series'],
                    'true_hurst': data['hurst']
                })
    
    def test_pino_trainer(self):
        """Test PINO trainer."""
        trainer = FractionalPINOTrainer(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            modes=8,
            learning_rate=0.001,
            device=self.device
        )
        
        # Train for a few epochs
        history = trainer.train(
            self.training_data,
            epochs=10,
            early_stopping_patience=5,
            save_model=False,
            verbose=False
        )
        
        self.assertIsInstance(history, dict)
        self.assertIn('train_total_loss', history)
        
        # Test estimation
        test_data = self.data_generator.generate_fbm(n_points=500, hurst=0.6)
        estimated_hurst = trainer.estimate(test_data['time_series'])
        
        if estimated_hurst is not None:
            self.assertIsInstance(estimated_hurst, float)
            self.assertGreater(estimated_hurst, 0)
            self.assertLess(estimated_hurst, 1)
    
    def test_neural_ode_trainer(self):
        """Test Neural ODE trainer."""
        trainer = NeuralFractionalODETrainer(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            alpha=0.5,
            learning_rate=0.001,
            device=self.device
        )
        
        # Train for a few epochs
        history = trainer.train(
            self.training_data,
            epochs=10,
            early_stopping_patience=5,
            save_model=False,
            verbose=False
        )
        
        self.assertIsInstance(history, dict)
        self.assertIn('train_total_loss', history)
        
        # Test estimation
        test_data = self.data_generator.generate_fbm(n_points=500, hurst=0.6)
        estimated_hurst = trainer.estimate(test_data['time_series'])
        
        if estimated_hurst is not None:
            self.assertIsInstance(estimated_hurst, float)
            self.assertGreater(estimated_hurst, 0)
            self.assertLess(estimated_hurst, 1)
    
    def test_neural_sde_trainer(self):
        """Test Neural SDE trainer."""
        trainer = NeuralFractionalSDETrainer(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            hurst=0.7,
            learning_rate=0.001,
            device=self.device
        )
        
        # Train for a few epochs
        history = trainer.train(
            self.training_data,
            epochs=10,
            early_stopping_patience=5,
            save_model=False,
            verbose=False
        )
        
        self.assertIsInstance(history, dict)
        self.assertIn('train_total_loss', history)
        
        # Test estimation
        test_data = self.data_generator.generate_fbm(n_points=500, hurst=0.6)
        estimated_hurst = trainer.estimate(test_data['time_series'])
        
        if estimated_hurst is not None:
            self.assertIsInstance(estimated_hurst, float)
            self.assertGreater(estimated_hurst, 0)
            self.assertLess(estimated_hurst, 1)


class TestModelPersistence(unittest.TestCase):
    """Test model persistence system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelPersistenceManager(base_dir=self.temp_dir)
        
        # Create a simple test model
        self.test_model = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
        
        self.test_config = ModelConfig(
            model_type='test',
            input_dim=1,
            hidden_dims=[10],
            output_dim=1,
            learning_rate=0.001,
            epochs=10
        )
        
        self.test_history = {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [1.1, 0.9, 0.7]
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_model_saving(self):
        """Test model saving."""
        model_id = self.model_manager.save_model(
            model=self.test_model,
            config=self.test_config,
            training_history=self.test_history,
            description="Test model",
            tags=['test', 'demo']
        )
        
        self.assertIsInstance(model_id, str)
        self.assertGreater(len(model_id), 0)
        
        # Check that model was saved
        model_info = self.model_manager.get_model_info(model_id)
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info['description'], "Test model")
        self.assertIn('test', model_info['tags'])
    
    def test_model_loading(self):
        """Test model loading."""
        # Save model
        model_id = self.model_manager.save_model(
            model=self.test_model,
            config=self.test_config,
            training_history=self.test_history,
            description="Test model"
        )
        
        # Load model
        loaded_model, loaded_config, loaded_metadata = self.model_manager.load_model(model_id)
        
        self.assertIsInstance(loaded_model, torch.nn.Module)
        self.assertEqual(loaded_config.model_type, self.test_config.model_type)
        self.assertEqual(loaded_config.input_dim, self.test_config.input_dim)
        self.assertIsInstance(loaded_metadata, dict)
    
    def test_quick_save_load(self):
        """Test quick save and load functions."""
        # Quick save
        model_id = quick_save_model(
            model=self.test_model,
            config=self.test_config,
            training_history=self.test_history,
            description="Quick test model"
        )
        
        # Quick load
        loaded_model, loaded_config, loaded_metadata = quick_load_model(model_id)
        
        self.assertIsInstance(loaded_model, torch.nn.Module)
        self.assertEqual(loaded_config.model_type, self.test_config.model_type)
    
    def test_model_listing(self):
        """Test model listing."""
        # Save multiple models
        for i in range(3):
            self.model_manager.save_model(
                model=self.test_model,
                config=self.test_config,
                training_history=self.test_history,
                description=f"Test model {i}",
                tags=['test']
            )
        
        # List models
        models_df = self.model_manager.list_models(tags=['test'])
        
        self.assertIsInstance(models_df, pd.DataFrame)
        self.assertGreaterEqual(len(models_df), 3)
    
    def test_model_deletion(self):
        """Test model deletion."""
        # Save model
        model_id = self.model_manager.save_model(
            model=self.test_model,
            config=self.test_config,
            training_history=self.test_history,
            description="Test model for deletion"
        )
        
        # Delete model
        self.model_manager.delete_model(model_id)
        
        # Check that model is deleted
        models_df = self.model_manager.list_models()
        model_ids = models_df['model_id'].tolist()
        self.assertNotIn(model_id, model_ids)


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_generator = FractionalDataGenerator(seed=42)
        
        # Generate test data
        self.test_data = []
        for hurst in [0.1, 0.5, 0.9]:
            for _ in range(3):
                data = self.data_generator.generate_fbm(n_points=500, hurst=hurst)
                self.test_data.append({
                    'time_series': data['time_series'],
                    'true_hurst': data['hurst']
                })
    
    def test_benchmark_metrics(self):
        """Test benchmark metrics calculation."""
        true_values = np.array([0.1, 0.5, 0.9])
        estimated_values = np.array([0.12, 0.48, 0.88])
        
        metrics = BenchmarkMetrics.calculate_metrics(true_values, estimated_values)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('bias', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['r2'], 0.5)  # Good correlation
        self.assertLess(metrics['mae'], 0.1)    # Low error
    
    def test_statistical_testing(self):
        """Test statistical testing utilities."""
        errors1 = np.random.normal(0, 0.1, 100)
        errors2 = np.random.normal(0.05, 0.1, 100)
        
        # Paired t-test
        t_test_result = StatisticalTesting.paired_t_test(errors1, errors2)
        
        self.assertIsInstance(t_test_result, dict)
        self.assertIn('statistic', t_test_result)
        self.assertIn('p_value', t_test_result)
        self.assertIn('significant', t_test_result)
        
        # Mann-Whitney test
        mw_test_result = StatisticalTesting.mann_whitney_test(errors1, errors2)
        
        self.assertIsInstance(mw_test_result, dict)
        self.assertIn('statistic', mw_test_result)
        self.assertIn('p_value', mw_test_result)
        self.assertIn('significant', mw_test_result)
    
    def test_automated_benchmark(self):
        """Test automated benchmark."""
        # Create dummy estimators
        def dummy_estimator1(time_series):
            return 0.5
        
        def dummy_estimator2(time_series):
            return 0.6
        
        estimators = {
            'dummy1': dummy_estimator1,
            'dummy2': dummy_estimator2
        }
        
        # Run benchmark
        benchmark = AutomatedBenchmark(save_results=False, n_jobs=1)
        results = benchmark.run_parallel_benchmark(estimators, self.test_data)
        
        self.assertIsInstance(results, dict)
        self.assertIn('dummy1', results)
        self.assertIn('dummy2', results)
        
        # Check results structure
        for estimator_name, result in results.items():
            self.assertIn('estimator', result)
            self.assertIn('mae', result)
            self.assertIn('success_rate', result)
            self.assertEqual(result['estimator'], estimator_name)


class TestVisualization(unittest.TestCase):
    """Test visualization utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = FractionalVisualizer()
        self.data_generator = FractionalDataGenerator(seed=42)
        
        # Generate test data
        self.test_data = self.data_generator.generate_fbm(n_points=1000, hurst=0.7)
        self.time_series = self.test_data['time_series']
    
    def test_training_curves_plot(self):
        """Test training curves plotting."""
        training_history = {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3]
        }
        
        fig = self.visualizer.plot_training_curves(
            training_history, 
            show_plot=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
    
    def test_data_exploration_plot(self):
        """Test data exploration plotting."""
        fig = self.visualizer.plot_data_exploration(
            self.time_series,
            hurst=0.7,
            show_plot=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
    
    def test_model_comparison_plot(self):
        """Test model comparison plotting."""
        # Create dummy results
        results_df = pd.DataFrame({
            'estimator': ['PINN', 'PINO', 'DFA', 'R/S'],
            'true_hurst': [0.7, 0.7, 0.7, 0.7],
            'estimated_hurst': [0.68, 0.72, 0.65, 0.75],
            'absolute_error': [0.02, 0.02, 0.05, 0.05],
            'model_type': ['neural', 'neural', 'classical', 'classical']
        })
        
        fig = self.visualizer.plot_model_comparison(
            results_df,
            show_plot=False
        )
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipelines."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_generator = FractionalDataGenerator(seed=42)
        self.device = 'cpu'
        
        # Generate comprehensive test data
        self.test_data = []
        for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for data_type in ['fbm', 'fgn']:
                for _ in range(2):
                    if data_type == 'fbm':
                        data = self.data_generator.generate_fbm(n_points=500, hurst=hurst)
                    else:
                        data = self.data_generator.generate_fgn(n_points=500, hurst=hurst)
                    
                    self.test_data.append({
                        'time_series': data['time_series'],
                        'true_hurst': data['hurst']
                    })
    
    def test_complete_pipeline(self):
        """Test complete estimation pipeline."""
        # 1. Data generation
        self.assertGreater(len(self.test_data), 0)
        
        # 2. Classical estimation
        classical_suite = ClassicalEstimatorSuite()
        classical_results = []
        
        for data in self.test_data:
            estimates = classical_suite.estimate_all(data['time_series'])
            for estimator_name, estimate in estimates.items():
                if estimate is not None:
                    classical_results.append({
                        'estimator': estimator_name,
                        'true_hurst': data['true_hurst'],
                        'estimated_hurst': estimate,
                        'absolute_error': abs(estimate - data['true_hurst'])
                    })
        
        self.assertGreater(len(classical_results), 0)
        
        # 3. ML estimation
        ml_suite = MLEstimatorSuite()
        ml_suite.train_all(self.test_data[:10])  # Train on subset
        
        ml_results = []
        for data in self.test_data:
            estimates = ml_suite.estimate_all(data['time_series'])
            for estimator_name, estimate in estimates.items():
                if estimate is not None:
                    ml_results.append({
                        'estimator': estimator_name,
                        'true_hurst': data['true_hurst'],
                        'estimated_hurst': estimate,
                        'absolute_error': abs(estimate - data['true_hurst'])
                    })
        
        self.assertGreater(len(ml_results), 0)
        
        # 4. Neural estimation (PINN)
        pinn_estimator = PINNEstimator(
            input_dim=1,
            hidden_dims=[32, 64, 32],
            output_dim=1,
            learning_rate=0.001,
            device=self.device
        )
        
        pinn_estimator.build_model()
        pinn_estimator.train(
            self.test_data[:5],  # Train on small subset
            epochs=5,
            early_stopping_patience=3,
            save_model=False,
            verbose=False
        )
        
        neural_results = []
        for data in self.test_data[:3]:  # Test on subset
            estimate = pinn_estimator.estimate(data['time_series'])
            if estimate is not None:
                neural_results.append({
                    'estimator': 'PINN',
                    'true_hurst': data['true_hurst'],
                    'estimated_hurst': estimate,
                    'absolute_error': abs(estimate - data['true_hurst'])
                })
        
        self.assertGreater(len(neural_results), 0)
        
        # 5. Combine and analyze results
        all_results = classical_results + ml_results + neural_results
        
        # Calculate overall metrics
        true_values = [r['true_hurst'] for r in all_results]
        estimated_values = [r['estimated_hurst'] for r in all_results]
        
        metrics = BenchmarkMetrics.calculate_metrics(
            np.array(true_values), 
            np.array(estimated_values)
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
    
    def test_model_comparison_framework(self):
        """Test model comparison framework."""
        framework = ModelComparisonFramework()
        
        # Add models
        framework.add_pinn_model(
            name="Test PINN",
            input_dim=1,
            hidden_dims=[32, 64, 32],
            output_dim=1,
            learning_rate=0.001
        )
        
        framework.add_pino_model(
            name="Test PINO",
            input_dim=1,
            hidden_dims=[32, 64, 32],
            modes=8,
            learning_rate=0.001
        )
        
        self.assertEqual(len(framework.models), 2)
        
        # Train models
        training_data = self.test_data[:5]
        framework.train_all_models(training_data, epochs=5, verbose=False)
        
        # Evaluate models
        test_data = self.test_data[5:8]
        results = framework.evaluate_all_models(test_data)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)


class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_generator = FractionalDataGenerator(seed=42)
    
    def test_large_data_handling(self):
        """Test handling of large datasets."""
        # Generate large dataset
        large_data = []
        for hurst in [0.1, 0.5, 0.9]:
            for _ in range(10):
                data = self.data_generator.generate_fbm(n_points=2000, hurst=hurst)
                large_data.append({
                    'time_series': data['time_series'],
                    'true_hurst': data['hurst']
                })
        
        # Test classical estimators on large data
        classical_suite = ClassicalEstimatorSuite()
        
        start_time = time.time()
        for data in large_data[:5]:  # Test on subset
            estimates = classical_suite.estimate_all(data['time_series'])
            self.assertIsInstance(estimates, dict)
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 60)  # Less than 60 seconds
    
    def test_memory_usage(self):
        """Test memory usage with large models."""
        # Test PINN with large architecture
        large_pinn = PINNEstimator(
            input_dim=1,
            hidden_dims=[256, 512, 512, 256],
            output_dim=1,
            learning_rate=0.001,
            device='cpu'
        )
        
        large_pinn.build_model()
        
        # Generate test data
        test_data = self.data_generator.generate_fbm(n_points=1000, hurst=0.7)
        
        # Should be able to process data without memory issues
        estimate = large_pinn.estimate(test_data['time_series'])
        self.assertIsNotNone(estimate)


def run_all_tests():
    """Run all tests."""
    print("Running Comprehensive Test Suite for Fractional PINN Project")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataGenerators,
        TestClassicalEstimators,
        TestMLEstimators,
        TestPINNEstimator,
        TestNeuralModels,
        TestModelPersistence,
        TestBenchmarking,
        TestVisualization,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import time
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The fractional PINN project is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED! ❌")
        print("Please check the failures and errors above.")
    
    sys.exit(0 if success else 1)
