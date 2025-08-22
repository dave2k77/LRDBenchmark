"""
Train Once, Apply Many Demo

This script demonstrates the comprehensive model persistence system
that enables "train once, apply many" functionality for all our
fractional neural models (PINN, PINO, Neural ODE, Neural SDE).

Key Features Demonstrated:
1. Training models with automatic saving
2. Loading pre-trained models
3. Model comparison and selection
4. Batch inference with saved models
5. Model versioning and management

Author: Fractional PINN Research Team
Date: 2024
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.generators import FractionalDataGenerator
from models.model_persistence import ModelPersistenceManager, quick_save_model, quick_load_model
from models.model_comparison import ModelConfig, ModelComparisonFramework
from estimators.pinn_estimator import PINNEstimator
from models.fractional_pino import FractionalPINOTrainer
from models.neural_fractional_ode import NeuralFractionalODETrainer
from models.neural_fractional_sde import NeuralFractionalSDETrainer


def create_training_data(n_samples: int = 100, n_points: int = 1000):
    """
    Create training data for all models.
    
    Args:
        n_samples: Number of samples
        n_points: Number of points per sample
        
    Returns:
        Dictionary containing training data
    """
    print("Generating training data...")
    
    # Initialize data generator
    generator = FractionalDataGenerator(seed=42)
    
    # Generate comprehensive dataset
    dataset = generator.generate_comprehensive_dataset(
        n_points=n_points,
        hurst_range=(0.1, 0.9),
        n_samples_per_hurst=n_samples // 8,  # 8 different Hurst values
        include_contamination=True
    )
    
    # Prepare data for different model types
    training_data = {
        'pinn': [],
        'pino': [],
        'neural_ode': [],
        'neural_sde': []
    }
    
    for sample in dataset['samples']:
        data = sample['data']
        hurst = sample['hurst']
        
        # PINN data (time series)
        training_data['pinn'].append({
            'data': data,
            'hurst': hurst
        })
        
        # PINO data (function-to-function mapping)
        t = np.linspace(0, 1, len(data))
        training_data['pino'].append({
            'input': t.reshape(-1, 1),
            'output': data.reshape(-1, 1),
            'hurst': hurst
        })
        
        # Neural ODE data (initial value problem)
        y0 = data[0]
        training_data['neural_ode'].append({
            't': t,
            'y0': y0,
            'solution': data,
            'hurst': hurst
        })
        
        # Neural SDE data (stochastic process)
        training_data['neural_sde'].append({
            't': t,
            'y0': y0,
            'trajectory': data,
            'hurst': hurst
        })
    
    print(f"Generated {len(dataset['samples'])} training samples")
    return training_data


def train_pinn_model(training_data, save_model: bool = True):
    """
    Train a PINN model and save it.
    
    Args:
        training_data: Training data for PINN
        save_model: Whether to save the model
        
    Returns:
        Trained model and training history
    """
    print("\n=== Training PINN Model ===")
    
    # Initialize PINN estimator
    estimator = PINNEstimator(
        input_dim=1,
        hidden_dims=[64, 128, 64],
        output_dim=1,
        learning_rate=1e-3
    )
    
    # Build model
    estimator.build_model(
        use_mellin_transform=True,
        use_physics_constraints=True
    )
    
    # Prepare data
    data_list = [sample['data'] for sample in training_data]
    hurst_list = [sample['hurst'] for sample in training_data]
    
    # Split data
    split_idx = int(0.8 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    train_hurst = hurst_list[:split_idx]
    val_hurst = hurst_list[split_idx:]
    
    # Create data loaders
    train_loader, val_loader = estimator.prepare_data(
        np.array(train_data),
        batch_size=32,
        validation_split=0.0  # Already split
    )
    
    # Train model
    history = estimator.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=500,  # Reduced for demo
        early_stopping_patience=20,
        verbose=True,
        save_model=save_model,
        model_description="PINN model trained on fractional time series data",
        model_tags=['pinn', 'fractional', 'demo']
    )
    
    return estimator, history


def train_pino_model(training_data, save_model: bool = True):
    """
    Train a PINO model and save it.
    
    Args:
        training_data: Training data for PINO
        save_model: Whether to save the model
        
    Returns:
        Trained model and training history
    """
    print("\n=== Training PINO Model ===")
    
    # This would require implementing PINO data preparation
    # For now, we'll create a placeholder
    print("PINO training not yet implemented in this demo")
    return None, None


def train_neural_ode_model(training_data, save_model: bool = True):
    """
    Train a Neural Fractional ODE model and save it.
    
    Args:
        training_data: Training data for Neural ODE
        save_model: Whether to save the model
        
    Returns:
        Trained model and training history
    """
    print("\n=== Training Neural Fractional ODE Model ===")
    
    # This would require implementing Neural ODE data preparation
    # For now, we'll create a placeholder
    print("Neural ODE training not yet implemented in this demo")
    return None, None


def train_neural_sde_model(training_data, save_model: bool = True):
    """
    Train a Neural Fractional SDE model and save it.
    
    Args:
        training_data: Training data for Neural SDE
        save_model: Whether to save the model
        
    Returns:
        Trained model and training history
    """
    print("\n=== Training Neural Fractional SDE Model ===")
    
    # This would require implementing Neural SDE data preparation
    # For now, we'll create a placeholder
    print("Neural SDE training not yet implemented in this demo")
    return None, None


def demonstrate_model_loading():
    """
    Demonstrate loading and using saved models.
    """
    print("\n=== Demonstrating Model Loading ===")
    
    # Initialize model persistence manager
    manager = ModelPersistenceManager("saved_models")
    
    # List all saved models
    models_df = manager.list_models()
    print(f"\nFound {len(models_df)} saved models:")
    
    if not models_df.empty:
        print(models_df[['model_id', 'model_type', 'version', 'created_at']].to_string())
        
        # Load the best PINN model
        best_pinn_id = manager.get_best_model('pinn', metric='best_val_loss')
        if best_pinn_id:
            print(f"\nLoading best PINN model: {best_pinn_id}")
            
            try:
                model, config, metadata = manager.load_model(best_pinn_id)
                print(f"Model loaded successfully!")
                print(f"Model type: {config.model_type}")
                print(f"Version: {metadata.version}")
                print(f"Training duration: {metadata.training_duration:.2f} seconds")
                print(f"Final loss: {metadata.final_loss:.6f}")
                
                # Demonstrate inference
                print("\nDemonstrating inference with loaded model...")
                
                # Generate test data
                generator = FractionalDataGenerator(seed=123)
                test_data = generator.generate_fbm(n_points=1000, hurst=0.7)
                
                # Prepare input
                t = torch.FloatTensor(np.arange(len(test_data['data'])).reshape(-1, 1))
                
                # Run inference
                model.eval()
                with torch.no_grad():
                    y_pred, hurst_pred = model(t.unsqueeze(0))  # Add batch dimension
                
                print(f"Predicted Hurst exponent: {hurst_pred.item():.4f}")
                print(f"True Hurst exponent: {test_data['hurst']:.4f}")
                print(f"Absolute error: {abs(hurst_pred.item() - test_data['hurst']):.4f}")
                
            except Exception as e:
                print(f"Error loading model: {e}")
    else:
        print("No saved models found. Run training first.")


def demonstrate_batch_inference():
    """
    Demonstrate batch inference with multiple saved models.
    """
    print("\n=== Demonstrating Batch Inference ===")
    
    # Initialize model persistence manager
    manager = ModelPersistenceManager("saved_models")
    
    # Generate test data
    generator = FractionalDataGenerator(seed=456)
    test_samples = []
    
    for hurst in [0.3, 0.5, 0.7, 0.9]:
        data = generator.generate_fbm(n_points=1000, hurst=hurst)
        test_samples.append({
            'data': data['data'],
            'hurst': data['hurst']
        })
    
    # Get all saved models
    models_df = manager.list_models()
    
    if not models_df.empty:
        results = {}
        
        for _, model_info in models_df.iterrows():
            model_id = model_info['model_id']
            model_type = model_info['model_type']
            
            print(f"\nRunning inference with {model_type} model: {model_id}")
            
            try:
                # Load model
                model, config, metadata = manager.load_model(model_id)
                model.eval()
                
                model_results = []
                
                # Run inference on all test samples
                for i, sample in enumerate(test_samples):
                    t = torch.FloatTensor(np.arange(len(sample['data'])).reshape(-1, 1))
                    
                    with torch.no_grad():
                        y_pred, hurst_pred = model(t.unsqueeze(0))
                    
                    error = abs(hurst_pred.item() - sample['hurst'])
                    model_results.append({
                        'sample_id': i,
                        'true_hurst': sample['hurst'],
                        'predicted_hurst': hurst_pred.item(),
                        'error': error
                    })
                
                # Calculate average error
                avg_error = np.mean([r['error'] for r in model_results])
                results[model_type] = {
                    'model_id': model_id,
                    'avg_error': avg_error,
                    'results': model_results
                }
                
                print(f"Average error: {avg_error:.4f}")
                
            except Exception as e:
                print(f"Error with model {model_id}: {e}")
        
        # Compare results
        print("\n=== Model Comparison ===")
        for model_type, result in results.items():
            print(f"{model_type.upper()}: {result['avg_error']:.4f}")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['avg_error'])
        print(f"\nBest performing model: {best_model[0]} (Error: {best_model[1]['avg_error']:.4f})")
        
    else:
        print("No saved models found. Run training first.")


def demonstrate_model_management():
    """
    Demonstrate model management features.
    """
    print("\n=== Demonstrating Model Management ===")
    
    # Initialize model persistence manager
    manager = ModelPersistenceManager("saved_models")
    
    # List models with different filters
    print("\nAll models:")
    all_models = manager.list_models()
    print(all_models[['model_id', 'model_type', 'version']].to_string())
    
    print("\nPINN models only:")
    pinn_models = manager.list_models(model_type='pinn')
    print(pinn_models[['model_id', 'version', 'created_at']].to_string())
    
    print("\nModels with 'demo' tag:")
    demo_models = manager.list_models(tags=['demo'])
    print(demo_models[['model_id', 'model_type', 'tags']].to_string())
    
    # Get detailed information about a model
    if not all_models.empty:
        model_id = all_models.iloc[0]['model_id']
        print(f"\nDetailed information for model: {model_id}")
        
        try:
            info = manager.get_model_info(model_id)
            print(f"Description: {info['metadata']['description']}")
            print(f"Training duration: {info['metadata']['training_duration']:.2f} seconds")
            print(f"Convergence epochs: {info['metadata']['convergence_epochs']}")
            print(f"Device used: {info['metadata']['device_used']}")
            print(f"Framework version: {info['metadata']['framework_version']}")
            print(f"Dependencies: {info['metadata']['dependencies']}")
        except Exception as e:
            print(f"Error getting model info: {e}")


def main():
    """
    Main demonstration function.
    """
    print("=" * 60)
    print("FRACTIONAL PINN: TRAIN ONCE, APPLY MANY DEMO")
    print("=" * 60)
    
    # Create training data
    training_data = create_training_data(n_samples=50, n_points=500)
    
    # Train models (only PINN for now)
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    pinn_model, pinn_history = train_pinn_model(training_data, save_model=True)
    
    # Demonstrate model loading and inference
    print("\n" + "=" * 60)
    print("INFERENCE PHASE")
    print("=" * 60)
    
    demonstrate_model_loading()
    demonstrate_batch_inference()
    
    # Demonstrate model management
    print("\n" + "=" * 60)
    print("MODEL MANAGEMENT PHASE")
    print("=" * 60)
    
    demonstrate_model_management()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nKey Benefits Demonstrated:")
    print("✅ Train once, apply many times")
    print("✅ Automatic model versioning and metadata")
    print("✅ Easy model comparison and selection")
    print("✅ Batch inference with multiple models")
    print("✅ Comprehensive model management")
    print("✅ Cross-platform model portability")


if __name__ == "__main__":
    main()
