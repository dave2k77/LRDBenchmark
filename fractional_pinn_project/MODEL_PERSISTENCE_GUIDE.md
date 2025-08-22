# Model Persistence System Guide

## Overview

The Fractional PINN project includes a comprehensive model persistence system that enables **"train once, apply many"** functionality. This system allows you to save trained models with all their components and metadata, then load and use them for inference without retraining.

## Key Features

### 🎯 **Core Functionality**
- **Complete Model Saving**: Save models with weights, configurations, training history, and metadata
- **Automatic Versioning**: Each saved model gets a unique version and timestamp
- **Model Registry**: Centralized tracking of all saved models
- **Cross-Platform Compatibility**: Models can be exported and imported across different systems
- **Fallback Support**: Graceful degradation if persistence system is unavailable

### 📊 **Model Management**
- **Model Discovery**: List and search saved models by type, tags, or performance metrics
- **Best Model Selection**: Automatically find the best performing model based on various metrics
- **Detailed Metadata**: Track training duration, convergence, device used, dependencies
- **Model Comparison**: Compare performance across different models and versions

### 🔄 **Advanced Features**
- **Model Export/Import**: Portable model format for sharing and deployment
- **Model Deletion**: Clean up old or unused models
- **Training History**: Preserve complete training curves and loss components
- **Configuration Persistence**: Save and restore exact model configurations

## Architecture

### Core Components

```
fractional_pinn_project/src/models/
├── model_persistence.py          # Main persistence system
├── model_comparison.py           # Model comparison framework
├── fractional_pinn.py            # PINN model (updated with persistence)
├── fractional_pino.py            # PINO model (updated with persistence)
├── neural_fractional_ode.py      # Neural ODE model (updated with persistence)
└── neural_fractional_sde.py      # Neural SDE model (updated with persistence)
```

### Directory Structure

```
saved_models/
├── model_registry.json           # Central registry of all models
├── checkpoints/                  # Model state dictionaries
│   ├── pinn_20241201_143022.pth
│   ├── pino_20241201_143045.pth
│   └── ...
├── metadata/                     # Model metadata
│   ├── pinn_20241201_143022.json
│   ├── pino_20241201_143045.json
│   └── ...
├── configs/                      # Model configurations
│   ├── pinn_20241201_143022.json
│   ├── pino_20241201_143045.json
│   └── ...
└── history/                      # Training histories
    ├── pinn_20241201_143022.json
    ├── pino_20241201_143045.json
    └── ...
```

## Usage Guide

### 1. Training with Automatic Saving

All neural models now support automatic saving during training:

```python
from estimators.pinn_estimator import PINNEstimator

# Initialize estimator
estimator = PINNEstimator()
estimator.build_model()

# Train with automatic saving
history = estimator.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1000,
    save_model=True,  # Enable automatic saving
    model_description="PINN model for fractional time series analysis",
    model_tags=['pinn', 'fractional', 'production']
)
```

### 2. Loading Saved Models

```python
from models.model_persistence import ModelPersistenceManager

# Initialize manager
manager = ModelPersistenceManager("saved_models")

# List all saved models
models_df = manager.list_models()
print(models_df)

# Load a specific model
model, config, metadata = manager.load_model("pinn_20241201_143022")

# Use the loaded model for inference
model.eval()
with torch.no_grad():
    prediction = model(input_data)
```

### 3. Finding the Best Model

```python
# Get the best PINN model based on validation loss
best_pinn_id = manager.get_best_model('pinn', metric='best_val_loss')

# Get the best model of any type based on training duration
fastest_model_id = manager.get_best_model(metric='training_duration')

# Load and use the best model
model, config, metadata = manager.load_model(best_pinn_id)
```

### 4. Batch Inference with Multiple Models

```python
# Get all saved models
models_df = manager.list_models()

results = {}
for _, model_info in models_df.iterrows():
    model_id = model_info['model_id']
    model_type = model_info['model_type']
    
    # Load model
    model, config, metadata = manager.load_model(model_id)
    
    # Run inference
    predictions = []
    for test_sample in test_data:
        pred = model(test_sample)
        predictions.append(pred)
    
    # Store results
    results[model_type] = {
        'model_id': model_id,
        'predictions': predictions,
        'avg_error': calculate_error(predictions, true_values)
    }

# Compare results
for model_type, result in results.items():
    print(f"{model_type}: {result['avg_error']:.4f}")
```

### 5. Model Management

```python
# List models with filters
pinn_models = manager.list_models(model_type='pinn')
demo_models = manager.list_models(tags=['demo'])
recent_models = manager.list_models().head(5)

# Get detailed information
info = manager.get_model_info("pinn_20241201_143022")
print(f"Description: {info['metadata']['description']}")
print(f"Training duration: {info['metadata']['training_duration']:.2f} seconds")
print(f"Final loss: {info['metadata']['final_loss']:.6f}")

# Delete old models
manager.delete_model("old_model_id")

# Export model for sharing
manager.export_model("model_id", "exported_model.zip")

# Import model from another system
new_model_id = manager.import_model("imported_model.zip")
```

## Integration with Training Pipelines

### PINN Estimator

```python
from estimators.pinn_estimator import PINNEstimator

estimator = PINNEstimator()
estimator.build_model()

# Training with persistence
history = estimator.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1000,
    save_model=True,
    model_description="Production PINN model",
    model_tags=['production', 'pinn']
)
```

### PINO Trainer

```python
from models.fractional_pino import FractionalPINOTrainer

trainer = FractionalPINOTrainer(model)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1000,
    save_model=True,
    model_description="Production PINO model",
    model_tags=['production', 'pino']
)
```

### Neural Fractional ODE Trainer

```python
from models.neural_fractional_ode import NeuralFractionalODETrainer

trainer = NeuralFractionalODETrainer(model)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1000,
    save_model=True,
    model_description="Production Neural ODE model",
    model_tags=['production', 'neural_ode']
)
```

### Neural Fractional SDE Trainer

```python
from models.neural_fractional_sde import NeuralFractionalSDETrainer

trainer = NeuralFractionalSDETrainer(model)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=1000,
    save_model=True,
    model_description="Production Neural SDE model",
    model_tags=['production', 'neural_sde']
)
```

## Best Practices

### 1. **Descriptive Model Names and Tags**
```python
# Good
model_description="PINN model trained on financial time series with 0.1-0.9 Hurst range"
model_tags=['pinn', 'financial', 'production', 'v2.0']

# Avoid
model_description="model"
model_tags=['test']
```

### 2. **Regular Model Cleanup**
```python
# List old models
old_models = manager.list_models().tail(10)

# Delete models older than 30 days
for _, model_info in old_models.iterrows():
    created_date = pd.to_datetime(model_info['created_at'])
    if (datetime.now() - created_date).days > 30:
        manager.delete_model(model_info['model_id'])
```

### 3. **Model Versioning Strategy**
```python
# Use semantic versioning for production models
model_tags=['production', 'v1.2.0', 'stable']

# Use descriptive tags for experimental models
model_tags=['experimental', 'mellin_transform', 'new_architecture']
```

### 4. **Performance Tracking**
```python
# Track model performance over time
best_models = []
for model_type in ['pinn', 'pino', 'neural_ode', 'neural_sde']:
    best_id = manager.get_best_model(model_type, metric='best_val_loss')
    if best_id:
        info = manager.get_model_info(best_id)
        best_models.append({
            'type': model_type,
            'id': best_id,
            'loss': info['metadata']['best_val_loss'],
            'date': info['metadata']['created_at']
        })
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```python
   # Check if model exists
   models_df = manager.list_models()
   if model_id not in models_df['model_id'].values:
       print(f"Model {model_id} not found")
   ```

2. **Import Errors**
   ```python
   # The system includes fallback support
   # If persistence system fails, models are saved with simple torch.save()
   ```

3. **Device Mismatch**
   ```python
   # Load model with specific device
   model, config, metadata = manager.load_model(model_id, device='cuda')
   ```

4. **Memory Issues**
   ```python
   # Load model to CPU first, then move to GPU if needed
   model, config, metadata = manager.load_model(model_id, device='cpu')
   model = model.to('cuda')
   ```

## Performance Considerations

### Storage Optimization
- Models are automatically compressed using PyTorch's efficient serialization
- Training histories are stored separately to reduce checkpoint size
- Metadata is stored in human-readable JSON format

### Loading Speed
- Model state dictionaries are loaded directly without reconstruction
- Configurations are cached for faster subsequent loads
- Registry is kept in memory for quick model discovery

### Memory Usage
- Models are loaded on-demand to minimize memory footprint
- Training histories can be loaded separately if not needed
- Large models can be loaded to CPU first, then moved to GPU

## Future Enhancements

### Planned Features
1. **Model Compression**: Automatic model quantization and pruning
2. **Distributed Storage**: Support for cloud storage backends
3. **Model Serving**: Integration with model serving frameworks
4. **A/B Testing**: Built-in support for model comparison experiments
5. **Automated Cleanup**: Intelligent model lifecycle management

### Extensibility
The persistence system is designed to be easily extensible:
- New model types can be added by implementing the required interfaces
- Custom metadata fields can be added to track domain-specific information
- Export/import formats can be customized for different deployment scenarios

## Conclusion

The model persistence system provides a robust foundation for the "train once, apply many" paradigm in the Fractional PINN project. It enables efficient model management, easy deployment, and comprehensive tracking of model performance across different architectures and training runs.

By integrating this system into your training pipelines, you can:
- Save significant computational resources by avoiding retraining
- Maintain a clear history of model development
- Easily compare and select the best performing models
- Deploy models consistently across different environments
- Share and collaborate on trained models with team members

The system is designed to be production-ready while maintaining ease of use for research and development workflows.
