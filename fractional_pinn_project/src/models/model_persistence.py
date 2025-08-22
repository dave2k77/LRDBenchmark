"""
Model Persistence System

This module provides comprehensive model saving and loading functionality
for the "train once, apply many" paradigm. It handles saving and loading
of trained models, configurations, training history, and metadata.

Key Features:
1. Save/load trained models with all components
2. Configuration persistence
3. Training history and metadata storage
4. Model versioning and checkpointing
5. Automatic model discovery and loading
6. Cross-platform compatibility

Author: Fractional PINN Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import pickle
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Type
import warnings
from pathlib import Path
import hashlib
import zipfile
from dataclasses import dataclass, asdict
import logging

# Import our custom modules
from .model_comparison import ModelConfig, EvaluationMetrics
from .fractional_pinn import FractionalPINN
from .fractional_pino import FractionalPINO
from .neural_fractional_ode import NeuralFractionalODE
from .neural_fractional_sde import NeuralFractionalSDE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for saved models."""
    model_name: str
    model_type: str
    version: str
    created_at: str
    training_duration: float
    final_loss: float
    best_val_loss: float
    convergence_epochs: int
    device_used: str
    framework_version: str
    dependencies: Dict[str, str]
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelCheckpoint:
    """Complete model checkpoint."""
    config: ModelConfig
    metadata: ModelMetadata
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, List[float]]] = None
    evaluation_metrics: Optional[EvaluationMetrics] = None
    model_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': asdict(self.config),
            'metadata': self.metadata.to_dict(),
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'scheduler_state': self.scheduler_state,
            'training_history': self.training_history,
            'evaluation_metrics': self.evaluation_metrics.to_dict() if self.evaluation_metrics else None,
            'model_hash': self.model_hash
        }


class ModelPersistenceManager:
    """
    Manager for saving and loading trained models.
    
    This class handles the complete lifecycle of model persistence,
    including saving, loading, versioning, and metadata management.
    """
    
    def __init__(self, base_dir: str = "saved_models"):
        """
        Initialize the model persistence manager.
        
        Args:
            base_dir: Base directory for saved models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_dir / "checkpoints").mkdir(exist_ok=True)
        (self.base_dir / "metadata").mkdir(exist_ok=True)
        (self.base_dir / "configs").mkdir(exist_ok=True)
        (self.base_dir / "history").mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry = self._load_model_registry()
    
    def _load_model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the model registry from disk."""
        registry_path = self.base_dir / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_model_registry(self) -> None:
        """Save the model registry to disk."""
        registry_path = self.base_dir / "model_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def _generate_model_hash(self, model_state: Dict[str, torch.Tensor]) -> str:
        """Generate a hash for the model state."""
        # Convert model state to bytes for hashing
        state_bytes = pickle.dumps(model_state)
        return hashlib.md5(state_bytes).hexdigest()
    
    def _get_framework_version(self) -> str:
        """Get the current framework version."""
        return "1.0.0"  # You can implement version detection here
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current dependency versions."""
        return {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__
        }
    
    def save_model(self,
                  model: nn.Module,
                  config: ModelConfig,
                  training_history: Optional[Dict[str, List[float]]] = None,
                  evaluation_metrics: Optional[EvaluationMetrics] = None,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                  training_duration: float = 0.0,
                  description: str = "",
                  tags: List[str] = None,
                  version: str = None) -> str:
        """
        Save a trained model with all its components.
        
        Args:
            model: Trained model
            config: Model configuration
            training_history: Training history
            evaluation_metrics: Evaluation metrics
            optimizer: Optimizer state
            scheduler: Scheduler state
            training_duration: Training duration in seconds
            description: Model description
            tags: Model tags
            version: Model version (auto-generated if None)
            
        Returns:
            Model ID
        """
        # Generate model ID and version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{config.model_type}_{timestamp}"
        if version is None:
            version = f"v1.0.0_{timestamp}"
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_id,
            model_type=config.model_type,
            version=version,
            created_at=datetime.now().isoformat(),
            training_duration=training_duration,
            final_loss=training_history.get('total_loss', [0])[-1] if training_history else 0.0,
            best_val_loss=training_history.get('val_total_loss', [float('inf')])[-1] if training_history else float('inf'),
            convergence_epochs=len(training_history.get('total_loss', [])) if training_history else 0,
            device_used=str(next(model.parameters()).device),
            framework_version=self._get_framework_version(),
            dependencies=self._get_dependencies(),
            description=description,
            tags=tags or []
        )
        
        # Get model state
        model_state = model.state_dict()
        model_hash = self._generate_model_hash(model_state)
        
        # Get optimizer and scheduler states
        optimizer_state = optimizer.state_dict() if optimizer else None
        scheduler_state = scheduler.state_dict() if scheduler else None
        
        # Create checkpoint
        checkpoint = ModelCheckpoint(
            config=config,
            metadata=metadata,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            training_history=training_history,
            evaluation_metrics=evaluation_metrics,
            model_hash=model_hash
        )
        
        # Save checkpoint
        checkpoint_path = self.base_dir / "checkpoints" / f"{model_id}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata separately
        metadata_path = self.base_dir / "metadata" / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save config separately
        config_path = self.base_dir / "configs" / f"{model_id}.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Save training history separately
        if training_history:
            history_path = self.base_dir / "history" / f"{model_id}.json"
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
        
        # Update registry
        self.model_registry[model_id] = {
            'model_type': config.model_type,
            'version': version,
            'created_at': metadata.created_at,
            'checkpoint_path': str(checkpoint_path),
            'metadata_path': str(metadata_path),
            'config_path': str(config_path),
            'history_path': str(history_path) if training_history else None,
            'model_hash': model_hash,
            'description': description,
            'tags': tags or []
        }
        
        self._save_model_registry()
        
        logger.info(f"Model saved successfully: {model_id}")
        return model_id
    
    def load_model(self, model_id: str, device: str = 'auto') -> Tuple[nn.Module, ModelConfig, ModelMetadata]:
        """
        Load a saved model.
        
        Args:
            model_id: Model ID to load
            device: Device to load model on
            
        Returns:
            Tuple of (model, config, metadata)
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        registry_entry = self.model_registry[model_id]
        checkpoint_path = Path(registry_entry['checkpoint_path'])
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Set device
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # Create model based on type
        model = self._create_model_from_config(checkpoint.config)
        model.load_state_dict(checkpoint.model_state)
        model = model.to(device)
        
        logger.info(f"Model loaded successfully: {model_id}")
        return model, checkpoint.config, checkpoint.metadata
    
    def _create_model_from_config(self, config: ModelConfig) -> nn.Module:
        """Create a model instance from configuration."""
        if config.model_type == 'pinn':
            from .fractional_pinn import create_fractional_pinn
            return create_fractional_pinn(
                input_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                output_dim=config.output_dim,
                use_mellin_transform=config.use_mellin_transform,
                use_physics_constraints=config.use_physics_constraints
            )
        elif config.model_type == 'pino':
            from .fractional_pino import create_fractional_pino
            return create_fractional_pino(
                input_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                output_dim=config.output_dim,
                modes=config.modes,
                use_mellin_transform=config.use_mellin_transform,
                use_physics_constraints=config.use_physics_constraints
            )
        elif config.model_type == 'neural_ode':
            from .neural_fractional_ode import create_neural_fractional_ode
            return create_neural_fractional_ode(
                input_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                output_dim=config.output_dim,
                alpha=config.alpha,
                use_mellin_transform=config.use_mellin_transform,
                use_physics_constraints=config.use_physics_constraints
            )
        elif config.model_type == 'neural_sde':
            from .neural_fractional_sde import create_neural_fractional_sde
            return create_neural_fractional_sde(
                input_dim=config.input_dim,
                hidden_dims=config.hidden_dims,
                output_dim=config.output_dim,
                hurst=config.hurst,
                fbm_method=config.fbm_method,
                use_mellin_transform=config.use_mellin_transform,
                use_physics_constraints=config.use_physics_constraints
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def list_models(self, model_type: Optional[str] = None, tags: Optional[List[str]] = None) -> pd.DataFrame:
        """
        List all saved models.
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags
            
        Returns:
            DataFrame with model information
        """
        models = []
        
        for model_id, info in self.model_registry.items():
            if model_type and info['model_type'] != model_type:
                continue
            
            if tags and not any(tag in info['tags'] for tag in tags):
                continue
            
            models.append({
                'model_id': model_id,
                'model_type': info['model_type'],
                'version': info['version'],
                'created_at': info['created_at'],
                'description': info['description'],
                'tags': ', '.join(info['tags']),
                'model_hash': info['model_hash']
            })
        
        df = pd.DataFrame(models)
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('created_at', ascending=False)
        
        return df
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary with model information
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        registry_entry = self.model_registry[model_id]
        
        # Load metadata
        metadata_path = Path(registry_entry['metadata_path'])
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load config
        config_path = Path(registry_entry['config_path'])
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load training history if available
        training_history = None
        if registry_entry['history_path']:
            history_path = Path(registry_entry['history_path'])
            if history_path.exists():
                with open(history_path, 'r') as f:
                    training_history = json.load(f)
        
        return {
            'model_id': model_id,
            'metadata': metadata,
            'config': config,
            'training_history': training_history,
            'registry_entry': registry_entry
        }
    
    def delete_model(self, model_id: str) -> None:
        """
        Delete a saved model.
        
        Args:
            model_id: Model ID to delete
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        registry_entry = self.model_registry[model_id]
        
        # Delete files
        files_to_delete = [
            registry_entry['checkpoint_path'],
            registry_entry['metadata_path'],
            registry_entry['config_path']
        ]
        
        if registry_entry['history_path']:
            files_to_delete.append(registry_entry['history_path'])
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove from registry
        del self.model_registry[model_id]
        self._save_model_registry()
        
        logger.info(f"Model deleted successfully: {model_id}")
    
    def export_model(self, model_id: str, export_path: str) -> None:
        """
        Export a model to a portable format.
        
        Args:
            model_id: Model ID to export
            export_path: Path for the exported file
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        registry_entry = self.model_registry[model_id]
        
        # Create temporary directory for export
        temp_dir = Path(f"temp_export_{model_id}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy all model files
            files_to_copy = [
                registry_entry['checkpoint_path'],
                registry_entry['metadata_path'],
                registry_entry['config_path']
            ]
            
            if registry_entry['history_path']:
                files_to_copy.append(registry_entry['history_path'])
            
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, temp_dir)
            
            # Create export manifest
            manifest = {
                'model_id': model_id,
                'export_date': datetime.now().isoformat(),
                'files': [Path(f).name for f in files_to_copy if os.path.exists(f)]
            }
            
            with open(temp_dir / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create zip file
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(temp_dir))
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
        
        logger.info(f"Model exported successfully to: {export_path}")
    
    def import_model(self, import_path: str) -> str:
        """
        Import a model from a portable format.
        
        Args:
            import_path: Path to the imported file
            
        Returns:
            Model ID of the imported model
        """
        # Create temporary directory for import
        temp_dir = Path(f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract zip file
            with zipfile.ZipFile(import_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Read manifest
            manifest_path = temp_dir / 'manifest.json'
            if not manifest_path.exists():
                raise ValueError("Invalid export file: missing manifest.json")
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Load checkpoint to get model info
            checkpoint_files = [f for f in manifest['files'] if f.endswith('.pth')]
            if not checkpoint_files:
                raise ValueError("Invalid export file: missing checkpoint file")
            
            checkpoint_path = temp_dir / checkpoint_files[0]
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Generate new model ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_id = f"{checkpoint.config.model_type}_imported_{timestamp}"
            
            # Copy files to appropriate locations
            for file_name in manifest['files']:
                source_path = temp_dir / file_name
                if file_name.endswith('.pth'):
                    dest_path = self.base_dir / "checkpoints" / f"{new_model_id}.pth"
                elif file_name.endswith('_metadata.json'):
                    dest_path = self.base_dir / "metadata" / f"{new_model_id}.json"
                elif file_name.endswith('_config.json'):
                    dest_path = self.base_dir / "configs" / f"{new_model_id}.json"
                elif file_name.endswith('_history.json'):
                    dest_path = self.base_dir / "history" / f"{new_model_id}.json"
                else:
                    continue
                
                shutil.copy2(source_path, dest_path)
            
            # Update registry
            self.model_registry[new_model_id] = {
                'model_type': checkpoint.config.model_type,
                'version': checkpoint.metadata.version,
                'created_at': datetime.now().isoformat(),
                'checkpoint_path': str(self.base_dir / "checkpoints" / f"{new_model_id}.pth"),
                'metadata_path': str(self.base_dir / "metadata" / f"{new_model_id}.json"),
                'config_path': str(self.base_dir / "configs" / f"{new_model_id}.json"),
                'history_path': str(self.base_dir / "history" / f"{new_model_id}.json") if checkpoint.training_history else None,
                'model_hash': checkpoint.model_hash,
                'description': f"Imported from {import_path}",
                'tags': ['imported']
            }
            
            self._save_model_registry()
            
            logger.info(f"Model imported successfully: {new_model_id}")
            return new_model_id
        
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    
    def get_best_model(self, model_type: str, metric: str = 'best_val_loss') -> Optional[str]:
        """
        Get the best model of a specific type based on a metric.
        
        Args:
            model_type: Type of model to search for
            metric: Metric to optimize ('best_val_loss', 'final_loss', 'training_duration')
            
        Returns:
            Model ID of the best model, or None if no models found
        """
        models = self.list_models(model_type=model_type)
        
        if models.empty:
            return None
        
        # Get detailed info for all models
        model_scores = []
        for model_id in models['model_id']:
            info = self.get_model_info(model_id)
            metadata = info['metadata']
            
            if metric == 'best_val_loss':
                score = metadata['best_val_loss']
            elif metric == 'final_loss':
                score = metadata['final_loss']
            elif metric == 'training_duration':
                score = metadata['training_duration']
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            model_scores.append((model_id, score))
        
        # Sort by score (lower is better for losses, higher is better for duration)
        reverse = (metric == 'training_duration')
        model_scores.sort(key=lambda x: x[1], reverse=reverse)
        
        return model_scores[0][0] if model_scores else None


# Convenience functions
def get_model_manager(base_dir: str = "saved_models") -> ModelPersistenceManager:
    """
    Get a model persistence manager instance.
    
    Args:
        base_dir: Base directory for saved models
        
    Returns:
        Model persistence manager
    """
    return ModelPersistenceManager(base_dir)


def quick_save_model(model: nn.Module,
                    config: ModelConfig,
                    training_history: Optional[Dict[str, List[float]]] = None,
                    description: str = "",
                    tags: List[str] = None) -> str:
    """
    Quick save a model with minimal parameters.
    
    Args:
        model: Trained model
        config: Model configuration
        training_history: Training history
        description: Model description
        tags: Model tags
        
    Returns:
        Model ID
    """
    manager = get_model_manager()
    return manager.save_model(
        model=model,
        config=config,
        training_history=training_history,
        description=description,
        tags=tags
    )


def quick_load_model(model_id: str, device: str = 'auto') -> Tuple[nn.Module, ModelConfig, ModelMetadata]:
    """
    Quick load a model.
    
    Args:
        model_id: Model ID to load
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, metadata)
    """
    manager = get_model_manager()
    return manager.load_model(model_id, device)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Model Persistence System...")
    
    # Create manager
    manager = get_model_manager("test_models")
    
    # Test listing models
    models_df = manager.list_models()
    print(f"Found {len(models_df)} saved models")
    
    if not models_df.empty:
        print("\nAvailable models:")
        print(models_df[['model_id', 'model_type', 'version', 'created_at']].to_string())
    
    print("Model persistence system test completed successfully!")
