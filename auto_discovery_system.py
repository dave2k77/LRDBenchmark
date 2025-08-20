#!/usr/bin/env python3
"""
Auto-Discovery System for DataExploratoryProject

This system automatically discovers new estimators and data generators,
then integrates them into benchmarking and example scripts.
"""

import os
import sys
import importlib
import inspect
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoDiscoverySystem:
    """Automated system for discovering and integrating new components."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.discovered_components = {
            'data_generators': {},
            'estimators': {},
            'neural_components': {}
        }
        self.component_registry_file = self.project_root / "component_registry.json"
        self.last_discovery_file = self.project_root / ".last_discovery"
        
    def discover_components(self) -> Dict[str, Any]:
        """Discover all available components in the project."""
        logger.info("Starting component discovery...")
        
        # Discover data generators
        self.discovered_components['data_generators'] = self._discover_data_generators()
        
        # Discover estimators
        self.discovered_components['estimators'] = self._discover_estimators()
        
        # Discover neural components
        self.discovered_components['neural_components'] = self._discover_neural_components()
        
        # Save discovery results
        self._save_discovery_results()
        
        logger.info(f"Discovery complete: {len(self.discovered_components['data_generators'])} data generators, "
                   f"{len(self.discovered_components['estimators'])} estimators, "
                   f"{len(self.discovered_components['neural_components'])} neural components")
        
        return self.discovered_components
    
    def _discover_data_generators(self) -> Dict[str, Any]:
        """Discover all data generator classes."""
        generators = {}
        
        # Look for data generator modules
        data_model_paths = [
            self.project_root / "models" / "data_models",
            self.project_root / "models" / "generators"
        ]
        
        for base_path in data_model_paths:
            if not base_path.exists():
                continue
                
            for model_dir in base_path.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('_'):
                    # Look for model files
                    for model_file in model_dir.glob("*_model.py"):
                        try:
                            module_path = model_file.relative_to(self.project_root)
                            module_name = str(module_path).replace(os.sep, '.').replace('.py', '')
                            
                            # Import module
                            module = importlib.import_module(module_name)
                            
                            # Find classes that inherit from BaseModel or have generate method
                            for name, obj in inspect.getmembers(module, inspect.isclass):
                                if (hasattr(obj, 'generate') or 
                                    (hasattr(obj, '__bases__') and 
                                     any('BaseModel' in str(base) for base in obj.__bases__))):
                                    
                                    # Get constructor parameters
                                    params = self._extract_constructor_params(obj)
                                    
                                    generators[name] = {
                                        'class_name': name,
                                        'module_name': module_name,
                                        'file_path': str(model_file),
                                        'constructor_params': params,
                                        'type': 'stochastic' if 'neural' not in module_name.lower() else 'neural',
                                        'description': self._extract_class_docstring(obj)
                                    }
                                    
                        except Exception as e:
                            logger.warning(f"Failed to analyze {model_file}: {e}")
        
        return generators
    
    def _discover_estimators(self) -> Dict[str, Any]:
        """Discover all estimator classes."""
        estimators = {}
        
        # Look for estimator modules
        estimator_paths = [
            self.project_root / "analysis",
            self.project_root / "models" / "estimators"
        ]
        
        for base_path in estimator_paths:
            if not base_path.exists():
                continue
                
            # Recursively search for estimator files
            for estimator_file in base_path.rglob("*_estimator.py"):
                try:
                    module_path = estimator_file.relative_to(self.project_root)
                    module_name = str(module_path).replace(os.sep, '.').replace('.py', '')
                    
                    # Import module
                    module = importlib.import_module(module_name)
                    
                    # Find classes that have estimate method
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, 'estimate'):
                            # Determine category from file path
                            category = self._determine_estimator_category(estimator_file)
                            
                            # Get constructor parameters
                            params = self._extract_constructor_params(obj)
                            
                            estimators[name] = {
                                'class_name': name,
                                'module_name': module_name,
                                'file_path': str(estimator_file),
                                'constructor_params': params,
                                'category': category,
                                'description': self._extract_class_docstring(obj)
                            }
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze {estimator_file}: {e}")
        
        return estimators
    
    def _discover_neural_components(self) -> Dict[str, Any]:
        """Discover neural network components."""
        neural_components = {}
        
        # Look for neural components
        neural_paths = [
            self.project_root / "models" / "data_models" / "neural_fsde",
            self.project_root / "models" / "neural"
        ]
        
        for base_path in neural_paths:
            if not base_path.exists():
                continue
                
            for neural_file in base_path.rglob("*.py"):
                if neural_file.name.startswith('_'):
                    continue
                    
                try:
                    module_path = neural_file.relative_to(self.project_root)
                    module_name = str(module_path).replace(os.sep, '.').replace('.py', '')
                    
                    # Import module
                    module = importlib.import_module(module_name)
                    
                    # Find relevant classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (hasattr(obj, 'simulate') or 
                            hasattr(obj, 'generate') or
                            'Neural' in name or
                            'FSDE' in name):
                            
                            params = self._extract_constructor_params(obj)
                            
                            neural_components[name] = {
                                'class_name': name,
                                'module_name': module_name,
                                'file_path': str(neural_file),
                                'constructor_params': params,
                                'type': 'neural',
                                'description': self._extract_class_docstring(obj)
                            }
                            
                except Exception as e:
                    logger.warning(f"Failed to analyze {neural_file}: {e}")
        
        return neural_components
    
    def _extract_constructor_params(self, cls) -> Dict[str, Any]:
        """Extract constructor parameters from a class."""
        try:
            # Get __init__ method
            init_method = cls.__init__
            if hasattr(init_method, '__code__'):
                # Get parameter names
                param_names = init_method.__code__.co_varnames[1:init_method.__code__.co_argcount]
                
                # Get default values
                defaults = init_method.__defaults__ or ()
                default_dict = {}
                
                # Map defaults to parameters
                for i, default in enumerate(defaults):
                    param_index = len(param_names) - len(defaults) + i
                    if param_index < len(param_names):
                        default_dict[param_names[param_index]] = default
                
                # Create parameter info
                params = {}
                for param in param_names:
                    if param != 'self':
                        params[param] = {
                            'type': 'any',  # Could be enhanced with type hints
                            'default': default_dict.get(param, None),
                            'required': param not in default_dict
                        }
                
                return params
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to extract constructor params for {cls.__name__}: {e}")
            return {}
    
    def _extract_class_docstring(self, cls) -> str:
        """Extract class docstring."""
        try:
            return cls.__doc__ or ""
        except:
            return ""
    
    def _determine_estimator_category(self, file_path: Path) -> str:
        """Determine estimator category from file path."""
        path_str = str(file_path).lower()
        
        if 'temporal' in path_str:
            return 'temporal'
        elif 'spectral' in path_str:
            return 'spectral'
        elif 'wavelet' in path_str:
            return 'wavelet'
        elif 'multifractal' in path_str:
            return 'multifractal'
        elif 'high_performance' in path_str:
            return 'high_performance'
        else:
            return 'other'
    
    def _save_discovery_results(self) -> None:
        """Save discovery results to registry file."""
        discovery_data = {
            'timestamp': datetime.now().isoformat(),
            'components': self.discovered_components,
            'summary': {
                'data_generators': len(self.discovered_components['data_generators']),
                'estimators': len(self.discovered_components['estimators']),
                'neural_components': len(self.discovered_components['neural_components'])
            }
        }
        
        with open(self.component_registry_file, 'w', encoding='utf-8') as f:
            json.dump(discovery_data, f, indent=2)
        
        # Save timestamp of last discovery
        with open(self.last_discovery_file, 'w', encoding='utf-8') as f:
            f.write(datetime.now().isoformat())
    
    def check_for_new_components(self) -> bool:
        """Check if there are new components since last discovery."""
        if not self.last_discovery_file.exists():
            return True
        
        # Load last discovery timestamp
        with open(self.last_discovery_file, 'r', encoding='utf-8') as f:
            last_discovery = datetime.fromisoformat(f.read().strip())
        
        # Check if any relevant files have been modified since last discovery
        relevant_paths = [
            self.project_root / "models" / "data_models",
            self.project_root / "analysis",
            self.project_root / "models" / "estimators"
        ]
        
        for path in relevant_paths:
            if path.exists():
                for file_path in path.rglob("*.py"):
                    if file_path.stat().st_mtime > last_discovery.timestamp():
                        return True
        
        return False
    
    def update_benchmark_scripts(self) -> None:
        """Update benchmark scripts with discovered components."""
        logger.info("Updating benchmark scripts...")
        
        # Update comprehensive benchmark script
        self._update_comprehensive_benchmark()
        
        # Update example scripts
        self._update_example_scripts()
        
        # Update test scripts
        self._update_test_scripts()
        
        logger.info("Benchmark scripts updated successfully")
    
    def _update_comprehensive_benchmark(self) -> None:
        """Update the comprehensive benchmark script."""
        benchmark_file = self.project_root / "comprehensive_estimator_benchmark.py"
        
        if not benchmark_file.exists():
            logger.warning("Comprehensive benchmark file not found")
            return
        
        # Read current benchmark file
        try:
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(benchmark_file, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Update imports section
        updated_content = self._update_benchmark_imports(content)
        
        # Update data generators initialization
        updated_content = self._update_data_generators_init(updated_content)
        
        # Update estimators initialization
        updated_content = self._update_estimators_init(updated_content)
        
        # Write updated content
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        logger.info(f"Updated {benchmark_file}")
    
    def _update_benchmark_imports(self, content: str) -> str:
        """Update imports section in benchmark script."""
        # Find imports section
        import_pattern = r'(# Import data models.*?)(# Import all available estimators.*?)(try:)'
        
        # Generate new imports
        new_imports = "# Import data models\n"
        
        # Add data generator imports
        for name, info in self.discovered_components['data_generators'].items():
            if info['type'] != 'neural':
                new_imports += f"from {info['module_name']} import {info['class_name']}\n"
        
        # Add neural component imports
        new_imports += "\n# Try to import neural components\n"
        for name, info in self.discovered_components['neural_components'].items():
            new_imports += f"try:\n"
            new_imports += f"    from {info['module_name']} import {info['class_name']}\n"
            new_imports += f"    {name.upper()}_AVAILABLE = True\n"
            new_imports += f"except ImportError:\n"
            new_imports += f"    {name.upper()}_AVAILABLE = False\n\n"
        
        # Add estimator imports
        new_imports += "# Import all available estimators\n"
        for name, info in self.discovered_components['estimators'].items():
            new_imports += f"try:\n"
            new_imports += f"    from {info['module_name']} import {info['class_name']}\n"
            new_imports += f"    {name.upper()}_AVAILABLE = True\n"
            new_imports += f"except ImportError:\n"
            new_imports += f"    {name.upper()}_AVAILABLE = False\n\n"
        
        # Replace imports section
        content = re.sub(import_pattern, new_imports, content, flags=re.DOTALL)
        
        return content
    
    def _update_data_generators_init(self, content: str) -> str:
        """Update data generators initialization in benchmark script."""
        # Find the _initialize_data_generators method
        method_pattern = r'def _initialize_data_generators\(self\) -> Dict\[str, Any\]:(.*?)return generators'
        
        # Generate new method content
        new_method = """        \"\"\"Initialize all available data generators with correct parameters.\"\"\"
        print("Initializing data generators...")
        
        generators = {}
        
        # Traditional stochastic models"""
        
        # Add data generators
        for name, info in self.discovered_components['data_generators'].items():
            if info['type'] == 'stochastic':
                new_method += f"\n        generators['{name}'] = {{\n"
                new_method += f"            'class': {info['class_name']},\n"
                new_method += f"            'params': {self._format_params(info['constructor_params'])},\n"
                new_method += f"            'type': 'stochastic'\n"
                new_method += f"        }}"
        
        # Add neural components
        if self.discovered_components['neural_components']:
            new_method += "\n        \n        # Neural components (if available)"
            for name, info in self.discovered_components['neural_components'].items():
                new_method += f"\n        if {name.upper()}_AVAILABLE:\n"
                new_method += f"            try:\n"
                new_method += f"                generators['{name}'] = {{\n"
                new_method += f"                    'class': {info['class_name']},\n"
                new_method += f"                    'params': {self._format_params(info['constructor_params'])},\n"
                new_method += f"                    'type': 'neural'\n"
                new_method += f"                }}\n"
                new_method += f"            except Exception as e:\n"
                new_method += f"                print(f\"Warning: {name} not available: {{e}}\")"
        
        new_method += "\n        \n        print(f\"  Initialized {len(generators)} data generators\")\n        return generators"
        
        # Replace method
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        
        return content
    
    def _update_estimators_init(self, content: str) -> str:
        """Update estimators initialization in benchmark script."""
        # Find the _initialize_estimators method
        method_pattern = r'def _initialize_estimators\(self\) -> Dict\[str, Any\]:(.*?)return estimators'
        
        # Generate new method content
        new_method = """        \"\"\"Initialize all available estimators.\"\"\"
        print("Initializing estimators...")
        
        estimators = {}
        
        # Group estimators by category"""
        
        # Group estimators by category
        categories = {}
        for name, info in self.discovered_components['estimators'].items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((name, info))
        
        # Add estimators by category
        for category, estimators_list in categories.items():
            new_method += f"\n        # {category.title()} estimators"
            for name, info in estimators_list:
                new_method += f"\n        if {name.upper()}_AVAILABLE:\n"
                new_method += f"            estimators['{name}'] = {{\n"
                new_method += f"                'class': {info['class_name']},\n"
                new_method += f"                'params': {self._format_params(info['constructor_params'])},\n"
                new_method += f"                'category': '{info['category']}'\n"
                new_method += f"            }}"
        
        new_method += "\n        \n        print(f\"  Initialized {len(estimators)} estimators\")\n        return estimators"
        
        # Replace method
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        
        return content
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for code generation."""
        if not params:
            return "{}"
        
        formatted = "{"
        for param_name, param_info in params.items():
            if param_info['required']:
                formatted += f"'{param_name}': value, "  # Placeholder for required params
            elif param_info['default'] is not None:
                formatted += f"'{param_name}': {repr(param_info['default'])}, "
        formatted = formatted.rstrip(", ") + "}"
        
        return formatted
    
    def _update_example_scripts(self) -> None:
        """Update example scripts with discovered components."""
        # Find example scripts
        example_paths = [
            self.project_root / "demos",
            self.project_root / "examples"
        ]
        
        for base_path in example_paths:
            if not base_path.exists():
                continue
            
            for example_file in base_path.rglob("*.py"):
                if example_file.name.endswith("_demo.py") or example_file.name.endswith("_example.py"):
                    self._update_single_example_script(example_file)
    
    def _update_single_example_script(self, file_path: Path) -> None:
        """Update a single example script."""
        try:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Update imports if needed
            updated_content = self._update_example_imports(content, file_path)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Updated example script: {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to update {file_path}: {e}")
    
    def _update_example_imports(self, content: str, file_path: Path) -> str:
        """Update imports in example script."""
        # This is a simplified version - could be enhanced based on specific needs
        return content
    
    def _update_test_scripts(self) -> None:
        """Update test scripts with discovered components."""
        test_paths = [
            self.project_root / "tests",
            self.project_root / "test"
        ]
        
        for base_path in test_paths:
            if not base_path.exists():
                continue
            
            for test_file in base_path.rglob("test_*.py"):
                self._update_single_test_script(test_file)
    
    def _update_single_test_script(self, file_path: Path) -> None:
        """Update a single test script."""
        try:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Update test imports if needed
            updated_content = self._update_test_imports(content, file_path)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Updated test script: {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to update {file_path}: {e}")
    
    def _update_test_imports(self, content: str, file_path: Path) -> str:
        """Update imports in test script."""
        # This is a simplified version - could be enhanced based on specific needs
        return content
    
    def generate_component_summary(self) -> str:
        """Generate a summary of discovered components."""
        summary = "# Component Discovery Summary\n\n"
        summary += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Data Generators
        summary += "## Data Generators\n\n"
        for name, info in self.discovered_components['data_generators'].items():
            summary += f"### {name}\n"
            summary += f"- **Type**: {info['type']}\n"
            summary += f"- **Module**: `{info['module_name']}`\n"
            summary += f"- **File**: `{info['file_path']}`\n"
            if info['description']:
                summary += f"- **Description**: {info['description'].strip()}\n"
            summary += f"- **Parameters**: {list(info['constructor_params'].keys())}\n\n"
        
        # Estimators
        summary += "## Estimators\n\n"
        for name, info in self.discovered_components['estimators'].items():
            summary += f"### {name}\n"
            summary += f"- **Category**: {info['category']}\n"
            summary += f"- **Module**: `{info['module_name']}`\n"
            summary += f"- **File**: `{info['file_path']}`\n"
            if info['description']:
                summary += f"- **Description**: {info['description'].strip()}\n"
            summary += f"- **Parameters**: {list(info['constructor_params'].keys())}\n\n"
        
        # Neural Components
        if self.discovered_components['neural_components']:
            summary += "## Neural Components\n\n"
            for name, info in self.discovered_components['neural_components'].items():
                summary += f"### {name}\n"
                summary += f"- **Type**: {info['type']}\n"
                summary += f"- **Module**: `{info['module_name']}`\n"
                summary += f"- **File**: `{info['file_path']}`\n"
                if info['description']:
                    summary += f"- **Description**: {info['description'].strip()}\n"
                summary += f"- **Parameters**: {list(info['constructor_params'].keys())}\n\n"
        
        return summary
    
    def run_auto_update(self) -> None:
        """Run the complete auto-update process."""
        logger.info("Starting auto-update process...")
        
        # Check if new components exist
        if self.check_for_new_components():
            logger.info("New components detected, running discovery...")
            
            # Discover components
            self.discover_components()
            
            # Update scripts
            self.update_benchmark_scripts()
            
            # Generate summary
            summary = self.generate_component_summary()
            with open(self.project_root / "COMPONENT_SUMMARY.md", 'w', encoding='utf-8') as f:
                f.write(summary)
            
            logger.info("Auto-update complete!")
        else:
            logger.info("No new components detected, skipping update.")

def main():
    """Main function to run the auto-discovery system."""
    print("=== AUTO-DISCOVERY SYSTEM FOR DATAEXPLORATORYPROJECT ===\n")
    
    # Initialize system
    discovery_system = AutoDiscoverySystem()
    
    # Run auto-update
    discovery_system.run_auto_update()
    
    # Print summary
    print("\n=== DISCOVERY SUMMARY ===")
    components = discovery_system.discovered_components
    print(f"Data Generators: {len(components['data_generators'])}")
    print(f"Estimators: {len(components['estimators'])}")
    print(f"Neural Components: {len(components['neural_components'])}")
    
    print("\nComponent summary saved to: COMPONENT_SUMMARY.md")
    print("Component registry saved to: component_registry.json")

if __name__ == "__main__":
    main()
