#!/usr/bin/env python3
"""
Comprehensive import testing script for the DataExploratoryProject.
Tests all imports at all levels to prevent future import issues.
"""

import sys
import os
import importlib
import traceback
from typing import List, Dict, Tuple

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_import(module_name: str, description: str = "") -> Tuple[bool, str]:
    """
    Test importing a specific module.
    
    Args:
        module_name: Name of the module to import
        description: Description of what this module is for
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        module = importlib.import_module(module_name)
        return True, f"✓ Successfully imported {module_name}"
    except Exception as e:
        error_msg = f"✗ Failed to import {module_name}: {str(e)}"
        if description:
            error_msg += f" ({description})"
        return False, error_msg

def test_class_import(module_name: str, class_name: str, description: str = "") -> Tuple[bool, str]:
    """
    Test importing a specific class from a module.
    
    Args:
        module_name: Name of the module to import from
        class_name: Name of the class to import
        description: Description of what this class is for
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        module = importlib.import_module(module_name)
        class_obj = getattr(module, class_name)
        return True, f"✓ Successfully imported {class_name} from {module_name}"
    except Exception as e:
        error_msg = f"✗ Failed to import {class_name} from {module_name}: {str(e)}"
        if description:
            error_msg += f" ({description})"
        return False, error_msg

def test_third_party_imports() -> List[Tuple[bool, str]]:
    """Test all third-party package imports."""
    print("=" * 60)
    print("TESTING THIRD-PARTY PACKAGE IMPORTS")
    print("=" * 60)
    
    third_party_packages = [
        ("numpy", "Core numerical computing"),
        ("scipy", "Scientific computing"),
        ("pandas", "Data manipulation"),
        ("matplotlib", "Plotting library"),
        ("seaborn", "Statistical plotting"),
        ("plotly", "Interactive plotting"),
        ("statsmodels", "Statistical models"),
        ("sklearn", "Machine learning"),
        ("jax", "High-performance computing"),
        ("jaxlib", "JAX library"),
        ("numba", "JIT compilation"),
        ("pywt", "Wavelet transforms"),
        ("pytest", "Testing framework"),
        ("tqdm", "Progress bars"),
        ("joblib", "Parallel computing"),
    ]
    
    results = []
    for package, description in third_party_packages:
        success, message = test_import(package, description)
        results.append((success, message))
        print(message)
    
    return results

def test_project_imports() -> List[Tuple[bool, str]]:
    """Test all project-specific imports."""
    print("\n" + "=" * 60)
    print("TESTING PROJECT-SPECIFIC IMPORTS")
    print("=" * 60)
    
    # Define all the imports to test
    imports_to_test = [
        # Package level imports
        ("models", "Main models package"),
        ("models.data_models", "Data models subpackage"),
        ("models.estimators", "Estimators subpackage"),
        ("tests", "Tests package"),
        ("analysis", "Analysis package"),
        
        # Base classes
        ("models.data_models.base_model", "Base model class"),
        ("models.estimators.base_estimator", "Base estimator class"),
        
        # Model implementations
        ("models.data_models.fbm", "fBm model package"),
        ("models.data_models.fbm.fbm_model", "fBm model implementation"),
        ("models.data_models.fgn", "fGn model package"),
        ("models.data_models.fgn.fgn_model", "fGn model implementation"),
        ("models.data_models.mrw", "MRW model package"),
        ("models.data_models.mrw.mrw_model", "MRW model implementation"),
        ("models.data_models.arfima", "ARFIMA model package"),
        ("models.data_models.arfima.arfima_model", "ARFIMA model implementation"),
        
        # Estimator implementations
        ("analysis.temporal.dfa", "DFA estimator package"),
        ("analysis.temporal.dfa.dfa_estimator", "DFA estimator implementation"),
    ]
    
    results = []
    for module, description in imports_to_test:
        success, message = test_import(module, description)
        results.append((success, message))
        print(message)
    
    return results

def test_class_imports() -> List[Tuple[bool, str]]:
    """Test importing specific classes from the project."""
    print("\n" + "=" * 60)
    print("TESTING CLASS IMPORTS")
    print("=" * 60)
    
    class_imports = [
        ("models.data_models.base_model", "BaseModel", "Abstract base class for models"),
        ("models.data_models.fbm.fbm_model", "FractionalBrownianMotion", "fBm model class"),
        ("models.data_models.fgn.fgn_model", "FractionalGaussianNoise", "fGn model class"),
        ("models.data_models.mrw.mrw_model", "MultifractalRandomWalk", "MRW model class"),
        ("models.data_models.arfima.arfima_model", "ARFIMAModel", "ARFIMA model class"),
        ("models.estimators.base_estimator", "BaseEstimator", "Abstract base class for estimators"),
        ("analysis.temporal.dfa.dfa_estimator", "DFAEstimator", "DFA estimator class"),
    ]
    
    results = []
    for module, class_name, description in class_imports:
        success, message = test_class_import(module, class_name, description)
        results.append((success, message))
        print(message)
    
    return results

def test_instantiation() -> List[Tuple[bool, str]]:
    """Test instantiating classes to ensure they work properly."""
    print("\n" + "=" * 60)
    print("TESTING CLASS INSTANTIATION")
    print("=" * 60)
    
    results = []
    
    # Test fBm model instantiation
    try:
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
        results.append((True, "✓ Successfully instantiated FractionalBrownianMotion"))
    except Exception as e:
        results.append((False, f"✗ Failed to instantiate FractionalBrownianMotion: {str(e)}"))
    
    # Test fGn model instantiation
    try:
        from models.data_models.fgn.fgn_model import FractionalGaussianNoise
        fgn = FractionalGaussianNoise(H=0.7, sigma=1.0)
        results.append((True, "✓ Successfully instantiated FractionalGaussianNoise"))
    except Exception as e:
        results.append((False, f"✗ Failed to instantiate FractionalGaussianNoise: {str(e)}"))
    
    # Test MRW model instantiation
    try:
        from models.data_models.mrw.mrw_model import MultifractalRandomWalk
        mrw = MultifractalRandomWalk(H=0.7, lambda_param=0.3, sigma=1.0)
        results.append((True, "✓ Successfully instantiated MultifractalRandomWalk"))
    except Exception as e:
        results.append((False, f"✗ Failed to instantiate MultifractalRandomWalk: {str(e)}"))
    
    # Test ARFIMA model instantiation
    try:
        from models.data_models.arfima.arfima_model import ARFIMAModel
        arfima = ARFIMAModel(d=0.3)  # Pure fractional integration
        results.append((True, "✓ Successfully instantiated ARFIMAModel"))
    except Exception as e:
        results.append((False, f"✗ Failed to instantiate ARFIMAModel: {str(e)}"))
    
    # Test DFA estimator instantiation
    try:
        from analysis.temporal.dfa.dfa_estimator import DFAEstimator
        dfa = DFAEstimator(min_box_size=4, max_box_size=100)
        results.append((True, "✓ Successfully instantiated DFAEstimator"))
    except Exception as e:
        results.append((False, f"✗ Failed to instantiate DFAEstimator: {str(e)}"))
    
    for success, message in results:
        print(message)
    
    return results

def test_basic_functionality() -> List[Tuple[bool, str]]:
    """Test basic functionality of instantiated objects."""
    print("\n" + "=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    results = []
    
    # Test fBm generation
    try:
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        import numpy as np
        
        fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
        data = fbm.generate(n=1000, seed=42)
        
        if isinstance(data, np.ndarray) and len(data) == 1000:
            results.append((True, "✓ Successfully generated fBm data"))
        else:
            results.append((False, "✗ fBm data generation failed - wrong type or length"))
    except Exception as e:
        results.append((False, f"✗ fBm data generation failed: {str(e)}"))
    
    # Test fGn generation
    try:
        from models.data_models.fgn.fgn_model import FractionalGaussianNoise
        import numpy as np
        
        fgn = FractionalGaussianNoise(H=0.7, sigma=1.0)
        data = fgn.generate(n=1000, seed=42)
        
        if isinstance(data, np.ndarray) and len(data) == 1000:
            results.append((True, "✓ Successfully generated fGn data"))
        else:
            results.append((False, "✗ fGn data generation failed - wrong type or length"))
    except Exception as e:
        results.append((False, f"✗ fGn data generation failed: {str(e)}"))
    
    # Test MRW generation
    try:
        from models.data_models.mrw.mrw_model import MultifractalRandomWalk
        import numpy as np
        
        mrw = MultifractalRandomWalk(H=0.7, lambda_param=0.3, sigma=1.0)
        data = mrw.generate(n=1000, seed=42)
        
        if isinstance(data, np.ndarray) and len(data) == 1000:
            results.append((True, "✓ Successfully generated MRW data"))
        else:
            results.append((False, "✗ MRW data generation failed - wrong type or length"))
    except Exception as e:
        results.append((False, f"✗ MRW data generation failed: {str(e)}"))
    
    # Test ARFIMA generation
    try:
        from models.data_models.arfima.arfima_model import ARFIMAModel
        import numpy as np
        
        arfima = ARFIMAModel(d=0.3)  # Pure fractional integration
        data = arfima.generate(n=1000, seed=42)
        
        if isinstance(data, np.ndarray) and len(data) == 1000:
            results.append((True, "✓ Successfully generated ARFIMA data"))
        else:
            results.append((False, "✗ ARFIMA data generation failed - wrong type or length"))
    except Exception as e:
        results.append((False, f"✗ ARFIMA data generation failed: {str(e)}"))
    
    # Test DFA estimation
    try:
        from analysis.temporal.dfa.dfa_estimator import DFAEstimator
        import numpy as np
        
        # Generate some test data
        np.random.seed(42)
        test_data = np.cumsum(np.random.randn(1000))
        
        dfa = DFAEstimator(min_box_size=4, max_box_size=100)
        results_dict = dfa.estimate(test_data)
        
        if 'hurst_parameter' in results_dict:
            results.append((True, "✓ Successfully performed DFA estimation"))
        else:
            results.append((False, "✗ DFA estimation failed - missing hurst_parameter"))
    except Exception as e:
        results.append((False, f"✗ DFA estimation failed: {str(e)}"))
    
    # Test R/S estimation
    try:
        from analysis.temporal.rs.rs_estimator import RSEstimator
        import numpy as np
        
        # Generate some test data
        np.random.seed(42)
        test_data = np.cumsum(np.random.randn(1000))
        
        rs = RSEstimator(min_window_size=10, max_window_size=100)
        results_dict = rs.estimate(test_data)
        
        if 'hurst_parameter' in results_dict:
            results.append((True, "✓ Successfully performed R/S estimation"))
        else:
            results.append((False, "✗ R/S estimation failed - missing hurst_parameter"))
    except Exception as e:
        results.append((False, f"✗ R/S estimation failed: {str(e)}"))
    
    # Test Higuchi estimation
    try:
        from analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator
        import numpy as np
        
        # Generate some test data
        np.random.seed(42)
        test_data = np.cumsum(np.random.randn(1000))
        
        higuchi = HiguchiEstimator(min_k=2, max_k=50)
        results_dict = higuchi.estimate(test_data)
        
        if 'fractal_dimension' in results_dict and 'hurst_parameter' in results_dict:
            results.append((True, "✓ Successfully performed Higuchi estimation"))
        else:
            results.append((False, "✗ Higuchi estimation failed - missing parameters"))
    except Exception as e:
        results.append((False, f"✗ Higuchi estimation failed: {str(e)}"))
    
    # Test DMA estimation
    try:
        from analysis.temporal.dma.dma_estimator import DMAEstimator
        import numpy as np
        
        # Generate some test data
        np.random.seed(42)
        test_data = np.cumsum(np.random.randn(1000))
        
        dma = DMAEstimator(min_window_size=4, max_window_size=100)
        results_dict = dma.estimate(test_data)
        
        if 'hurst_parameter' in results_dict:
            results.append((True, "✓ Successfully performed DMA estimation"))
        else:
            results.append((False, "✗ DMA estimation failed - missing hurst_parameter"))
    except Exception as e:
        results.append((False, f"✗ DMA estimation failed: {str(e)}"))
    
    # Test spectral estimators imports and basic estimation
    try:
        from analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator
        from analysis.spectral.whittle.whittle_estimator import WhittleEstimator
        from analysis.spectral.gph.gph_estimator import GPHEstimator
        import numpy as np

        np.random.seed(7)
        # Create fractional Gaussian noise by differencing fBm
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        fbm = FractionalBrownianMotion(H=0.6)
        series = fbm.generate(4096)
        fgn = np.diff(series)

        per = PeriodogramEstimator(min_freq_ratio=0.01, max_freq_ratio=0.1)
        whi = WhittleEstimator()
        gph = GPHEstimator()

        per_res = per.estimate(fgn)
        whi_res = whi.estimate(fgn)
        gph_res = gph.estimate(fgn)

        cond = (
            "hurst_parameter" in per_res
            and "hurst_parameter" in whi_res
            and "hurst_parameter" in gph_res
        )
        if cond:
            results.append((True, "✓ Successfully performed spectral estimations (Periodogram/Whittle/GPH)"))
        else:
            results.append((False, "✗ Spectral estimations missing expected outputs"))
    except Exception as e:
        results.append((False, f"✗ Spectral estimators failed: {str(e)}"))

    for success, message in results:
        print(message)
    
    return results

def test_import_paths() -> List[Tuple[bool, str]]:
    """Test that import paths are correctly set up."""
    print("\n" + "=" * 60)
    print("TESTING IMPORT PATHS")
    print("=" * 60)
    
    results = []
    
    # Test that we can import from different levels
    try:
        # Test importing from project root
        sys.path.insert(0, project_root)
        from models.data_models.fbm.fbm_model import FractionalBrownianMotion
        results.append((True, "✓ Import from project root works"))
    except Exception as e:
        results.append((False, f"✗ Import from project root failed: {str(e)}"))
    
    # Test relative imports within packages
    try:
        # This tests that __init__.py files are properly set up
        import models.data_models
        import models.estimators
        import analysis.temporal.dfa
        import analysis.temporal.rs
        import analysis.temporal.higuchi
        import analysis.temporal.dma
        results.append((True, "✓ Package __init__.py files work correctly"))
    except Exception as e:
        results.append((False, f"✗ Package __init__.py files failed: {str(e)}"))
    
    for success, message in results:
        print(message)
    
    return results

def main():
    """Run all import tests."""
    print("COMPREHENSIVE IMPORT TESTING")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[0]}")
    print()
    
    all_results = []
    
    # Run all tests
    all_results.extend(test_third_party_imports())
    all_results.extend(test_project_imports())
    all_results.extend(test_class_imports())
    all_results.extend(test_instantiation())
    all_results.extend(test_basic_functionality())
    all_results.extend(test_import_paths())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for success, _ in all_results if success)
    total_tests = len(all_results)
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL IMPORTS WORK CORRECTLY! 🎉")
        print("No import issues detected. The project structure is properly set up.")
    else:
        print("\n⚠️  SOME IMPORTS FAILED! ⚠️")
        print("Please check the failed imports above and fix any issues.")
        
        # Show failed tests
        print("\nFailed tests:")
        for success, message in all_results:
            if not success:
                print(f"  {message}")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
