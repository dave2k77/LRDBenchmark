#!/usr/bin/env python3
"""
Comprehensive API Demo for LRDBench
Demonstrates all major components and usage patterns
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_data_models():
    """Demonstrate data model usage."""
    print("🔬 DATA MODELS DEMO")
    print("=" * 50)
    
    try:
        # Import data models
        from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
        from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
        from lrdbench.models.data_models.arfima.arfima_model import ARFIMAModel
        from lrdbench.models.data_models.mrw.mrw_model import MultifractalRandomWalk
        
        # Generate fBm data
        print("📊 Generating Fractional Brownian Motion...")
        fbm = FractionalBrownianMotion(H=0.7, sigma=1.0)
        fbm_data = fbm.generate(1000, seed=42)
        print(f"   Generated {len(fbm_data)} fBm data points with H=0.7")
        
        # Generate fGn data
        print("📊 Generating Fractional Gaussian Noise...")
        fgn = FractionalGaussianNoise(H=0.7, sigma=1.0)
        fgn_data = fgn.generate(1000, seed=42)
        print(f"   Generated {len(fgn_data)} fGn data points with H=0.7")
        
        # Generate ARFIMA data
        print("📊 Generating ARFIMA data...")
        arfima = ARFIMAModel(d=0.3, ar_params=[0.5], ma_params=[0.2])
        arfima_data = arfima.generate(1000, seed=42)
        print(f"   Generated {len(arfima_data)} ARFIMA data points with d=0.3")
        
        # Generate MRW data
        print("📊 Generating Multifractal Random Walk...")
        mrw = MultifractalRandomWalk(H=0.7, lambda_param=0.5, sigma=1.0)
        mrw_data = mrw.generate(1000, seed=42)
        print(f"   Generated {len(mrw_data)} MRW data points with H=0.7")
        
        return {
            'fbm': fbm_data,
            'fgn': fgn_data,
            'arfima': arfima_data,
            'mrw': mrw_data
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the package is properly installed: pip install lrdbench")
        return None

def demo_classical_estimators(data_dict):
    """Demonstrate classical estimator usage."""
    print("\n🔬 CLASSICAL ESTIMATORS DEMO")
    print("=" * 50)
    
    try:
        # Import classical estimators
        from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator
        from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
        from lrdbench.analysis.spectral.gph.gph_estimator import GPHEstimator
        from lrdbench.analysis.wavelet.cwt.cwt_estimator import CWTEstimator
        
        estimators = {
            'R/S': RSEstimator(),
            'DFA': DFAEstimator(),
            'GPH': GPHEstimator(),
            'CWT': CWTEstimator()
        }
        
        results = {}
        
        for name, estimator in estimators.items():
            print(f"🔍 Testing {name} estimator...")
            
            # Test on fBm data
            result = estimator.estimate(data_dict['fbm'])
            if result.get('hurst_parameter') is not None:
                h_est = result['hurst_parameter']
                error = abs(h_est - 0.7)
                print(f"   fBm: H_est={h_est:.4f}, Error={error:.4f}")
            else:
                print(f"   fBm: Failed to estimate")
            
            # Test on fGn data
            result = estimator.estimate(data_dict['fgn'])
            if result.get('hurst_parameter') is not None:
                h_est = result['hurst_parameter']
                error = abs(h_est - 0.7)
                print(f"   fGn: H_est={h_est:.4f}, Error={error:.4f}")
            else:
                print(f"   fGn: Failed to estimate")
            
            results[name] = result
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

def demo_ml_estimators(data_dict):
    """Demonstrate ML estimator usage."""
    print("\n🔬 MACHINE LEARNING ESTIMATORS DEMO")
    print("=" * 50)
    
    try:
        # Import ML estimators
        from lrdbench.analysis.machine_learning.random_forest_estimator import RandomForestEstimator
        from lrdbench.analysis.machine_learning.gradient_boosting_estimator import GradientBoostingEstimator
        from lrdbench.analysis.machine_learning.svr_estimator import SVREstimator
        
        estimators = {
            'RandomForest': RandomForestEstimator(),
            'GradientBoosting': GradientBoostingEstimator(),
            'SVR': SVREstimator()
        }
        
        results = {}
        
        for name, estimator in estimators.items():
            print(f"🔍 Testing {name} estimator...")
            
            # Test on fBm data
            result = estimator.estimate(data_dict['fbm'])
            if result.get('hurst_parameter') is not None:
                h_est = result['hurst_parameter']
                error = abs(h_est - 0.7)
                print(f"   fBm: H_est={h_est:.4f}, Error={error:.4f}")
            else:
                print(f"   fBm: Failed to estimate")
            
            results[name] = result
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

def demo_neural_estimators(data_dict):
    """Demonstrate neural estimator usage."""
    print("\n🔬 NEURAL NETWORK ESTIMATORS DEMO")
    print("=" * 50)
    
    try:
        # Import neural estimators
        from lrdbench.analysis.machine_learning.cnn_estimator import CNNEstimator
        from lrdbench.analysis.machine_learning.transformer_estimator import TransformerEstimator
        
        estimators = {
            'CNN': CNNEstimator(),
            'Transformer': TransformerEstimator()
        }
        
        results = {}
        
        for name, estimator in estimators.items():
            print(f"🔍 Testing {name} estimator...")
            
            # Test on fBm data
            result = estimator.estimate(data_dict['fbm'])
            if result.get('hurst_parameter') is not None:
                h_est = result['hurst_parameter']
                error = abs(h_est - 0.7)
                print(f"   fBm: H_est={h_est:.4f}, Error={error:.4f}")
            else:
                print(f"   fBm: Failed to estimate")
            
            results[name] = result
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

def demo_pre_trained_models(data_dict):
    """Demonstrate pre-trained model usage."""
    print("\n🔬 PRE-TRAINED MODELS DEMO")
    print("=" * 50)
    
    try:
        # Import pre-trained models
        from lrdbench.models.pretrained_models.cnn_pretrained import CNNPretrainedModel
        from lrdbench.models.pretrained_models.transformer_pretrained import TransformerPretrainedModel
        from lrdbench.models.pretrained_models.ml_pretrained import RandomForestPretrainedModel
        
        models = {
            'CNNPretrained': CNNPretrainedModel(input_length=500),
            'TransformerPretrained': TransformerPretrainedModel(input_length=500),
            'RandomForestPretrained': RandomForestPretrainedModel()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"🔍 Testing {name}...")
            
            # Test on fBm data
            result = model.estimate(data_dict['fbm'])
            if result.get('hurst_parameter') is not None:
                h_est = result['hurst_parameter']
                error = abs(h_est - 0.7)
                print(f"   fBm: H_est={h_est:.4f}, Error={error:.4f}")
            else:
                print(f"   fBm: Failed to estimate")
            
            results[name] = result
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

def demo_comprehensive_benchmark():
    """Demonstrate the comprehensive benchmark system."""
    print("\n🔬 COMPREHENSIVE BENCHMARK DEMO")
    print("=" * 50)
    
    try:
        from lrdbench.analysis.benchmark import ComprehensiveBenchmark
        
        # Initialize benchmark system
        benchmark = ComprehensiveBenchmark(output_dir="demo_benchmark_results")
        
        print("🚀 Running comprehensive benchmark...")
        
        # Run a quick benchmark
        results = benchmark.run_comprehensive_benchmark(
            data_length=500,  # Shorter for demo
            save_results=True
        )
        
        print(f"✅ Benchmark completed!")
        print(f"   Success rate: {results['success_rate']:.1%}")
        print(f"   Total tests: {results['total_tests']}")
        print(f"   Successful: {results['successful_tests']}")
        
        # Show top performers
        print("\n🏆 Top performing estimators:")
        for model_name, model_data in results['results'].items():
            if 'estimator_results' in model_data:
                successful = [r for r in model_data['estimator_results'] if r['success']]
                if successful:
                    best = min(successful, key=lambda x: x.get('error', float('inf')))
                    print(f"   {model_name}: {best['estimator']} (Error: {best.get('error', 'N/A'):.4f})")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

def demo_contamination_testing(data_dict):
    """Demonstrate contamination testing."""
    print("\n🔬 CONTAMINATION TESTING DEMO")
    print("=" * 50)
    
    try:
        from lrdbench.analysis.benchmark import ComprehensiveBenchmark
        
        benchmark = ComprehensiveBenchmark()
        
        # Test different contamination types
        contamination_types = ['additive_gaussian', 'outliers', 'trend']
        
        for cont_type in contamination_types:
            print(f"🔍 Testing with {cont_type} contamination...")
            
            results = benchmark.run_classical_benchmark(
                data_length=500,
                contamination_type=cont_type,
                contamination_level=0.2,
                save_results=False
            )
            
            print(f"   Success rate: {results['success_rate']:.1%}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

def main():
    """Run the comprehensive demo."""
    print("🚀 LRDBench Comprehensive API Demo")
    print("=" * 60)
    print("This demo showcases all major components of LRDBench")
    print("=" * 60)
    
    # Demo 1: Data Models
    data_dict = demo_data_models()
    if data_dict is None:
        print("❌ Data models demo failed. Exiting.")
        return
    
    # Demo 2: Classical Estimators
    classical_results = demo_classical_estimators(data_dict)
    
    # Demo 3: ML Estimators
    ml_results = demo_ml_estimators(data_dict)
    
    # Demo 4: Neural Estimators
    neural_results = demo_neural_estimators(data_dict)
    
    # Demo 5: Pre-trained Models
    pre_trained_results = demo_pre_trained_models(data_dict)
    
    # Demo 6: Comprehensive Benchmark
    benchmark_results = demo_comprehensive_benchmark()
    
    # Demo 7: Contamination Testing
    contamination_results = demo_contamination_testing(data_dict)
    
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE DEMO COMPLETED!")
    print("=" * 60)
    
    if all([classical_results, ml_results, neural_results, 
             pre_trained_results, benchmark_results, contamination_results]):
        print("✅ All demos completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Explore the generated data and results")
        print("   2. Check the 'demo_benchmark_results' directory")
        print("   3. Try different parameters and contamination types")
        print("   4. Use the ComprehensiveBenchmark for your own analysis")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
    
    print("\n📚 For more information, see:")
    print("   - API Reference: documentation/api_reference/")
    print("   - Examples: examples/")
    print("   - Project README: README.md")

if __name__ == "__main__":
    main()
