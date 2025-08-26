#!/usr/bin/env python3
"""
Test script for LRDBenchmark Web Dashboard
This script tests the core functionality without running the full Streamlit app.
"""

import sys
import os

# Add parent directory to path to import lrdbench
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from lrdbench import (
            FBMModel, FGNModel, ARFIMAModel, MRWModel,
            ComprehensiveBenchmark
        )
        print("✅ LRDBenchmark imported successfully")
    except ImportError as e:
        print(f"❌ LRDBenchmark import failed: {e}")
        print("   Please install with: pip install lrdbenchmark")
        return False
    
    return True

def test_data_generation():
    """Test data generation functionality."""
    print("\n🧪 Testing data generation...")
    
    try:
        from lrdbench import FBMModel, FGNModel, ARFIMAModel, MRWModel
        
        # Test FBM
        fbm = FBMModel(H=0.7, sigma=1.0)
        fbm_data = fbm.generate(100, seed=42)
        print(f"✅ FBM data generated: {len(fbm_data)} points")
        
        # Test FGN
        fgn = FGNModel(H=0.6, sigma=1.0)
        fgn_data = fgn.generate(100, seed=42)
        print(f"✅ FGN data generated: {len(fgn_data)} points")
        
        # Test ARFIMA
        arfima = ARFIMAModel(d=0.3, sigma=1.0)
        arfima_data = arfima.generate(100, seed=42)
        print(f"✅ ARFIMA data generated: {len(arfima_data)} points")
        
        # Test MRW
        mrw = MRWModel(H=0.7, lambda_param=0.1, sigma=1.0)
        mrw_data = mrw.generate(100, seed=42)
        print(f"✅ MRW data generated: {len(mrw_data)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_benchmark():
    """Test benchmarking functionality."""
    print("\n🧪 Testing benchmarking...")
    
    try:
        from lrdbench import ComprehensiveBenchmark
        
        benchmark = ComprehensiveBenchmark()
        print("✅ ComprehensiveBenchmark created successfully")
        
        # Test with small data length for quick test
        results = benchmark.run_comprehensive_benchmark(data_length=100)
        print(f"✅ Benchmark completed: {len(results)} estimators tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality."""
    print("\n🧪 Testing visualization...")
    
    try:
        import plotly.graph_objects as go
        import numpy as np
        
        # Create a simple plot
        data = np.random.randn(100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data, mode='lines'))
        fig.update_layout(title="Test Plot")
        
        print("✅ Plotly visualization created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 LRDBenchmark Web Dashboard Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Generation Test", test_data_generation),
        ("Benchmark Test", test_benchmark),
        ("Visualization Test", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        print("\n🚀 To run the dashboard:")
        print("   cd web_dashboard")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   - Install missing packages: pip install streamlit plotly")
        print("   - Install LRDBenchmark: pip install lrdbenchmark")
        print("   - Check Python environment and dependencies")

if __name__ == "__main__":
    main()
