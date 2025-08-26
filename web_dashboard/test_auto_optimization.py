#!/usr/bin/env python3
"""
Test script to verify auto-optimization system is working.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_auto_optimization():
    """Test that auto-optimization system can be imported and used."""
    
    print("🧪 Testing Auto-Optimization System...")
    
    try:
        # Test 1: Import AutoOptimizedEstimator
        print("1. Testing import...")
        from lrdbench.analysis.auto_optimized_estimator import AutoOptimizedEstimator
        print("   ✅ AutoOptimizedEstimator imported successfully!")
        
        # Test 2: Create an auto-optimized estimator
        print("2. Testing estimator creation...")
        estimator = AutoOptimizedEstimator('dfa')
        print(f"   ✅ Created {estimator.estimator_type.upper()} estimator")
        print(f"   ✅ Optimization level: {estimator.optimization_level}")
        
        # Test 3: Generate test data
        print("3. Testing data generation...")
        from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise
        fgn = FractionalGaussianNoise(H=0.7)
        test_data = fgn.generate(1000, seed=42)
        print(f"   ✅ Generated {len(test_data)} data points")
        
        # Test 4: Run estimation
        print("4. Testing estimation...")
        result = estimator.estimate(test_data)
        print(f"   ✅ Estimation successful: H = {result['hurst_parameter']:.3f}")
        
        print("\n🎉 All tests passed! Auto-optimization system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_auto_optimization()
    sys.exit(0 if success else 1)
