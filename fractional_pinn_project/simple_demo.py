"""
Simple Demo for Fractional PINN Project

This script demonstrates the core functionality with minimal dependencies.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main demo function."""
    print("Simple Fractional PINN Demo")
    print("=" * 50)
    
    try:
        # Test data generation
        print("\n1. Testing data generation...")
        from data.generators import FractionalDataGenerator
        
        generator = FractionalDataGenerator(seed=42)
        
        # Generate sample data
        fbm_data = generator.generate_fbm(n_points=1000, hurst=0.7)
        fgn_data = generator.generate_fgn(n_points=1000, hurst=0.7)
        
        print(f"✅ Generated fBm data: {len(fbm_data['data'])} points, Hurst={fbm_data['hurst']}")
        print(f"✅ Generated fGn data: {len(fgn_data['data'])} points, Hurst={fgn_data['hurst']}")
        
        # Test classical estimators
        print("\n2. Testing classical estimators...")
        from estimators.classical_estimators import ClassicalEstimatorSuite
        
        classical_suite = ClassicalEstimatorSuite()
        
        # Test on fBm data
        estimates = classical_suite.estimate_all(fbm_data['data'])
        
        print("Classical estimator results:")
        for estimator_name, estimate in estimates.items():
            if estimate is not None and isinstance(estimate, dict) and 'hurst' in estimate:
                error = abs(estimate['hurst'] - fbm_data['hurst'])
                print(f"  {estimator_name}: {estimate['hurst']:.4f} (error: {error:.4f})")
            elif estimate is not None:
                error = abs(estimate - fbm_data['hurst'])
                print(f"  {estimator_name}: {estimate:.4f} (error: {error:.4f})")
        
        # Test ML estimators
        print("\n3. Testing ML estimators...")
        try:
            from estimators.ml_estimators import MLEstimatorSuite
            
            ml_suite = MLEstimatorSuite()
            
            # Create training data
            training_data = []
            hurst_values = []
            for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for _ in range(3):
                    data = generator.generate_fbm(n_points=500, hurst=hurst)
                    training_data.append(data['data'])
                    hurst_values.append(data['hurst'])
            
            # Train ML models
            ml_suite.train_all(training_data, hurst_values)
            
            # Test on new data
            test_data = generator.generate_fbm(n_points=500, hurst=0.6)
            ml_estimates = ml_suite.estimate_all(test_data['data'])
            
            print("ML estimator results:")
            for estimator_name, estimate in ml_estimates.items():
                if estimate is not None:
                    error = abs(estimate - test_data['hurst'])
                    print(f"  {estimator_name}: {estimate:.4f} (error: {error:.4f})")
        except Exception as e:
            print(f"❌ ML estimators failed: {e}")
            ml_estimates = {}
            test_data = generator.generate_fbm(n_points=500, hurst=0.6)
        
        # Test PINN estimator
        print("\n4. Testing PINN estimator...")
        pinn_estimate = None
        try:
            from estimators.pinn_estimator import PINNEstimator
            
            pinn_estimator = PINNEstimator(
                input_dim=1,
                hidden_dims=[32, 64, 32],
                output_dim=1,
                learning_rate=0.001,
                device='cpu'
            )
            
            pinn_estimator.build_model()
            
            # Create PINN training data
            pinn_training_data = []
            for i in range(min(5, len(training_data))):
                pinn_training_data.append({
                    'time_series': training_data[i],
                    'true_hurst': hurst_values[i]
                })
            
            # Train for a few epochs
            history = pinn_estimator.train(
                pinn_training_data,
                epochs=10,
                early_stopping_patience=5,
                save_model=False,
                verbose=False
            )
            
            # Test estimation
            pinn_estimate = pinn_estimator.estimate(test_data['data'])
            if pinn_estimate is not None:
                error = abs(pinn_estimate - test_data['hurst'])
                print(f"  PINN: {pinn_estimate:.4f} (error: {error:.4f})")
            
            print("✅ PINN training completed successfully")
            
        except Exception as e:
            print(f"❌ PINN test failed: {e}")
        
        # Create summary
        print("\n5. Creating summary...")
        
        # Collect all results
        all_results = []
        
        # Classical results
        for estimator_name, estimate in estimates.items():
            if estimate is not None and isinstance(estimate, dict) and 'hurst' in estimate:
                all_results.append({
                    'estimator': estimator_name,
                    'type': 'classical',
                    'estimate': estimate['hurst'],
                    'true_hurst': fbm_data['hurst'],
                    'error': abs(estimate['hurst'] - fbm_data['hurst'])
                })
            elif estimate is not None:
                all_results.append({
                    'estimator': estimator_name,
                    'type': 'classical',
                    'estimate': estimate,
                    'true_hurst': fbm_data['hurst'],
                    'error': abs(estimate - fbm_data['hurst'])
                })
        
        # ML results
        for estimator_name, estimate in ml_estimates.items():
            if estimate is not None:
                all_results.append({
                    'estimator': estimator_name,
                    'type': 'ml',
                    'estimate': estimate,
                    'true_hurst': test_data['hurst'],
                    'error': abs(estimate - test_data['hurst'])
                })
        
        # PINN results
        if pinn_estimate is not None:
            all_results.append({
                'estimator': 'PINN',
                'type': 'neural',
                'estimate': pinn_estimate,
                'true_hurst': test_data['hurst'],
                'error': abs(pinn_estimate - test_data['hurst'])
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(all_results)
        
        print("\nPerformance Summary:")
        print(results_df[['estimator', 'type', 'estimate', 'error']].to_string(index=False))
        
        # Find best estimator
        best_estimator = results_df.loc[results_df['error'].idxmin()]
        print(f"\n🏆 Best estimator: {best_estimator['estimator']} (error: {best_estimator['error']:.4f})")
        
        # Save results
        results_df.to_csv('demo_results.csv', index=False)
        print("\n✅ Results saved to 'demo_results.csv'")
        
        # Create simple plot
        print("\n6. Creating visualization...")
        plt.figure(figsize=(12, 8))
        
        # Plot time series
        plt.subplot(2, 2, 1)
        plt.plot(fbm_data['data'][:200])
        plt.title(f'fBm Time Series (H={fbm_data["hurst"]})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        # Plot estimates vs true
        plt.subplot(2, 2, 2)
        for estimator_type in ['classical', 'ml', 'neural']:
            subset = results_df[results_df['type'] == estimator_type]
            if not subset.empty:
                plt.scatter(subset['true_hurst'], subset['estimate'], 
                           label=estimator_type, alpha=0.7)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        plt.xlabel('True Hurst')
        plt.ylabel('Estimated Hurst')
        plt.title('Estimates vs True Values')
        plt.legend()
        
        # Plot errors
        plt.subplot(2, 2, 3)
        plt.bar(results_df['estimator'], results_df['error'])
        plt.xlabel('Estimator')
        plt.ylabel('Absolute Error')
        plt.title('Estimation Errors')
        plt.xticks(rotation=45)
        
        # Plot by type
        plt.subplot(2, 2, 4)
        type_errors = results_df.groupby('type')['error'].mean()
        plt.bar(type_errors.index, type_errors.values)
        plt.xlabel('Estimator Type')
        plt.ylabel('Mean Error')
        plt.title('Mean Errors by Type')
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to 'demo_results.png'")
        
        print("\n🎉 Demo completed successfully!")
        print(f"Generated {len(training_data)} training samples")
        print(f"Tested {len(results_df)} estimators")
        print(f"Best performance: {best_estimator['estimator']} ({best_estimator['error']:.4f} error)")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
