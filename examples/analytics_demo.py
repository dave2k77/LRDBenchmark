#!/usr/bin/env python3
"""
Analytics Demo for LRDBench

This script demonstrates the comprehensive analytics capabilities of LRDBench,
including usage tracking, performance monitoring, error analysis, and workflow tracking.
"""

import time
import numpy as np
from lrdbench import (
    ComprehensiveBenchmark,
    enable_analytics,
    get_analytics_summary,
    generate_analytics_report,
    track_usage,
    monitor_performance,
    track_errors,
    track_workflow
)


def demo_basic_analytics():
    """Demonstrate basic analytics functionality"""
    print("🔍 **LRDBench Analytics Demo**\n")
    
    # Enable analytics (this happens automatically, but we can control it)
    enable_analytics(True, privacy_mode=True)
    print("✅ Analytics enabled with privacy mode\n")
    
    # Get initial analytics summary
    print("📊 **Initial Analytics Summary:**")
    print(get_analytics_summary(days=1))
    print()


def demo_usage_tracking():
    """Demonstrate usage tracking with decorators"""
    print("📈 **Usage Tracking Demo:**\n")
    
    # Example of tracking estimator usage
    @track_usage("demo_estimator", data_length=1000)
    def demo_estimator(data, window_size=10):
        """Demo estimator that simulates some computation"""
        time.sleep(0.1)  # Simulate computation
        return np.mean(data[:window_size])
    
    # Run the tracked estimator
    data = np.random.randn(1000)
    result = demo_estimator(data, window_size=20)
    print(f"✅ Demo estimator result: {result:.4f}")
    
    # Get updated analytics
    print("\n📊 **Updated Analytics Summary:**")
    print(get_analytics_summary(days=1))
    print()


def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("⚡ **Performance Monitoring Demo:**\n")
    
    # Example of monitoring performance
    @monitor_performance("demo_performance_test")
    def performance_test(data_size=10000):
        """Function to test performance monitoring"""
        # Simulate different performance characteristics
        if data_size > 5000:
            time.sleep(0.5)  # Slower for large data
        else:
            time.sleep(0.1)  # Faster for small data
        
        return np.random.randn(data_size)
    
    # Test with different data sizes
    print("Testing performance monitoring...")
    performance_test(1000)
    performance_test(8000)
    performance_test(2000)
    
    print("✅ Performance monitoring completed")
    print()


def demo_error_tracking():
    """Demonstrate error tracking"""
    print("🛡️ **Error Tracking Demo:**\n")
    
    # Example of tracking errors
    @track_errors("demo_error_test")
    def error_test(should_fail=False):
        """Function that may fail for testing error tracking"""
        if should_fail:
            raise ValueError("This is a test error for analytics")
        return "Success!"
    
    # Test error tracking
    print("Testing error tracking...")
    try:
        error_test(should_fail=False)
        print("✅ Normal execution tracked")
    except Exception as e:
        print(f"❌ Error caught: {e}")
    
    try:
        error_test(should_fail=True)
    except Exception as e:
        print(f"✅ Error execution tracked: {e}")
    
    print()


def demo_workflow_tracking():
    """Demonstrate workflow tracking"""
    print("🔄 **Workflow Tracking Demo:**\n")
    
    # Example of tracking workflow steps
    @track_workflow("data_preprocessing")
    def preprocess_data(data):
        """Simulate data preprocessing step"""
        time.sleep(0.1)
        return data * 2
    
    @track_workflow("estimation")
    def estimate_parameter(data):
        """Simulate parameter estimation step"""
        time.sleep(0.2)
        return np.mean(data)
    
    @track_workflow("validation")
    def validate_result(result):
        """Simulate result validation step"""
        time.sleep(0.05)
        return result > 0
    
    # Run a complete workflow
    print("Running complete workflow...")
    data = np.random.randn(1000)
    
    processed_data = preprocess_data(data)
    estimated_param = estimate_parameter(processed_data)
    is_valid = validate_result(estimated_param)
    
    print(f"✅ Workflow completed: parameter={estimated_param:.4f}, valid={is_valid}")
    print()


def demo_benchmark_with_analytics():
    """Demonstrate running a benchmark with analytics tracking"""
    print("🏃 **Benchmark with Analytics Demo:**\n")
    
    # Create benchmark instance
    benchmark = ComprehensiveBenchmark()
    
    # Run a small benchmark
    print("Running small benchmark with analytics tracking...")
    results = benchmark.run_comprehensive_benchmark(
        estimators=['DFA', 'GPH'],
        data_models=['FBM'],
        data_length=500,
        num_trials=3
    )
    
    print("✅ Benchmark completed with analytics tracking")
    print()


def demo_report_generation():
    """Demonstrate report generation"""
    print("📋 **Report Generation Demo:**\n")
    
    # Generate comprehensive report
    print("Generating comprehensive analytics report...")
    try:
        report_path = generate_analytics_report(days=1)
        print(f"✅ Report generated: {report_path}")
    except Exception as e:
        print(f"⚠️ Report generation failed: {e}")
        print("This is normal if no analytics data exists yet")
    
    print()


def main():
    """Main demo function"""
    print("🚀 Starting LRDBench Analytics Demo\n")
    print("=" * 50)
    
    # Run all demos
    demo_basic_analytics()
    demo_usage_tracking()
    demo_performance_monitoring()
    demo_error_tracking()
    demo_workflow_tracking()
    demo_benchmark_with_analytics()
    demo_report_generation()
    
    print("=" * 50)
    print("🎉 **Analytics Demo Completed!**\n")
    
    # Final analytics summary
    print("📊 **Final Analytics Summary:**")
    print(get_analytics_summary(days=1))
    
    print("\n💡 **What You Can Do Next:**")
    print("1. Check the analytics data in ~/.lrdbench/analytics/")
    print("2. Generate detailed reports using generate_analytics_report()")
    print("3. Use the @track_usage, @monitor_performance decorators in your code")
    print("4. Analyze workflow patterns and optimize your processes")
    print("5. Monitor error rates and improve reliability")


if __name__ == "__main__":
    main()
