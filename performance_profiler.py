#!/usr/bin/env python3
"""
Performance Profiling Script for LRDBench

This script profiles the performance of various estimators to identify bottlenecks
and optimization opportunities.
"""

import time
import cProfile
import pstats
import io
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import our estimators
from lrdbench.analysis.temporal.rs.rs_estimator import RSEstimator
from lrdbench.analysis.temporal.dfa.dfa_estimator import DFAEstimator
from lrdbench.analysis.temporal.dma.dma_estimator import DMAEstimator
from lrdbench.analysis.temporal.higuchi.higuchi_estimator import HiguchiEstimator
from lrdbench.analysis.spectral.gph.gph_estimator import GPHEstimator
from lrdbench.analysis.spectral.periodogram.periodogram_estimator import PeriodogramEstimator
from lrdbench.analysis.spectral.whittle.whittle_estimator import WhittleEstimator

# Import data models
from lrdbench.models.data_models.fbm.fbm_model import FractionalBrownianMotion
from lrdbench.models.data_models.fgn.fgn_model import FractionalGaussianNoise


class PerformanceProfiler:
    """Performance profiler for LRDBench estimators."""
    
    def __init__(self):
        self.results = {}
        self.profiling_data = {}
        
    def generate_test_data(self, n_samples: int = 10000, hurst: float = 0.7) -> np.ndarray:
        """Generate test data for profiling."""
        fgn = FractionalGaussianNoise(H=hurst)
        return fgn.generate(n_samples, seed=42)
    
    def profile_estimator(self, estimator, data: np.ndarray, name: str) -> Dict[str, Any]:
        """Profile a single estimator."""
        print(f"Profiling {name}...")
        
        # Time profiling
        start_time = time.time()
        
        # Create profiler
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run estimation
        try:
            result = estimator.estimate(data)
            success = True
        except Exception as e:
            print(f"Error in {name}: {e}")
            result = None
            success = False
        
        profiler.disable()
        
        # Get timing
        execution_time = time.time() - start_time
        
        # Get profiling stats
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        profiling_output = s.getvalue()
        
        return {
            'name': name,
            'execution_time': execution_time,
            'success': success,
            'result': result,
            'profiling_stats': profiling_output,
            'data_size': len(data)
        }
    
    def run_comprehensive_profiling(self, data_sizes: List[int] = None) -> Dict[str, Any]:
        """Run comprehensive profiling on all estimators."""
        if data_sizes is None:
            data_sizes = [1000, 5000, 10000]
        
        estimators = {
            'RS': RSEstimator(),
            'DFA': DFAEstimator(),
            'DMA': DMAEstimator(),
            'Higuchi': HiguchiEstimator(),
            'GPH': GPHEstimator(),
            'Periodogram': PeriodogramEstimator(),
            'Whittle': WhittleEstimator()
        }
        
        all_results = {}
        
        for size in data_sizes:
            print(f"\n{'='*50}")
            print(f"Profiling with data size: {size}")
            print(f"{'='*50}")
            
            # Generate data
            data = self.generate_test_data(size)
            
            size_results = {}
            for name, estimator in estimators.items():
                result = self.profile_estimator(estimator, data, name)
                size_results[name] = result
                
                if result['success']:
                    print(f"✅ {name}: {result['execution_time']:.4f}s")
                else:
                    print(f"❌ {name}: Failed")
            
            all_results[size] = size_results
        
        self.results = all_results
        return all_results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.results:
            return "No profiling results available."
        
        report = []
        report.append("# LRDBench Performance Profiling Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("")
        report.append("| Estimator | 1K | 5K | 10K | Status |")
        report.append("|-----------|----|----|----|--------|")
        
        # Get all estimator names
        all_estimators = set()
        for size_results in self.results.values():
            all_estimators.update(size_results.keys())
        
        for estimator_name in sorted(all_estimators):
            row = [estimator_name]
            success_count = 0
            
            for size in [1000, 5000, 10000]:
                if size in self.results and estimator_name in self.results[size]:
                    result = self.results[size][estimator_name]
                    if result['success']:
                        row.append(f"{result['execution_time']:.4f}s")
                        success_count += 1
                    else:
                        row.append("FAIL")
                else:
                    row.append("N/A")
            
            # Status
            if success_count == 3:
                row.append("✅ All Pass")
            elif success_count > 0:
                row.append(f"⚠️ {success_count}/3 Pass")
            else:
                row.append("❌ All Fail")
            
            report.append("| " + " | ".join(row) + " |")
        
        report.append("")
        
        # Detailed analysis
        report.append("## Detailed Analysis")
        report.append("")
        
        for size in sorted(self.results.keys()):
            report.append(f"### Data Size: {size}")
            report.append("")
            
            for name, result in self.results[size].items():
                report.append(f"#### {name}")
                report.append("")
                report.append(f"- **Execution Time**: {result['execution_time']:.4f}s")
                report.append(f"- **Success**: {result['success']}")
                report.append(f"- **Data Size**: {result['data_size']}")
                
                if result['success'] and result['result']:
                    report.append(f"- **Hurst Estimate**: {result['result'].get('hurst_parameter', 'N/A')}")
                
                report.append("")
                
                # Add profiling stats for slow estimators
                if result['execution_time'] > 0.1:  # More than 100ms
                    report.append("**Top 10 Function Calls:**")
                    report.append("```")
                    report.append(result['profiling_stats'])
                    report.append("```")
                    report.append("")
        
        return "\n".join(report)
    
    def create_performance_plots(self, save_path: str = "performance_plots.png"):
        """Create performance visualization plots."""
        if not self.results:
            print("No results to plot.")
            return
        
        # Prepare data for plotting
        sizes = sorted(self.results.keys())
        estimators = set()
        for size_results in self.results.values():
            estimators.update(size_results.keys())
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Execution time vs data size
        for estimator in sorted(estimators):
            times = []
            for size in sizes:
                if size in self.results and estimator in self.results[size]:
                    result = self.results[size][estimator]
                    if result['success']:
                        times.append(result['execution_time'])
                    else:
                        times.append(np.nan)
                else:
                    times.append(np.nan)
            
            ax1.plot(sizes, times, 'o-', label=estimator, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Data Size')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Data Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Performance comparison (bar chart)
        avg_times = {}
        for estimator in sorted(estimators):
            times = []
            for size in sizes:
                if size in self.results and estimator in self.results[size]:
                    result = self.results[size][estimator]
                    if result['success']:
                        times.append(result['execution_time'])
            
            if times:
                avg_times[estimator] = np.mean(times)
            else:
                avg_times[estimator] = np.nan
        
        estimators_list = list(avg_times.keys())
        avg_times_list = list(avg_times.values())
        
        bars = ax2.bar(estimators_list, avg_times_list, alpha=0.7)
        ax2.set_xlabel('Estimator')
        ax2.set_ylabel('Average Execution Time (seconds)')
        ax2.set_title('Average Performance Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_yscale('log')
        
        # Color bars based on performance
        for i, (estimator, time) in enumerate(zip(estimators_list, avg_times_list)):
            if time < 0.01:
                bars[i].set_color('green')
            elif time < 0.1:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plots saved to {save_path}")
        plt.show()


def main():
    """Main profiling function."""
    print("🚀 Starting LRDBench Performance Profiling")
    print("=" * 50)
    
    profiler = PerformanceProfiler()
    
    # Run comprehensive profiling
    results = profiler.run_comprehensive_profiling([1000, 5000, 10000])
    
    # Generate report
    report = profiler.generate_performance_report()
    
    # Save report
    with open("performance_report.md", "w", encoding='utf-8') as f:
        f.write(report)
    
    print("\n📊 Performance report saved to performance_report.md")
    
    # Create plots
    profiler.create_performance_plots()
    
    print("\n✅ Performance profiling completed!")
    
    return results


if __name__ == "__main__":
    main()
