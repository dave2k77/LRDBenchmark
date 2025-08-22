"""
Benchmarking Utilities for Fractional Parameter Estimation

This module provides automated benchmarking utilities for:
1. Standardized evaluation metrics
2. Statistical significance testing
3. Automated benchmark execution
4. Performance ranking and analysis
5. Robustness testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkMetrics:
    """
    Standardized evaluation metrics for fractional parameter estimation.
    """
    
    @staticmethod
    def calculate_metrics(true_values: np.ndarray, 
                         estimated_values: np.ndarray,
                         confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            true_values: True Hurst exponents
            estimated_values: Estimated Hurst exponents
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary of metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(true_values) | np.isnan(estimated_values))
        true_clean = true_values[mask]
        est_clean = estimated_values[mask]
        
        if len(true_clean) == 0:
            return {
                'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan,
                'bias': np.nan, 'std_error': np.nan, 'correlation': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan
            }
        
        # Basic metrics
        mae = mean_absolute_error(true_clean, est_clean)
        rmse = np.sqrt(mean_squared_error(true_clean, est_clean))
        mape = np.mean(np.abs((true_clean - est_clean) / true_clean)) * 100
        
        # R-squared
        r2 = r2_score(true_clean, est_clean)
        
        # Bias and standard error
        bias = np.mean(est_clean - true_clean)
        std_error = np.std(est_clean - true_clean)
        
        # Correlation
        correlation = np.corrcoef(true_clean, est_clean)[0, 1]
        
        # Confidence interval for bias
        n = len(true_clean)
        t_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        ci_width = t_value * std_error / np.sqrt(n)
        ci_lower = bias - ci_width
        ci_upper = bias + ci_width
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'bias': bias,
            'std_error': std_error,
            'correlation': correlation,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': n
        }
    
    @staticmethod
    def calculate_robustness_metrics(true_values: np.ndarray,
                                   estimated_values: np.ndarray,
                                   contamination_types: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate robustness metrics for different contamination types.
        
        Args:
            true_values: True Hurst exponents
            estimated_values: Estimated Hurst exponents
            contamination_types: List of contamination type labels
            
        Returns:
            Dictionary of metrics by contamination type
        """
        robustness_metrics = {}
        
        for contam_type in np.unique(contamination_types):
            mask = contamination_types == contam_type
            if np.sum(mask) > 0:
                metrics = BenchmarkMetrics.calculate_metrics(
                    true_values[mask], estimated_values[mask]
                )
                robustness_metrics[contam_type] = metrics
        
        return robustness_metrics


class StatisticalTesting:
    """
    Statistical testing utilities for benchmark comparisons.
    """
    
    @staticmethod
    def paired_t_test(estimator1_errors: np.ndarray, 
                     estimator2_errors: np.ndarray,
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform paired t-test between two estimators.
        
        Args:
            estimator1_errors: Errors from first estimator
            estimator2_errors: Errors from second estimator
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        mask = ~(np.isnan(estimator1_errors) | np.isnan(estimator2_errors))
        errors1 = estimator1_errors[mask]
        errors2 = estimator2_errors[mask]
        
        if len(errors1) < 2:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'effect_size': np.nan
            }
        
        # Paired t-test
        statistic, p_value = stats.ttest_rel(errors1, errors2)
        
        # Effect size (Cohen's d)
        d = np.mean(errors1 - errors2) / np.std(errors1 - errors2, ddof=1)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': d,
            'n_samples': len(errors1)
        }
    
    @staticmethod
    def mann_whitney_test(estimator1_errors: np.ndarray,
                         estimator2_errors: np.ndarray,
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test between two estimators.
        
        Args:
            estimator1_errors: Errors from first estimator
            estimator2_errors: Errors from second estimator
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        mask1 = ~np.isnan(estimator1_errors)
        mask2 = ~np.isnan(estimator2_errors)
        errors1 = estimator1_errors[mask1]
        errors2 = estimator2_errors[mask2]
        
        if len(errors1) < 1 or len(errors2) < 1:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False
            }
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(errors1, errors2, alternative='two-sided')
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'n_samples_1': len(errors1),
            'n_samples_2': len(errors2)
        }
    
    @staticmethod
    def friedman_test(estimator_errors: Dict[str, np.ndarray],
                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Friedman test for multiple estimators.
        
        Args:
            estimator_errors: Dictionary of errors for each estimator
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        # Prepare data for Friedman test
        estimators = list(estimator_errors.keys())
        min_samples = min(len(errors) for errors in estimator_errors.values())
        
        # Create aligned arrays
        aligned_errors = []
        for i in range(min_samples):
            row = []
            for estimator in estimators:
                errors = estimator_errors[estimator]
                if i < len(errors) and not np.isnan(errors[i]):
                    row.append(errors[i])
                else:
                    row.append(np.nan)
            if not any(np.isnan(row)):
                aligned_errors.append(row)
        
        if len(aligned_errors) < 2:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False
            }
        
        aligned_errors = np.array(aligned_errors)
        
        # Friedman test
        statistic, p_value = stats.friedmanchisquare(*aligned_errors.T)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'n_samples': len(aligned_errors),
            'n_estimators': len(estimators)
        }


class AutomatedBenchmark:
    """
    Automated benchmark execution and analysis.
    """
    
    def __init__(self, 
                 save_results: bool = True,
                 results_dir: str = "benchmark_results",
                 n_jobs: int = -1):
        """
        Initialize automated benchmark.
        
        Args:
            save_results: Whether to save results
            results_dir: Directory for results
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.save_results = save_results
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set number of jobs
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = min(n_jobs, mp.cpu_count())
        
        logger.info(f"AutomatedBenchmark initialized with {self.n_jobs} jobs")
    
    def run_estimator_benchmark(self,
                               estimator_func: callable,
                               test_data: List[Dict[str, Any]],
                               estimator_name: str,
                               **kwargs) -> Dict[str, Any]:
        """
        Run benchmark for a single estimator.
        
        Args:
            estimator_func: Function that estimates Hurst exponent
            test_data: List of test data dictionaries
            estimator_name: Name of the estimator
            **kwargs: Additional arguments for estimator
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running benchmark for {estimator_name}")
        start_time = time.time()
        
        results = {
            'estimator': estimator_name,
            'true_hurst': [],
            'estimated_hurst': [],
            'estimation_time': [],
            'errors': [],
            'successful_estimations': 0,
            'total_estimations': len(test_data)
        }
        
        for i, data in enumerate(test_data):
            try:
                # Time the estimation
                est_start = time.time()
                estimated_hurst = estimator_func(data['time_series'], **kwargs)
                est_time = time.time() - est_start
                
                if estimated_hurst is not None and not np.isnan(estimated_hurst):
                    results['true_hurst'].append(data['true_hurst'])
                    results['estimated_hurst'].append(estimated_hurst)
                    results['estimation_time'].append(est_time)
                    results['errors'].append(abs(estimated_hurst - data['true_hurst']))
                    results['successful_estimations'] += 1
                
            except Exception as e:
                logger.warning(f"Estimation failed for {estimator_name} on sample {i}: {e}")
                continue
        
        # Calculate metrics
        if results['successful_estimations'] > 0:
            metrics = BenchmarkMetrics.calculate_metrics(
                np.array(results['true_hurst']),
                np.array(results['estimated_hurst'])
            )
            results.update(metrics)
        else:
            results.update({
                'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2': np.nan,
                'bias': np.nan, 'std_error': np.nan, 'correlation': np.nan
            })
        
        results['total_time'] = time.time() - start_time
        results['success_rate'] = results['successful_estimations'] / results['total_estimations']
        
        logger.info(f"Completed {estimator_name}: {results['successful_estimations']}/{results['total_estimations']} successful")
        
        return results
    
    def run_parallel_benchmark(self,
                              estimators: Dict[str, callable],
                              test_data: List[Dict[str, Any]],
                              **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmark for multiple estimators in parallel.
        
        Args:
            estimators: Dictionary of estimator names and functions
            test_data: List of test data dictionaries
            **kwargs: Additional arguments for estimators
            
        Returns:
            Dictionary of benchmark results for each estimator
        """
        logger.info(f"Running parallel benchmark for {len(estimators)} estimators")
        
        results = {}
        
        if self.n_jobs == 1:
            # Sequential execution
            for name, estimator_func in estimators.items():
                results[name] = self.run_estimator_benchmark(
                    estimator_func, test_data, name, **kwargs
                )
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_estimator = {}
                
                for name, estimator_func in estimators.items():
                    future = executor.submit(
                        self.run_estimator_benchmark,
                        estimator_func, test_data, name, **kwargs
                    )
                    future_to_estimator[future] = name
                
                for future in as_completed(future_to_estimator):
                    estimator_name = future_to_estimator[future]
                    try:
                        results[estimator_name] = future.result()
                    except Exception as e:
                        logger.error(f"Benchmark failed for {estimator_name}: {e}")
                        results[estimator_name] = {
                            'estimator': estimator_name,
                            'error': str(e),
                            'successful_estimations': 0,
                            'total_estimations': len(test_data)
                        }
        
        return results
    
    def create_performance_ranking(self, 
                                 benchmark_results: Dict[str, Dict[str, Any]],
                                 metric: str = 'mae') -> pd.DataFrame:
        """
        Create performance ranking based on specified metric.
        
        Args:
            benchmark_results: Results from benchmark
            metric: Metric to rank by ('mae', 'rmse', 'mape', 'r2')
            
        Returns:
            DataFrame with performance ranking
        """
        ranking_data = []
        
        for estimator_name, results in benchmark_results.items():
            if 'error' not in results:  # Skip failed estimators
                ranking_data.append({
                    'estimator': estimator_name,
                    metric: results.get(metric, np.nan),
                    'success_rate': results.get('success_rate', 0),
                    'total_time': results.get('total_time', np.nan),
                    'n_samples': results.get('n_samples', 0)
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        if metric in ['mae', 'rmse', 'mape']:
            # Lower is better
            ranking_df = ranking_df.sort_values(metric)
        else:
            # Higher is better (e.g., R²)
            ranking_df = ranking_df.sort_values(metric, ascending=False)
        
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df
    
    def perform_statistical_analysis(self,
                                   benchmark_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            benchmark_results: Results from benchmark
            
        Returns:
            Dictionary with statistical analysis results
        """
        logger.info("Performing statistical analysis")
        
        # Prepare error data
        estimator_errors = {}
        for estimator_name, results in benchmark_results.items():
            if 'error' not in results and 'errors' in results:
                estimator_errors[estimator_name] = np.array(results['errors'])
        
        if len(estimator_errors) < 2:
            logger.warning("Insufficient data for statistical analysis")
            return {}
        
        # Friedman test for overall differences
        friedman_result = StatisticalTesting.friedman_test(estimator_errors)
        
        # Pairwise comparisons
        estimators = list(estimator_errors.keys())
        pairwise_results = {}
        
        for i, est1 in enumerate(estimators):
            for j, est2 in enumerate(estimators):
                if i < j:
                    pair_name = f"{est1}_vs_{est2}"
                    
                    # Paired t-test
                    t_test_result = StatisticalTesting.paired_t_test(
                        estimator_errors[est1], estimator_errors[est2]
                    )
                    
                    # Mann-Whitney test
                    mw_test_result = StatisticalTesting.mann_whitney_test(
                        estimator_errors[est1], estimator_errors[est2]
                    )
                    
                    pairwise_results[pair_name] = {
                        'paired_t_test': t_test_result,
                        'mann_whitney_test': mw_test_result
                    }
        
        return {
            'friedman_test': friedman_result,
            'pairwise_tests': pairwise_results,
            'n_estimators': len(estimators),
            'n_samples': len(next(iter(estimator_errors.values())))
        }
    
    def generate_benchmark_report(self,
                                benchmark_results: Dict[str, Dict[str, Any]],
                                statistical_analysis: Dict[str, Any],
                                save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            benchmark_results: Results from benchmark
            statistical_analysis: Statistical analysis results
            save_path: Path to save report
            
        Returns:
            String containing the report
        """
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance ranking
        report.append("PERFORMANCE RANKING")
        report.append("-" * 40)
        for metric in ['mae', 'rmse', 'mape', 'r2']:
            ranking = self.create_performance_ranking(benchmark_results, metric)
            report.append(f"\n{metric.upper()} Ranking:")
            report.append(ranking[['estimator', metric, 'success_rate', 'rank']].to_string(index=False))
        
        # Statistical analysis
        if statistical_analysis:
            report.append("\n\nSTATISTICAL ANALYSIS")
            report.append("-" * 40)
            
            # Friedman test
            friedman = statistical_analysis.get('friedman_test', {})
            if friedman:
                report.append(f"Friedman Test:")
                report.append(f"  Statistic: {friedman.get('statistic', 'N/A'):.4f}")
                report.append(f"  p-value: {friedman.get('p_value', 'N/A'):.4f}")
                report.append(f"  Significant: {friedman.get('significant', 'N/A')}")
            
            # Pairwise tests
            pairwise = statistical_analysis.get('pairwise_tests', {})
            if pairwise:
                report.append(f"\nPairwise Comparisons:")
                for pair_name, tests in pairwise.items():
                    report.append(f"\n  {pair_name}:")
                    t_test = tests.get('paired_t_test', {})
                    if t_test.get('significant'):
                        report.append(f"    Paired t-test: p={t_test.get('p_value', 'N/A'):.4f} (SIGNIFICANT)")
                    else:
                        report.append(f"    Paired t-test: p={t_test.get('p_value', 'N/A'):.4f}")
        
        # Detailed results
        report.append("\n\nDETAILED RESULTS")
        report.append("-" * 40)
        for estimator_name, results in benchmark_results.items():
            report.append(f"\n{estimator_name}:")
            if 'error' in results:
                report.append(f"  Error: {results['error']}")
            else:
                report.append(f"  MAE: {results.get('mae', 'N/A'):.4f}")
                report.append(f"  RMSE: {results.get('rmse', 'N/A'):.4f}")
                report.append(f"  R²: {results.get('r2', 'N/A'):.4f}")
                report.append(f"  Success Rate: {results.get('success_rate', 'N/A'):.2%}")
                report.append(f"  Total Time: {results.get('total_time', 'N/A'):.2f}s")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Benchmark report saved to {save_path}")
        
        return report_text


class RobustnessTester:
    """
    Robustness testing utilities for estimators.
    """
    
    def __init__(self, data_generator):
        """
        Initialize robustness tester.
        
        Args:
            data_generator: Data generator instance
        """
        self.data_generator = data_generator
    
    def test_noise_robustness(self,
                            estimator_func: callable,
                            base_data: Dict[str, Any],
                            noise_levels: List[float],
                            n_trials: int = 10) -> Dict[str, List[float]]:
        """
        Test robustness to different noise levels.
        
        Args:
            estimator_func: Function that estimates Hurst exponent
            base_data: Base time series data
            noise_levels: List of noise levels to test
            n_trials: Number of trials per noise level
            
        Returns:
            Dictionary with results for each noise level
        """
        results = {level: [] for level in noise_levels}
        true_hurst = base_data['true_hurst']
        
        for noise_level in noise_levels:
            for _ in range(n_trials):
                # Add noise
                noisy_data = self.data_generator.apply_contamination(
                    base_data, contamination_type='noise', noise_level=noise_level
                )
                
                try:
                    estimated_hurst = estimator_func(noisy_data['time_series'])
                    if estimated_hurst is not None and not np.isnan(estimated_hurst):
                        error = abs(estimated_hurst - true_hurst)
                        results[noise_level].append(error)
                except:
                    continue
        
        return results
    
    def test_outlier_robustness(self,
                               estimator_func: callable,
                               base_data: Dict[str, Any],
                               outlier_fractions: List[float],
                               n_trials: int = 10) -> Dict[str, List[float]]:
        """
        Test robustness to different outlier fractions.
        
        Args:
            estimator_func: Function that estimates Hurst exponent
            base_data: Base time series data
            outlier_fractions: List of outlier fractions to test
            n_trials: Number of trials per fraction
            
        Returns:
            Dictionary with results for each outlier fraction
        """
        results = {fraction: [] for fraction in outlier_fractions}
        true_hurst = base_data['true_hurst']
        
        for fraction in outlier_fractions:
            for _ in range(n_trials):
                # Add outliers
                outlier_data = self.data_generator.apply_contamination(
                    base_data, contamination_type='outliers', outlier_fraction=fraction
                )
                
                try:
                    estimated_hurst = estimator_func(outlier_data['time_series'])
                    if estimated_hurst is not None and not np.isnan(estimated_hurst):
                        error = abs(estimated_hurst - true_hurst)
                        results[fraction].append(error)
                except:
                    continue
        
        return results


def quick_benchmark(estimators: Dict[str, callable],
                   test_data: List[Dict[str, Any]],
                   save_results: bool = True,
                   results_dir: str = "quick_benchmark_results") -> Dict[str, Any]:
    """
    Quick benchmark function for simple comparisons.
    
    Args:
        estimators: Dictionary of estimator names and functions
        test_data: List of test data dictionaries
        save_results: Whether to save results
        results_dir: Directory for results
        
    Returns:
        Dictionary with benchmark results
    """
    benchmark = AutomatedBenchmark(save_results=save_results, results_dir=results_dir)
    
    # Run benchmark
    results = benchmark.run_parallel_benchmark(estimators, test_data)
    
    # Create ranking
    ranking = benchmark.create_performance_ranking(results)
    
    # Statistical analysis
    stats_analysis = benchmark.perform_statistical_analysis(results)
    
    # Generate report
    report = benchmark.generate_benchmark_report(results, stats_analysis)
    
    return {
        'results': results,
        'ranking': ranking,
        'statistical_analysis': stats_analysis,
        'report': report
    }


if __name__ == "__main__":
    # Example usage
    print("Benchmarking Utilities for Fractional Parameter Estimation")
    print("=" * 60)
    
    # Example estimator function
    def dummy_estimator(time_series, **kwargs):
        """Dummy estimator for testing."""
        return np.random.uniform(0.1, 0.9)
    
    # Example test data
    test_data = [
        {'time_series': np.random.randn(1000), 'true_hurst': 0.7},
        {'time_series': np.random.randn(1000), 'true_hurst': 0.5},
        {'time_series': np.random.randn(1000), 'true_hurst': 0.3}
    ]
    
    # Example estimators
    estimators = {
        'dummy_1': dummy_estimator,
        'dummy_2': dummy_estimator
    }
    
    # Run quick benchmark
    benchmark_results = quick_benchmark(estimators, test_data)
    
    print("Benchmark completed!")
    print(f"Results saved to: {benchmark_results['ranking']}")
