"""
Visualization Utilities for Fractional Parameter Estimation

This module provides comprehensive visualization tools for:
1. Training curves and convergence analysis
2. Model performance comparisons
3. Data exploration and analysis
4. Benchmark results visualization
5. Interactive plots and dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FractionalVisualizer:
    """
    Comprehensive visualization class for fractional parameter estimation.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
        # Color schemes
        self.colors = {
            'neural': '#1f77b4',
            'classical': '#ff7f0e',
            'ml': '#2ca02c',
            'pinn': '#d62728',
            'pino': '#9467bd',
            'neural_ode': '#8c564b',
            'neural_sde': '#e377c2'
        }
        
        logger.info("FractionalVisualizer initialized")
    
    def plot_training_curves(self, 
                           training_history: Dict[str, List[float]], 
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> plt.Figure:
        """
        Plot training curves for neural models.
        
        Args:
            training_history: Dictionary containing training metrics
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16)
        
        # Training loss
        if 'train_loss' in training_history:
            axes[0, 0].plot(training_history['train_loss'], label='Training Loss', color='blue')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Validation loss
        if 'val_loss' in training_history:
            axes[0, 1].plot(training_history['val_loss'], label='Validation Loss', color='red')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'learning_rate' in training_history:
            axes[1, 0].plot(training_history['learning_rate'], label='Learning Rate', color='green')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined loss plot
        if 'train_loss' in training_history and 'val_loss' in training_history:
            axes[1, 1].plot(training_history['train_loss'], label='Training Loss', color='blue', alpha=0.7)
            axes[1, 1].plot(training_history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
            axes[1, 1].set_title('Training vs Validation Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_model_comparison(self, 
                            results_df: pd.DataFrame,
                            metric: str = 'absolute_error',
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        Create comprehensive model comparison plots.
        
        Args:
            results_df: DataFrame with benchmark results
            metric: Metric to compare ('absolute_error', 'relative_error', 'rmse')
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Performance Comparison - {metric.replace("_", " ").title()}', fontsize=16)
        
        # 1. Overall performance comparison
        ax1 = axes[0, 0]
        performance_summary = results_df.groupby('estimator')[metric].mean().sort_values()
        colors = [self.colors.get(est.split('_')[0].lower(), 'gray') for est in performance_summary.index]
        performance_summary.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Mean Performance by Estimator')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Performance by model type
        ax2 = axes[0, 1]
        model_performance = results_df.groupby('model_type')[metric].mean()
        model_colors = [self.colors.get(model_type, 'gray') for model_type in model_performance.index]
        model_performance.plot(kind='bar', ax=ax2, color=model_colors)
        ax2.set_title('Performance by Model Type')
        ax2.set_ylabel(metric.replace('_', ' ').title())
        
        # 3. Box plot of errors
        ax3 = axes[0, 2]
        results_df.boxplot(column=metric, by='model_type', ax=ax3)
        ax3.set_title('Error Distribution by Model Type')
        ax3.set_xlabel('Model Type')
        ax3.set_ylabel(metric.replace('_', ' ').title())
        
        # 4. True vs Estimated scatter plot
        ax4 = axes[1, 0]
        for model_type in results_df['model_type'].unique():
            subset = results_df[results_df['model_type'] == model_type]
            ax4.scatter(subset['true_hurst'], subset['estimated_hurst'], 
                       alpha=0.6, label=model_type, s=20, color=self.colors.get(model_type, 'gray'))
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_title('True vs Estimated Hurst Exponent')
        ax4.set_xlabel('True Hurst')
        ax4.set_ylabel('Estimated Hurst')
        ax4.legend()
        
        # 5. Performance by data type
        ax5 = axes[1, 1]
        if 'config' in results_df.columns:
            results_df['data_type'] = results_df['config'].str.split('_').str[0]
            data_performance = results_df.groupby('data_type')[metric].mean()
            data_performance.plot(kind='bar', ax=ax5, color='orange')
            ax5.set_title('Performance by Data Type')
            ax5.set_ylabel(metric.replace('_', ' ').title())
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Performance by contamination
        ax6 = axes[1, 2]
        if 'config' in results_df.columns:
            results_df['contamination'] = results_df['config'].str.split('_').str[-1]
            contam_performance = results_df.groupby('contamination')[metric].mean()
            contam_performance.plot(kind='bar', ax=ax6, color='purple')
            ax6.set_title('Performance by Contamination Type')
            ax6.set_ylabel(metric.replace('_', ' ').title())
            ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_data_exploration(self, 
                            time_series: np.ndarray,
                            hurst: float,
                            title: str = "Time Series Analysis",
                            save_path: Optional[str] = None,
                            show_plot: bool = True) -> plt.Figure:
        """
        Create comprehensive time series exploration plots.
        
        Args:
            time_series: Time series data
            hurst: True Hurst exponent
            title: Plot title
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{title} (H={hurst:.3f})', fontsize=16)
        
        # 1. Time series plot
        ax1 = axes[0, 0]
        ax1.plot(time_series, alpha=0.7)
        ax1.set_title('Time Series')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram
        ax2 = axes[0, 1]
        ax2.hist(time_series, bins=50, alpha=0.7, density=True)
        ax2.set_title('Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.grid(True, alpha=0.3)
        
        # 3. Autocorrelation
        ax3 = axes[0, 2]
        max_lag = min(50, len(time_series) // 4)
        lags = range(1, max_lag + 1)
        autocorr = [np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1] for lag in lags]
        ax3.plot(lags, autocorr, 'o-', alpha=0.7)
        ax3.set_title('Autocorrelation')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('Autocorrelation')
        ax3.grid(True, alpha=0.3)
        
        # 4. Power spectral density
        ax4 = axes[1, 0]
        fft = np.fft.fft(time_series)
        freqs = np.fft.fftfreq(len(time_series))
        psd = np.abs(fft)**2
        positive_freqs = freqs > 0
        ax4.loglog(freqs[positive_freqs], psd[positive_freqs], alpha=0.7)
        ax4.set_title('Power Spectral Density')
        ax4.set_xlabel('Frequency')
        ax4.set_ylabel('Power')
        ax4.grid(True, alpha=0.3)
        
        # 5. Increments
        ax5 = axes[1, 1]
        increments = np.diff(time_series)
        ax5.plot(increments, alpha=0.7)
        ax5.set_title('Increments')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Increment')
        ax5.grid(True, alpha=0.3)
        
        # 6. Increment distribution
        ax6 = axes[1, 2]
        ax6.hist(increments, bins=50, alpha=0.7, density=True)
        ax6.set_title('Increment Distribution')
        ax6.set_xlabel('Increment')
        ax6.set_ylabel('Density')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data exploration saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_benchmark_summary(self, 
                             benchmark_results: Dict[str, Any],
                             save_path: Optional[str] = None,
                             show_plot: bool = True) -> plt.Figure:
        """
        Create comprehensive benchmark summary visualization.
        
        Args:
            benchmark_results: Results from performance benchmark
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        df = benchmark_results['all_results']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Comprehensive Benchmark Summary', fontsize=16)
        
        # 1. Overall performance ranking
        ax1 = axes[0, 0]
        performance_ranking = df.groupby('estimator')['absolute_error'].mean().sort_values()
        colors = [self.colors.get(est.split('_')[0].lower(), 'gray') for est in performance_ranking.index]
        performance_ranking.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_title('Estimator Performance Ranking')
        ax1.set_xlabel('Mean Absolute Error')
        
        # 2. Performance by model type with error bars
        ax2 = axes[0, 1]
        model_stats = df.groupby('model_type')['absolute_error'].agg(['mean', 'std']).sort_values('mean')
        model_stats['mean'].plot(kind='bar', ax=ax2, yerr=model_stats['std'], 
                               capsize=5, color=[self.colors.get(mt, 'gray') for mt in model_stats.index])
        ax2.set_title('Performance by Model Type')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Error distribution comparison
        ax3 = axes[1, 0]
        for model_type in df['model_type'].unique():
            subset = df[df['model_type'] == model_type]['absolute_error']
            ax3.hist(subset, alpha=0.6, label=model_type, bins=30, 
                    color=self.colors.get(model_type, 'gray'))
        ax3.set_title('Error Distribution Comparison')
        ax3.set_xlabel('Absolute Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. Performance vs data length
        ax4 = axes[1, 1]
        if 'config' in df.columns:
            df['n_points'] = df['config'].str.extract(r'n(\d+)').astype(int)
            length_performance = df.groupby('n_points')['absolute_error'].mean()
            ax4.plot(length_performance.index, length_performance.values, 'o-', linewidth=2, markersize=8)
            ax4.set_title('Performance vs Data Length')
            ax4.set_xlabel('Number of Points')
            ax4.set_ylabel('Mean Absolute Error')
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance vs Hurst value
        ax5 = axes[2, 0]
        hurst_performance = df.groupby('true_hurst')['absolute_error'].mean()
        ax5.plot(hurst_performance.index, hurst_performance.values, 'o-', linewidth=2, markersize=8)
        ax5.set_title('Performance vs Hurst Value')
        ax5.set_xlabel('True Hurst Exponent')
        ax5.set_ylabel('Mean Absolute Error')
        ax5.grid(True, alpha=0.3)
        
        # 6. Robustness analysis (performance under contamination)
        ax6 = axes[2, 1]
        if 'config' in df.columns:
            df['contamination'] = df['config'].str.split('_').str[-1]
            contam_performance = df.groupby('contamination')['absolute_error'].mean()
            contam_performance.plot(kind='bar', ax=ax6, color='purple')
            ax6.set_title('Robustness Analysis')
            ax6.set_ylabel('Mean Absolute Error')
            ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Benchmark summary saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   benchmark_results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Plotly dashboard for benchmark results.
        
        Args:
            benchmark_results: Results from performance benchmark
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure
        """
        df = benchmark_results['all_results']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Performance Ranking', 'Model Type Comparison',
                'Error Distribution', 'True vs Estimated',
                'Performance by Data Length', 'Robustness Analysis'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Performance ranking
        performance_ranking = df.groupby('estimator')['absolute_error'].mean().sort_values()
        fig.add_trace(
            go.Bar(x=performance_ranking.values, y=performance_ranking.index, 
                   orientation='h', name='Performance Ranking'),
            row=1, col=1
        )
        
        # 2. Model type comparison
        model_performance = df.groupby('model_type')['absolute_error'].mean()
        fig.add_trace(
            go.Bar(x=model_performance.index, y=model_performance.values, name='Model Type'),
            row=1, col=2
        )
        
        # 3. Error distribution
        for model_type in df['model_type'].unique():
            subset = df[df['model_type'] == model_type]['absolute_error']
            fig.add_trace(
                go.Histogram(x=subset, name=model_type, opacity=0.7),
                row=2, col=1
            )
        
        # 4. True vs Estimated scatter
        for model_type in df['model_type'].unique():
            subset = df[df['model_type'] == model_type]
            fig.add_trace(
                go.Scatter(x=subset['true_hurst'], y=subset['estimated_hurst'],
                          mode='markers', name=model_type, opacity=0.6),
                row=2, col=2
            )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Prediction',
                      line=dict(dash='dash', color='black')),
            row=2, col=2
        )
        
        # 5. Performance by data length
        if 'config' in df.columns:
            df['n_points'] = df['config'].str.extract(r'n(\d+)').astype(int)
            length_performance = df.groupby('n_points')['absolute_error'].mean()
            fig.add_trace(
                go.Scatter(x=length_performance.index, y=length_performance.values,
                          mode='lines+markers', name='Data Length'),
                row=3, col=1
            )
        
        # 6. Robustness analysis
        if 'config' in df.columns:
            df['contamination'] = df['config'].str.split('_').str[-1]
            contam_performance = df.groupby('contamination')['absolute_error'].mean()
            fig.add_trace(
                go.Bar(x=contam_performance.index, y=contam_performance.values, name='Contamination'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Benchmark Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Mean Absolute Error", row=1, col=1)
        fig.update_xaxes(title_text="Model Type", row=1, col=2)
        fig.update_xaxes(title_text="Absolute Error", row=2, col=1)
        fig.update_xaxes(title_text="True Hurst", row=2, col=2)
        fig.update_xaxes(title_text="Number of Points", row=3, col=1)
        fig.update_xaxes(title_text="Contamination Type", row=3, col=2)
        
        fig.update_yaxes(title_text="Estimator", row=1, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Estimated Hurst", row=2, col=2)
        fig.update_yaxes(title_text="Mean Absolute Error", row=3, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error", row=3, col=2)
        
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=False)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, 
                              feature_importance: Dict[str, float],
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> plt.Figure:
        """
        Plot feature importance for ML models.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance for Hurst Exponent Estimation')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_confidence_intervals(self, 
                                true_values: np.ndarray,
                                estimated_values: np.ndarray,
                                confidence_level: float = 0.95,
                                save_path: Optional[str] = None,
                                show_plot: bool = True) -> plt.Figure:
        """
        Plot confidence intervals for estimates.
        
        Args:
            true_values: True Hurst exponents
            estimated_values: Estimated Hurst exponents
            confidence_level: Confidence level for intervals
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate confidence intervals
        errors = estimated_values - true_values
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Calculate confidence interval
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_width = z_score * std_error / np.sqrt(len(errors))
        
        # Plot true vs estimated
        ax.scatter(true_values, estimated_values, alpha=0.6, s=30)
        
        # Add perfect prediction line
        min_val = min(true_values.min(), estimated_values.min())
        max_val = max(true_values.max(), estimated_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        
        # Add confidence bands
        x_range = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_range, x_range + mean_error - ci_width, x_range + mean_error + ci_width,
                       alpha=0.3, color='red', label=f'{confidence_level*100:.0f}% Confidence Interval')
        ax.plot(x_range, x_range + mean_error, 'r-', alpha=0.7, label=f'Mean Error: {mean_error:.3f}')
        
        ax.set_xlabel('True Hurst Exponent')
        ax.set_ylabel('Estimated Hurst Exponent')
        ax.set_title('Estimation Accuracy with Confidence Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence intervals plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig


def create_quick_comparison_plot(results_df: pd.DataFrame, 
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a quick comparison plot for benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    visualizer = FractionalVisualizer()
    return visualizer.plot_model_comparison(results_df, save_path=save_path)


def create_training_visualization(training_history: Dict[str, List[float]], 
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create training visualization.
    
    Args:
        training_history: Training history dictionary
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    visualizer = FractionalVisualizer()
    return visualizer.plot_training_curves(training_history, save_path=save_path)


if __name__ == "__main__":
    # Example usage
    print("Fractional Parameter Estimation Visualization Utilities")
    print("=" * 60)
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = np.random.randn(1000)
    
    visualizer = FractionalVisualizer()
    
    # Example: Plot data exploration
    visualizer.plot_data_exploration(
        sample_data, 
        hurst=0.7, 
        title="Sample Time Series Analysis",
        save_path="sample_data_exploration.png"
    )
    
    print("Visualization utilities ready for use!")
