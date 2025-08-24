# LRDBench Analytics System

The LRDBench Analytics System provides comprehensive tracking and analysis of how LRDBench is used in production environments. It helps developers and users understand usage patterns, identify performance bottlenecks, track errors, and optimize workflows.

## 🚀 Quick Start

```python
from lrdbench import (
    enable_analytics,
    get_analytics_summary,
    generate_analytics_report
)

# Enable analytics (enabled by default)
enable_analytics(True, privacy_mode=True)

# Get quick summary
summary = get_analytics_summary(days=30)
print(summary)

# Generate comprehensive report
report_path = generate_analytics_report(days=30)
```

## 📊 What Gets Tracked

### 1. **Usage Tracking**
- Which estimators are used most frequently
- Parameter combinations and common values
- Data length distributions
- Success/failure rates
- User session patterns

### 2. **Performance Monitoring**
- Execution times for each estimator
- Memory usage patterns
- CPU utilization
- Performance trends over time
- Bottleneck identification

### 3. **Error Analysis**
- Error types and frequencies
- Failure modes by estimator
- Error correlation analysis
- Reliability scoring
- Improvement recommendations

### 4. **Workflow Analysis**
- User workflow patterns
- Common estimator sequences
- Workflow complexity distribution
- Optimization opportunities
- Feature usage analysis

## 🛠️ Usage Examples

### Basic Usage Tracking

```python
from lrdbench import track_usage

@track_usage("my_estimator", data_length=1000)
def my_custom_estimator(data, window_size=10):
    # Your estimator logic here
    return result
```

### Performance Monitoring

```python
from lrdbench import monitor_performance

@monitor_performance("my_function")
def my_function(data):
    # Your function logic here
    return result
```

### Error Tracking

```python
from lrdbench import track_errors

@track_errors("my_estimator")
def my_estimator(data):
    try:
        # Your estimator logic here
        return result
    except Exception as e:
        # Error will be automatically tracked
        raise
```

### Workflow Tracking

```python
from lrdbench import track_workflow

@track_workflow("data_preprocessing")
def preprocess_data(data):
    return processed_data

@track_workflow("estimation")
def estimate_parameter(data):
    return parameter

# Run workflow
data = load_data()
processed = preprocess_data(data)
result = estimate_parameter(processed)
```

## 📈 Analytics Dashboard

The `AnalyticsDashboard` class provides a unified interface for accessing all analytics data:

```python
from lrdbench import get_analytics_dashboard

dashboard = get_analytics_dashboard()

# Get comprehensive summary
summary = dashboard.get_comprehensive_summary(days=30)

# Generate individual reports
usage_report = dashboard.generate_usage_report(days=30)
performance_report = dashboard.generate_performance_report(days=30)
reliability_report = dashboard.generate_reliability_report(days=30)
workflow_report = dashboard.generate_workflow_report(days=30)

# Generate comprehensive report
comprehensive_report = dashboard.generate_comprehensive_report(days=30)

# Create visualizations
plots = dashboard.create_visualizations(days=30)

# Export all data
exports = dashboard.export_all_data(days=30)
```

## 🔧 Configuration

### Storage Location
Analytics data is stored in `~/.lrdbench/analytics/` by default. You can customize this:

```python
from lrdbench.analytics import UsageTracker

tracker = UsageTracker(storage_path="/custom/path/analytics")
```

### Privacy Settings
The analytics system includes privacy-preserving features:

```python
from lrdbench import enable_analytics

# Enable with privacy mode (default)
enable_analytics(True, privacy_mode=True)

# Disable privacy mode for detailed tracking
enable_analytics(True, privacy_mode=False)
```

### Disabling Analytics
To completely disable analytics:

```python
from lrdbench import enable_analytics

enable_analytics(False)
```

## 📋 Data Structure

### Usage Events
```python
@dataclass
class UsageEvent:
    timestamp: str
    event_type: str
    estimator_name: str
    parameters: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str]
    data_length: int
    user_id: Optional[str]
    session_id: str
```

### Performance Metrics
```python
@dataclass
class PerformanceMetrics:
    timestamp: str
    estimator_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    data_length: int
    parameters: Dict[str, str]
```

### Error Events
```python
@dataclass
class ErrorEvent:
    timestamp: str
    estimator_name: str
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    parameters: Dict[str, str]
    data_length: int
    user_id: Optional[str]
    session_id: str
```

## 📊 Report Generation

### Usage Report
- Total events and unique users
- Most popular estimators
- Parameter usage patterns
- Data length distributions
- Common errors

### Performance Report
- Execution time statistics
- Memory usage patterns
- Performance trends
- Bottleneck identification

### Reliability Report
- Error rates and types
- Failure mode analysis
- Reliability scoring
- Improvement recommendations

### Workflow Report
- Workflow patterns and complexity
- Popular estimator sequences
- Feature usage analysis
- Optimization recommendations

## 🎯 Use Cases

### 1. **Development and Testing**
- Track which estimators are most used during development
- Monitor performance during testing
- Identify common failure modes
- Optimize development workflows

### 2. **Production Monitoring**
- Monitor real-world usage patterns
- Track performance in production environments
- Identify reliability issues
- Optimize user workflows

### 3. **Research and Analysis**
- Analyze estimator popularity
- Study parameter usage patterns
- Investigate performance characteristics
- Understand user behavior

### 4. **Quality Assurance**
- Monitor error rates
- Track reliability metrics
- Identify improvement opportunities
- Validate fixes and optimizations

## 🔒 Privacy and Security

### Data Protection
- User IDs are hashed by default
- Sensitive parameters are sanitized
- Data is stored locally by default
- No personal information is collected

### Compliance
- GDPR compliant data handling
- User consent for data collection
- Easy data deletion
- Transparent data usage

## 🚀 Advanced Features

### Custom Analytics
```python
from lrdbench.analytics import UsageTracker

tracker = UsageTracker()

# Custom tracking
tracker.track_estimator_usage(
    estimator_name="custom_estimator",
    parameters={"window_size": 20},
    execution_time=1.5,
    success=True,
    data_length=1000
)
```

### Batch Analysis
```python
from lrdbench.analytics import get_analytics_dashboard

dashboard = get_analytics_dashboard()

# Export data for external analysis
exports = dashboard.export_all_data(days=90)

# Create custom visualizations
plots = dashboard.create_visualizations(days=30)
```

### Integration with External Tools
```python
# Export to JSON for external analysis
dashboard.export_all_data(output_dir="./analytics_export")

# Generate reports in markdown format
dashboard.generate_comprehensive_report(output_dir="./reports")
```

## 📚 API Reference

### Core Functions
- `enable_analytics(enable, privacy_mode)` - Enable/disable analytics
- `get_analytics_summary(days)` - Get quick summary
- `generate_analytics_report(days)` - Generate comprehensive report

### Decorators
- `@track_usage(estimator_name, **kwargs)` - Track estimator usage
- `@monitor_performance(estimator_name)` - Monitor performance
- `@track_errors(estimator_name)` - Track errors
- `@track_workflow(step_type)` - Track workflow steps

### Classes
- `UsageTracker` - Usage tracking functionality
- `PerformanceMonitor` - Performance monitoring
- `ErrorAnalyzer` - Error analysis
- `WorkflowAnalyzer` - Workflow analysis
- `AnalyticsDashboard` - Unified dashboard interface

## 🐛 Troubleshooting

### Common Issues

1. **Analytics not working**
   - Check if analytics is enabled: `enable_analytics(True)`
   - Verify storage directory permissions
   - Check for import errors

2. **Performance impact**
   - Analytics overhead is minimal (<1ms per event)
   - Background processing runs every 5 minutes
   - Data is automatically cleaned up after 90 days

3. **Storage issues**
   - Default storage: `~/.lrdbench/analytics/`
   - Check disk space
   - Verify write permissions

4. **Import errors**
   - Install required dependencies: `pip install psutil matplotlib seaborn pandas networkx`
   - Check Python version compatibility

### Getting Help

- Check the analytics data files in the storage directory
- Use `get_analytics_summary()` to verify data collection
- Generate reports to identify issues
- Check the main LRDBench documentation

## 🔮 Future Enhancements

- Real-time analytics dashboard
- Machine learning insights
- Predictive analytics
- Integration with external monitoring tools
- Advanced visualization options
- Cloud-based analytics (optional)
- Custom metric definitions
- Alert system for anomalies

---

For more information, see the main LRDBench documentation and examples.
