# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Repository Structure**: Ensure your repository has the correct structure

## 🏗️ Repository Structure

Your repository should have this structure for deployment:

```
your-repo/
├── web_dashboard/
│   ├── streamlit_app.py          # Main Streamlit app
│   ├── requirements.txt          # Python dependencies
│   └── .streamlit/
│       └── config.toml           # Streamlit configuration
├── lrdbench/                     # Your package code
├── setup.py                      # Package setup
└── README.md
```

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Ensure all files are committed**:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Verify your `requirements.txt`** includes all dependencies:
   ```txt
   streamlit>=1.28.0
   numpy>=1.21.0
   pandas>=1.3.0
   scipy>=1.7.0
   plotly>=5.15.0
   matplotlib>=3.5.0
   seaborn>=0.11.0
   numba>=0.56.0
   pywavelets>=1.1.0
   torch>=1.9.0
   jax>=0.3.0
   jaxlib>=0.3.0
   psutil>=5.8.0
   networkx>=2.6.0
   -e ..
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure your app**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `web_dashboard/streamlit_app.py`
   - **App URL**: Choose a custom URL (optional)

5. **Click "Deploy!"**

### Step 3: Monitor Deployment

1. **Watch the build logs** for any errors
2. **Common issues and solutions**:
   - **Import errors**: Check `requirements.txt`
   - **Path issues**: Ensure file paths are correct
   - **Memory issues**: Optimize your app for cloud deployment

## 🔧 Configuration Options

### Streamlit Configuration (`.streamlit/config.toml`)

```toml
[global]
developmentMode = false

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Environment Variables

You can set environment variables in Streamlit Cloud:
- Go to your app settings
- Add environment variables as needed
- Restart the app to apply changes

## 🚀 Advanced Deployment Options

### Custom Domain

1. **Purchase a domain** (e.g., from Namecheap, GoDaddy)
2. **Configure DNS** to point to your Streamlit app
3. **Add custom domain** in Streamlit Cloud settings

### Multiple Apps

You can deploy multiple apps from the same repository:
- **Main app**: `web_dashboard/streamlit_app.py`
- **Lightweight app**: `web_dashboard/lightweight_dashboard.py`
- **Test app**: `web_dashboard/simple_test_app.py`

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Check `requirements.txt` includes all dependencies
   - Ensure local package is installed with `-e ..`

2. **Path Issues**:
   - Use relative paths in your app
   - Test locally before deploying

3. **Memory Issues**:
   - Optimize data loading
   - Use caching for expensive operations
   - Limit data size for cloud deployment

4. **Performance Issues**:
   - Use `@st.cache_data` for data caching
   - Optimize computations
   - Use lazy loading for large datasets

### Debugging

1. **Check build logs** in Streamlit Cloud
2. **Test locally** with same environment
3. **Use Streamlit's debugging features**:
   ```python
   st.write("Debug info:", variable)
   st.exception(e)  # For error handling
   ```

## 📊 Monitoring and Analytics

### App Analytics

Streamlit Cloud provides:
- **Usage statistics**
- **Performance metrics**
- **Error logs**
- **User analytics**

### Custom Analytics

You can add custom analytics to your app:
```python
import streamlit as st

# Track page views
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
st.session_state.page_views += 1

# Display analytics
st.sidebar.metric("Page Views", st.session_state.page_views)
```

## 🔄 Continuous Deployment

### Automatic Updates

1. **Push to main branch** triggers automatic redeployment
2. **Monitor deployment status** in Streamlit Cloud
3. **Test changes** before pushing to main

### Version Control

1. **Use semantic versioning** for releases
2. **Tag important releases** in Git
3. **Document changes** in README

## 🎯 Best Practices

### Performance

1. **Cache expensive operations**:
   ```python
   @st.cache_data
   def expensive_computation(data):
       return result
   ```

2. **Use lazy loading** for large datasets
3. **Optimize imports** and dependencies
4. **Monitor memory usage**

### Security

1. **Don't expose sensitive data** in your app
2. **Use environment variables** for secrets
3. **Validate user inputs**
4. **Follow security best practices**

### User Experience

1. **Add loading indicators** for long operations
2. **Provide clear error messages**
3. **Use consistent styling**
4. **Optimize for mobile devices**

## 📞 Support

### Resources

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repository

### Getting Help

1. **Check Streamlit documentation**
2. **Search community forums**
3. **Create GitHub issues** for bugs
4. **Contact Streamlit support** for cloud issues

---

**Happy Deploying! 🚀**
