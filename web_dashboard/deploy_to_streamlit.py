#!/usr/bin/env python3
"""
🚀 Streamlit Cloud Deployment Helper

This script validates your repository structure and prepares it for Streamlit Cloud deployment.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_requirements():
    """Check if requirements.txt has necessary dependencies."""
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"❌ Requirements file missing: {req_file}")
        return False
    
    with open(req_file, 'r') as f:
        content = f.read()
    
    required_deps = [
        "streamlit",
        "numpy", 
        "pandas",
        "plotly"
    ]
    
    missing_deps = []
    for dep in required_deps:
        if dep not in content:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing dependencies in requirements.txt: {missing_deps}")
        return False
    else:
        print("✅ All required dependencies found in requirements.txt")
        return True

def check_git_status():
    """Check git status and ensure all changes are committed."""
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Not in a git repository")
            return False
        
        # Check for uncommitted changes
        if "nothing to commit" in result.stdout:
            print("✅ All changes committed")
            return True
        else:
            print("⚠️  Uncommitted changes detected:")
            print(result.stdout)
            return False
            
    except FileNotFoundError:
        print("❌ Git not found. Please install git.")
        return False

def check_streamlit_config():
    """Check if Streamlit configuration exists."""
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    if os.path.exists(config_file):
        print(f"✅ Streamlit config found: {config_file}")
        return True
    else:
        print(f"⚠️  Streamlit config missing: {config_file}")
        print("   Creating basic config...")
        
        # Create config directory
        os.makedirs(config_dir, exist_ok=True)
        
        # Create basic config
        config_content = """[global]
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
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created Streamlit config: {config_file}")
        return True

def main():
    """Main validation function."""
    print("🚀 Streamlit Cloud Deployment Validation")
    print("=" * 50)
    
    # Change to web_dashboard directory
    if os.path.basename(os.getcwd()) != "web_dashboard":
        if os.path.exists("web_dashboard"):
            os.chdir("web_dashboard")
            print("📁 Changed to web_dashboard directory")
        else:
            print("❌ web_dashboard directory not found")
            return False
    
    # Check essential files
    essential_files = [
        ("streamlit_app.py", "Main Streamlit app"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    all_files_ok = True
    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_files_ok = False
    
    # Check requirements
    if not check_requirements():
        all_files_ok = False
    
    # Check Streamlit config
    if not check_streamlit_config():
        all_files_ok = False
    
    # Check git status
    if not check_git_status():
        print("⚠️  Please commit all changes before deploying")
        all_files_ok = False
    
    print("\n" + "=" * 50)
    
    if all_files_ok:
        print("🎉 All checks passed! Your app is ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Push your changes to GitHub:")
        print("   git push origin main")
        print("\n2. Go to https://share.streamlit.io")
        print("3. Sign in with GitHub")
        print("4. Click 'New app'")
        print("5. Select your repository")
        print("6. Set main file path to: web_dashboard/streamlit_app.py")
        print("7. Click 'Deploy!'")
    else:
        print("❌ Some checks failed. Please fix the issues above before deploying.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
