# PowerShell script to upload DataExploratoryProject to PyPI

Write-Host "🚀 PyPI Upload Script for DataExploratoryProject" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Check if we have the distribution files
if (-not (Test-Path "dist")) {
    Write-Host "❌ Error: No 'dist' directory found. Run 'python -m build' first." -ForegroundColor Red
    exit 1
}

$distFiles = Get-ChildItem "dist" -File
if ($distFiles.Count -eq 0) {
    Write-Host "❌ Error: No distribution files found in 'dist' directory." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Found distribution files:" -ForegroundColor Green
foreach ($file in $distFiles) {
    Write-Host "   - $($file.Name)" -ForegroundColor Cyan
}

Write-Host "`n📦 Step 1: Upload to TestPyPI (for testing)" -ForegroundColor Yellow
Write-Host "This will test if your package works before uploading to production PyPI." -ForegroundColor White

# Check if credentials are set
$username = $env:TWINE_USERNAME
$password = $env:TWINE_PASSWORD

if (-not $username -or -not $password) {
    Write-Host "`n⚠️  PyPI credentials not found in environment variables." -ForegroundColor Yellow
    Write-Host "Please set them using:" -ForegroundColor White
    Write-Host "   `$env:TWINE_USERNAME = 'your_username'" -ForegroundColor Cyan
    Write-Host "   `$env:TWINE_PASSWORD = 'your_api_token'" -ForegroundColor Cyan
    Write-Host "`nOr enter them when prompted." -ForegroundColor White
    
    $username = Read-Host "Enter your PyPI username"
    $password = Read-Host "Enter your PyPI API token" -AsSecureString
    $password = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($password))
}

Write-Host "`n🔐 Using credentials:" -ForegroundColor Green
Write-Host "   Username: $username" -ForegroundColor Cyan
Write-Host "   Password: [HIDDEN]" -ForegroundColor Cyan

# Set environment variables for this session
$env:TWINE_USERNAME = $username
$env:TWINE_PASSWORD = $password

Write-Host "`n📤 Uploading to TestPyPI..." -ForegroundColor Yellow

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully uploaded to TestPyPI!" -ForegroundColor Green
    Write-Host "`n🔍 Test the installation from TestPyPI:" -ForegroundColor Yellow
    Write-Host "   pip install --index-url https://test.pypi.org/simple/ data-exploratory-project" -ForegroundColor Cyan
    
    $continue = Read-Host "`nDo you want to continue to production PyPI? (y/n)"
    if ($continue -eq "y" -or $continue -eq "Y") {
        Write-Host "`n📤 Uploading to production PyPI..." -ForegroundColor Yellow
        python -m twine upload dist/*
        if ($LASTEXITCODE -eq 0) {
            Write-Host "🎉 SUCCESS! Package uploaded to production PyPI!" -ForegroundColor Green
            Write-Host "`n🔍 Test the installation:" -ForegroundColor Yellow
            Write-Host "   pip install data-exploratory-project" -ForegroundColor Cyan
        } else {
            Write-Host "❌ Failed to upload to production PyPI" -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ Failed to upload to TestPyPI" -ForegroundColor Red
}

Write-Host "`n📚 For more information:" -ForegroundColor Yellow
Write-Host "   - TestPyPI: https://test.pypi.org/" -ForegroundColor Cyan
Write-Host "   - PyPI: https://pypi.org/" -ForegroundColor Cyan
Write-Host "   - Twine docs: https://twine.readthedocs.io/" -ForegroundColor Cyan
