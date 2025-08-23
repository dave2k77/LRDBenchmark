@echo off
echo 🚀 PyPI Upload for DataExploratoryProject
echo ==========================================
echo.
echo Running PowerShell upload script...
echo.

powershell -ExecutionPolicy Bypass -File "upload_to_pypi.ps1"

echo.
echo Upload script completed.
pause
