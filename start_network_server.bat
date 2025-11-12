@echo off
echo ========================================
echo 🌐 PDF to Excel Extractor - Network Mode
echo ========================================
echo.
echo Starting server for network access...
echo Other users can access this at your computer's IP address
echo.

python network_config.py
echo.
echo Server stopped.
echo Press any key to exit...
pause > nul