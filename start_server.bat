@echo off
setlocal enabledelayedexpansion
echo ========================================
echo PDF Table Extractor - Server Startup
echo ========================================
echo.

echo Finding your network IP address(es)...
echo.

REM Get all local IP addresses
echo Your network IP address(es):
ipconfig | findstr /c:"IPv4 Address"
echo.

REM Get the first non-localhost IPv4 address (usually the main network adapter)
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address" ^| findstr /v "127.0.0.1"') do (
    set LOCAL_IP=%%a
    set LOCAL_IP=!LOCAL_IP:~1!
    goto :found
)
:found

if "%LOCAL_IP%"=="" (
    echo WARNING: Could not detect IP address automatically.
    echo Please check your IP address manually using: ipconfig
    echo.
    set LOCAL_IP=YOUR_IP_HERE
)

echo ========================================
echo Starting Streamlit server...
echo ========================================
echo.
echo IMPORTANT: Use one of these URLs to access the app:
echo.
echo   For THIS PC (localhost):  http://localhost:8501
echo   For OTHER PCs on network: http://%LOCAL_IP%:8501
echo.
echo If you have multiple IP addresses above, try:
echo   - 192.168.10.x or 192.168.1.x (usually main network)
echo   - Avoid 192.168.56.x (usually VirtualBox/Hyper-V)
echo.
echo The server is now starting...
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501

pause

