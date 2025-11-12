@echo off
setlocal
cd /d "%~dp0"
echo Starting PDF Table Extractor API on http://127.0.0.1:8000
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
endlocal