@echo off
cd /d "%~dp0"
echo Starting CareerForge-AI app...
echo When the app is ready, open: http://localhost:8501
echo Press Ctrl+C in this window to stop the app.
echo.
streamlit run app.py
pause
