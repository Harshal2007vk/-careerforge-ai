@echo off
echo ============================================
echo   CareerForge-AI - Quick Setup
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] Installing Python packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Try instead: python -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo.
echo [2/2] Training the recommender model (creates career_model.pkl for the app)...
python train_model.py
if errorlevel 1 (
    echo Training had an error. Check messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo To run the app, double-click:  run_app.bat
echo Or in this folder open a terminal and type:  streamlit run app.py
echo.
pause
