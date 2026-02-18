@echo off
setlocal
echo ===================================================
echo    🚀 AI TICKET INTELLIGENT SYSTEM - LAUNCHER
echo ===================================================
echo.

:: Kill any existing streamlit processes to free the port
taskkill /f /im streamlit.exe >nul 2>&1

:: Check for venv and activate
if exist "venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [1/3] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
)

:: Check if streamlit is installed
python -m streamlit --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Streamlit not found in current environment.
    echo [2/3] Installing required libraries...
    pip install streamlit pandas numpy joblib scipy scikit-learn
) else (
    echo [2/3] Environment check passed.
)

echo [3/3] Starting AI Dashboard...
echo.
echo ---------------------------------------------------
echo IMPORTANT: If http://localhost:8501 does not work, 
echo please try this address:
echo 👉 http://127.0.0.1:8501
echo ---------------------------------------------------
echo.

:: Start streamlit with explicit 127.0.0.1 to avoid localhost resolution issues
python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start Streamlit. 
    echo Please ensure you have Python 3.8+ installed.
)

pause
endlocal
