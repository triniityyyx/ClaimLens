@echo off
title ClaimLens — Fake News Detection & Fact Verification System
color 0A

echo ======================================================================
echo                 CLAIMLENS - THINK BEFORE YOU TRUST
echo    Machine Learning Fake News Detection & Fact Verification System
echo ======================================================================
echo.

:: Check for py launcher or python
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3.11
) else (
    set PYTHON_CMD=python
)

echo [1/3] Checking environment with %PYTHON_CMD%...
%PYTHON_CMD% --version >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.11 is not available. Please install Python 3.11.
    pause
    exit /b 1
)

:: Check if trained model exists
if not exist "models\claimlens_model.joblib" (
    echo [2/3] Trained ML model not found. Training model now...
    %PYTHON_CMD% train_model.py
) else (
    echo [2/3] Trained ML model found.
)

echo [3/3] Launching ClaimLens Streamlit Application...
echo.
echo Application will open in your default browser at http://localhost:8501
echo Press Ctrl+C in this terminal window to stop the server.
echo.

%PYTHON_CMD% -m streamlit run app.py --server.port 8501

pause
