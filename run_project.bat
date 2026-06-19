@echo off
setlocal

cd /d "%~dp0"

echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python is not installed or is only the Microsoft Store shortcut.
    echo Install Python from https://www.python.org/downloads/
    echo During installation, tick: Add python.exe to PATH
    echo Then close this window and run run_project.bat again.
    echo.
    pause
    exit /b 1
)

echo Creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo Installing requirements...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo Training model...
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 (
    echo Model training failed.
    pause
    exit /b 1
)

echo Starting Streamlit app...
".venv\Scripts\python.exe" -m streamlit run app.py

endlocal
