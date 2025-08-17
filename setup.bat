@echo off
REM Image Captioning Pipeline Setup Script for Windows

echo Setting up Image Captioning Pipeline...

REM Check if Python 3 is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 is required but not installed.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Copy environment template
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo Please edit .env file with your API keys before running the pipeline.
) else (
    echo .env file already exists.
)

echo.
echo Setup completed successfully!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Activate the virtual environment: venv\Scripts\activate.bat
echo 3. Run the pipeline: python pipeline.py --help
echo.
pause
