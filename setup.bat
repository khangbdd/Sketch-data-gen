@echo off
REM Image Captioning Pipeline Setup Script for Windows

echo Setting up Image Captioning Pipeline...

REM Check if conda is installed
conda --version >nul 2>&1
if errorlevel 1 (
    echo Error: Conda is required but not installed.
    echo Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

REM Create conda environment
echo Creating conda environment 'sketch-data'...
conda create -n sketch-data python=3.9 -y

REM Activate conda environment
echo Activating conda environment...
call conda activate sketch-data

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
echo 2. Activate the conda environment: conda activate sketch-data
echo 3. Run the pipeline: python pipeline.py --help
echo.
pause
