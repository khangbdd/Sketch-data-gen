#!/bin/bash

# Image Captioning Pipeline Clean Setup Script
# This script removes existing environments and sets up fresh

echo "Clean setup for Image Captioning Pipeline..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is required but not installed."
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

# Initialize conda for bash/zsh
eval "$(conda shell.bash hook)"

# Remove existing conda environment if it exists
echo "Removing existing 'sketch-data' environment if it exists..."
conda env remove -n sketch-data -y 2>/dev/null || echo "No existing environment to remove."

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create fresh conda environment
echo "Creating fresh conda environment 'sketch-data'..."
conda create -n sketch-data python=3.9 -y

# Activate conda environment
echo "Activating conda environment..."
conda activate sketch-data

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Copy environment template
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys before running the pipeline."
else
    echo ".env file already exists."
fi

# Make pipeline executable
chmod +x pipeline.py

echo ""
echo "Clean setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Activate the conda environment: conda activate sketch-data"
echo "3. Run the pipeline: python pipeline.py --help"
echo ""