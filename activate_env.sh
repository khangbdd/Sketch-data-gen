#!/bin/bash
# Quick activation script for the sketch-data conda environment
# Usage: source activate_env.sh

# Initialize conda for bash/zsh
eval "$(conda shell.bash hook)"

# Activate the sketch-data environment
conda activate sketch-data

echo "Activated conda environment: sketch-data"
echo "You can now run: python pipeline.py --help"
