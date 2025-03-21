#!/bin/bash
# Script to set up and test Phase 1 in a virtual environment

# Exit on error
set -e

echo "Setting up virtual environment for KastenRAG testing..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment in ./venv"
    python3 -m venv venv
else
    echo "Using existing virtual environment in ./venv"
fi

# Activate virtual environment
echo "Activating virtual environment"
source venv/bin/activate

# Install dependencies
echo "Installing dependencies"
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Create test directories
echo "Setting up test directories"
mkdir -p data/sample_audio
mkdir -p test_output/logs

# Run the test script
echo -e "\n===== Running Phase 1 Test =====\n"
python test_output.py

# Deactivate virtual environment
echo -e "\nTest completed. Deactivating virtual environment"
deactivate

echo "You can reactivate the virtual environment with: source venv/bin/activate"