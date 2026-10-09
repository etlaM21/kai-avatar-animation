#!/bin/bash
echo "=========================================="
echo "Setting up Local Virtual Environment (WSL)..."
echo "=========================================="

# 1. Create the virtual environment using copies instead of symlinks
python3 -m venv venv --copies

# 2. Activate it
source ./venv/bin/activate

# 3. Safely upgrade pip and install build tools inside the venv
python3 -m pip install --upgrade pip setuptools wheel

# 4. Install the requirements
pip install -r requirements.txt

echo -e "\nLocal setup complete! To activate this environment in the future, run:"
echo "source ./venv/bin/activate"

read -p "Press [Enter] key to continue..."