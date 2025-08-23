#!/bin/bash
#
# This script cleans up the lora-sftuner project by:
# 1. Uninstalling the pip package if a virtual environment exists.
# 2. Deleting the Python virtual environment directory (.venv).
# 3. Removing common Python cache and build artifact directories.
# 4. Optionally, deleting output directories for models and data.

set -e # Exit immediately if a command exits with a non-zero status.

VENV_DIR=".venv"
PROJECT_NAME="lora-sftuner"

echo "--- Starting cleanup for ${PROJECT_NAME} ---"

# --- Step 1: Uninstall the pip package ---
if [ -d "$VENV_DIR" ]; then
    echo "--- Uninstalling pip package from $VENV_DIR ---"
    # Activate the environment to ensure we use the correct pip
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    # Use pip to properly uninstall the package in editable mode
    pip uninstall -y "$PROJECT_NAME"
    deactivate
else
    echo "--- Virtual environment not found, skipping pip uninstall ---"
fi

# --- Step 2: Remove Python artifacts and virtual environment ---
echo "--- Removing Python artifacts and virtual environment ---"
rm -rf "$VENV_DIR"
rm -rf .pytest_cache
rm -rf build dist *.egg-info
find . -type d -name "__pycache__" -exec rm -r {} +

# --- Step 3: Optional cleanup for output directories ---
# This part is commented out by default to prevent accidental data loss.
# Uncomment the following lines if you want to also delete all generated
# models, datasets, and other outputs.
#
# read -p "Do you want to delete all output data and models? [y/N] " -n 1 -r
# echo # Move to a new line
# if [[ $REPLY =~ ^[Yy]$ ]]; then
#     echo "--- Removing output directories (out/, data/, models/) ---"
#     rm -rf out/
#     rm -rf data/
#     rm -rf models/
# fi

echo "✅ Cleanup complete!"

