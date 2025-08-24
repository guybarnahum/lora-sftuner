#!/bin/bash
#
# This script cleans up the lora-sftuner project by:
# 1. Logging out of the Hugging Face CLI.
# 2. Uninstalling the pip package.
# 3. Deleting the Python virtual environment directory (.venv).
# 4. Removing common Python cache and build artifact directories.

set -e # Exit immediately if a command exits with a non-zero status.

VENV_DIR=".venv"
PROJECT_NAME="lora-sftuner"

echo "--- Starting cleanup for ${PROJECT_NAME} ---"

# --- Step 1: De-authenticate and Uninstall ---
if [ -d "$VENV_DIR" ]; then
    echo "--- Activating environment for cleanup ---"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    
    # Check if hf command is installed before trying to log out
    if command -v hf &> /dev/null; then
        echo "--- Logging out of Hugging Face CLI ---"
        hf auth logout
    else
        echo "--- hf command not found, skipping logout ---"
    fi
    
    echo "--- Uninstalling pip package ---"
    pip uninstall -y "$PROJECT_NAME"
    deactivate
else
    echo "--- Virtual environment not found, skipping de-authentication and uninstall ---"
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

if [[ -x "./clean_llama.sh" ]]; then
  echo "Found clean_llama.sh, calling it to remove llama.cpp installation..."
  # Pass through any flags provided to clean.sh to clean_llama.sh
  ./clean_llama.sh "$@"
else
  echo "• clean_llama.sh not found or not executable; skipping llama.cpp removal"
fi

echo "✅ Cleanup complete!"
