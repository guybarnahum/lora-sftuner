#!/bin/bash
#
# This script automates the setup of the lora-sftuner project by:
# 1. Finding a compatible Python version (3.11 or 3.12 preferred).
# 2. Creating a Python virtual environment in ./.venv using it.
# 3. Activating the environment.
# 4. Detecting if an NVIDIA GPU is available.
# 5. Installing the project with the correct dependencies (CPU or CUDA).
# 6. Optionally installing extras for document processing.

set -e # Exit immediately if a command exits with a non-zero status.

VENV_DIR=".venv"
PYTHON_BIN="" # This will be determined by the script

# --- Step 1: Find a compatible Python interpreter ---
echo "--- Searching for a compatible Python version (3.11 or 3.12 preferred) ---"
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_BIN="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
    echo "⚠️  Warning: python3.11 or python3.12 not found. Falling back to the default 'python3'."
    echo "   This may cause issues if your default Python is not a compatible version (e.g., 3.13+)."
else
    echo "❌ Error: No Python interpreter found. Please install Python 3.11 or 3.12."
    exit 1
fi
echo "✅ Using Python interpreter: $($PYTHON_BIN --version)"


# --- Step 2: Create and Activate Virtual Environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creating virtual environment at $VENV_DIR ---"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "--- Virtual environment already exists at $VENV_DIR ---"
fi

echo "--- Activating virtual environment ---"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# --- Step 3: Detect Hardware and Install Core Dependencies ---
echo "--- Installing project dependencies ---"

# Check for nvidia-smi to determine if a GPU is present
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Installing with CUDA support..."
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e ".[cuda]"
else
    echo "🖥️ No NVIDIA GPU detected. Installing with CPU support..."
    # Install with the CPU extra
    pip install -e ".[cpu]"
fi

# --- Step 4: Install Optional Document Processing Dependencies ---
echo ""
read -p "Do you want to install support for document processing (PDF, DOCX, etc.)? [y/N] " -n 1 -r
echo # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "--- Installing optional 'docs' dependencies ---"
    pip install -e ".[docs]"
fi

echo ""
echo "✅ Setup complete!"
echo "To activate the environment in the future, run: source $VENV_DIR/bin/activate"
