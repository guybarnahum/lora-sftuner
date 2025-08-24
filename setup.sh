#!/bin/bash
#
# This script automates the setup of the lora-sftuner project by:
# 1. Sourcing the .env file for secrets.
# 2. Finding a compatible Python version.
# 3. Creating and activating a virtual environment.
# 4. Installing dependencies for the correct hardware (CPU/CUDA).
# 5. Authenticating the Hugging Face CLI for access to gated models.
# 6. Optionally installing extras for document processing.
# 7. Optionally installing GGUF conversion dependencies.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Step 1: Load .env file if it exists ---
if [ -f ".env" ]; then
    echo "--- Sourcing .env file ---"
    # Use set -a to export all variables to the environment
    set -a
    source .env
    set +a
fi

VENV_DIR=".venv"
PYTHON_BIN=""

# --- Step 2: Find a compatible Python interpreter ---
echo "--- Searching for a compatible Python version (3.11 or 3.12 preferred) ---"
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3.12 &> /dev/null; then
    PYTHON_BIN="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
    echo "⚠️  Warning: python3.11 or python3.12 not found. Falling back to the default 'python3'."
else
    echo "❌ Error: No Python interpreter found. Please install Python 3.11 or 3.12."
    exit 1
fi
echo "✅ Using Python interpreter: $($PYTHON_BIN --version)"


# --- Step 3: Create and Activate Virtual Environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creating virtual environment at $VENV_DIR ---"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "--- Virtual environment already exists at $VENV_DIR ---"
fi

echo "--- Activating virtual environment ---"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# --- Step 4: Install Core Dependencies ---
echo "--- Installing project dependencies ---"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Installing with CUDA support..."
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e ".[cuda]"
else
    echo "🖥️ No NVIDIA GPU detected. Installing with CPU support..."
    pip install -e ".[cpu]"
fi

# --- Step 5: Authenticate Hugging Face CLI ---
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "--- Authenticating Hugging Face CLI ---"
    huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"
else
    echo "⚠️  Warning: HUGGINGFACE_HUB_TOKEN not found in .env. You may need to log in manually for gated models."
fi

# --- Step 6: Install Optional Dependencies ---
echo ""
read -p "Do you want to install support for document processing (PDF, DOCX, etc.)? [y/N] " -n 1 -r
echo # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "--- Installing optional 'docs' dependencies ---"
    pip install -e ".[docs]"
fi

# --- Step 7: Install Optional GGUF Conversion Dependencies ---
echo ""
read -p "Do you want to install support for Ollama/GGUF export (llama-cpp-python)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "--- Installing optional 'gguf' dependencies ---"
    pip install -e ".[gguf]"
fi

echo ""
echo "✅ Setup complete!"
echo "To activate the environment in the future, run: source $VENV_DIR/bin/activate"
