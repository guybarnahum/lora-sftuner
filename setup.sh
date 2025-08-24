#!/bin/bash
#
# This script automates the setup of the lora-sftuner project by:
# 1. Sourcing the .env file for secrets.
# 2. Finding a compatible Python version.
# 3. Installing system build dependencies (Linux & macOS).
# 4. Creating and activating a virtual environment.
# 5. Installing dependencies for the correct hardware (CPU/CUDA).
# 6. Authenticating the Hugging Face CLI for access to gated models.
# 7. Optionally installing extras for document processing.
# 8. Optionally installing GGUF conversion dependencies.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Step 1: Load .env file if it exists ---
if [ -f ".env" ]; then
    echo "--- Sourcing .env file ---"
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

# --- Step 3: Install system build dependencies ---
if [[ "$(uname -s)" == "Linux" ]]; then
    # Install general build tools if missing
    if ! command -v g++ &> /dev/null || ! command -v cmake &> /dev/null; then
        echo "--- Build tools (g++, cmake) not found. Attempting to install... ---"
        sudo apt-get update && sudo apt-get install -y build-essential g++ cmake
    fi
    # Install CUDA Toolkit if nvidia-smi is present but nvcc is missing
    if command -v nvidia-smi &> /dev/null && ! command -v nvcc &> /dev/null; then
        echo "--- NVIDIA GPU detected, but CUDA Toolkit (nvcc) is missing. Attempting to install... ---"
        wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update
        rm cuda-keyring_1.1-1_all.deb
        sudo apt-get -y install cuda-toolkit-12-4
    fi
    
    # Ensure CUDA is in the PATH
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
        if ! grep -q "CUDA_PATH" ~/.profile; then
            echo "--- Adding CUDA to PATH in ~/.profile ---"
            echo '' >> ~/.profile
            echo '# Add CUDA to PATH' >> ~/.profile
            echo "export CUDA_PATH=${CUDA_PATH}" >> ~/.profile
            echo 'export PATH="${CUDA_PATH}/bin:${PATH}"' >> ~/.profile
            echo 'export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"' >> ~/.profile
            echo "Please log out and back in for changes to take full effect."
        fi
        # Source for the current session
        export CUDA_PATH=${CUDA_PATH}
        export PATH="${CUDA_PATH}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
    fi
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    if ! xcode-select -p &> /dev/null; then
        echo "--- Xcode Command Line Tools not found. Attempting to install... ---"
        xcode-select --install
    fi
fi

# --- Step 4: Create and Activate Virtual Environment ---
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

# --- Step 5: Install Core Dependencies ---
echo "--- Installing project dependencies ---"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Installing with CUDA support..."
    pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e ".[cuda]"
else
    echo "🖥️ No NVIDIA GPU detected. Installing with CPU support..."
    pip install -e ".[cpu]"
fi

# --- Step 6: Authenticate Hugging Face CLI ---
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "--- Authenticating Hugging Face CLI ---"
    hf auth login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential
else
    echo "⚠️  Warning: HUGGINGFACE_HUB_TOKEN not found in .env. You may need to log in manually for gated models."
fi

# --- Step 7: Install Optional Dependencies ---
echo ""
read -p "Do you want to install support for document processing (PDF, DOCX, etc.)? [y/N] " -n 1 -r
echo # Move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "--- Installing optional 'docs' dependencies ---"
    pip install -e ".[docs]"
fi

# --- Step 8: Install Optional GGUF Conversion Dependencies ---
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
