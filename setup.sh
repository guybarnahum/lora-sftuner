#!/usr/bin/env bash
#
# This script automates the setup of the lora-sftuner project by:
# 1. Sourcing the .env file for secrets.
# 2. Finding a compatible Python version.
# 3. Installing system build dependencies (Linux & macOS).
# 4. Creating and activating a virtual environment.
# 5. Installing dependencies for the correct hardware (CPU/CUDA).
# 6. Authenticating the Hugging Face CLI for access to gated models.
# 7. Optionally installing extras for document processing.
# 8. Optionally installing llama.cpp (for GGUF conversion) via setup_llama.sh.
#
set -e

# ------------- Auto-yes handling (no functionality changes to steps) -------------
AUTO_YES="${AUTO_YES:-0}"
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

ask_yes_no() {
  # preserves your prompts; returns 0 for yes, 1 for no
  local prompt="$1"
  if [[ "$AUTO_YES" == "1" ]]; then
    echo "Auto-yes: $prompt -> yes"
    return 0
  fi
  # mimic your single-char prompt behavior
  read -p "$prompt " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# ------------- Safer colors + cleanup -------------
if tput setaf 0 >/dev/null 2>&1; then
  COLOR_GRAY="$(tput setaf 8)"
  COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[90m'
  COLOR_RESET=$'\033[0m'
fi

cleanup_render() {
  # Always reset color & cursor, clear active line
  printf '\r\033[K%s' "${COLOR_RESET}"
  tput cnorm 2>/dev/null || true
}
trap cleanup_render EXIT INT TERM

# ------------- Improved run_and_log (ANSI-safe, truncation-safe, last-line preview) -------------
run_and_log() {
  local log_file
  log_file=$(mktemp)
  local description="$1"
  shift

  printf "⏳ %s\n" "$description"
  tput civis 2>/dev/null || true

  local prev_render=""
  local cols
  cols=$(tput cols 2>/dev/null || echo 120)

  (
    frames=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' )
    i=0
    while :; do
      # Read last line and strip ANSI from tool output
      local last_line=""
      if [[ -s "$log_file" ]]; then
        last_line=$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g')
      fi

      # Build plain text first (safe to truncate)
      local plain_prefix="${frames[i]} ${description} : "
      local plain="${plain_prefix}${last_line}"
      if (( ${#plain} > cols )); then
        plain="${plain:0:cols-1}"
      fi

      # Color only the tail (the last_line portion that survived)
      local visible_tail=""
      if (( ${#plain} >= ${#plain_prefix} )); then
        visible_tail="${plain:${#plain_prefix}}"
      fi
      local visible_head="${plain:0:${#plain_prefix}}"

      local render="${COLOR_RESET}${visible_head}${COLOR_GRAY}${visible_tail}${COLOR_RESET}"

      if [[ "$render" != "$prev_render" ]]; then
        printf '\r\033[K%s' "$render"
        prev_render="$render"
      fi

      i=$(( (i + 1) % ${#frames[@]} ))
      sleep 0.25
    done
  ) &
  local spinner_pid=$!

  # Run command, capture output
  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" &>/dev/null || true
    wait "$spinner_pid" &>/dev/null || true
    printf '\r\033[K%s' "${COLOR_RESET}"
    printf "❌ %s failed.\n" "$description"
    echo "ERROR LOG :"
    cat "$log_file"
    echo "END OF ERROR LOG"
    rm -f "$log_file"
    exit 1
  fi

  kill "$spinner_pid" &>/dev/null || true
  wait "$spinner_pid" &>/dev/null || true
  printf '\r\033[K%s' "${COLOR_RESET}"
  printf '✅ %s\n' "$description"
  rm -f "$log_file"
}

# --- Step 1: Load .env file if it exists ---
if [ -f ".env" ]; then
  echo "Sourcing .env file "
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

VENV_DIR=".venv"
PYTHON_BIN=""

# --- Step 2: Find a compatible Python interpreter ---
echo "Searching for a compatible Python version (3.11 or 3.12 preferred) "
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
  if ! command -v g++ &> /dev/null || ! command -v cmake &> /dev/null; then
    run_and_log "Updating package list" sudo apt-get update
    run_and_log "Installing build tools (g++, cmake)" sudo apt-get install -y build-essential g++ cmake
  fi
  if command -v nvidia-smi &> /dev/null && ! command -v nvcc &> /dev/null; then
    echo "NVIDIA GPU detected, but CUDA Toolkit (nvcc) is missing. Attempting to install... "
    run_and_log "Downloading CUDA keyring" wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    run_and_log "Installing CUDA keyring" sudo dpkg -i cuda-keyring_1.1-1_all.deb
    run_and_log "Updating package list for CUDA" sudo apt-get update
    rm cuda-keyring_1.1-1_all.deb
    run_and_log "Installing CUDA toolkit" sudo apt-get -y install cuda-toolkit-12-4
  fi

  if [ -d "/usr/local/cuda" ]; then
    CUDA_PATH="/usr/local/cuda"
    if ! grep -q "CUDA_PATH" ~/.profile; then
      echo "Adding CUDA to PATH in ~/.profile "
      {
        echo ''
        echo '# Add CUDA to PATH'
        echo "export CUDA_PATH=${CUDA_PATH}"
        echo 'export PATH="${CUDA_PATH}/bin:${PATH}"'
        echo 'export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"'
      } >> ~/.profile
    fi
    export CUDA_PATH=${CUDA_PATH}
    export PATH="${CUDA_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
  fi
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! xcode-select -p &> /dev/null; then
    echo "Xcode Command Line Tools not found. Attempting to install... "
    xcode-select --install
  fi
fi

# --- Step 4: Create and Activate Virtual Environment ---
if [ ! -d "$VENV_DIR" ]; then
  run_and_log "Creating virtual environment" "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Activating virtual environment "
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

run_and_log "Upgrading pip" pip install --upgrade pip

# --- Step 5: Install Core Dependencies ---
echo "Installing project dependencies "
if command -v nvidia-smi &> /dev/null; then
  run_and_log "Installing CUDA dependencies" pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e ".[cuda]"
else
  run_and_log "Installing CPU dependencies" pip install -e ".[cpu]"
fi

# --- Step 6: Authenticate Hugging Face CLI ---
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  run_and_log "Authenticating Hugging Face CLI" hf auth login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential
fi

# --- Step 7: Optional docs extras (unchanged functionality; now supports auto-yes) ---
echo ""
if ask_yes_no "Do you want to install support for document processing (PDF, DOCX, etc.)? [y/N]"; then
  run_and_log "Installing optional 'docs' dependencies" pip install -e ".[docs]"
fi

# --- Step 8: Optional llama.cpp install via setup_llama.sh (replaces llama-cpp-python extras) ---
echo ""
if ask_yes_no "Do you want to install llama.cpp for GGUF conversion (run ./setup_llama.sh)? [y/N]"; then
  if [[ -x "./setup_llama.sh" ]]; then
    if [[ "$AUTO_YES" == "1" ]]; then
      run_and_log "Installing llama.cpp via setup_llama.sh (auto-yes)" ./setup_llama.sh --yes
    else
      run_and_log "Installing llama.cpp via setup_llama.sh" ./setup_llama.sh
    fi
  else
    echo "❌ setup_llama.sh not found or not executable. Place it next to this script (chmod +x setup_llama.sh)."
  fi
fi

echo ""
echo "✅ Setup complete!"
echo "To activate the environment in the future, run: source $VENV_DIR/bin/activate"
