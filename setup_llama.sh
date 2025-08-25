#!/usr/bin/env bash
#
# setup_llama.sh — Install llama.cpp + converter deps, and (optionally) your project's deps via pyproject extras.
# - macOS (Intel & Apple Silicon) and Linux (incl. Ubuntu on GCP T4)
# - Detects CUDA (Linux/NVIDIA), Metal (macOS/Apple Silicon), or CPU fallback
# - Builds llama.cpp binaries
# - Creates Python venv for converter & (optionally) installs your project's extras: [cpu] or [cuda]
#
set -euo pipefail

# Always run under bash (re-exec if started from zsh/sh)
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

# ---------- User-configurable knobs ----------
# If you want to SKIP installing your project's extras from pyproject.toml, set:
#   export USE_PROJECT_PYPROJECT=0
USE_PROJECT_PYPROJECT="${USE_PROJECT_PYPROJECT:-1}"
PROJECT_DIR="${PROJECT_DIR:-$PWD}"   # directory containing your pyproject.toml (defaults to current dir)

# ---------- Colors ----------

if tput setaf 0 >/dev/null 2>&1; then
  COLOR_GRAY="$(tput setaf 8)"
  COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[90m'
  COLOR_RESET=$'\033[0m'
fi

# ---------- Pretty runner with spinner + last-line preview (gray) ----------
# Ensure cursor + colors reset on any exit
cleanup_render() {
  printf '\r\033[K%s' "${COLOR_RESET}"
  (tput cnorm >/dev/null 2>&1) || true
}
trap cleanup_render EXIT INT TERM

run_and_log() {
  local log_file
  log_file=$(mktemp)
  local description="$1"; shift

  printf "⏳ %s\n" "$description"
  (tput civis >/dev/null 2>&1) || true

  local prev_render=""
  local cols
  cols=$(tput cols 2>/dev/null || echo 120)

  (
    frames=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' )
    i=0
    while :; do
      # 1) Read last log line and STRIP any ANSI escape sequences it may contain
      #    This regex removes CSI sequences like ESC [ ... cmd
      local last_line=""
      if [[ -s "$log_file" ]]; then
        last_line=$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g')
      fi

      # 2) Build a PLAIN (no color) version first, for safe truncation
      #    Format: [spinner] [desc] : [last_line]
      local plain_prefix="${frames[i]} ${description} : "
      local plain="${plain_prefix}${last_line}"

      # 3) Truncate the PLAIN string to terminal width (reserve 1 char headroom)
      if (( ${#plain} > cols )); then
        plain="${plain:0:cols-1}"
      fi

      # 4) Re-apply GRAY **only to the tail segment** (the last_line portion that survived)
      #    Figure out how much of last_line remains after truncation
      local visible_tail=""
      if (( ${#plain} >= ${#plain_prefix} )); then
        visible_tail="${plain:${#plain_prefix}}"
      else
        # Prefix itself was truncated; no gray tail remains
        visible_tail=""
      fi
      local visible_head="${plain:0:${#plain_prefix}}"

      # Combine with colors on the tail only; ALWAYS end with RESET
      local render="${COLOR_RESET}${visible_head}${COLOR_GRAY}${visible_tail}${COLOR_RESET}"

      # 5) Only repaint if changed; clear the line before printing
      if [[ "$render" != "$prev_render" ]]; then
        printf '\r\033[K%s' "$render"
        prev_render="$render"
      fi

      i=$(( (i + 1) % ${#frames[@]} ))
      sleep 0.25
    done
  ) &
  local spinner_pid=$!

  # Run the command, teeing output into log_file
  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" >/dev/null 2>&1 || true
    wait "$spinner_pid" >/dev/null 2>&1 || true
    printf '\r\033[K%s' "${COLOR_RESET}"
    printf "❌ %s failed.\n" "$description"
    echo "ERROR LOG:"
    cat "$log_file"
    echo "END OF ERROR LOG"
    rm -f "$log_file"
    # cleanup_render will run via trap
    exit 1
  fi

  kill "$spinner_pid" >/dev/null 2>&1 || true
  wait "$spinner_pid" >/dev/null 2>&1 || true
  printf '\r\033[K%s' "${COLOR_RESET}"
  printf '✅ %s\n' "$description"
  rm -f "$log_file"
  # leaving cursor restore + reset to the trap is fine
}

# ---------- Defaults ----------
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
BUILD_DIR="$LLAMA_DIR/build"
PY_VENV_DIR="$LLAMA_DIR/.venv"
INSTALL_BIN_DIR="${INSTALL_BIN_DIR:-$HOME/.local/bin}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

OS="$(uname -s)"
ARCH="$(uname -m)"
HAVE_CUDA="false"
HAVE_METAL="false"
USE_OPENBLAS="false"
declare -a CMAKE_FLAGS=()

echo "Detecting platform features ..."
if [[ "$OS" == "Linux" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected (nvidia-smi present). Will try CUDA/cuBLAS build."
    HAVE_CUDA="true"
  fi
elif [[ "$OS" == "Darwin" ]]; then
  if [[ "$ARCH" == "arm64" ]]; then
    echo "Apple Silicon detected. Will enable Metal acceleration."
    HAVE_METAL="true"
  else
    echo "Intel macOS detected. CPU-only build."
  fi
fi

# ---------- Pick Python ----------
PYTHON_BIN=""
echo "Searching for a Python interpreter (3.10+ recommended) ..."
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"; break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "❌ No Python 3 found. Please install Python 3.10+ and re-run."
  exit 1
fi
echo "✅ Using: $($PYTHON_BIN --version)"

# ---------- Build deps ----------
if [[ "$OS" == "Linux" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    run_and_log "Updating apt package list" sudo apt-get update
    run_and_log "Installing build tools (git, cmake, g++, make, python-venv)" \
      sudo apt-get install -y git cmake build-essential g++ make python3-venv python3-pip
    if ! dpkg -s libopenblas-dev >/dev/null 2>&1; then
      read -r -p "Install OpenBLAS for faster CPU builds? [y/N] " REPLY_OB
      if [[ "${REPLY_OB:-}" =~ ^[Yy]$ ]]; then
        run_and_log "Installing OpenBLAS" sudo apt-get install -y libopenblas-dev
        USE_OPENBLAS="true"
      fi
    else
      USE_OPENBLAS="true"
    fi
    if [[ "$HAVE_CUDA" == "true" ]] && ! command -v nvcc >/dev/null 2>&1; then
      echo "⚠️  CUDA detected (nvidia-smi), but nvcc not found. Building with GGML_CUDA=ON anyway."
    fi
  else
    echo "⚠️ Non-apt Linux—ensure git, cmake, g++, make, python venv are installed."
  fi
elif [[ "$OS" == "Darwin" ]]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "Installing Xcode Command Line Tools (required)..."
    xcode-select --install || true
    echo "If build tools are still missing after install, re-run this script."
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
      run_and_log "Installing cmake via Homebrew" brew install cmake
    else
      echo "⚠️ Homebrew not found and cmake missing. Install Homebrew (https://brew.sh) or cmake manually."
      exit 1
    fi
  fi
fi

# ---------- Clone or update llama.cpp ----------
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  run_and_log "Cloning llama.cpp into $LLAMA_DIR" git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
  pushd "$LLAMA_DIR" >/dev/null
  run_and_log "Updating llama.cpp (git pull)" git pull --ff-only
  popd >/dev/null
fi

# ---------- Configure acceleration flags ----------
if [[ "$OS" == "Linux" ]]; then
  [[ "$HAVE_CUDA" == "true" ]] && CMAKE_FLAGS+=("-DGGML_CUDA=ON")
  [[ "$USE_OPENBLAS" == "true" ]] && CMAKE_FLAGS+=("-DGGML_BLAS=ON" "-DGGML_BLAS_VENDOR=OpenBLAS")
elif [[ "$OS" == "Darwin" ]]; then
  [[ "$HAVE_METAL" == "true" ]] && CMAKE_FLAGS+=("-DGGML_METAL=ON")
fi

# ---------- Build llama.cpp ----------
mkdir -p "$BUILD_DIR"
pushd "$BUILD_DIR" >/dev/null
run_and_log "Configuring CMake ($CMAKE_BUILD_TYPE)" \
  cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" ${CMAKE_FLAGS[@]+"${CMAKE_FLAGS[@]}"}
CPU_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
run_and_log "Building llama.cpp binaries" cmake --build . --config "$CMAKE_BUILD_TYPE" -j"$CPU_CORES"
popd >/dev/null

# ---------- Symlink binaries ----------
mkdir -p "$INSTALL_BIN_DIR"
for b in llama-cli llama-server llama-quantize ; do
  [[ -f "$BUILD_DIR/bin/$b" ]] && ln -sf "$BUILD_DIR/bin/$b" "$INSTALL_BIN_DIR/$b"
done

# ---------- Ensure INSTALL_BIN_DIR in PATH ----------
case "${SHELL:-}" in
  */zsh) SHELL_RC="$HOME/.zshrc" ;;
  */bash) SHELL_RC="$HOME/.bashrc" ;;
  *) SHELL_RC="$HOME/.profile" ;;
esac
NEED_EXPORT="true"
IFS=':' read -ra PATH_ARR <<< "$PATH"
for p in "${PATH_ARR[@]}"; do
  [[ "$p" == "$INSTALL_BIN_DIR" ]] && NEED_EXPORT="false"
done
if [[ "$NEED_EXPORT" == "true" ]]; then
  echo ""
  read -r -p "Add $INSTALL_BIN_DIR to your PATH in $(basename "$SHELL_RC")? [Y/n] " REPLY_PATH
  if [[ ! "${REPLY_PATH:-}" =~ ^[Nn]$ ]]; then
    { echo ""; echo "# Added by setup_llama.sh"; echo "export PATH=\"$INSTALL_BIN_DIR:\$PATH\""; } >> "$SHELL_RC"
    echo "✅ Added to PATH in $SHELL_RC. Open a new shell or: source \"$SHELL_RC\""
  else
    echo "ℹ️ Skipped PATH export. Binaries are in: $INSTALL_BIN_DIR"
  fi
fi

# ---------- Python venv ----------
if [[ ! -d "$PY_VENV_DIR" ]]; then
  run_and_log "Creating Python venv for conversion tools" "$PYTHON_BIN" -m venv "$PY_VENV_DIR"
fi
# shellcheck source=/dev/null
source "$PY_VENV_DIR/bin/activate"

# Upgrade pip early (optional but recommended)
run_and_log "Upgrading pip/setuptools/wheel" \
  pip install --upgrade pip setuptools wheel

unset PIP_EXTRA_INDEX_URL
unset PIP_INDEX_URL

# ---------- (Optional) Install your project's deps via pyproject extras ----------
if [[ "$USE_PROJECT_PYPROJECT" == "1" && -f "$PROJECT_DIR/pyproject.toml" ]]; then
  pushd "$PROJECT_DIR" >/dev/null
  if [[ "$OS" == "Darwin" ]]; then
    # Your pyproject pins Torch==2.2.2 on macOS under [project.optional-dependencies].cpu
    run_and_log "Installing project extras [cpu] from pyproject.toml" \
      pip install -e ".[cpu]"
  elif [[ "$OS" == "Linux" && "$HAVE_CUDA" == "true" ]]; then
    # Your pyproject's [cuda] extra with cu121 index
    run_and_log "Installing project extras [cuda] from pyproject.toml" \
      pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e ".[cuda]"
  else
    # Linux CPU-only or other cases → [cpu]
    run_and_log "Installing project extras [cpu] from pyproject.toml" \
      pip install -e ".[cpu]"
  fi
  popd >/dev/null
else
  echo "Skipping pyproject extras (USE_PROJECT_PYPROJECT=$USE_PROJECT_PYPROJECT or missing pyproject.toml)."
fi

# ---------- Converter deps (PyPI stable only) ----------
# Exact deps needed for convert_hf_to_gguf.py (no torch required)
CONVERTER_PKGS=(
  'numpy>=1.26,<2'
  'sentencepiece>=0.2.0,<0.3'
  'transformers>=4.45,<5'
  'gguf>=0.16'
  'protobuf>=4.21,<5'
  'mistral-common>=1.8.3'
  'safetensors>=0.4'
  'tokenizers>=0.15'
)

# Be explicit about using only PyPI, and clear any extra index env vars
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL PIP_FIND_LINKS
run_and_log "Installing Python requirements (converter only, PyPI stable)" \
  env PIP_EXTRA_INDEX_URL= PIP_INDEX_URL=https://pypi.org/simple \
  pip install --index-url https://pypi.org/simple "${CONVERTER_PKGS[@]}"

# ---------- Summary ----------
echo ""
echo "🎉 llama.cpp setup complete!"
echo ""
echo "Binaries:"
echo "  • $BUILD_DIR/bin/llama-cli"
echo "  • $BUILD_DIR/bin/llama-quantize"
echo "  • $BUILD_DIR/bin/llama-server"
echo "Symlinks:"
echo "  • $INSTALL_BIN_DIR/llama-cli"
echo "  • $INSTALL_BIN_DIR/llama-quantize"
echo "  • $INSTALL_BIN_DIR/llama-server"
echo ""
echo "Converter script:"
echo "  • $LLAMA_DIR/convert_hf_to_gguf.py"
echo "Venv: $PY_VENV_DIR"
echo ""
echo "Examples:"
echo "  source \"$PY_VENV_DIR/bin/activate\""
echo "  python \"$LLAMA_DIR/convert_hf_to_gguf.py\" /path/to/merged-hf-model --outfile out/model-f16.gguf --outtype f16"
echo "  \"$BUILD_DIR/bin/llama-quantize\" out/model-f16.gguf out/model-Q5_K_M.gguf Q5_K_M"
echo ""
if [[ "$HAVE_CUDA" == "true" ]]; then
  echo "  \"$BUILD_DIR/bin/llama-cli\" -m out/model-Q5_K_M.gguf --n-gpu-layers 100 -p 'Hello'"
elif [[ "$HAVE_METAL" == "true" ]]; then
  echo "  \"$BUILD_DIR/bin/llama-cli\" -m out/model-Q5_K_M.gguf -p 'Hello'"
else
  echo "  \"$BUILD_DIR/bin/llama-cli\" -m out/model-Q5_K_M.gguf -p 'Hello'"
fi
echo ""
echo "Tip: Use your pyproject extras to control Torch versions:"
echo "  macOS:   pip install -e '.[cpu]'    # (Torch 2.2.2 per your file)"
echo "  Linux+GPU: pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e '.[cuda]'"
