#!/usr/bin/env bash
#
# setup_ollama.sh — Install Ollama on Linux or macOS, enable service (Linux), verify install.
# - Supports AUTO_YES=1 or --yes to skip prompts.
# - Supports AUTO_SERVE=1 or --serve to start ollama serve (macOS runs foreground).
# - Resilient PATH handling on macOS: finds ollama and exports PATH for this run; can persist to shell rc.
# - Safe spinner that shows the last line of the command log in gray.
#
set -euo pipefail

# --- Parse args / env toggles ---
AUTO_YES="${AUTO_YES:-0}"
AUTO_SERVE="${AUTO_SERVE:-0}"
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES=1 ;;
    --serve)  AUTO_SERVE=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

ask_yes_no() {
  local prompt="$1"
  if [[ "$AUTO_YES" == "1" ]]; then
    echo "Auto-yes: $prompt -> yes"
    return 0
  fi
  read -p "$prompt " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# --- Colors + cleanup ---
if tput setaf 0 >/dev/null 2>&1; then
  COLOR_GRAY="$(tput setaf 8)"
  COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[90m'
  COLOR_RESET=$'\033[0m'
fi

cleanup_render() {
  printf '\r\033[K%s' "${COLOR_RESET}"
  tput cnorm >/dev/null 2>&1 || true
}
trap cleanup_render EXIT INT TERM

# --- Spinner with last-line preview ---
run_and_log() {
  local log_file; log_file="$(mktemp)"
  local description="$1"; shift

  printf "⏳ %s\n" "$description"
  tput civis >/dev/null 2>&1 || true

  local prev_render="" cols
  cols="$(tput cols 2>/dev/null || echo 120)"

  (
    local i=0
    local frames=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' )
    while :; do
      local last_line=""
      if [[ -s "$log_file" ]]; then
        last_line="$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g')"
      fi
      local prefix="${frames[i]} ${description} : "
      local plain="${prefix}${last_line}"
      if (( ${#plain} > cols )); then plain="${plain:0:cols-1}"; fi
      local head="${plain:0:${#prefix}}"
      local tail=""
      if (( ${#plain} >= ${#prefix} )); then tail="${plain:${#prefix}}"; fi
      local render="${COLOR_RESET}${head}${COLOR_GRAY}${tail}${COLOR_RESET}"
      if [[ "$render" != "$prev_render" ]]; then
        printf '\r\033[K%s' "$render"
        prev_render="$render"
      fi
      i=$(( (i + 1) % ${#frames[@]} ))
      sleep 0.2
    done
  ) &
  local spinner_pid=$!

  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" >/dev/null 2>&1 || true
    wait "$spinner_pid" >/dev/null 2>&1 || true
    printf '\r\033[K%s' "${COLOR_RESET}"
    printf "❌ %s failed.\n" "$description"
    echo "ERROR LOG:"
    cat "$log_file"
    echo "END OF ERROR LOG"
    rm -f "$log_file"
    exit 1
  fi

  kill "$spinner_pid" >/dev/null 2>&1 || true
  wait "$spinner_pid" >/dev/null 2>&1 || true
  printf '\r\033[K%s' "${COLOR_RESET}"
  printf "✅ %s\n" "$description"
  rm -f "$log_file"
}

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "Detecting platform ..."
echo "• OS: $OS, ARCH: $ARCH"

# --- Install Ollama ---
install_linux() {
  if ! command -v curl >/dev/null 2>&1; then
    run_and_log "Installing curl (apt)" bash -lc 'sudo apt-get update && sudo apt-get install -y curl'
  fi

  run_and_log "Installing Ollama (official script)" \
    bash -lc 'curl -fsSL https://ollama.com/install.sh | sh'

  # Enable + start service
  if command -v systemctl >/dev/null 2>&1; then
    run_and_log "Enabling ollama systemd service" sudo systemctl enable ollama
    run_and_log "Starting ollama service" sudo systemctl start ollama
  fi
}

install_macos() {
  if command -v brew >/dev/null 2>&1; then
    run_and_log "Installing Ollama via Homebrew" brew install ollama
  else
    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: need curl (or Homebrew) on macOS." >&2
      exit 1
    fi
    run_and_log "Installing Ollama (official script)" \
      bash -lc 'curl -fsSL https://ollama.com/install.sh | sh'
  fi
}

# --- Ensure we can find the ollama binary now (don’t depend on PATH refresh) ---
find_ollama_bin() {
  # 1) If already in PATH
  if command -v ollama >/dev/null 2>&1; then
    command -v ollama
    return 0
  fi

  # 2) Try brew prefix (covers Intel: /usr/local, Apple Silicon: /opt/homebrew)
  local bp
  bp="$(brew --prefix 2>/dev/null || true)"
  for d in \
    "$bp/bin" \
    "/usr/local/bin" \
    "/opt/homebrew/bin" \
    "/usr/bin" \
    "/bin"
  do
    if [[ -x "$d/ollama" ]]; then
      echo "$d/ollama"
      return 0
    fi
  done

  # 3) As a last resort, check inside app bundle (rare)
  if [[ -x "/Applications/Ollama.app/Contents/MacOS/ollama" ]]; then
    echo "/Applications/Ollama.app/Contents/MacOS/ollama"
    return 0
  fi

  return 1
}

persist_path_hint() {
  local bin_dir="$1"
  # Choose the user’s rc file
  local rc="$HOME/.zshrc"
  if [[ "${SHELL:-}" == *bash* ]]; then rc="$HOME/.bashrc"; fi

  if ask_yes_no "Add $bin_dir to your PATH in $(basename "$rc")? [y/N]"; then
    {
      echo ""
      echo "# Added by setup_ollama.sh"
      echo "export PATH=\"$bin_dir:\$PATH\""
    } >> "$rc"
    echo "✅ Added to PATH in $rc. Open a new shell or: source \"$rc\""
  else
    echo "ℹ️ Skipped PATH persist."
  fi
}

case "$OS" in
  Linux)  install_linux ;;
  Darwin) install_macos ;;
  *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

# Try to locate binary and make it available to *this* script invocation
OLLAMA_BIN="$(find_ollama_bin || true)"
if [[ -z "${OLLAMA_BIN:-}" ]]; then
  echo "❌ Ollama installed but not found on PATH yet."
  echo "   Try opening a new terminal, or add Homebrew’s bin to PATH:"
  echo "     eval \"\$($(brew --prefix)/bin/brew shellenv)\""
  exit 1
fi

# Make sure current process can call it (prepend its dir to PATH)
export PATH="$(dirname "$OLLAMA_BIN"):$PATH"

# --- Verify install ---
run_and_log "Checking ollama version" "$OLLAMA_BIN" --version

# --- Offer to persist PATH if needed (only if not already in PATH earlier) ---
if ! command -v ollama >/dev/null 2>&1; then
  BIN_DIR="$(dirname "$OLLAMA_BIN")"
  # make it available for this shell now too
  export PATH="$BIN_DIR:$PATH"
  persist_path_hint "$BIN_DIR"
fi

# --- GPU hint (Linux + NVIDIA) ---
if [[ "$OS" == "Linux" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA driver detected:"
    nvidia-smi || true
  else
    echo "ℹ️ No NVIDIA driver detected. CPU-only inference will be used unless drivers are installed."
  fi
fi

# --- Optionally start the server ---
if [[ "$AUTO_SERVE" == "1" ]]; then
  if [[ "$OS" == "Linux" ]]; then
    # Already started via systemd; print status
    if command -v systemctl >/dev/null 2>&1; then
      echo "ℹ️ Ollama systemd service should be running:"
      systemctl --no-pager status ollama || true
    fi
  else
    echo "Starting 'ollama serve' in foreground (Ctrl+C to stop) ..."
    exec "$OLLAMA_BIN" serve
  fi
else
  echo ""
  echo "Next steps:"
  if [[ "$OS" == "Linux" ]]; then
    echo "  • Service is enabled & started:    systemctl status ollama"
  else
    echo "  • Start the server manually:       ollama serve"
  fi
  echo "  • Pull a small model:              ollama pull llama3:8b"
  echo "  • Run a prompt:                    ollama run llama3:8b -q 'Hello from Ollama!'"
fi

# --- After install steps ---
BREW_PREFIX=""
if command -v brew >/dev/null 2>&1; then
  BREW_PREFIX="$(brew --prefix)"
fi

# On macOS, add Homebrew bin path if not already in PATH
if [[ "$OS" == "Darwin" && -n "$BREW_PREFIX" ]]; then
  BREW_BIN="$BREW_PREFIX/bin"
  if ! echo "$PATH" | grep -q "$BREW_BIN"; then
    export PATH="$BREW_BIN:$PATH"
    echo "Added $BREW_BIN to current PATH."
    # Persist for future zsh shells:
    if [[ -f "$HOME/.zshrc" ]]; then
      if ! grep -q "$BREW_BIN" "$HOME/.zshrc"; then
        echo "export PATH=\"$BREW_BIN:\$PATH\"" >> "$HOME/.zshrc"
        echo "Added $BREW_BIN to ~/.zshrc for future shells."
      fi
    else
      echo "export PATH=\"$BREW_BIN:\$PATH\"" >> "$HOME/.zshrc"
      echo "Created ~/.zshrc and added $BREW_BIN to PATH."
    fi
  fi
fi

# --- Verify install after PATH fix ---
if ! command -v ollama >/dev/null 2>&1; then
  echo "⚠️  ollama binary still not found in PATH. Try opening a new shell or run:"
  echo "  export PATH=\"${BREW_PREFIX:-/usr/local}/bin:\$PATH\""
else
  echo "⏳ Checking ollama version"
  ollama --version
fi
