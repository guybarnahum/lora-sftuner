#!/usr/bin/env bash
#
# clean_ollama.sh — Uninstall Ollama and remove its data (optional).
# - Stops & disables services (systemd on Linux, launchd/Homebrew on macOS).
# - Removes binaries and shared assets (Intel & Apple Silicon paths).
# - Optionally deletes ~/.ollama model cache.
# - Supports AUTO_YES=1 or --yes / -y to skip prompts.
#
set -euo pipefail

AUTO_YES="${AUTO_YES:-0}"

# Correct arg parsing — do NOT synthesize an empty arg when none given
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES=1 ;;
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

remove_path_if_exists() {
  local target="$1"
  local use_sudo="${2:-0}"
  if [[ -e "$target" || -L "$target" ]]; then
    if [[ "$use_sudo" == "1" ]]; then
      run_and_log "Removing $target" sudo rm -rf "$target"
    else
      run_and_log "Removing $target" rm -rf "$target"
    fi
  fi
}

# --- Stop services + remove files ---
if [[ "$OS" == "Linux" ]]; then
  if command -v systemctl >/dev/null 2>&1; then
    run_and_log "Stopping ollama service" sudo systemctl stop ollama || true
    run_and_log "Disabling ollama service" sudo systemctl disable ollama || true
    if [[ -f /etc/systemd/system/ollama.service ]]; then
      run_and_log "Removing systemd unit" sudo rm -f /etc/systemd/system/ollama.service
      run_and_log "Reloading systemd" sudo systemctl daemon-reload
    fi
  fi
  remove_path_if_exists "/usr/local/bin/ollama" 1
  remove_path_if_exists "/usr/share/ollama" 1
  remove_path_if_exists "/var/lib/ollama" 1
  remove_path_if_exists "/var/log/ollama" 1

elif [[ "$OS" == "Darwin" ]]; then
  # Try Homebrew first, but use a shell so '|| true' is honored and disable brew auto behaviors
  if command -v brew >/dev/null 2>&1; then
    run_and_log "Stopping Ollama (brew services)" \
      bash -lc 'brew services stop ollama >/dev/null 2>&1 || true'

    # Try formula uninstall
    run_and_log "Uninstalling Ollama (Homebrew formula)" \
      bash -lc 'env HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1 \
        brew list --formula ollama >/dev/null 2>&1 && brew uninstall --force ollama || true'

    # Try cask uninstall
    run_and_log "Uninstalling Ollama (Homebrew cask)" \
      bash -lc 'env HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 \
        brew list --cask ollama >/dev/null 2>&1 && brew uninstall --cask --force ollama || true'
  fi

  # Launchd (system + user)
  if [[ -f /Library/LaunchDaemons/io.ollama.plist ]]; then
    run_and_log "Unload launchd job (system)" sudo launchctl bootout system /Library/LaunchDaemons/io.ollama.plist || true
    remove_path_if_exists "/Library/LaunchDaemons/io.ollama.plist" 1
  fi
  if [[ -f "$HOME/Library/LaunchAgents/io.ollama.plist" ]]; then
    run_and_log "Unload launchd job (user)" launchctl bootout gui/"$UID" "$HOME/Library/LaunchAgents/io.ollama.plist" || true
    remove_path_if_exists "$HOME/Library/LaunchAgents/io.ollama.plist" 0
  fi

  # Remove binaries & data from both Intel (/usr/local) and Apple Silicon (/opt/homebrew)
  remove_path_if_exists "/usr/local/bin/ollama" 1
  remove_path_if_exists "/usr/local/share/ollama" 1
  remove_path_if_exists "/usr/local/var/ollama" 1

  remove_path_if_exists "/opt/homebrew/bin/ollama" 1
  remove_path_if_exists "/opt/homebrew/share/ollama" 1
  remove_path_if_exists "/opt/homebrew/var/ollama" 1
fi

# --- Optional: remove user cache/models ---
if ask_yes_no "Remove local models/cache in ~/.ollama ? [y/N]"; then
  remove_path_if_exists "$HOME/.ollama" 0
else
  echo "ℹ️ Kept ~/.ollama (models & cache)."
fi

echo ""
echo "🧹 Ollama has been removed (best-effort)."
echo "If any files remain due to custom install locations, you can remove them manually."
