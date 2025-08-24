#!/usr/bin/env bash
# clean_llama.sh — remove llama.cpp install, symlinks, and PATH edits safely.
# Usage:
#   bash clean_llama.sh [--dry-run] [--yes] [--system]
#     --dry-run  : show what would be removed, but do nothing
#     --yes      : proceed without interactive prompts
#     --system   : also remove system-wide items under /usr/local and /etc (sudo)
set -euo pipefail

# -------- Config (matches setup_llama.sh defaults) --------
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
BUILD_DIR="$LLAMA_DIR/build"
USER_BIN="$HOME/.local/bin"
SYSTEM_BIN="/usr/local/llama/bin"
SYSTEM_PROFILED="/etc/profile.d/llama.sh"

# Detect primary shell rc
SHELL_RC="$HOME/.zshrc"
[[ ${SHELL:-} == *bash* ]] && SHELL_RC="$HOME/.bashrc"

# -------- Flags --------
DRY_RUN=0
ASSUME_YES=0
DO_SYSTEM=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y)  ASSUME_YES=1 ;;
    --system)  DO_SYSTEM=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1;;
  esac
done

say() { printf "%s\n" "$*"; }
act() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf "[dry-run] %s\n" "$*"
  else
    eval "$@"
  fi
}
confirm() {
  local msg="$1"
  if [[ $ASSUME_YES -eq 1 ]]; then
    return 0
  fi
  read -r -p "$msg [y/N] " ans
  [[ $ans =~ ^[Yy]$ ]]
}

# -------- Helpers --------
remove_symlink_if_points_to() {
  # $1 = symlink path, $2 = must point under this dir
  local link="$1" must_root="$2"
  if [[ -L "$link" ]]; then
    local target
    target="$(readlink "$link")"
    case "$target" in
      "$must_root"/*)
        act "rm -f \"$link\""
        say "✔ removed symlink: $link -> $target"
        ;;
      *)
        say "• skipped (points elsewhere): $link -> $target"
        ;;
    esac
  else
    say "• skipped (not a symlink): $link"
  fi
}

remove_line_block_from_rc() {
  # Removes the PATH block we added:
  #   # llama.cpp bins
  #   export PATH="$HOME/.local/bin:$PATH"
  local rc="$1"
  if [[ -f "$rc" ]]; then
    if grep -q '^# llama.cpp bins' "$rc"; then
      if [[ $DRY_RUN -eq 1 ]]; then
        say "[dry-run] would remove PATH block from $rc"
      else
        # macOS sed needs backup suffix; GNU sed accepts -i'' too.
        sed -i.bak '/^# llama.cpp bins/,+1 d' "$rc" || true
        say "✔ cleaned PATH block in $rc (backup: ${rc}.bak)"
      fi
    else
      say "• PATH block not found in $rc"
    fi
  else
    say "• shell rc not found: $rc"
  fi
}

# -------- Start --------
say "Cleaning llama.cpp installation (user scope)"
say "  LLAMA_DIR = $LLAMA_DIR"
say "  USER_BIN  = $USER_BIN"
say "  SHELL_RC  = $SHELL_RC"
[[ $DO_SYSTEM -eq 1 ]] && say "  [system] SYSTEM_BIN = $SYSTEM_BIN, PROFILED = $SYSTEM_PROFILED"
[[ $DRY_RUN -eq 1 ]] && say "  (dry-run mode)"

# 1) Remove user symlinks if they point into $BUILD_DIR/bin
for b in llama-cli llama-quantize llama-server; do
  remove_symlink_if_points_to "$USER_BIN/$b" "$BUILD_DIR/bin"
done

# 2) Remove PATH edits from rc
remove_line_block_from_rc "$SHELL_RC"

# 3) Optionally remove repo (and venv/build)
if [[ -d "$LLAMA_DIR" ]]; then
  if confirm "Remove repository directory and venv at $LLAMA_DIR ?"; then
    act "rm -rf \"$LLAMA_DIR\""
    say "✔ removed $LLAMA_DIR"
  else
    say "• kept $LLAMA_DIR"
  fi
else
  say "• repo not found: $LLAMA_DIR"
fi

# 4) System-wide cleanup (optional)
if [[ $DO_SYSTEM -eq 1 ]]; then
  say "Cleaning system-wide items (requires sudo)"
  # Remove system symlinks (only if they point under our BUILD_DIR/bin)
  for b in llama-cli llama-quantize llama-server; do
    if [[ -L "$SYSTEM_BIN/$b" ]]; then
      target="$(readlink "$SYSTEM_BIN/$b")"
      if [[ "$target" == "$BUILD_DIR/bin/"* ]]; then
        act "sudo rm -f \"$SYSTEM_BIN/$b\""
        say "✔ removed system symlink: $SYSTEM_BIN/$b -> $target"
      else
        say "• skipped system symlink (points elsewhere): $SYSTEM_BIN/$b -> $target"
      fi
    else
      say "• system link missing or not a symlink: $SYSTEM_BIN/$b"
    fi
  done

  # Remove empty dir if possible
  if [[ -d "$SYSTEM_BIN" ]]; then
    act "sudo rmdir \"$SYSTEM_BIN\" 2>/dev/null || true"
  fi

  # Remove profile.d script if it contains our PATH line
  if [[ -f "$SYSTEM_PROFILED" ]]; then
    if grep -q '/usr/local/llama/bin' "$SYSTEM_PROFILED"; then
      act "sudo rm -f \"$SYSTEM_PROFILED\""
      say "✔ removed $SYSTEM_PROFILED"
    else
      say "• kept $SYSTEM_PROFILED (does not look like ours)"
    fi
  else
    say "• no $SYSTEM_PROFILED found"
  fi
fi

say ""
say "Done."
say "Notes:"
say "  • Open a new terminal (or 'source $SHELL_RC') to refresh PATH."
say "  • If you used other shells (fish, etc.), remove their PATH entries manually."
say "  • Reinstall later with your setup_llama.sh."

