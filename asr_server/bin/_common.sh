# Shared by start.sh / stop.sh / status.sh. Sourced, not executed.
# ROOT is the run directory (bin/ config/ lib/ logs/ model/ data/, made by scripts/update.sh)
# or the source tree (code in src/ instead of lib/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Local settings of this installation (interpreter, GPU), if any. See bin/env.sh.example.
if [ -f "$ROOT/bin/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/bin/env.sh"
fi
if [ -d "$ROOT/lib/asr_server" ]; then CODE_DIR="$ROOT/lib"; else CODE_DIR="$ROOT/src"; fi
PID_FILE="$ROOT/logs/asr-server.pid"
OUT_FILE="$ROOT/logs/asr-server.out"
PYTHON="${PYTHON:-python3}"
CONFIG="${ASR_SERVER_CONFIG:-$ROOT/config/server.yaml}"
LOGGING_CONFIG="${ASR_SERVER_LOGGING_CONFIG:-$ROOT/config/logging.yaml}"

# Prints the pid and returns 0 only when the pid file points at a live asr_server process.
running_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [ -r "/proc/$pid/cmdline" ] && ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'asr_server'; then
    return 1          # pid was reused by another program
  fi
  echo "$pid"
}
