#!/usr/bin/env bash
# Start asr_server in the background. Environment overrides:
#   PYTHON, ASR_SERVER_CONFIG, ASR_SERVER_LOGGING_CONFIG
# bin/env.sh, when present, is read first (interpreter, CUDA_VISIBLE_DEVICES).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if pid="$(running_pid)"; then
  echo "asr_server is already running (pid $pid)"
  exit 0
fi

cd "$ROOT"
mkdir -p "$ROOT/logs"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Fail here, in the foreground, if the configuration is invalid.
"$PYTHON" -m asr_server --config "$CONFIG" --logging-config "$LOGGING_CONFIG" --check-config > /dev/null

# Log records go to the rotated file only; $OUT_FILE keeps whatever else is printed
# (startup tracebacks, output of third-party libraries).
nohup "$PYTHON" -m asr_server --config "$CONFIG" --logging-config "$LOGGING_CONFIG" \
  --no-console-log >> "$OUT_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1
if pid="$(running_pid)"; then
  echo "asr_server started (pid $pid). Model loading continues in the background."
  echo "log: $ROOT/logs/asr-server.log    stdout/stderr: $OUT_FILE"
else
  echo "asr_server exited during startup; see $OUT_FILE" >&2
  rm -f "$PID_FILE"
  exit 1
fi
