#!/usr/bin/env bash
# Stop asr_server: SIGTERM, wait up to 15 s. Use --force to send SIGKILL after that.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if ! pid="$(running_pid)"; then
  echo "asr_server is not running"
  rm -f "$PID_FILE"
  exit 0
fi

kill -TERM "$pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "asr_server stopped (pid $pid)"
    exit 0
  fi
  sleep 0.5
done

if [ "${1:-}" = "--force" ]; then
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "asr_server killed (pid $pid)"
  exit 0
fi
echo "asr_server (pid $pid) did not stop within 15 s; run: bin/stop.sh --force" >&2
exit 1
