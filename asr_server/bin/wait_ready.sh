#!/usr/bin/env bash
# Wait until the started server listens on its port. The port opens after the models are loaded.
# ASR_START_WAIT_SEC: seconds to wait (default 300); 0 = only check that the process is alive.
# The result is written to VERSION as health=ok or health=failed.
# Exit code: 0 = ready (or not checked), 3 = not running, 4 = still not ready after the wait.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_upgrade.sh"

check_settings
wait_sec="${ASR_START_WAIT_SEC:-300}"

record() {
  if [ -f "$ROOT/VERSION" ]; then record_health "$ROOT/VERSION" "$1"; fi
}

# True only for a listening socket of process $1: another program on the port does not count.
listens_on_port() {
  "$PYTHON" - "$1" "$port" 2>/dev/null <<'PY'
import os
import sys

pid, port = sys.argv[1], int(sys.argv[2])
listening = set()
for table in ('/proc/net/tcp', '/proc/net/tcp6'):
    try:
        with open(table) as f:
            rows = f.readlines()[1:]
    except OSError:
        continue
    for row in rows:
        cols = row.split()
        if cols[3] == '0A' and int(cols[1].rsplit(':', 1)[1], 16) == port:
            listening.add(f'socket:[{cols[9]}]')
fds = f'/proc/{pid}/fd'
sys.exit(0 if any(os.readlink(f'{fds}/{fd}') in listening for fd in os.listdir(fds)) else 1)
PY
}

port="$(PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m asr_server \
          --config "$CONFIG" --logging-config "" --check-config 2>/dev/null | sed -n 's/^port = //p')" \
  || port=""

deadline=$((SECONDS + wait_sec))
while :; do
  if ! pid="$(running_pid)"; then
    record failed
    echo "asr_server is not running" >&2
    exit 3
  fi
  if [ "$wait_sec" = 0 ]; then
    echo "asr_server is running (pid $pid); readiness not checked (ASR_START_WAIT_SEC=0)"
    exit 0
  fi
  if listens_on_port "$pid"; then
    record ok
    echo "asr_server is ready on port $port (pid $pid)"
    exit 0
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    record failed
    echo "asr_server (pid $pid) is still not listening on port $port after $wait_sec s" >&2
    exit 4
  fi
  sleep 2
done
