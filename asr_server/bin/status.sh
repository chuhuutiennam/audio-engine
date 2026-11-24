#!/usr/bin/env bash
# Exit code 0 = running, 3 = not running.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if pid="$(running_pid)"; then
  echo "asr_server is running (pid $pid)"
  port="$(PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m asr_server \
            --config "$CONFIG" --logging-config "" --check-config 2>/dev/null | sed -n 's/^port = //p')" \
    || port=""        # a configuration that is not valid now does not stop the running server
  if [ -n "$port" ] && command -v ss > /dev/null 2>&1; then
    if ss -lnt 2>/dev/null | grep -q ":$port "; then
      echo "listening on port $port"
    else
      echo "not listening on port $port yet (models may still be loading)"
    fi
  fi
  exit 0
fi
echo "asr_server is not running"
exit 3
