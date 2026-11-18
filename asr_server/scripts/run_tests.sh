#!/usr/bin/env bash
# Run the server tests, then the eval-client tests. Neither needs a GPU or model checkpoints.
# The two suites run in separate interpreters: each ships its own copy of speech_pb2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"
echo "== server tests =="
"$PYTHON" -m pytest tests -q "$@"
echo "== eval client tests =="
(cd "$ROOT/tools/eval" && "$PYTHON" -m pytest tests -q "$@")
