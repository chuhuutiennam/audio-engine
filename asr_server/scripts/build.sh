#!/usr/bin/env bash
# Build the deployable tree of asr_server in build/app. It has the layout of the run directory:
#   bin/  lib/asr_server/  config/  VERSION
# Usage: scripts/build.sh [--test]
#   --test   run the test suites first (needs pytest)
# Environment: PYTHON (default python3), used for the syntax check and the tests.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
STAGE="$ROOT/build/app"

run_tests=0
for arg in "$@"; do
  case "$arg" in
    --test) run_tests=1 ;;
    -h|--help) sed -n '2,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ "$run_tests" = 1 ]; then
  PYTHON="$PYTHON" bash "$ROOT/scripts/run_tests.sh"
fi

# Syntax check of the sources; leaves no .pyc files behind.
"$PYTHON" - "$ROOT/src/asr_server" <<'PY'
import pathlib
import sys

for path in sorted(pathlib.Path(sys.argv[1]).rglob('*.py')):
    compile(path.read_bytes(), str(path), 'exec')
PY

rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/lib" "$STAGE/config"
cp -R "$ROOT/src/asr_server" "$STAGE/lib/asr_server"
find "$STAGE/lib" -name '__pycache__' -type d -prune -exec rm -rf {} +
# shellcheck disable=SC1091
source "$ROOT/bin/_upgrade.sh"
for name in $BIN_SCRIPTS; do
  cp "$ROOT/bin/$name" "$STAGE/bin/$name"
done
chmod +x "$STAGE"/bin/*.sh
cp "$ROOT/config/server.yaml" "$ROOT/config/logging.yaml" "$STAGE/config/"

version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
{
  echo "name=asr_server"
  echo "version=${version:-unknown}"
  echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source=$ROOT"
} > "$STAGE/VERSION"
echo "built asr_server ${version:-unknown} -> $STAGE"
