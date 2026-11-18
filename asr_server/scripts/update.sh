#!/usr/bin/env bash
# Install or update asr_server in its run directory (app home).
#
# Usage: scripts/update.sh [--app-home DIR] [--restart] [--no-build]
#   --app-home DIR  run directory; default: $ASR_APP_HOME, else /u01/ims/audio-engine
#   --restart       stop a running server, update, start it again
#   --no-build      install the existing build/app instead of running scripts/build.sh first
#
# Run directory after the first install:
#   bin/     start.sh stop.sh status.sh (replaced on every update)
#            env.sh = local settings (created once, never overwritten)
#   config/  server.yaml logging.yaml (created once, never overwritten)
#            *.dist = the defaults of the installed version, for comparison
#   lib/     asr_server/ = application code (replaced); asr_server.prev/ = the version before
#   logs/ model/ data/  (created once, never touched)
#
# Nothing in the run directory is changed when the configuration found there is not valid
# for the new version, or when the server is running and --restart was not given.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_HOME="${ASR_APP_HOME:-/u01/ims/audio-engine}"
STAGE="$ROOT/build/app"

restart=0
build=1
while [ $# -gt 0 ]; do
  case "$1" in
    --app-home)
      [ $# -ge 2 ] || { echo "--app-home needs a directory" >&2; exit 2; }
      APP_HOME="$2"; shift 2 ;;
    --restart) restart=1; shift ;;
    --no-build) build=0; shift ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$APP_HOME" in
  /) echo "app home must not be /" >&2; exit 2 ;;
  /*) APP_HOME="${APP_HOME%/}" ;;
  *) echo "app home must be an absolute path: $APP_HOME" >&2; exit 2 ;;
esac

# Use the interpreter configured for this installation, if there is one.
if [ -f "$APP_HOME/bin/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$APP_HOME/bin/env.sh"
fi
PYTHON="${PYTHON:-python3}"

if [ "$build" = 1 ]; then
  PYTHON="$PYTHON" bash "$ROOT/scripts/build.sh"
fi
if [ ! -d "$STAGE/lib/asr_server" ]; then
  echo "nothing built in $STAGE; run scripts/build.sh" >&2
  exit 1
fi

was_running=0
if [ -f "$APP_HOME/bin/status.sh" ] && bash "$APP_HOME/bin/status.sh" > /dev/null 2>&1; then
  was_running=1
  if [ "$restart" != 1 ]; then
    echo "asr_server is running in $APP_HOME; stop it first ($APP_HOME/bin/stop.sh) or use --restart" >&2
    exit 1
  fi
fi

mkdir -p "$APP_HOME/bin" "$APP_HOME/config" "$APP_HOME/lib" "$APP_HOME/logs" \
         "$APP_HOME/model" "$APP_HOME/data"

# Configuration: write the defaults of this version as *.dist; create the real file only once.
# The shipped default model directory /opt/models becomes <app home>/model.
install_config() {
  local name="$1" src="$STAGE/config/$1" dst="$APP_HOME/config/$1"
  if [ "$name" = server.yaml ]; then
    awk -v old="/opt/models" -v new="$APP_HOME/model" '{
      out = ""
      while ((i = index($0, old)) > 0) {
        out = out substr($0, 1, i - 1) new
        $0 = substr($0, i + length(old))
      }
      print out $0
    }' "$src" > "$dst.dist"
  else
    cp "$src" "$dst.dist"
  fi
  if [ -f "$dst" ]; then
    echo "kept     $dst"
  else
    cp "$dst.dist" "$dst"
    echo "created  $dst"
  fi
}
install_config server.yaml
install_config logging.yaml

# Validate the configuration of the run directory against the NEW code before touching the old code.
if ! settings="$(PYTHONPATH="$STAGE/lib${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m asr_server \
      --config "$APP_HOME/config/server.yaml" --logging-config "$APP_HOME/config/logging.yaml" \
      --check-config 2>&1)"; then
  printf '%s\n' "$settings" >&2
  echo "the configuration in $APP_HOME/config is not valid for this version; the application was not updated" >&2
  echo "compare with $APP_HOME/config/server.yaml.dist" >&2
  exit 1
fi

if [ "$was_running" = 1 ]; then
  bash "$APP_HOME/bin/stop.sh"
fi

# Code: copy next to the old version, then swap. One previous version is kept for rollback.
rm -rf "$APP_HOME/lib/asr_server.new"
cp -R "$STAGE/lib/asr_server" "$APP_HOME/lib/asr_server.new"
if [ -d "$APP_HOME/lib/asr_server" ]; then
  rm -rf "$APP_HOME/lib/asr_server.prev"
  mv "$APP_HOME/lib/asr_server" "$APP_HOME/lib/asr_server.prev"
fi
mv "$APP_HOME/lib/asr_server.new" "$APP_HOME/lib/asr_server"

for name in _common.sh start.sh stop.sh status.sh env.sh.example; do
  cp "$STAGE/bin/$name" "$APP_HOME/bin/$name"
done
chmod +x "$APP_HOME/bin/_common.sh" "$APP_HOME/bin/start.sh" "$APP_HOME/bin/stop.sh" "$APP_HOME/bin/status.sh"
if [ -f "$APP_HOME/bin/env.sh" ]; then
  echo "kept     $APP_HOME/bin/env.sh"
else
  cp "$STAGE/bin/env.sh.example" "$APP_HOME/bin/env.sh"
  echo "created  $APP_HOME/bin/env.sh"
fi

{
  cat "$STAGE/VERSION"
  echo "installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$APP_HOME/VERSION"
version="$(sed -n 's/^version=//p' "$APP_HOME/VERSION")"
echo "asr_server $version installed in $APP_HOME"

# Model files named by the configuration: report what is missing (the server refuses to start without them).
setting() { printf '%s\n' "$settings" | sed -n "s/^$1 = '\(.*\)'\$/\1/p"; }
missing=0
asr_model="$(setting model)"
punct_model="$(setting punct_model)"
if [ -n "$asr_model" ] && [ ! -d "$asr_model" ]; then echo "MISSING  ASR model directory: $asr_model"; missing=1; fi
if [ -n "$punct_model" ] && [ ! -d "$punct_model" ]; then echo "MISSING  punctuation model directory: $punct_model"; missing=1; fi
if [ "$(setting vad_backend)" = silero ]; then
  silero_model="$(setting silero_model)"
  if [ -n "$silero_model" ] && [ ! -f "$silero_model" ]; then echo "MISSING  Silero VAD file: $silero_model"; missing=1; fi
fi

if [ "$was_running" = 1 ]; then
  bash "$APP_HOME/bin/start.sh"
elif [ "$missing" = 1 ]; then
  echo "copy the missing models, then start with: $APP_HOME/bin/start.sh"
else
  echo "start with: $APP_HOME/bin/start.sh"
fi
