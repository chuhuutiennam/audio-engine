#!/usr/bin/env bash
# Install or update asr_server in its run directory (app home).
#
# Usage: scripts/update.sh [--app-home DIR] [--restart] [--no-build] [--ignore-old-config]
#   --app-home DIR       run directory; default: $ASR_APP_HOME, else /u01/ims/audio-engine
#   --restart            stop a running server, update, start the server and wait until it is ready
#   --no-build           install the existing build/app instead of running scripts/build.sh first
#   --ignore-old-config  update even if the installed code cannot read the configuration; that
#                        saved version is marked failed, because it cannot be rolled back to
#
# Run directory after the first install:
#   bin/     start.sh stop.sh status.sh rollback.sh wait_ready.sh (replaced on every update)
#            env.sh = local settings (created once, never overwritten)
#   config/  server.yaml logging.yaml (created once, never overwritten)
#            *.dist = the defaults of the installed version, for comparison
#            *.dist.new = the defaults of a version whose update was refused
#   lib/     asr_server/ = application code (replaced); asr_server.prev/ = the version before
#   backup/  <date>-<time>/ = the versions replaced by the last updates, for bin/rollback.sh
#   logs/ model/ data/  (created once, never touched)
#
# Settings in bin/env.sh: ASR_BACKUP_KEEP (saved versions, default 10),
# ASR_START_WAIT_SEC (wait after the restart, default 300; 0 = do not check readiness).
#
# The installed version is not changed when the configuration found there is not valid for the new
# version or for the installed version, or when the server is running and --restart was not given.
# A server that fails after the restart is reported and NOT rolled back.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_HOME="${ASR_APP_HOME:-/u01/ims/audio-engine}"
STAGE="$ROOT/build/app"
# shellcheck disable=SC1091
source "$ROOT/bin/_upgrade.sh"

restart=0
build=1
ignore_old_config=0
while [ $# -gt 0 ]; do
  case "$1" in
    --app-home)
      [ $# -ge 2 ] || { echo "--app-home needs a directory" >&2; exit 2; }
      APP_HOME="$2"; shift 2 ;;
    --restart) restart=1; shift ;;
    --no-build) build=0; shift ;;
    --ignore-old-config) ignore_old_config=1; shift ;;
    -h|--help) show_usage "${BASH_SOURCE[0]}"; exit 0 ;;
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
check_settings

if [ "$build" = 1 ]; then
  PYTHON="$PYTHON" bash "$ROOT/scripts/build.sh"
fi
for name in lib/asr_server/__main__.py config/server.yaml config/logging.yaml VERSION \
            $(printf 'bin/%s ' $BIN_SCRIPTS); do
  if [ ! -e "$STAGE/$name" ]; then
    echo "nothing built in $STAGE (missing $name); run scripts/build.sh" >&2
    exit 1
  fi
done

# The pid file decides, not the installed scripts: they may be older or damaged.
was_running=0
if (source "$ROOT/bin/_common.sh"; PID_FILE="$APP_HOME/logs/asr-server.pid"; running_pid > /dev/null); then
  was_running=1
  if [ "$restart" != 1 ]; then
    echo "asr_server is running in $APP_HOME; stop it first ($APP_HOME/bin/stop.sh) or use --restart" >&2
    exit 1
  fi
fi

mkdir -p "$APP_HOME/bin" "$APP_HOME/config" "$APP_HOME/lib" "$APP_HOME/logs" \
         "$APP_HOME/model" "$APP_HOME/data" "$APP_HOME/backup"

# Configuration: write the defaults of this version as *.dist.new; create the real file only once.
# They become *.dist when the update goes ahead, so that a refused update leaves *.dist alone.
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
    }' "$src" > "$dst.dist.new"
  else
    cp "$src" "$dst.dist.new"
  fi
  if [ -f "$dst" ]; then
    echo "kept     $dst"
  else
    cp "$dst.dist.new" "$dst"
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
  echo "compare with $APP_HOME/config/server.yaml.dist.new" >&2
  exit 1
fi

# The saved version is only useful if its code can read the configuration saved with it.
old_config_valid=1
if [ -d "$APP_HOME/lib/asr_server" ]; then
  if ! output="$(PYTHONPATH="$APP_HOME/lib${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m asr_server \
        --config "$APP_HOME/config/server.yaml" --logging-config "$APP_HOME/config/logging.yaml" \
        --check-config 2>&1)"; then
    printf '%s\n' "$output" >&2
    if [ "$ignore_old_config" != 1 ]; then
      echo "the installed version cannot read $APP_HOME/config, so the update could not be rolled back; the application was not updated" >&2
      echo "remove the keys of the new version, update, then add them again (or use --ignore-old-config)" >&2
      exit 1
    fi
    old_config_valid=0
  fi

  backup_id="$(new_backup_id "$APP_HOME/backup")"
  save_version "$APP_HOME" "$APP_HOME/backup/$backup_id"
  echo "saved    $APP_HOME/backup/$backup_id"
  if [ "$old_config_valid" = 0 ]; then
    record_health "$APP_HOME/backup/$backup_id/VERSION" failed
    echo "warning: $backup_id cannot read its configuration and is marked failed" >&2
  fi
  prune_versions "$APP_HOME/backup" "${ASR_BACKUP_KEEP:-10}"
fi

mv "$APP_HOME/config/server.yaml.dist.new" "$APP_HOME/config/server.yaml.dist"
mv "$APP_HOME/config/logging.yaml.dist.new" "$APP_HOME/config/logging.yaml.dist"

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

for name in $BIN_SCRIPTS; do
  cp "$STAGE/bin/$name" "$APP_HOME/bin/$name"
  case "$name" in
    *.sh) chmod +x "$APP_HOME/bin/$name" ;;
  esac
done
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
# Only for the tasks that are on (asr.enabled, noise_reduction.enabled).
setting() { printf '%s\n' "$settings" | sed -n "s/^$1 = '\(.*\)'\$/\1/p"; }
flag() { printf '%s\n' "$settings" | sed -n "s/^$1 = //p"; }      # True | False
missing=0
if [ "$(flag asr_enabled)" != False ]; then
  asr_model="$(setting model)"
  punct_model="$(setting punct_model)"
  if [ -n "$asr_model" ] && [ ! -d "$asr_model" ]; then echo "MISSING  ASR model directory: $asr_model"; missing=1; fi
  if [ -n "$punct_model" ] && [ ! -d "$punct_model" ]; then echo "MISSING  punctuation model directory: $punct_model"; missing=1; fi
  if [ "$(setting vad_backend)" = silero ]; then
    silero_model="$(setting silero_model)"
    if [ -n "$silero_model" ] && [ ! -f "$silero_model" ]; then echo "MISSING  Silero VAD file: $silero_model"; missing=1; fi
  fi
fi
if [ "$(flag noise_enabled)" = True ]; then
  noise_model="$(setting noise_model)"
  if [ -n "$noise_model" ] && [ ! -f "$noise_model" ]; then echo "MISSING  DPDFNet-2 model file: $noise_model"; missing=1; fi
  # The denoiser classes came with sherpa-onnx 2.x (2.0.2 tested); an import error is shown as is.
  if ! sherpa_error="$("$PYTHON" -c 'import sherpa_onnx as s
assert hasattr(s, "OnlineSpeechDenoiser") and hasattr(s, "OfflineSpeechDenoiserDpdfNetModelConfig"), \
    "sherpa-onnx %s has no OnlineSpeechDenoiser" % getattr(s, "__version__", "?")' 2>&1)"; then
    echo "MISSING  Python package sherpa-onnx==2.0.2 (noise_reduction.enabled; see requirements.txt):" \
         "$(printf '%s\n' "$sherpa_error" | tail -n 1)"
    missing=1
  fi
fi

if [ "$was_running" = 1 ] || { [ "$restart" = 1 ] && [ "$missing" = 0 ]; }; then
  bash "$APP_HOME/bin/start.sh" || true    # bin/wait_ready.sh reports a server that did not start
  if ! bash "$APP_HOME/bin/wait_ready.sh"; then
    echo "asr_server $version did not become ready; it was NOT rolled back" >&2
    echo "check:        $APP_HOME/logs/asr-server.log and $APP_HOME/logs/asr-server.out" >&2
    echo "to go back:   $APP_HOME/bin/rollback.sh" >&2
    exit 1
  fi
elif [ "$missing" = 1 ]; then
  echo "copy the missing models, then start with: $APP_HOME/bin/start.sh"
else
  echo "start with: $APP_HOME/bin/start.sh"
fi
