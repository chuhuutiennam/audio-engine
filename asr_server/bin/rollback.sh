#!/usr/bin/env bash
# Go back to a version saved in backup/ by scripts/update.sh.
# Usage: bin/rollback.sh [--list | BACKUP_ID]
#   (no argument)  the newest saved version that passed its check after a restart
#   BACKUP_ID      that saved version, as shown by --list
#   --list         show the saved versions and change nothing
# Restores the code, bin/, config/ and VERSION, starts the server and waits until it is ready.
# The replaced version is kept in backup/rolled-back/<date>-<time>/.
# Nothing is changed when the saved configuration is not valid for the saved code.
# Exit code: 0 = restored and ready, 1 = failed, 2 = wrong usage.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_upgrade.sh"

BACKUP_DIR="$ROOT/backup"
REPLACED_DIR="$BACKUP_DIR/rolled-back"

list_versions() {
  local id health mark default
  default="$(default_version "$BACKUP_DIR")"
  printf '%-16s %-10s %-21s %-8s\n' ID VERSION BUILT_AT HEALTH
  for id in $(saved_versions "$BACKUP_DIR"); do
    health="$(backup_field "$BACKUP_DIR" "$id" health)"
    mark=""
    if [ "$id" = "$default" ]; then mark="<- default"; fi
    printf '%-16s %-10s %-21s %-8s %s\n' "$id" "$(backup_field "$BACKUP_DIR" "$id" version)" \
      "$(backup_field "$BACKUP_DIR" "$id" built_at)" "${health:-unknown}" "$mark"
  done
}

if [ $# -gt 1 ]; then
  echo "usage: bin/rollback.sh [--list | BACKUP_ID]" >&2
  exit 2
fi
target=""
case "${1:-}" in
  --list) list_versions; exit 0 ;;
  -h|--help) show_usage "${BASH_SOURCE[0]}"; exit 0 ;;
  "") ;;
  *)
    if ! [[ "$1" =~ $BACKUP_ID_PATTERN ]] || [ ! -d "$BACKUP_DIR/$1/lib/asr_server" ]; then
      echo "no saved version named: $1 (see bin/rollback.sh --list)" >&2
      exit 2
    fi
    target="$1" ;;
esac
check_settings
if [ -z "$target" ]; then
  target="$(default_version "$BACKUP_DIR")"
  if [ -z "$target" ]; then
    echo "no saved version to go back to" >&2
    exit 1
  fi
fi
saved="$BACKUP_DIR/$target"

if ! output="$(PYTHONPATH="$saved/lib${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m asr_server \
      --config "$saved/config/server.yaml" --logging-config "$saved/config/logging.yaml" \
      --check-config 2>&1)"; then
  printf '%s\n' "$output" >&2
  echo "the saved configuration of $target is not valid for its code; nothing was changed" >&2
  exit 1
fi

trap 'rm -rf "$ROOT/lib/asr_server.new"' EXIT
rm -rf "$ROOT/lib/asr_server.new"
cp -R "$saved/lib/asr_server" "$ROOT/lib/asr_server.new"

mkdir -p "$REPLACED_DIR"
replaced="$REPLACED_DIR/$(new_backup_id "$REPLACED_DIR")"
save_version "$ROOT" "$replaced"
echo "saved    $replaced"
prune_versions "$REPLACED_DIR" "${ASR_BACKUP_KEEP:-10}"

bash "$ROOT/bin/stop.sh"

rm -rf "$ROOT/lib/asr_server"
mv "$ROOT/lib/asr_server.new" "$ROOT/lib/asr_server"
copy_version_files "$saved" "$ROOT"
echo "restored $target (asr_server $(backup_field "$BACKUP_DIR" "$target" version), built $(backup_field "$BACKUP_DIR" "$target" built_at))"

bash "$ROOT/bin/start.sh" || true          # bin/wait_ready.sh reports a server that did not start
if ! bash "$ROOT/bin/wait_ready.sh"; then
  echo "asr_server did not become ready after the rollback" >&2
  echo "check: $ROOT/logs/asr-server.log and $ROOT/logs/asr-server.out" >&2
  exit 1
fi
