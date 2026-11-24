# Shared by scripts/build.sh, scripts/update.sh, bin/rollback.sh and bin/wait_ready.sh.
# Sourced, not executed.
#
# backup/<date>-<time>/              a version replaced by an update (UTC; see new_backup_id)
# backup/rolled-back/<date>-<time>/  a version replaced by a rollback, kept to be examined

BIN_SCRIPTS="_common.sh _upgrade.sh start.sh stop.sh status.sh rollback.sh wait_ready.sh env.sh.example"
# This file, rollback.sh and wait_ready.sh are not saved: a rollback keeps the newest ones.
VERSION_FILES="bin/_common.sh bin/start.sh bin/stop.sh bin/status.sh bin/env.sh.example bin/env.sh
               config/server.yaml config/logging.yaml config/server.yaml.dist config/logging.yaml.dist
               VERSION"
BACKUP_ID_PATTERN='^[0-9]{8}-[0-9]{6}$'

# Prints the comment at the top of script $1.
show_usage() {
  awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$1"
}

# A leading zero is refused: bash arithmetic would read the value as octal.
check_settings() {
  if ! [[ "${ASR_BACKUP_KEEP:-10}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ASR_BACKUP_KEEP must be a number of versions, 1 or more: $ASR_BACKUP_KEEP" >&2
    exit 2
  fi
  if ! [[ "${ASR_START_WAIT_SEC:-300}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ASR_START_WAIT_SEC must be a number of seconds: $ASR_START_WAIT_SEC" >&2
    exit 2
  fi
}

# Copies the files of a version from directory $1 to directory $2.
copy_version_files() {
  local name
  for name in $VERSION_FILES; do
    if [ -f "$1/$name" ]; then cp "$1/$name" "$2/$name"; fi
  done
}

# Saves the version installed in $1 as the new directory $2.
save_version() {
  rm -rf "$2.new"
  mkdir -p "$2.new/lib" "$2.new/bin" "$2.new/config"
  if [ -d "$1/lib/asr_server" ]; then cp -R "$1/lib/asr_server" "$2.new/lib/asr_server"; fi
  copy_version_files "$1" "$2.new"
  mv "$2.new" "$2"
}

# Prints the name $2 seconds after name $1 (YYYYmmdd-HHMMSS, UTC); fails when $1 is no real date
# or the result is no valid name (needs GNU date).
shift_backup_id() {
  local out
  out="$(date -u -d "${1:0:4}-${1:4:2}-${1:6:2} ${1:9:2}:${1:11:2}:${1:13:2} UTC + $2 seconds" \
         +%Y%m%d-%H%M%S 2>/dev/null)" || return 1
  [[ "$out" =~ $BACKUP_ID_PATTERN ]] || return 1
  echo "$out"
}

# Prints an unused name for a saved version in directory $1: the UTC time, so that a clock change
# cannot make a newer version sort before an older one, and always after the newest saved name
# (names given before UTC was used are in local time, up to 14 hours ahead of UTC). A saved name
# that is no real date, or lies more than a day ahead of the clock, is ignored with a warning.
new_backup_id() {
  local id now limit name newest=""
  now="$(date -u +%Y%m%d-%H%M%S)"
  limit="$(shift_backup_id "$now" 86400)"
  for name in $(saved_versions "$1"); do
    if [[ "$name" > "$limit" ]] || ! shift_backup_id "$name" 0 > /dev/null; then
      echo "warning: saved version name $name ignored for the order (no date within a day of now)" >&2
      continue
    fi
    newest="$name"
    break
  done
  id="$now"
  if [ -n "$newest" ] && [[ ! "$id" > "$newest" ]]; then
    id="$(shift_backup_id "$newest" 1)" || return 1
  fi
  while [ -e "$1/$id" ]; do          # only an ignored name can be in the way; ids only go up
    id="$(shift_backup_id "$id" 1)" || return 1
  done
  echo "$id"
}

# Prints the saved versions of directory $1, newest first.
saved_versions() {
  [ -d "$1" ] || return 0
  ls -1 "$1" | { grep -E "$BACKUP_ID_PATTERN" || true; } | sort -r
}

# Prints field $3 of the VERSION file of saved version $2 in directory $1.
backup_field() {
  [ -f "$1/$2/VERSION" ] || return 0
  sed -n "s/^$3=//p" "$1/$2/VERSION" | head -1
}

# Prints the version to go back to: the newest that passed its check after a restart,
# else the newest that has no recorded result. A version marked failed is never chosen.
default_version() {
  local id fallback=""
  for id in $(saved_versions "$1"); do
    [ -d "$1/$id/lib/asr_server" ] || continue
    case "$(backup_field "$1" "$id" health)" in
      ok) echo "$id"; return 0 ;;
      failed) ;;
      *) if [ -z "$fallback" ]; then fallback="$id"; fi ;;
    esac
  done
  if [ -n "$fallback" ]; then echo "$fallback"; fi
}

# Keeps at most $2 saved versions in directory $1. The version to go back to is always one of them,
# so a series of failed updates cannot push the last working version out.
prune_versions() {
  local id protected kept=0
  protected="$(default_version "$1")"
  if [ -n "$protected" ]; then kept=1; fi
  for id in $(saved_versions "$1"); do
    if [ "$id" = "$protected" ]; then continue; fi
    if [ "$kept" -lt "$2" ]; then
      kept=$((kept + 1))
    else
      rm -rf "${1:?}/$id"
      echo "removed  $1/$id"
    fi
  done
}

# Writes health=$2 in the VERSION file $1.
record_health() {
  { grep -v '^health=' "$1" 2>/dev/null || true; echo "health=$2"; } > "$1.tmp"
  mv "$1.tmp" "$1"
}
