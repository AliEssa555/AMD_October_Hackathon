#!/usr/bin/env bash
# Pull a finished run off the GPU Droplet, verify it arrived, then print the destroy step.
#
# Run it FROM THE LAPTOP (Git Bash is fine), not on the Droplet:
#
#   bash infra/fetch_results.sh root@<droplet-ip>
#   bash infra/fetch_results.sh root@<droplet-ip> --wait     # poll until the job finishes
#
# This is the other half of hard rule 0.4: nothing important lives on the Droplet, and the
# Droplet cannot push to us, so we pull. --wait exists so the owner can start the pull
# before the run ends and come back to a Droplet that is safe to destroy.
#
# It never destroys anything. Destroying is a deliberate console action by the owner
# (CLAUDE.md "Approved deviation from rule 0.2").
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_BUNDLE_DIR="${REMOTE_BUNDLE_DIR:-/work/results}"
REMOTE_RUNS_DIR="${REMOTE_RUNS_DIR:-/work/runs}"
LOCAL_DIR="${LOCAL_DIR:-$REPO_ROOT/runs/from-droplet}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT=0

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

HOST="${1:-}"
[[ -n "$HOST" ]] || die "usage: bash infra/fetch_results.sh root@<droplet-ip> [--wait]"
shift
while (($#)); do
  case "$1" in
    --wait) WAIT=1; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
remote() { ssh "${SSH_OPTS[@]}" "$HOST" "$@"; }

bold "Connection"
remote true || die "cannot ssh to $HOST (is the Droplet up, and is your key on it?)"
printf 'uptime: %s\n' "$(remote uptime -p 2>/dev/null || echo '?')"

# ---------------------------------------------------------------------- wait ---
if ((WAIT)); then
  bold "Waiting for a .done marker in $REMOTE_BUNDLE_DIR (poll ${POLL_SECONDS}s)"
  printf 'Ctrl-C is safe: it stops the polling, not the run.\n'
  while :; do
    if remote "ls $REMOTE_BUNDLE_DIR/*.done >/dev/null 2>&1"; then
      printf '\nmarker found\n'
      break
    fi
    # Show the last line of the newest log so the wait is not blind.
    tail_line="$(remote "tail -n 1 \$(ls -t /work/yolo-der/logs/*.log 2>/dev/null | head -1) 2>/dev/null" || true)"
    printf '\r\033[K  %s' "${tail_line:0:110}"
    sleep "$POLL_SECONDS"
  done
fi

# -------------------------------------------------------------------- bundles --
bold "Remote artifacts"
remote "ls -lh $REMOTE_BUNDLE_DIR 2>/dev/null" || printf 'no %s yet\n' "$REMOTE_BUNDLE_DIR"
remote "cat $REMOTE_BUNDLE_DIR/*.done 2>/dev/null" || true

mkdir -p "$LOCAL_DIR"
bold "Pulling into $LOCAL_DIR"
command -v rsync >/dev/null || die "rsync not found; install it (Git Bash: pacman -S rsync) or scp by hand"

# Two passes. The bundle first, because it is the self-contained copy that matters; then
# the raw runs tree, which is nicer to browse and lets MLflow be opened directly.
pulled_any=0
if rsync -ah --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
     "$HOST:$REMOTE_BUNDLE_DIR/" "$LOCAL_DIR/bundles/"; then
  pulled_any=1
else
  printf '\033[33m! no bundle pulled\033[0m\n'
fi
if rsync -ah --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
     "$HOST:$REMOTE_RUNS_DIR/" "$LOCAL_DIR/runs/"; then
  pulled_any=1
else
  printf '\033[33m! no runs tree pulled\033[0m\n'
fi
((pulled_any)) || die "nothing was pulled -- do NOT destroy the Droplet yet"

# ------------------------------------------------------------------- verify ---
bold "What arrived"
du -sh "$LOCAL_DIR"/* 2>/dev/null
weights="$(find "$LOCAL_DIR" -name 'best.pt' -o -name 'last.pt' 2>/dev/null | wc -l)"
results="$(find "$LOCAL_DIR" -name 'results.csv' 2>/dev/null | wc -l)"
dbs="$(find "$LOCAL_DIR" -name '*.db' 2>/dev/null | wc -l)"
printf 'checkpoints %s | results.csv %s | mlflow db %s\n' "$weights" "$results" "$dbs"
for bundle in "$LOCAL_DIR"/bundles/*.tar.zst "$LOCAL_DIR"/bundles/*.tar.gz; do
  [[ -f "$bundle" ]] || continue
  if tar -tf "$bundle" >/dev/null 2>&1; then
    printf '\033[32m  ok\033[0m %s reads cleanly (%s entries)\n' \
      "$(basename "$bundle")" "$(tar -tf "$bundle" | wc -l)"
  else
    printf '\033[31m  CORRUPT\033[0m %s -- re-pull before destroying\n' "$(basename "$bundle")"
  fi
done

if ((weights == 0 && results == 0)); then
  printf '\n\033[31m! no checkpoints and no results.csv arrived. Something is wrong --\n'
  printf '  keep the Droplet and look at %s/runs.\033[0m\n' "$LOCAL_DIR"
  exit 1
fi

cat <<BANNER

  Results are on this machine. The Droplet is now expendable.

  DESTROY IT: https://cloud.digitalocean.com/droplets -> the Droplet -> Destroy
  Powering off does NOT stop billing. Only Destroy does (hard rule 0.2).

  Then, locally:
    mlflow ui --backend-store-uri sqlite:///$LOCAL_DIR/runs/mlflow.db

BANNER
