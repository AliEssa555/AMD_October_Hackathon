#!/usr/bin/env bash
# Run one training job on the GPU Droplet under a hard wall-clock watchdog, then bundle
# every artifact for the pull home (spec 8.7, reshaped for the manual-launch deviation).
#
# Run it ON THE DROPLET, from /work/yolo-der:
#
#   RUNS_DIR=/work/runs MAX_RUNTIME_HOURS=1 \
#     bash infra/run_job.sh --config baseline --epochs-override 3
#
# Everything after the recognised flags below is passed through to
# `python -m yolo_der.train`, so --fraction, --batch, --resume etc. all work.
#
# Safeguards, in the order they fire (CLAUDE.md "Approved deviation from rule 0.2"):
#   1. `timeout` hard-kills training at MAX_RUNTIME_HOURS, so a hung dataloader cannot
#      quietly burn the credit.
#   2. An EXIT trap bundles logs + weights + MLflow store *on every path* -- success,
#      crash, timeout, Ctrl-C -- because artifacts must be safe before the Droplet dies
#      (hard rule 0.4). The bundle is written before the banner, never after.
#   3. A loud banner states the exit status and the exact commands to pull and destroy.
#   4. --poweroff is available for an unattended run, but refuses unless the bundle was
#      pushed to a remote, since powering off makes a local bundle unreachable while
#      still billing.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-/work/runs}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-6}"
# sqlite, not a file store: MLflow 3 deprecated the file store and the model registry
# needs a database backend. Four slashes -- sqlite:/// + the absolute /work path.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$RUNS_DIR/mlflow.db}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-yolo-der-exdark}"
# Optional: an s3-compatible destination for the bundle, e.g. s3://bucket/path.
# Unset in the current setup (no Spaces); the bundle is pulled over SSH instead.
RESULTS_S3_URI="${RESULTS_S3_URI:-}"
POWEROFF=0
DRY_RUN=0

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 2; }

PASS_THROUGH=()
while (($#)); do
  case "$1" in
    --poweroff) POWEROFF=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --max-hours) MAX_RUNTIME_HOURS="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) PASS_THROUGH+=("$1"); shift ;;
  esac
done
((${#PASS_THROUGH[@]})) || die "nothing to run; pass at least --config <variant>"

[[ "$MAX_RUNTIME_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "MAX_RUNTIME_HOURS must be a number"
# awk, not python3: the watchdog must be computable before we trust anything about the
# interpreter, and this script is also dry-run from the owner's Windows Git Bash.
TIMEOUT_SECONDS="$(awk -v h="$MAX_RUNTIME_HOURS" 'BEGIN{printf "%d", h * 3600}')"
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && ((TIMEOUT_SECONDS > 0))   || die "MAX_RUNTIME_HOURS must be > 0 (got '$MAX_RUNTIME_HOURS')"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB="job-$STAMP"
LOG_DIR="$REPO_ROOT/logs"
LOG="$LOG_DIR/$JOB.log"
# Deliberately NOT inside RUNS_DIR: tar would be archiving the file it is writing.
# fetch_results.sh looks here, so the two must agree on the path.
BUNDLE_DIR="${BUNDLE_DIR:-/work/results}"
BUNDLE="$BUNDLE_DIR/$JOB.tar.zst"
DONE_MARKER="$BUNDLE_DIR/$JOB.done"
mkdir -p "$LOG_DIR" "$RUNS_DIR" "$BUNDLE_DIR"

# ------------------------------------------------------------------ the trap --
# Defined before training starts so there is no window in which a crash loses the logs.
bundled=0
bundle_results() {
  local status=$1
  ((bundled)) && return 0
  bundled=1
  bold "Bundling results (exit status $status)"

  # --ignore-failed-read: a timeout can leave a half-written checkpoint, and a bundle
  # missing one file beats no bundle at all. `reports` may legitimately not exist yet.
  local sources=(-C "$RUNS_DIR" .)
  [[ -d "$REPO_ROOT/logs" ]] && sources+=(-C "$REPO_ROOT" logs)
  [[ -d "$REPO_ROOT/reports" ]] && sources+=(-C "$REPO_ROOT" reports)

  # zstd if present, gzip otherwise. Checked rather than assumed: `tar -I zstd` on a host
  # without zstd exits non-zero having already created an empty file, which looks exactly
  # like a successful bundle to anyone reading `ls -lh`.
  if command -v zstd >/dev/null 2>&1; then
    tar --ignore-failed-read -I 'zstd -3 -T0' -cf "$BUNDLE" "${sources[@]}" 2>/dev/null
  else
    BUNDLE="${BUNDLE%.tar.zst}.tar.gz"
    printf '  zstd not installed, falling back to gzip\n'
    tar --ignore-failed-read -czf "$BUNDLE" "${sources[@]}" 2>/dev/null
  fi

  # This check is what stands between the owner and destroying a Droplet whose results
  # never actually left it (hard rule 0.4), so it is loud and it names the fix.
  local size=0
  [[ -f "$BUNDLE" ]] && size="$(wc -c <"$BUNDLE" | tr -d ' ')"
  if [[ "$size" -gt 0 ]] && tar -tf "$BUNDLE" >/dev/null 2>&1; then
    printf '\033[32m  ok\033[0m %s (%s)\n' "$BUNDLE" "$(du -h "$BUNDLE" | cut -f1)"
  else
    printf '\033[31m  ! the bundle is empty or unreadable (%s bytes). DO NOT destroy the\n' "$size"
    printf '    Droplet: copy %s off by hand first.\033[0m\n' "$RUNS_DIR"
  fi
  printf '%s\n' "status=$status" "job=$JOB" "bundle=$BUNDLE" "bundle_bytes=$size" \
    > "$DONE_MARKER"
}

pushed=0
STATUS=0
on_exit() {
  local status=$?
  bundle_results "$status"
  final_banner "$status"
  if ((POWEROFF)); then
    if ((pushed)); then
      printf '\npowering off in 60s (the bundle is safe at %s)\n' "$RESULTS_S3_URI"
      sleep 60 && poweroff
    else
      printf '\033[33m! --poweroff ignored: the bundle was never pushed, so powering off\n'
      printf '  would make it unreachable while the Droplet still bills. Pull it, then\n'
      printf '  destroy the Droplet in the console.\033[0m\n'
    fi
  fi
}

final_banner() {
  local status=$1 colour=32 word=SUCCEEDED
  if ((status == 124)); then colour=33; word="TIMED OUT after ${MAX_RUNTIME_HOURS}h (watchdog)"
  elif ((status != 0));  then colour=31; word="FAILED (exit $status)"; fi
  printf '\n\033[%sm' "$colour"
  printf '########################################################################\n'
  printf '##  %-66s##\n' "RUN $word"
  printf '##  %-66s##\n' "job $JOB"
  printf '########################################################################\n'
  printf '\033[0m'
  printf 'bundle: %s\n' "$BUNDLE"
  cat <<BANNER

  From the laptop, pull the results and then DESTROY the Droplet:

    bash infra/fetch_results.sh root@<droplet-ip>

  A powered-off Droplet is STILL BILLED. Only destroying it in the console at
  https://cloud.digitalocean.com/droplets stops the meter (hard rule 0.2).

BANNER
}
trap on_exit EXIT

# --------------------------------------------------------------- the run plan --
bold "Run plan"
COST="$(awk -v h="$MAX_RUNTIME_HOURS" -v r="${GPU_HOURLY_USD:-1.99}" 'BEGIN{printf "%.2f", h * r}')"
cat <<PLAN
  command        python -m yolo_der.train ${PASS_THROUGH[*]}
  repo           $REPO_ROOT  ($(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?'))
  runs dir       $RUNS_DIR
  watchdog       ${MAX_RUNTIME_HOURS}h  (${TIMEOUT_SECONDS}s, SIGTERM then SIGKILL after 120s)
  worst-case     ${MAX_RUNTIME_HOURS}h x \$${GPU_HOURLY_USD:-1.99}/hr = \$${COST}
  mlflow         $MLFLOW_TRACKING_URI
  log            $LOG
  bundle         $BUNDLE
PLAN

# Not fatal in a dry-run on a laptop; fatal below if we are actually about to train.
GPU_LINE="$(python3 - 2>/dev/null <<'PY'
import torch
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
print(f"{name}  hip={torch.version.hip}")
PY
)" || GPU_LINE=""
printf '  gpu            %s
' "${GPU_LINE:-<no python3/torch here>}"

if ((DRY_RUN)); then
  bold "--dry-run: nothing was launched"
  trap - EXIT
  exit 0
fi

# A paid run with no GPU is the single most expensive mistake available here, so this
# is a hard stop rather than a warning.
case "$GPU_LINE" in
  ""|*NONE*|*"hip=None"*) die "no ROCm GPU visible -- refusing to start a paid run ($GPU_LINE)" ;;
esac

# ------------------------------------------------------------------ training ---
bold "Training"
export RUNS_DIR
# `timeout` is safeguard 1. --signal=TERM lets Ultralytics' own handlers write last.pt;
# --kill-after is the backstop for a process that ignores TERM. Exit 124 means it fired.
set -o pipefail
timeout --signal=TERM --kill-after=120s "${TIMEOUT_SECONDS}s" \
  python3 -u -m yolo_der.train "${PASS_THROUGH[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
exit "$STATUS"
