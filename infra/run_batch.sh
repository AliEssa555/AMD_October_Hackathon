#!/usr/bin/env bash
# Run several variants back to back on one GPU Droplet (spec 9.3 batches).
#
# Run it ON THE DROPLET, from /work/yolo-der:
#
#   BATCH_MAX_HOURS=2 bash infra/in_container.sh bash infra/run_batch.sh der enh der_enh
#   BATCH_MAX_HOURS=2 bash infra/in_container.sh bash infra/run_batch.sh baseline:1 baseline:2
#
# A variant may carry a seed as `variant:seed` (default 0). Seeds are how a delta
# becomes a result rather than an anecdote: the ExDark ablation's whole spread across
# seven variants was 0.0056 mAP50, which is the same order as the deltas being claimed.
#
# Sequential, not parallel, and deliberately so: a 4-way test on this hardware stalled.
# Four processes spawn ~104 dataloader workers against 20 vCPU and contend on a single
# MIOpen kernel cache -- the GPU reported 100% busy while every log sat frozen. One
# process already finishes a 100-epoch run in about ten minutes, so there is nothing to
# gain. Measured 17 Oct 2026, recorded in CLAUDE.md.
#
# Two nested caps, because the per-run watchdog alone does not bound a batch:
#   * each run gets RUN_MAX_HOURS (default 0.4 h, ~2.4x the measured 10 min);
#   * the whole batch gets BATCH_MAX_HOURS (default 2 h), so worst-case spend is one
#     number the owner can be quoted up front rather than runs x watchdog.
# Variants are ordered by the caller; put the ones the write-up needs first, because a
# batch that runs out of time loses the tail, not the head.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-/work/runs}"
RUN_MAX_HOURS="${RUN_MAX_HOURS:-0.4}"
BATCH_MAX_HOURS="${BATCH_MAX_HOURS:-2}"
GPU_HOURLY_USD="${GPU_HOURLY_USD:-1.99}"
SUMMARY="${SUMMARY:-$RUNS_DIR/batch-summary.tsv}"

bold() { printf '\n\033[1m######## %s\033[0m\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 2; }

(($#)) || die "name at least one variant, e.g. der enh der_enh"
VARIANTS=("$@")

BATCH_SECONDS="$(awk -v h="$BATCH_MAX_HOURS" 'BEGIN{printf "%d", h * 3600}')"
COST="$(awk -v h="$BATCH_MAX_HOURS" -v r="$GPU_HOURLY_USD" 'BEGIN{printf "%.2f", h * r}')"

bold "Batch plan"
cat <<PLAN
  variants       ${VARIANTS[*]}
  runs           ${#VARIANTS[@]}
  per-run cap    ${RUN_MAX_HOURS} h
  batch cap      ${BATCH_MAX_HOURS} h  (${BATCH_SECONDS}s)
  worst case     ${BATCH_MAX_HOURS} h x \$${GPU_HOURLY_USD}/hr = \$${COST}
  runs dir       $RUNS_DIR
  summary        $SUMMARY
PLAN

mkdir -p "$RUNS_DIR"
[[ -f "$SUMMARY" ]] || printf 'variant\tstatus\tseconds\tmAP50\tmAP50_95\n' > "$SUMMARY"

batch_start=$(date +%s)
cd "$REPO_ROOT"

for variant in "${VARIANTS[@]}"; do
  elapsed=$(( $(date +%s) - batch_start ))
  remaining=$(( BATCH_SECONDS - elapsed ))
  if (( remaining < 120 )); then
    bold "Batch cap reached; skipping $variant and everything after it"
    printf '%s\tSKIPPED_BATCH_CAP\t0\t\t\n' "$variant" >> "$SUMMARY"
    continue
  fi

  # `variant:seed`, seed 0 when omitted.
  seed="${variant##*:}"
  if [[ "$seed" == "$variant" ]]; then seed=0; else variant="${variant%%:*}"; fi
  run_name="$variant-s$seed"

  bold "$run_name  (${elapsed}s into the batch, ${remaining}s of budget left)"
  run_start=$(date +%s)

  # Each run keeps its own watchdog, EXIT trap and bundle: one variant crashing must not
  # cost the batch the runs that already succeeded (hard rule 0.4).
  RUNS_DIR="$RUNS_DIR" MAX_RUNTIME_HOURS="$RUN_MAX_HOURS" GPU_HOURLY_USD="$GPU_HOURLY_USD" \
    bash "$REPO_ROOT/infra/run_job.sh" --config "$variant" --seed "$seed"
  status=$?
  seconds=$(( $(date +%s) - run_start ))

  # Read the achieved metrics straight from results.csv, so the summary does not depend
  # on parsing Ultralytics' progress output.
  read -r map50 map5095 < <(
    python3 - "$RUNS_DIR/$run_name/results.csv" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(" "); raise SystemExit
rows = list(csv.DictReader(p.open()))
if not rows:
    print(" "); raise SystemExit
k50 = next((c for c in rows[0] if "mAP50(B)" in c), None)
k95 = next((c for c in rows[0] if "mAP50-95(B)" in c), None)
best = max(rows, key=lambda r: float(r[k50] or 0))
print(f"{float(best[k50]):.4f} {float(best[k95]):.4f}")
PY
  )

  case $status in
    0)   word=OK ;;
    124) word=TIMEOUT ;;
    *)   word="FAILED_$status" ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\n' "$run_name" "$word" "$seconds" "${map50:-}" "${map5095:-}" \
    >> "$SUMMARY"
  printf '\n>>> %s: %s in %ss  mAP50=%s mAP50-95=%s\n' \
    "$run_name" "$word" "$seconds" "${map50:-?}" "${map5095:-?}"
done

bold "Batch finished in $(( ($(date +%s) - batch_start) / 60 )) min"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
printf '\nRemember: destroying the Droplet in the console is what stops the bill.\n'
