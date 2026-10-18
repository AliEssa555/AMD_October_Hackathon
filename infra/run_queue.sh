#!/usr/bin/env bash
# Chain the rest of the night's work so it survives this SSH session ending.
#
# Run it ON THE DROPLET, detached, and **inside the container**:
#
#   cd /work/yolo-der
#   nohup bash infra/in_container.sh bash infra/run_queue.sh > /work/queue.log 2>&1 < /dev/null &
#
# The `in_container.sh` wrapper is not optional. The AMD image ships no torch on the
# host, so a queue launched directly on the host fails every run instantly with
# "no ROCm GPU visible -- refusing to start a paid run" and then tries to import
# yolo_der from a Python that has never heard of it. Learned twice now: once when
# bootstrap.sh first ran, and once here, launching this very script.
#
# It waits for any batch already running, then trains, then evaluates, then collects
# everything into /work/takehome. Each stage logs its own file, and a stage failing does
# not stop the ones after it: a half-finished night still leaves artifacts on disk and a
# take-home tree to pull.
#
# Why this exists: the supervision, not the training, is what broke last time. The BDD
# batch finished in 78 minutes and the Droplet then idled for ~85 more, because the
# session watching it ended and the watcher died with it while `nohup`-ed training
# carried on. There is no reaper (CLAUDE.md, approved deviation from rule 0.2), so the
# work has to be queued on the machine rather than driven from a laptop.
#
# It never destroys the Droplet. That stays a deliberate console action by the owner.
set -uo pipefail

REPO=/work/yolo-der
RUNS=/work/runs
export RUNS_DIR="$RUNS"
export GPU_HOURLY_USD="${GPU_HOURLY_USD:-1.99}"

cd "$REPO"

# Refuse early rather than failing seven runs in four seconds. This script is the one
# that chains the night together, so a mislaunch here wastes the whole window.
if ! python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  printf '[31mrun_queue.sh is not inside the ROCm container (no GPU-capable torch).
'
  printf 'Relaunch it as:
'
  printf '  nohup bash infra/in_container.sh bash infra/run_queue.sh > /work/queue.log 2>&1 &[0m
'
  exit 2
fi

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
bold()  { printf '\n\033[1m######## %s  [%s]\033[0m\n' "$1" "$(stamp)"; }

bold "Waiting for any batch already running"
waited=0
while pgrep -f 'run_batch.sh' >/dev/null; do
  sleep 60
  waited=$((waited + 1))
  (( waited % 10 == 0 )) && printf '  still waiting, %s min\n' "$waited"
  (( waited > 240 )) && { echo "  gave up waiting after 4 h"; break; }
done
printf '  clear after %s min\n' "$waited"

# ---------------------------------------------------------------- 1. training ----
# The E1 entropy sweep first: it is the question nobody has actually asked, and it is
# cheap. Then seeds, which turn every delta already measured into a result or a non-result.
# 0.6 h per run against a measured ~10 min for 100 ExDark epochs, i.e. 3.5x margin.
# The BDD batch used 1.2 h against a 73-minute der_enh run -- 1.15x -- and the
# watchdog duly killed it two epochs short. Module variants run ~40% slower per epoch
# than the baseline they were timed against, so the margin has to cover that.
bold "Stage 1: entropy sweep + seeds"
RUN_MAX_HOURS=0.6 BATCH_MAX_HOURS=2.5 SUMMARY="$RUNS/night-summary.tsv" \
  bash "$REPO/infra/run_batch.sh" \
    der_enh_ent01 der_enh_ent10 der_enh_ent50 \
    baseline:1 baseline:2 der_enh_sevcond:1 der_enh_sevcond:2 \
  > /work/night_batch.log 2>&1
printf '  stage 1 exit %s\n' "$?"
cat "$RUNS/night-summary.tsv" 2>/dev/null

# -------------------------------------------------------------- 2. evaluation ----
# Everything that has a best.pt, on its own dataset. The BDD runs carry 11 condition
# bins (tod_night, wx_rainy, ...), which is the comparison the dataset switch was for.
bold "Stage 2: evaluation"
python3 -m yolo_der.evaluate \
  --runs-dir "$RUNS" --device 0 --batch 32 --reports "$REPO/reports/eval" \
  > /work/eval_all.log 2>&1
printf '  evaluation exit %s\n' "$?"
tail -5 /work/eval_all.log

bold "Stage 2b: entropy and gates on the new runs"
for d in "$RUNS"/der_enh_ent*-s0 "$RUNS"/der_enh_sevcond-s*; do
  [ -d "$d" ] || continue
  python3 -m yolo_der.analysis.entropy --run "$d" \
    --data "$REPO/data/prepared/data.yaml" --severity "$REPO/data/severity.csv" \
    --limit 64 --device 0 --out "$REPO/reports/entropy" >> /work/entropy_all.log 2>&1
  python3 -m yolo_der.analysis.record_gates --run "$d" \
    --data "$REPO/data/prepared/data.yaml" --severity "$REPO/data/severity.csv" \
    --limit 200 --device 0 --out "$REPO/reports/gates" >> /work/gates_all.log 2>&1
done
printf '  entropy/gates done\n'

# ----------------------------------------------------------------- 3. reports ----
bold "Stage 3: tables and figures"
python3 -m yolo_der.evaluate --aggregate --reports "$REPO/reports/eval" --out "$REPO/reports" \
  >> /work/eval_all.log 2>&1
python3 -m yolo_der.analysis.plots --reports "$REPO/reports" --out "$REPO/reports/figures" \
  >> /work/eval_all.log 2>&1
ls -1 "$REPO/reports/figures" 2>/dev/null | sed 's/^/  figure: /'

# ----------------------------------------------------------------- 4. collect ----
# Hard rule 0.4: nothing important may live only on this Droplet.
bold "Stage 4: collect everything portable"
rm -rf /work/takehome
python3 "$REPO/scripts/collect_results.py" --runs "$RUNS" --out /work/takehome 2>&1 | tail -12
tar -I 'zstd -3 -T0' -cf /work/takehome.tar.zst -C /work takehome 2>/dev/null
ls -lh /work/takehome.tar.zst 2>/dev/null

bold "QUEUE FINISHED"
cat <<BANNER

  Pull it:
    scp root@<ip>:/work/takehome.tar.zst .

  Then DESTROY the Droplet in the console. It is idle from now on, and idle is
  billed at \$${GPU_HOURLY_USD}/hr exactly like training.

BANNER
