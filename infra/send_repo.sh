#!/usr/bin/env bash
# Send this repo to a GPU Droplet over SSH, without needing rsync.
#
# Run it FROM THE LAPTOP, from the repo root:
#
#   bash infra/send_repo.sh root@<droplet-ip>
#
# Git for Windows ships no rsync but does ship tar and ssh, so this streams a tarball
# through the SSH connection instead. That is the whole trick; there is nothing to
# install. It is also the route to use whenever the GitHub repo is private or unpushed,
# which is why bootstrap.sh points at it when a clone is not possible.
#
# What is deliberately left behind:
#   .venv    Windows binaries, useless on Linux and ~1 GB
#   data     the Droplet rebuilds it from Roboflow, far quicker than this upload
#   runs     previous local runs; results travel the other way, via fetch_results.sh
#   mlruns   the local MLflow store, same reason
# .git is KEPT (under a megabyte) so bootstrap.sh can report the exact commit it is
# running, which is what ties an MLflow run to a revision.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_REPO="${REMOTE_REPO:-/work/yolo-der}"

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

HOST="${1:-}"
[[ -n "$HOST" ]] || die "usage: bash infra/send_repo.sh root@<droplet-ip>"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)

bold "Connection"
ssh "${SSH_OPTS[@]}" "$HOST" true \
  || die "cannot ssh to $HOST. Is the Droplet up, and is your key the one you chose in the create form?"
printf 'ok\n'

# The leading ./ matters: plain `--exclude=data` silently matches nothing under GNU
# tar, which is how a 10 MB upload becomes a 256 MB one.
#
# Run this with GNU tar (Git Bash), NOT with Windows tar.exe. Windows ships bsdtar,
# whose --exclude matches *every* path component, so `--exclude=./data` there also
# drops src/yolo_der/data and the upload arrives looking complete while the package is
# quietly missing a subpackage. That cost a confusing ModuleNotFoundError on a billing
# Droplet; the REQUIRED check below is what catches it either way.
EXCLUDES=(
  --exclude=./.venv
  --exclude=./data
  --exclude=./runs
  --exclude=./mlruns
  # DVC keeps a second full copy of the dataset here. Found the hard way: it was 96%
  # of the tarball, and the dataset is rebuilt on the Droplet anyway.
  --exclude=./.dvc/cache
  --exclude=./.pytest_cache
  --exclude=./.ruff_cache
  --exclude=./yolov8n.pt
)

# Every exclude is a chance to remove something load-bearing, and the failure shows up
# only later as an import error on the Droplet, so the manifest is checked here.
REQUIRED=(
  ./pyproject.toml
  ./src/yolo_der/__init__.py
  ./src/yolo_der/train.py
  ./src/yolo_der/trainer.py
  ./src/yolo_der/tracking.py
  ./src/yolo_der/data/prepare.py
  ./src/yolo_der/data/download.py
  ./src/yolo_der/data/severity.py
  ./src/yolo_der/models/der.py
  ./src/yolo_der/models/enhancenet.py
  ./src/yolo_der/models/inject.py
  ./configs/baseline.yaml
  ./infra/bootstrap.sh
  ./infra/run_job.sh
  ./infra/in_container.sh
)

bold "Checking the archive manifest"
MANIFEST="$(tar -tz "${EXCLUDES[@]}" -C "$REPO_ROOT" . 2>/dev/null)"
absent=()
for f in "${REQUIRED[@]}"; do
  grep -qxF "$f" <<<"$MANIFEST" || absent+=("$f")
done
if ((${#absent[@]})); then
  printf '  missing from the archive:\n'
  printf '    %s\n' "${absent[@]}"
  die "an exclude pattern removed files the package needs. If you are running this
  under Windows tar.exe (bsdtar), use Git Bash instead: bsdtar applies --exclude to
  every path component."
fi
printf '  all %s required paths present\n' "${#REQUIRED[@]}"

bold "Sending $REPO_ROOT -> $HOST:$REMOTE_REPO"
# Size first, so a surprise (a forgotten checkpoint, a stray dataset) is visible before
# the upload starts rather than halfway through it.
SIZE="$(tar -cz "${EXCLUDES[@]}" -C "$REPO_ROOT" . 2>/dev/null | wc -c)"
printf 'compressed size: %s KB\n' "$((SIZE / 1024))"
if ((SIZE > 200 * 1024 * 1024)); then
  die "that is larger than expected (>200 MB). Check for a dataset or checkpoint that
  should be excluded before uploading it over a home connection."
fi

ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE_REPO'" || die "cannot create $REMOTE_REPO"
tar -cz "${EXCLUDES[@]}" -C "$REPO_ROOT" . \
  | ssh "${SSH_OPTS[@]}" "$HOST" "tar -xz -C '$REMOTE_REPO'" \
  || die "the transfer failed; nothing on the Droplet should be trusted -- re-run it"

bold "Verifying"
# Checking the commit rather than just "did tar exit 0": a truncated stream can still
# unpack into something that looks plausible.
LOCAL_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
REMOTE_SHA="$(ssh "${SSH_OPTS[@]}" "$HOST" "git -C '$REMOTE_REPO' rev-parse HEAD 2>/dev/null" || true)"
printf 'local  %s\nremote %s\n' "$LOCAL_SHA" "${REMOTE_SHA:-<none>}"
[[ "$LOCAL_SHA" == "$REMOTE_SHA" ]] || die "the remote checkout does not match this one"

ssh "${SSH_OPTS[@]}" "$HOST" "ls '$REMOTE_REPO/infra/bootstrap.sh' >/dev/null" \
  || die "bootstrap.sh is missing on the Droplet"
printf '\033[32m  ok\033[0m same commit, bootstrap.sh present\n'

cat <<BANNER

  Next, on the Droplet:

    ssh $HOST
    ROBOFLOW_API_KEY=<key> bash $REMOTE_REPO/infra/bootstrap.sh

  That step is free. Nothing starts training until you run infra/run_job.sh.

BANNER
