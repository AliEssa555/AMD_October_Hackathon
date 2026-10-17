#!/usr/bin/env bash
# Prepare a freshly created AMD MI300X GPU Droplet for training (spec 8.6, reshaped for
# the manual-launch deviation recorded in CLAUDE.md).
#
# Run it INSIDE the ROCm PyTorch container, via the host-side wrapper:
#
#   bash /work/yolo-der/infra/in_container.sh bash infra/bootstrap.sh
#
# Not on the host. The AMD image ships no torch on the host at all -- `python3` there
# is a bare system 3.12 -- and PyTorch lives only inside the preloaded
# rocm/pytorch:latest image. in_container.sh explains the layout and bind-mounts /work
# so the paths match on both sides.
#
# Idempotent: re-running it re-verifies everything and skips what is already done.
#
# It does NOT start training and does NOT create any cloud resource. Training is
# infra/run_job.sh, which the banner at the end tells you how to call.
set -euo pipefail

# Fixed paths. They must be byte-identical on every Droplet: Ultralytics writes absolute
# paths into last.pt, so a resume on a differently-laid-out machine looks for a dataset
# that is not there (CLAUDE.md conventions, spec gotcha 7).
WORK=/work
REPO="$WORK/yolo-der"
RUNS="$WORK/runs"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/AliEssa555/AMD_October_Hackathon.git}"
GIT_REF="${GIT_REF:-main}"

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root"

# ------------------------------------------------------------------ hardware --
# First, because everything after it is wasted money if this is not an MI300X.
bold "GPU"
# rocm-smi is present in this image, but it is not the authority on whether *torch*
# can use the GPU, so a failure here is a warning and the torch check below is fatal.
if command -v rocm-smi >/dev/null; then
  rocm-smi --showproductname || printf '  rocm-smi failed\n'
  GPU_NAME="$(rocm-smi --showproductname --csv 2>/dev/null | awk -F, 'NR==2{print $2}')"
  printf 'product: %s\n' "${GPU_NAME:-unknown}"
  case "${GPU_NAME:-}" in
    *MI300*) ok "MI300X present" ;;
    *) printf '\033[33m! not an MI300X (%s)\033[0m\n' "${GPU_NAME:-?}" ;;
  esac
else
  printf '  no rocm-smi here; relying on the torch check\n'
fi

bold "PyTorch / HIP"
# The image ships torch built for ROCm. We must never reinstall torch: a pip resolver
# that pulls the CUDA wheel over it silently turns a $1.99/hr GPU run into a CPU run.
python3 - <<'PY' || die "torch/HIP check failed"
import sys, torch
print("torch        ", torch.__version__)
print("torch.version.hip", torch.version.hip)
print("cuda.is_available", torch.cuda.is_available())
print("device_count ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name  ", torch.cuda.get_device_name(0))
if not torch.version.hip:
    sys.exit("this torch is not a ROCm build -- refusing")
if not torch.cuda.is_available():
    sys.exit("torch cannot see the GPU")
PY
ok "ROCm torch sees the GPU"
TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)')"

# ------------------------------------------------------------------- system ---
bold "Command-line tools"
# Checked, not apt-installed. This container already has git, tar, zstd, awk and
# timeout, and `apt-get update` inside it is a slow way to confirm that. Anything
# genuinely missing is named here so the fix is obvious rather than a later crash.
missing=()
for cmd in git tar zstd awk timeout python3; do
  command -v "$cmd" >/dev/null || missing+=("$cmd")
done
if ((${#missing[@]})); then
  printf 'installing: %s\n' "${missing[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq "${missing[@]}" >/dev/null \
    || die "could not install: ${missing[*]}"
fi
ok "git tar zstd awk timeout python3"

mkdir -p "$WORK" "$RUNS"

# --------------------------------------------------------------------- repo ---
bold "Repository"
# Three ways the repo can get here, in order of preference. The clone is only
# attempted if the remote is actually reachable: it may be private or not pushed yet,
# and a bare `git clone` failure is a confusing way to discover that at $1.99/hr.
if [[ -d "$REPO/.git" ]]; then
  # Already here: a re-run, or the owner rsynced it up (see docs/MANUAL_LAUNCH.md).
  if git -C "$REPO" fetch --quiet origin "$GIT_REF" 2>/dev/null; then
    git -C "$REPO" reset --hard --quiet "origin/$GIT_REF"
  else
    printf '  no reachable remote; using the working tree as it is\n'
  fi
elif git ls-remote --exit-code "$GIT_REPO_URL" >/dev/null 2>&1; then
  git clone --quiet --branch "$GIT_REF" "$GIT_REPO_URL" "$REPO"
else
  die "no repo at $REPO, and $GIT_REPO_URL is not reachable (private, or never
  pushed?). Send it up from the laptop instead, then re-run this script:
    rsync -ah --exclude .venv --exclude data --exclude runs \\
      ./ root@<droplet-ip>:/work/yolo-der/"
fi
printf 'at %s\n' "$(git -C "$REPO" log --oneline -1)"
ok "$REPO"

# ------------------------------------------------------------------- python ---
bold "Python dependencies"
# No venv: the image's system python owns the ROCm torch build, and a venv without
# --system-site-packages would hide it. Install into it, never over torch.
# Deliberately NOT upgrading pip: this is a conda environment that owns torch, and
# there is no reason to touch its package manager during a paid run.
# ultralytics with --no-deps so its opencv-python requirement cannot replace the
# headless wheel, and so its torch pin cannot touch the ROCm build. Everything it
# actually needs at runtime is listed in the [train] extra (see pyproject.toml).
python3 -m pip install --quiet --no-deps "ultralytics==8.3.*"
python3 -m pip install --quiet \
  "polars>=0.20.0" "mlflow==3.*" "boto3>=1.34" "opencv-python-headless>=4.9" \
  "pandas>=2.2" "pyarrow>=16.0" "scipy>=1.13" "matplotlib>=3.8" "roboflow>=1.1" \
  "pyyaml>=6.0" "tqdm>=4.66" "psutil>=5.9" "py-cpuinfo"
python3 -m pip install --quiet --no-deps -e "$REPO"

# NumPy 1.x, pinned deliberately. The ROCm torch 2.6.0 in this image was compiled
# against NumPy 1.x, and `pip install numpy>=1.26` resolves to 2.2.6, which makes torch
# print "Failed to initialize NumPy" and then fail on any array conversion -- i.e. on
# every image the dataloader touches. The local CPU wheel is fine with NumPy 2, so this
# constraint belongs here, on the Droplet, not in pyproject.toml.
python3 -m pip install --quiet "numpy<2" || die "could not install numpy<2"
python3 - <<'PY' || die "torch cannot use numpy; training would fail on the first batch"
import numpy, torch
x = torch.from_numpy(numpy.zeros((2, 2), dtype="float32"))
assert x.sum().item() == 0.0
print(f"  numpy {numpy.__version__} usable from torch {torch.__version__}")
PY
ok "numpy pinned below 2 and verified through torch"
ok "installed"

TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
[[ "$TORCH_BEFORE" == "$TORCH_AFTER" ]] \
  || die "pip replaced torch ($TORCH_BEFORE -> $TORCH_AFTER). The ROCm build is gone; rebuild the Droplet."
python3 -c 'import torch,sys; sys.exit(0 if torch.version.hip and torch.cuda.is_available() else 1)' \
  || die "torch lost the GPU after installing dependencies"
ok "torch still $TORCH_AFTER on HIP, GPU visible"

bold "Ultralytics settings"
# Both built-in loggers off: yolo_der.tracking owns MLflow (its metric-name sanitising and
# MLflow-3 model registration), and DVC has no remote here.
python3 - <<'PY'
from ultralytics import settings  # package level; ultralytics.utils has no `settings`
settings.update({"mlflow": False, "dvc": False})
print("mlflow:", settings.get("mlflow"), "dvc:", settings.get("dvc"))
PY
ok "built-in loggers disabled"

# --------------------------------------------------------------------- data ---
bold "Dataset"
# Rebuilt on the Droplet rather than copied from the laptop: the Roboflow download is a
# few hundred MB over the Droplet's link, which is far quicker than the owner's upload,
# and it keeps the dataset reproducible from params.yaml.
if [[ -f "$REPO/data/prepared/data.yaml" ]]; then
  ok "data/prepared already present"
else
  [[ -n "${ROBOFLOW_API_KEY:-}" ]] \
    || die "ROBOFLOW_API_KEY is unset. Re-run as: ROBOFLOW_API_KEY=... bash bootstrap.sh"
  cd "$REPO"
  python3 -m yolo_der.data.download --out data/raw
  python3 -m yolo_der.data.prepare --raw data/raw --out data/prepared
  python3 -m yolo_der.data.severity --data data/prepared --out data/severity.csv --bins data/bins
  ok "download + prepare + severity"
fi
# The split directories are read from data.yaml, not assumed. The Roboflow export
# calls the validation split "valid", so the obvious `data/prepared/val` check failed
# on a dataset that was in fact perfectly fine -- a false alarm is expensive here,
# because the natural reaction to it is to rebuild a 6,554-image dataset.
python3 - <<PY || die "the prepared dataset is not usable"
import pathlib, yaml

d = pathlib.Path("$REPO/data/prepared")
doc = yaml.safe_load((d / "data.yaml").read_text(encoding="utf-8"))
for key in ("train", "val", "test"):
    rel = doc[key]
    images = sorted((d / rel).glob("*"))
    labels = sorted((d / rel.replace("images", "labels")).glob("*.txt"))
    print(f'  {key:5s} {rel:16s} {len(images):5d} images  {len(labels):5d} labels')
    assert images, f'{key} ({rel}) is empty'
print('  nc', doc['nc'], doc['names'])
PY
ok "splits non-empty"

# ----------------------------------------------------------------- pretrained --
bold "Pretrained weights"
# Fetched now, not at train time: a failed download at minute 1 of a paid run is the
# cheapest failure to move earlier.
cd "$REPO"
python3 -c "from ultralytics.utils.downloads import attempt_download_asset as d; print(d('yolov8n.pt'))"
ok "yolov8n.pt"

bold "Ready"
cat <<BANNER

  repo   $REPO   ($(git -C "$REPO" rev-parse --short HEAD))
  data   $REPO/data/prepared
  runs   $RUNS

  Next, from $REPO:

    RUNS_DIR=$RUNS MAX_RUNTIME_HOURS=1 \
      bash infra/run_job.sh --config baseline --epochs-override 3

  run_job.sh carries the watchdog and bundles the results for the pull home.
  Remember: powering this Droplet off does NOT stop billing. Only destroying it
  in the console does (hard rule 0.2).

BANNER
