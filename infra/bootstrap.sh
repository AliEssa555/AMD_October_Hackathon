#!/usr/bin/env bash
# Prepare a freshly created AMD MI300X GPU Droplet for training (spec 8.6, reshaped for
# the manual-launch deviation recorded in CLAUDE.md).
#
# Run it ON THE DROPLET, as root, straight after the first SSH login:
#
#   ssh root@<droplet-ip>
#   curl -fsSL https://raw.githubusercontent.com/<owner>/yolo-der/main/infra/bootstrap.sh -o bootstrap.sh
#   bash bootstrap.sh            # or: GIT_REF=some-branch bash bootstrap.sh
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
command -v rocm-smi >/dev/null || die "rocm-smi missing -- wrong image? expected a ROCm image"
rocm-smi --showproductname || die "rocm-smi failed"
GPU_NAME="$(rocm-smi --showproductname --csv 2>/dev/null | awk -F, 'NR==2{print $2}')"
printf 'product: %s\n' "${GPU_NAME:-unknown}"
case "${GPU_NAME:-}" in
  *MI300*) ok "MI300X present" ;;
  *) printf '\033[33m! not an MI300X (%s) -- continuing, but check the plan\033[0m\n' "${GPU_NAME:-?}" ;;
esac

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
bold "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# zstd for the results bundle, rsync for the pull home, git-lfs not needed.
apt-get install -y -qq git rsync zstd jq >/dev/null
ok "git rsync zstd jq"

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
python3 -m pip install --quiet --upgrade pip
# ultralytics with --no-deps so its opencv-python requirement cannot replace the
# headless wheel, and so its torch pin cannot touch the ROCm build. Everything it
# actually needs at runtime is listed in the [train] extra (see pyproject.toml).
python3 -m pip install --quiet --no-deps "ultralytics==8.3.*"
python3 -m pip install --quiet \
  "polars>=0.20.0" "mlflow==3.*" "boto3>=1.34" "opencv-python-headless>=4.9" \
  "pandas>=2.2" "pyarrow>=16.0" "scipy>=1.13" "matplotlib>=3.8" "roboflow>=1.1" \
  "pyyaml>=6.0" "numpy>=1.26" "tqdm>=4.66" "psutil>=5.9" "py-cpuinfo"
python3 -m pip install --quiet --no-deps -e "$REPO"
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
from ultralytics.utils import SETTINGS, settings
settings.update({"mlflow": False, "dvc": False, "sync": False})
print("mlflow:", SETTINGS.get("mlflow"), "dvc:", SETTINGS.get("dvc"), "sync:", SETTINGS.get("sync"))
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
python3 - <<PY
import yaml, pathlib
d = pathlib.Path("$REPO/data/prepared")
doc = yaml.safe_load((d / "data.yaml").read_text(encoding="utf-8"))
for split in ("train", "val", "test"):
    n = len(list((d / split / "images").glob("*")))
    print(f"{split:5s} {n:5d} images")
    assert n, f"{split} is empty"
print("nc", doc["nc"], doc["names"])
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
