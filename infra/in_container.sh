#!/usr/bin/env bash
# Run a command inside the ROCm PyTorch container on the GPU Droplet.
#
# Run it ON THE DROPLET (host side), from anywhere:
#
#   bash /work/yolo-der/infra/in_container.sh bash infra/bootstrap.sh
#   bash /work/yolo-der/infra/in_container.sh bash infra/run_job.sh --config baseline
#   bash /work/yolo-der/infra/in_container.sh            # interactive shell
#
# Why this exists, discovered on the live Droplet rather than assumed: the AMD Developer
# Cloud "PyTorch on ROCm" image does **not** install torch on the host. `python3` there is
# a bare system 3.12 with no torch at all. PyTorch ships inside two preloaded Docker
# images instead:
#
#   rocm/pytorch:latest   torch 2.6.0+git684f6f2, HIP 6.4.43483, python 3.12.10 (conda)
#   rocm:latest           the same torch, wrapped in a Jupyter server that the image
#                         starts automatically as the container named `rocm`
#
# Both see the MI300X. We use rocm/pytorch because it is the plain one: no Jupyter, no
# ports, nothing competing for the GPU.
#
# This is also why spec Section 0's convention that the repo lives at /work/yolo-der and
# runs at /work/runs matters: /work is bind-mounted, so those paths are identical on the
# host and inside the container, and an Ultralytics checkpoint's absolute paths stay valid
# across both (gotcha 7).
set -uo pipefail

# Environment that must cross into the container. `docker run -e` only applies when
# the container is CREATED, and this script deliberately reuses a running one -- so
# anything the caller sets per-run has to be passed on `docker exec` too. Missing that
# made MAX_RUNTIME_HOURS=1 silently become the 6h default inside the container, i.e.
# the watchdog that caps spend was reading a different number from the one printed to
# the owner. Found on the live Droplet: the run plan said 6h x $1.99 = $11.94.
FORWARD=(
  MAX_RUNTIME_HOURS
  RUNS_DIR
  BUNDLE_DIR
  MLFLOW_TRACKING_URI
  MLFLOW_EXPERIMENT_NAME
  GPU_HOURLY_USD
  RESULTS_S3_URI
  ROBOFLOW_API_KEY
  PYTHONUNBUFFERED
)

# Build the -e flags from whatever is actually set, so an unset variable keeps the
# script's own default instead of being forced to an empty string.
env_flags() {
  local out=() name
  for name in "${FORWARD[@]}"; do
    if [[ -n "${!name-}" ]]; then out+=(-e "$name=${!name}"); fi
  done
  printf '%s\n' "${out[@]}"
}

IMAGE="${YOLO_DER_IMAGE:-rocm/pytorch:latest}"
NAME="${YOLO_DER_CONTAINER:-yolo-der}"
WORK=/work
REPO="$WORK/yolo-der"

bold() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

command -v docker >/dev/null || die "docker not found -- is this the AMD ROCm image?"
[[ -d "$REPO" ]] || die "$REPO does not exist; send the repo up first (infra/send_repo.sh)"

# --ipc=host and a large --shm-size are not optional: Ultralytics' dataloader workers
# share memory through /dev/shm, and the Docker default of 64 MB makes them die with an
# opaque "DataLoader worker (pid N) is killed by signal: Bus error".
# No --group-add render. Many ROCm examples include it, but docker resolves group
# names against the *image's* /etc/group, not the host's -- rocm/pytorch:latest has no
# `render` group, and docker then refuses the entire run with "Unable to find group
# render" instead of skipping it. Probing the host was the wrong fix for the same
# reason. `video` alone is enough: verified on this Droplet, torch inside the container
# reports the MI300X and torch.cuda.is_available() is True.

start_container() {
  local RUN_ENV=()
  mapfile -t RUN_ENV < <(PYTHONUNBUFFERED=1 env_flags)
  bold "Starting container $NAME from $IMAGE"
  docker run -d --name "$NAME" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host --shm-size 16G \
    --security-opt seccomp=unconfined \
    -v "$WORK:$WORK" -w "$REPO" \
    "${RUN_ENV[@]}" \
    "$IMAGE" sleep infinity >/dev/null \
    || die "could not start the container"
}

state="$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | tr -d '[:space:]')"
# Empty, not the literal "absent", is what a failed inspect actually leaves here once
# the whitespace is stripped -- the first version of this matched neither arm.
[[ -n "$state" ]] || state=absent
case "$state" in
  running) : ;;
  absent)  start_container ;;
  *)
    # A stopped container has the old environment baked in, including a stale or empty
    # ROBOFLOW_API_KEY, so it is replaced rather than restarted.
    bold "Replacing the $state container $NAME"
    docker rm -f "$NAME" >/dev/null
    start_container
    ;;
esac

# Confirm the GPU is visible *through the container* before handing it any work. The host
# seeing an MI300X via rocm-smi does not prove the container can use it: that needs
# /dev/kfd, /dev/dri and the video group, any one of which can be missing silently.
bold "GPU, from inside the container"
docker exec "$NAME" python3 -c '
import torch
assert torch.version.hip, "not a ROCm build"
assert torch.cuda.is_available(), "no GPU visible in the container"
print(f"  {torch.cuda.get_device_name(0)}")
print(f"  torch {torch.__version__}  hip {torch.version.hip}")
' || die "the container cannot see the GPU; check --device=/dev/kfd and --group-add video"

if (($# == 0)); then
  bold "Interactive shell in $NAME (exit to leave it running)"
  exec docker exec -it "$NAME" bash
fi

bold "Running in $NAME: $*"
# -i keeps stdin attached so a Ctrl-C reaches the process and its EXIT trap still fires,
# which is what gets the artifacts bundled on an interrupted run (hard rule 0.4).
EXEC_ENV=()
mapfile -t EXEC_ENV < <(env_flags)
printf '  forwarding: %s\n' "$(printf '%s ' "${EXEC_ENV[@]}" | sed 's/-e //g')"
exec docker exec -i -w "$REPO" "${EXEC_ENV[@]}" "$NAME" "$@"
