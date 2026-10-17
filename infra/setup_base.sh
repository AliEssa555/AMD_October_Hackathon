#!/usr/bin/env bash
# Create the long-lived infrastructure: VPC, two firewalls, and the hub Droplet (spec 8.2-8.4).
# Idempotent -- safe to re-run; it reuses anything that already exists.
#
#   bash infra/setup_base.sh [--dry-run]
#
# Not created here:
#   * the Spaces bucket   -- made once from the UI or aws-cli (see the note it prints)
#   * GPU Droplets        -- infra/launch_gpu.sh only, and never from CI (rule 0.1)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# shellcheck disable=SC1091
if [[ -f "$REPO_ROOT/.env" ]]; then set -a; source "$REPO_ROOT/.env"; set +a; fi

bold() { printf '\n\033[1m%s\033[0m\n' "$1"; }
die()  { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }
run()  { if ((DRY_RUN)); then printf '  [dry-run] %s\n' "$*"; else "$@"; fi; }

HUB_NAME="${HUB_NAME:-yolo-der-hub}"
HUB_SIZE="${HUB_SIZE:-s-1vcpu-2gb}"
HUB_IMAGE="${HUB_IMAGE:-ubuntu-24-04-x64}"
VPC_NAME="${VPC_NAME:-yolo-der-vpc}"
HUB_FW="${HUB_FW:-yolo-der-hub-fw}"
GPU_FW="${GPU_FW:-yolo-der-gpu-fw}"

# ---------------------------------------------------------------- pre-flight --
bold "Pre-flight"
command -v doctl >/dev/null || die "doctl not installed (docs/OWNER_CHECKLIST.md)"
[[ -n "${DO_REGION:-}" ]] || die "DO_REGION is unset in .env"
[[ -n "${DO_SSH_KEY_FINGERPRINT:-}" ]] || die "DO_SSH_KEY_FINGERPRINT is unset in .env"

printf 'active doctl context:\n'; doctl auth list
printf 'account: '; doctl account get --format Email,Status --no-header
bold "Balance -- this MUST show the \$100 credit (rule 0.20)"
doctl balance get

# Verify the region actually offers an MI300X slug before we build anything around it.
bold "MI300X availability in $DO_REGION"
if doctl compute size list --format Slug,Memory,VCPUs,PriceHourly --no-header \
     | grep -i 'mi300x' ; then
  printf '\n(the Regions column is not in this output; confirm with:\n'
  printf '  doctl compute size list -o json | jq -r \x27.[]|select(.slug|test("mi300x"))|"\\(.slug) \\(.regions)"\x27)\n'
else
  printf '\033[33m! no mi300x slug visible to this account.\033[0m\n'
  printf '  GPU Droplets may need to be enabled, or live in a different region/team.\n'
  printf '  The hub does not depend on it, so this is a warning, not a stop.\n'
fi

printf '\nContinue and create/reuse VPC + firewalls + hub in %s? [yes/NO] ' "$DO_REGION"
read -r reply
[[ "$reply" == "yes" ]] || die "aborted"

# ---------------------------------------------------------------------- VPC ---
bold "VPC $VPC_NAME"
VPC_UUID="$(doctl vpcs list --format ID,Name --no-header | awk -v n="$VPC_NAME" '$2==n{print $1}')"
if [[ -z "$VPC_UUID" ]]; then
  run doctl vpcs create --name "$VPC_NAME" --region "$DO_REGION" --wait
  VPC_UUID="$(doctl vpcs list --format ID,Name --no-header | awk -v n="$VPC_NAME" '$2==n{print $1}')"
  printf 'created: %s\n' "$VPC_UUID"
else
  printf 'exists: %s\n' "$VPC_UUID"
fi
VPC_CIDR="$(doctl vpcs get "$VPC_UUID" --format IPRange --no-header 2>/dev/null || echo '')"
printf 'CIDR: %s\n' "${VPC_CIDR:-unknown}"

# ----------------------------------------------------------------- hub box ----
# Created before the firewalls so the hub firewall can reference its private IP / tag.
bold "Hub Droplet $HUB_NAME ($HUB_SIZE, $HUB_IMAGE)"
printf 'price check: '
doctl compute size list --format Slug,Memory,VCPUs,PriceMonthly,PriceHourly --no-header \
  | awk -v s="$HUB_SIZE" '$1==s'
HUB_ID="$(doctl compute droplet list --format ID,Name --no-header | awk -v n="$HUB_NAME" '$2==n{print $1}')"
if [[ -z "$HUB_ID" ]]; then
  run doctl compute droplet create "$HUB_NAME" \
    --region "$DO_REGION" --size "$HUB_SIZE" --image "$HUB_IMAGE" \
    --vpc-uuid "$VPC_UUID" --ssh-keys "$DO_SSH_KEY_FINGERPRINT" \
    --tag-names yolo-der,hub \
    --user-data-file "$REPO_ROOT/infra/hub/cloud-init.yaml" \
    --enable-monitoring --wait
  HUB_ID="$(doctl compute droplet list --format ID,Name --no-header | awk -v n="$HUB_NAME" '$2==n{print $1}')"
  printf 'created: %s\n' "$HUB_ID"
else
  printf 'exists: %s\n' "$HUB_ID"
fi

if ((DRY_RUN)); then
  HUB_PUB="<dry-run>"; HUB_PRIV="<dry-run>"
else
  HUB_PUB="$(doctl compute droplet get "$HUB_ID" --format PublicIPv4 --no-header)"
  HUB_PRIV="$(doctl compute droplet get "$HUB_ID" --format PrivateIPv4 --no-header)"
fi
printf 'public: %s   private: %s\n' "$HUB_PUB" "$HUB_PRIV"

# --------------------------------------------------------------- firewalls ----
# Owner IP: inbound SSH is locked to it. Detected, overridable with OWNER_IP.
OWNER_IP="${OWNER_IP:-$(curl -fsS https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]')}"
[[ -n "$OWNER_IP" ]] || die "could not detect your public IP; set OWNER_IP in .env"
printf '\nowner IP for the SSH rules: %s\n' "$OWNER_IP"

bold "Firewall $HUB_FW (tag: hub)"
if doctl compute firewall list --format Name --no-header | grep -qx "$HUB_FW"; then
  printf 'exists -- not modified. Delete and re-run to change the rules.\n'
else
  # 22 from the owner only; 80/443 public (Caddy + Let's Encrypt); 5000 from the VPC only.
  run doctl compute firewall create \
    --name "$HUB_FW" \
    --tag-names hub \
    --inbound-rules "protocol:tcp,ports:22,address:${OWNER_IP}/32 protocol:tcp,ports:80,address:0.0.0.0/0,address:::/0 protocol:tcp,ports:443,address:0.0.0.0/0,address:::/0 protocol:tcp,ports:5000,address:${VPC_CIDR}" \
    --outbound-rules "protocol:tcp,ports:all,address:0.0.0.0/0,address:::/0 protocol:udp,ports:all,address:0.0.0.0/0,address:::/0 protocol:icmp,address:0.0.0.0/0,address:::/0"
fi

bold "Firewall $GPU_FW (tag: ephemeral-gpu)"
if doctl compute firewall list --format Name --no-header | grep -qx "$GPU_FW"; then
  printf 'exists -- not modified.\n'
else
  run doctl compute firewall create \
    --name "$GPU_FW" \
    --tag-names ephemeral-gpu \
    --inbound-rules "protocol:tcp,ports:22,address:${OWNER_IP}/32" \
    --outbound-rules "protocol:tcp,ports:all,address:0.0.0.0/0,address:::/0 protocol:udp,ports:all,address:0.0.0.0/0,address:::/0 protocol:icmp,address:0.0.0.0/0,address:::/0"
fi

# ------------------------------------------------------------------- bucket ---
bold "Spaces bucket"
if [[ -n "${SPACES_BUCKET:-}" ]]; then
  printf 'configured: %s  (endpoint %s)\n' "$SPACES_BUCKET" "${SPACES_ENDPOINT:-unset}"
  printf 'create it once, if it does not exist yet:\n'
  printf '  aws s3 mb s3://%s --endpoint-url %s\n' "$SPACES_BUCKET" "${SPACES_ENDPOINT:-<endpoint>}"
  printf 'prefixes used: dvc/ mlflow/ logs/ checkpoints/ onnx/ reports/ backups/\n'
else
  printf 'SPACES_BUCKET is unset. Pick yolo-der-<random6>, create it private in %s.\n' "$DO_REGION"
fi

# -------------------------------------------------------------------- next ----
bold "Write these into .env"
cat <<EOF
DO_VPC_UUID=$VPC_UUID
HUB_PUBLIC_IP=$HUB_PUB
HUB_PRIVATE_IP=$HUB_PRIV
SITE_ADDRESS=$(printf '%s' "$HUB_PUB" | tr '.' '-').sslip.io
MLFLOW_TRACKING_URI=http://$HUB_PRIV:5000
EOF

bold "Then deploy the hub services"
cat <<EOF
  bash infra/hub/deploy_hub.sh      # rsyncs infra/hub + .env, builds, starts mlflow + caddy
  ssh -N -L 5000:$HUB_PRIV:5000 root@$HUB_PUB   # tunnel, then MLflow at http://localhost:5000
EOF
