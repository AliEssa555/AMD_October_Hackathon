#!/usr/bin/env bash
# Show every yolo-der cloud resource, the bucket size, and the remaining credit.
# Run it at the start and end of every phase (spec 8.5, Section 13).
#
#   bash infra/status.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$REPO_ROOT/.env" ]] && set -a && source "$REPO_ROOT/.env" && set +a

bold() { printf '\n\033[1m%s\033[0m\n' "$1"; }
warn() { printf '\033[33m! %s\033[0m\n' "$1"; }

if ! command -v doctl >/dev/null 2>&1; then
  warn "doctl not installed -- see docs/OWNER_CHECKLIST.md"
  exit 1
fi

bold "doctl context"
doctl auth list 2>/dev/null || warn "no doctl contexts configured"
printf 'account: '
doctl account get --format Email,Status --no-header 2>/dev/null || warn "token invalid?"

bold "Balance (rule 0.20: confirm this is the credit account)"
doctl balance get 2>/dev/null || warn "could not read balance"

bold "Month-to-date usage"
doctl invoice list --format InvoiceUUID,Amount,InvoicePeriod --no-header 2>/dev/null | head -3 \
  || warn "no invoices yet"

bold "Droplets tagged yolo-der"
doctl compute droplet list --tag-name yolo-der \
  --format ID,Name,Status,Region,Size,PublicIPv4,PrivateIPv4,Tags,Created 2>/dev/null

bold "⚠️  Ephemeral GPU Droplets (these BILL while powered off -- rule 0.2)"
GPU_LIST="$(doctl compute droplet list --tag-name ephemeral-gpu \
  --format ID,Name,Status,Size,Created --no-header 2>/dev/null)"
if [[ -z "${GPU_LIST// /}" ]]; then
  printf '  none — good\n'
else
  printf '%s\n' "$GPU_LIST"
  warn "GPU Droplet(s) alive. 'bash infra/reaper.sh' destroys stale ones."
fi

bold "VPCs"
doctl vpcs list --format ID,Name,Region,IPRange --no-header 2>/dev/null | grep -i yolo-der \
  || printf '  none\n'

bold "Firewalls"
doctl compute firewall list --format ID,Name,Status,DropletIDs --no-header 2>/dev/null \
  | grep -i yolo-der || printf '  none\n'

bold "Volumes / snapshots (should be empty: the GPU box is rebuilt by cloud-init)"
doctl compute volume list --format ID,Name,Size,Region --no-header 2>/dev/null | grep -i yolo-der \
  || printf '  no volumes\n'
doctl compute snapshot list --format ID,Name,Size --no-header 2>/dev/null | grep -i yolo-der \
  || printf '  no snapshots\n'

bold "Spaces bucket ${SPACES_BUCKET:-<unset>}"
if [[ -n "${SPACES_BUCKET:-}" && -n "${SPACES_ENDPOINT:-}" ]] && command -v aws >/dev/null 2>&1; then
  aws s3 ls "s3://${SPACES_BUCKET}/" --endpoint-url "$SPACES_ENDPOINT" 2>/dev/null \
    || warn "could not list the bucket (keys set?)"
  printf 'total size: '
  aws s3 ls "s3://${SPACES_BUCKET}/" --recursive --summarize \
    --endpoint-url "$SPACES_ENDPOINT" 2>/dev/null | tail -2 | tr '\n' ' '
  printf '\n'
else
  printf '  skipped (needs SPACES_BUCKET, SPACES_ENDPOINT and aws-cli)\n'
fi

bold "GPU ledger"
tail -n 8 "$REPO_ROOT/reports/gpu_ledger.md" 2>/dev/null || printf '  missing\n'
