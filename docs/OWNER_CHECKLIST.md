# Owner checklist — things only you can do

Spec Section 8.1. Work top to bottom. Everything here goes into `.env` (never into git).
Tick the boxes as you go and tell me when a section is done; I'll verify each one with
`doctl` before creating anything.

---

## A. Find the account that actually holds the $100 credit  ⚠️ do this first

AMD Developer Cloud runs on DigitalOcean, but the credit may sit in a **separate team or
console** from your personal account. Creating resources in the wrong team spends real money.

- [ ] Log in and open **Billing**. Note which team name shows the $100 credit and its expiry
      (should be 19 Oct 2026).
- [ ] If you have more than one team, switch between them and confirm which one has it.
- [ ] Tell me the team name.

## B. API tokens

Create these in **the team from step A** (API → Tokens → Generate New Token).

- [ ] **`DIGITALOCEAN_ACCESS_TOKEN`** — full read/write. Name it `yolo-der-admin`.
      Lives on your laptop (`.env`) and in GitHub Actions secrets. Nowhere else.
- [ ] **`DO_SELF_DESTROY_TOKEN`** — custom scoped, name it `yolo-der-self-destroy`.
      Grant **only** `droplet: read` and `droplet: delete` (the scope names in the UI may
      differ slightly — pick the narrowest pair that allows reading and deleting Droplets).
      This is the only DO token that ever reaches a GPU Droplet, because user-data is
      readable from inside the Droplet via the metadata service.

Paste both into `.env`. If you'd rather not paste tokens into a chat, put them in `.env`
yourself and just tell me "done" — I only ever need to run commands, not see the values.

## C. SSH key

- [ ] You have an SSH keypair. If not: `ssh-keygen -t ed25519 -C "yolo-der"`.
- [ ] Upload the public key:
      `doctl compute ssh-key import yolo-der --public-key-file ~/.ssh/id_ed25519.pub`
- [ ] Put the fingerprint it prints into `DO_SSH_KEY_FINGERPRINT`.

## D. Spaces access key

- [ ] Create a Spaces access key (Spaces Object Storage → Access Keys).
- [ ] If the UI offers **limited / per-bucket** keys, scope it to the project bucket only.
      (The bucket doesn't exist yet — create the key after I create the bucket in Phase 2,
      or create a full-access key now and replace it later.)
- [ ] Put the pair into `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env`.
      Those exact names are what boto3, DVC and MLflow read.

## E. Spend guardrails

- [ ] Billing → **Spend alerts** → set an alert at **$85**.
- [ ] Check whether the account has a **payment method** on file. The billing page showed
      none. DigitalOcean may require one before it will create GPU Droplets, and anything
      beyond the $100 credit (or after 19 Oct) is charged to it. Decide now whether you're
      comfortable adding one.

## F. Roboflow API key  ← needed today, for Phase 1

- [ ] Get it from <https://app.roboflow.com/settings/api>.
- [ ] Put it in `ROBOFLOW_API_KEY` in `.env`.

## G. GitHub repository

- [ ] Create an empty repo (e.g. `yolo-der`). Public or private both work; if private, the
      GPU Droplet needs a read-only deploy key or a fine-grained PAT, which I'll wire up.
- [ ] Tell me the `owner/repo`, and I'll set `GIT_REPO_URL` and the GHCR image name.
- [ ] Add these Actions secrets when we reach the relevant phase:
      `DIGITALOCEAN_ACCESS_TOKEN` (Phase 4, for the reaper), then `HUB_HOST` and
      `HUB_SSH_KEY` (Phase 7, for API deploys).

## H. The paper PDF

- [ ] Have it to hand. In Phase 3 I'll ask you one question about Figures 1–3: whether DER
      feeds EnhanceNet *sequentially*, or whether DER's router directly supplies
      EnhanceNet's branch gate. The spec says default to sequential, but your reading of the
      figure decides it.

---

## Install these CLIs (your machine)

Currently missing: `doctl`, `dvc`, `aws`. Node 22 and Docker are already installed.

```powershell
# doctl — DigitalOcean CLI
winget install --id DigitalOcean.Doctl

# dvc + aws-cli come from the project venv once ./tasks.ps1 setup finishes:
#   .venv\Scripts\dvc.exe --version
# aws-cli (only needed for raw bucket pokes; DVC and boto3 don't need it)
winget install --id Amazon.AWSCLI
```

Then, once the token is in `.env`:

```powershell
doctl auth init --context yolo-der
doctl auth switch --context yolo-der
doctl balance get          # must show the $100 credit
doctl account get
```

`doctl balance get` showing the credit is the gate for Phase 2. Nothing gets created before
that passes — spec rule 0.20: always confirm the active context is the credit account.
