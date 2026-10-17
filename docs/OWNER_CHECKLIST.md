# Owner checklist — things only you can do

Spec Section 8.1. Work top to bottom. Everything here goes into `.env` (never into git).
Tick the boxes as you go and tell me when a section is done; I'll verify each one with
`doctl` before creating anything.

---

## Verified from the live console, 17 Oct 2026

Read off the `devcloud.amd.com/gpus/new` page you sent, so these no longer need checking:

| Thing | Value |
|---|---|
| Team | `My AMD Team`, param `bb4e16` |
| Project | `My AMD Home`, `fa26874e-0012-4b24-8af4-24d2a2a1de3a` |
| Credit | **$100.00 of $100.00 remaining**, "expiring in 2 days" |
| Region | `atl1` (Atlanta DC1). `ric1` greyed out: *GPUs currently unavailable* |
| Size | `gpu-mi300x1-192gb-devcloud`, **$1.99/GPU/hr**. The x8 plan is greyed out |
| Price | **$1.99/hr, not $2.59** — the credit buys ~50 GPU-hours |
| SSH key | one key present, `test`, id `60164577` |
| Paywall | feature flag `amd_no_paywall: true` — creation is **not** gated on a card |
| Spot | offered, and ATL1 is a supported spot region |
| Startup scripts | offered (this is the cloud-init hook the launcher needs) |

The console also warns: *"AMD GPU Droplets will be destroyed after credit is used if no
payment method is applied"* and *"AMD credit only covers GPU access. All other services
would be charged to your payment method."* So Spaces and the CPU hub Droplet are **not**
covered by the credit — that's a real, if small, out-of-pocket cost.

---

## A. ~~Find the account that actually holds the $100 credit~~ ✅ done

AMD Developer Cloud runs on DigitalOcean, but the credit may sit in a **separate team or
console** from your personal account. Creating resources in the wrong team spends real money.

- [x] `My AMD Team` holds the full $100, expiring 19 Oct 2026.

## B. ~~API tokens~~ — dropped, 17 Oct 2026

The console will not issue a token without a payment method and you have declined to
add one, so **there are no DO tokens** and nothing here to do. `doctl` is not used.

What this costs us: no scripted launch, no trap-on-exit destroy, and no reaper. What
replaces them is in `CLAUDE.md` → *Approved deviation from hard rule 0.2*, and the
procedure you actually follow is **`docs/MANUAL_LAUNCH.md`**.

The one consequence to keep in your head: **nothing will clean up a Droplet you
forget.** Powering it off does not stop the bill; only Destroy does.

## C. SSH key

- [x] The console already lists one key, `test` (id 60164577).
- [ ] Confirm the **private** half is on this laptop and works:
      `ssh-add -l`, or check `~/.ssh/`. Without it you cannot reach the Droplet at all,
      and a Droplet you cannot reach is a Droplet you can only destroy.
- [x] Checked: you have `ssh`, `scp` and `tar` but **no `rsync`**. Nothing to install --
      `send_repo.sh` and `fetch_results.sh` fall back to tar-over-ssh.

Nothing to upload with `doctl` (see B); you pick the key in the create form instead.

## D. ~~Spaces access key~~ — dropped, 17 Oct 2026

The $100 credit covers GPU Droplets only, and there is no card to pay for object
storage, so there is no bucket. MLflow runs on the GPU Droplet against sqlite with its
artifacts under `/work/runs`, and `infra/fetch_results.sh` pulls the lot home over SSH
before you destroy the Droplet. Hard rule 0.4 still holds; the destination changed.

## E. Spend guardrails

- [ ] Billing → **Spend alerts** → set an alert at **$85**.
- [x] Payment method: still none (sidebar Billing shows *Action Needed*), but the
      `amd_no_paywall` flag means GPU Droplet creation is **not** blocked by that. Nothing
      to do unless you want Spaces/the hub Droplet, which the credit does not cover.

## F. Roboflow API key  ← needed today, for Phase 1

- [ ] Get it from <https://app.roboflow.com/settings/api>.
- [ ] Put it in `ROBOFLOW_API_KEY` in `.env`.

## G. GitHub repository

- [ ] Create an empty repo (e.g. `yolo-der`). Public or private both work; if private, the
      GPU Droplet needs a read-only deploy key or a fine-grained PAT, which I'll wire up.
- [ ] Tell me the `owner/repo`, and I'll set `GIT_REPO_URL` and the GHCR image name.
- [ ] Add these Actions secrets when we reach the relevant phase:
      `ROBOFLOW_API_KEY` (the Droplet rebuilds the dataset itself), then
      `HUB_SSH_KEY` (Phase 7, for API deploys).

## H. The paper PDF

- [ ] Have it to hand. In Phase 3 I'll ask you one question about Figures 1–3: whether DER
      feeds EnhanceNet *sequentially*, or whether DER's router directly supplies
      EnhanceNet's branch gate. The spec says default to sequential, but your reading of the
      figure decides it.

---

## Install these CLIs (your machine)

Node 22 and Docker are already installed, and with sections B and D dropped there is
very little left to install.

- **`doctl`** — not needed. It cannot authenticate without a token.
- **`aws`** — not needed. No bucket.
- **`dvc`** — comes from the project venv (`.venv\Scripts\dvc.exe --version`), and is
  only used locally to version `data/`. The Droplet rebuilds the dataset from
  `params.yaml` instead of pulling a DVC remote.
- **`rsync`** — not needed. It is absent on this machine, so both transfer scripts use
  tar-over-ssh; they use rsync only if they happen to find it.

The old gate for Phase 2 was `doctl balance get` showing the credit. Without a token
the equivalent check is the console itself: the Billing page must show the $100 credit
and the create form must show the MI300X plan as available, both of which were verified
on 17 Oct 2026 and are recorded in the table at the top of this file.

---

## When you are ready to spend money

Read **`docs/MANUAL_LAUNCH.md`** once, end to end, before the first launch. It is the
whole procedure: create, bootstrap, train, pull, destroy — plus what to do when the SSH
connection drops or the watchdog fires.

I will not create a Droplet. Before every run I print the plan and hours × $1.99 and
wait for you to say yes, and then you click Create yourself (hard rule 0.1).