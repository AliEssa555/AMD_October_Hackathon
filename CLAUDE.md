# CLAUDE.md

Full brief: **[docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)**. Read it before writing code.
Sections 0–4 rules/context · 5–12 build specs · 13 phased tasks + acceptance · 14 gotchas · 15 glossary.

## Hard rules (spec Section 0 — these override everything else)

1. **Never create a GPU Droplet without explicit user confirmation in chat.** Before every launch,
   print the run plan and estimated cost (`hours × $1.99` -- verified, see the price-correction note in spec Section 3) and wait for an explicit "yes".
2. **A powered-off GPU Droplet is still billed.** Only *destroying* it stops billing. Every launch
   path must: destroy on exit (trap), carry a max-runtime watchdog, and be covered by the reaper.
3. **Never commit secrets.** `.env` locally, GitHub Actions secrets in CI. `.gitignore` must keep
   `.env`, `.env.*`, `*.pem`, `.dvc/config.local`, `mlruns/`, `runs/`, `*.db` out of git.
4. **Nothing important lives on the GPU Droplet.** Data from Spaces; checkpoints/metrics/plots to
   MLflow + Spaces *before* the Droplet dies.
5. **Tag every cloud resource `yolo-der`**; ephemeral GPU Droplets also get `ephemeral-gpu`.
6. **Verify, don't assume.** Droplet slugs, region availability, Ultralytics internals, Docker tags
   and MLflow flags are checked against live CLI output or installed source — not memory.
   Spec was written against Ultralytics 8.3.x and MLflow 3.x.
7. **Credits expire 19 Oct 2026.** Teardown (Phase 8) completes by 18 Oct.
8. **Hand-write the research code.** DER, EnhanceNet, severity metrics and extension losses are
   plain, commented PyTorch. Ultralytics supplies only backbone/neck/head, loss assignment,
   dataloading and NMS.
9. **Small, reviewable commits**, conventional-commit messages. Every phase ends with a commit and
   a short summary to the owner.

## Approved deviation from hard rule 0.2 (17 Oct 2026)

The DO console will not issue an API token without a payment method, and the owner has
declined to add one. **There is therefore no API token, so no trap-on-exit destroy and no
reaper.** The owner approved launching GPU Droplets by hand from the console instead.

Replacements, which every launch path must still carry:

1. An in-VM `timeout` watchdog that hard-kills training at `MAX_RUNTIME_HOURS`.
2. Artifacts bundled by an `EXIT` trap on *every* path -- success, crash, timeout,
   Ctrl-C -- before anything else, and the bundle's size verified (rule 0.4 stands).
3. A loud end-of-run banner naming the exit status, the bundle and the destroy step.
4. The owner pulls with `infra/fetch_results.sh`, then destroys the Droplet in the
   console. Powering off does **not** stop billing.

**Correction to safeguard 3, made while implementing it:** the original wording was
"banner, then `poweroff`". Automatic poweroff is wrong here. Without Spaces there is
nowhere for the Droplet to push to, so the bundle stays on its disk -- and a powered-off
Droplet is unreachable *and still billed*, i.e. poweroff would hide the artifacts while
the meter runs. `run_job.sh` therefore keeps `--poweroff` as an opt-in for the
unattended case and **refuses it unless the bundle was pushed to a remote**. The owner
pulling and then destroying is the path that actually satisfies rule 0.4.

`infra/{launch_gpu.sh,cloud-init.yaml.tmpl,reaper.sh}` are deleted, not stubbed: all
three need the token. `setup_base.sh` and `status.sh` are kept but unused (no hub
Droplet, no Spaces). The live path is `docs/MANUAL_LAUNCH.md` ->
`infra/bootstrap.sh` -> `infra/run_job.sh` -> `infra/fetch_results.sh`.

Why this is acceptable here: the account has **no card on file**, and the console states
*"AMD GPU Droplets will be destroyed after credit is used if no payment method is
applied"*. Worst-case spend is therefore capped at the $100 credit, which expires 19 Oct
2026 anyway. The runaway-cost scenario rule 0.2 was written to prevent cannot happen.

Rules 0.1 (explicit "yes" plus printed cost before any launch) and 0.4 are unaffected.

## Current state

Phases 0-4 complete. **Ran on a real MI300X on 17 Oct 2026.**

- Baseline is done: 100 epochs on ExDark, **mAP50 0.574 / mAP50-95 0.270** (best epoch
  90, 0.580). Pulled home to `runs/from-droplet/` (gitignored) with `results.csv`,
  `best.pt`, `last.pt`, `mlflow.db` and the plots. The MLflow run carries
  `gpu=AMD Instinct MI300X VF` and `hip=6.4.43483` -- the hardware evidence the
  hackathon needs.
- **Measured, replacing the spec's Section 9.3 estimates:** ~6 s/epoch steady state
  (epoch 1 is ~48 s of one-time MIOpen kernel compilation), 3.8 GB VRAM of 192 GB, and
  **9.9 minutes for a full 100-epoch run**. The seven-variant ablation is ~70 minutes,
  about $2.30 -- not the $75 the spec budgeted.
- **4-way concurrency does not work and is not needed.** Four processes spawn ~104
  dataloader workers on 20 vCPU and contend on one MIOpen kernel cache; the GPU read
  100% busy while the logs sat frozen. A single process is already fast enough.
- Spend so far: **~$2.80** of the $100 credit.
- The launch path is container-based, which the spec did not anticipate: the AMD image
  ships **no torch on the host**. See `docs/MANUAL_LAUNCH.md` ->
  `infra/send_repo.sh` -> `infra/in_container.sh` -> `infra/bootstrap.sh` ->
  `infra/run_job.sh`, and `scripts/unbundle.py` to open a bundle on Windows.
- **Next: the six remaining variants against the 0.574 baseline** (Phase 5), then the
  Phase 6 analysis. Needs an explicit "yes" plus a printed cost (rule 0.1).
- **Credits expire 19 Oct 2026, so 18 Oct is the last working day** (rule 0.7). Phases
  7-8 (export/API/dashboard) are the ones to drop if time runs short.

See `docs/PROJECT_SPEC.md` Section 13 for the full phase list.

## Conventions

- Python 3.11+, `ruff` for lint + format, type hints on public functions.
- Docstrings cite the paper equation they implement, e.g. `"""Paper Eq. 4: contrast branch."""`.
- Everything runs as `python -m yolo_der.<module>`; no notebooks in the critical path.
- Paths come from config or env — never hard-code `/content/...` or `/root/...`.
- In containers: repo at `/work/yolo-der`, data at `/work/yolo-der/data`, runs at `/work/runs`.
  These must be identical on every GPU Droplet (Ultralytics resume stores absolute paths).
- No `make` on the owner's Windows machine: `./tasks.ps1 <target>` mirrors the Makefile.
