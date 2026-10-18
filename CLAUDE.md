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

Phases 0-5 complete on ExDark, Phase 6 half written. **See
`docs/ENGINEERING_LOG.md`** for the full account, including every mistake and fix --
that file is the one to read first.

### The finding that matters

**The DER router does not route.** Normalised routing entropy is 0.9997-0.9999 (1.0 =
uniform) in every variant, with per-path weights hugging 0.25 for every image. The
paper's "dynamic" routing is a fixed equal-weight ensemble of four conv paths here.

- Not a dead module: ReZero scales moved to -0.33..+0.27, so the modules do train.
- Not a broken pattern in general: EnhanceNet's branch gate *is* non-uniform, H~0.82.
- The E1 intervention never fired: `train/ent_loss` logged 0.01498, which is exactly
  `0.01 * mean(3*log 4, 3*log 5)`, so the term is correct but ~100x too weak against a
  detection loss of order 2-6. **`entropy_reg` needs to be nearer 1.0** for the
  question to have been asked. This is the cheapest high-value experiment left.

### ExDark ablation (7 variants, 100 epochs, seed 0)

| variant | mAP50 | delta | +params% | ms/img |
|---|---|---|---|---|
| baseline | 0.5672 | -- | 0.00 | 24.35 |
| der | 0.5688 | +0.0016 (noise) | 7.20 | 30.23 |
| der_enh | 0.5657 | -0.0015 (noise) | 9.52 | 34.38 |
| der_enh_entreg | 0.5721 | +0.0049 (noise) | 9.52 | 34.97 |
| der_enh_sevcond | **0.5756** | **+0.0084** | 9.53 | 35.92 |

- **Noise floor 0.0056 mAP50**, measured as the spread across all seven variants on one
  seed. `evaluate.py` prints *(within noise)* below it; do not report smaller deltas.
- **+41% latency for no mAP gain** (24.35 -> 34.38 ms/img). Params stay under the
  paper's <10% claim, so a reproduction reporting only params would miss the real cost.
- The only delta above noise is `der_enh_sevcond` -- *our* E3 extension -- and its
  darkest-quartile gain is +0.036 (0.5610 -> 0.5973). One seed, 164 images: a lead.
- **Seeds are the gap.** Deltas of 0.002-0.008 against a 0.0056 floor cannot be settled
  on one seed. 3 seeds x 2 variants is ~25 min and ~$2.

### BDD100K (Addendum A)

Converted and training: 66,312 / 3,551 / 10,000 at 448 px, **62.4 s/epoch**
(1,056 img/s -- faster per image than ExDark, because 1280x720 caches to 448x252).
50 epochs = 52 min = $1.74/run. Four variants = 3.5 h = $6.90.

- `tod_night` has 3,929 test images, so the paper's night claim is testable.
- `wx_foggy` has **13**, so its fog claim is not. `evaluate.py` flags bins under 100.

### Operational facts

- **No torch on the host**; the AMD image keeps PyTorch inside Docker. Everything runs
  through `infra/in_container.sh`.
- **Pin `numpy<2` on the Droplet**: this ROCm torch was built against NumPy 1.x.
- **4-way concurrency stalls** (MIOpen cache contention, ~104 workers on 20 vCPU).
  Run variants sequentially via `infra/run_batch.sh`.
- **AMP is only live on GPU**, so CPU tests cannot see dtype bugs. `model.half()` on CPU
  reproduces them -- see `tests/test_amp_dtypes.py`.
- Spend ~$12 of $100. Credits expire **19 Oct 2026**; the billing page is authoritative.
- **There is no reaper.** A finished batch leaves the Droplet idle and billing; ~85
  minutes were lost that way. Check it yourself after every batch.

See `docs/PROJECT_SPEC.md` Section 13 for the phase list and
`docs/BDD100K_ADDENDUM.md` for the scale-up plan.

## Conventions

- Python 3.11+, `ruff` for lint + format, type hints on public functions.
- Docstrings cite the paper equation they implement, e.g. `"""Paper Eq. 4: contrast branch."""`.
- Everything runs as `python -m yolo_der.<module>`; no notebooks in the critical path.
- Paths come from config or env — never hard-code `/content/...` or `/root/...`.
- In containers: repo at `/work/yolo-der`, data at `/work/yolo-der/data`, runs at `/work/runs`.
  These must be identical on every GPU Droplet (Ultralytics resume stores absolute paths).
- No `make` on the owner's Windows machine: `./tasks.ps1 <target>` mirrors the Makefile.
