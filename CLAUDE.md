# CLAUDE.md

Full brief: **[docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md)**. Read it before writing code.
Sections 0–4 rules/context · 5–12 build specs · 13 phased tasks + acceptance · 14 gotchas · 15 glossary.

## Hard rules (spec Section 0 — these override everything else)

1. **Never create a GPU Droplet without explicit user confirmation in chat.** Before every launch,
   print the run plan and estimated cost (`hours × $2.59`) and wait for an explicit "yes".
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

## Current state

Phase 0 complete (repo bootstrap). See `docs/PROJECT_SPEC.md` Section 13 for the phase list.

## Conventions

- Python 3.11+, `ruff` for lint + format, type hints on public functions.
- Docstrings cite the paper equation they implement, e.g. `"""Paper Eq. 4: contrast branch."""`.
- Everything runs as `python -m yolo_der.<module>`; no notebooks in the critical path.
- Paths come from config or env — never hard-code `/content/...` or `/root/...`.
- In containers: repo at `/work/yolo-der`, data at `/work/yolo-der/data`, runs at `/work/runs`.
  These must be identical on every GPU Droplet (Ultralytics resume stores absolute paths).
- No `make` on the owner's Windows machine: `./tasks.ps1 <target>` mirrors the Makefile.
