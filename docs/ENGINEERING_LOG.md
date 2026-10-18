# Engineering log — YOLO-DER on AMD MI300X

What was built, what it found, and every mistake worth remembering. Written to be read
months from now, and to be talked through out loud.

Status at time of writing: **18 Oct 2026**, 17 commits, 8,084 lines of Python and shell,
297 tests passing. ExDark ablation complete (7 variants × 100 epochs). BDD100K ablation
training (4 variants × 50 epochs).

GPU spend ≈ **$12** of the $100 credit, derived from Droplet uptime
(1.4 h + 4.8 h × $1.99). The owner's billing page read $91.64 remaining mid-session, so
treat that page as the authority — uptime-derived figures ignore partial-hour rounding.

---

## 1. What the project is

Reproduce *"YOLO-DER: A Dynamic Enhancement Routing Framework for Adverse Weather Vehicle
Detection"* (Gao, Omar & Mahmuddin, *Electronics* 2025) on low-light data, training on an
**AMD Instinct MI300X** via AMD Developer Cloud, with industry-grade MLOps around it.

The paper adds two modules inside a YOLO backbone:

- **DER** (Dynamic Enhancement Routing, Eq. 1–2): a router looks at a feature map's global
  average pool, produces a softmax over *N* parallel conv paths, and mixes them. Soft
  routing, so every path runs.
- **EnhanceNet** (Eq. 3–9): five image-enhancement branches — tone, contrast, detail,
  white balance, identity — blended by a CBAM-style softmax gate.

Both are injected into a pretrained YOLOv8n at three points (P3, P4, after the last
backbone block) behind **ReZero** scales initialised to zero, so at step 0 the injected
model is numerically identical to stock YOLOv8n.

Three goals, in the owner's stated priority order: learning, MLOps quality, and a
hackathon deliverable proving AMD hardware was used.

---

## 2. What we actually found

The headline is a **negative result with a mechanism**, which is more interesting than a
weak positive would have been.

### 2.1 The router does not route

`record_gates.py` captures per-image routing weights and computes entropy normalised so
**1.0 means perfectly uniform**:

| variant | DER routing H | weight range | EnhanceNet gate H |
|---|---|---|---|
| `der` | 0.9997 | 0.231–0.266 | — |
| `der_enh` | 0.9999 | 0.239–0.258 | 0.8183 |
| `der_enh_entreg` | 0.9998 | 0.234–0.257 | 0.8152 |
| `der_enh_sevcond` | 0.9997 | 0.235–0.263 | 0.8242 |

Every variant's router outputs ≈0.25 per path for every image. The "dynamic" routing is,
in this reproduction, a **fixed equal-weight ensemble of four conv paths**.

Two candidate explanations are ruled out:

- **Not a dead module.** The ReZero scales moved well off zero (−0.33 to +0.27), so the
  modules train and contribute. (`der_routes_enh` shows `lambda_der` at exactly 0.0,
  correctly — DER is router-only there and builds no paths.)
- **Not a broken gate in general.** EnhanceNet's branch gate *is* non-uniform at H≈0.82.
  So enhancement gating discriminates and routing does not. That asymmetry is the
  interesting part: the same architecture pattern works in one module and not the other.

### 2.2 The entropy intervention never fired

Extension E1 adds `entropy_reg` to push the router to commit. `train/ent_loss` logged
**0.01498** and moved to 0.01496 over 100 epochs. That number is exactly:

```
0.01 × mean(3 × log 4, 3 × log 5) = 0.01 × 1.4979 = 0.014979
```

So the term is implemented correctly and is **~100× too weak** — about 0.3% of a detection
loss of order 2–6. The question "does a sharper router detect better?" was never asked.
Being able to show the arithmetic is what turns "it didn't work" into "here is precisely
why, and here is the fix".

### 2.3 The ablation table, with cost

| variant | mAP50 | Δ vs baseline | +params % | GFLOPs | ms/img |
|---|---|---|---|---|---|
| baseline | 0.5672 | — | 0.00 | 3.97 | **24.35** |
| der | 0.5688 | +0.0016 *(noise)* | 7.20 | 4.15 | 30.23 |
| der_enh | 0.5657 | −0.0015 *(noise)* | 9.52 | 4.14 | 34.38 |
| der_enh_entreg | 0.5721 | +0.0049 *(noise)* | 9.52 | 4.14 | 34.97 |
| **der_enh_sevcond** | **0.5756** | **+0.0084** | 9.53 | 4.14 | 35.92 |

Params stay under the paper's "<10%" claim. **Latency does not: +41%** for no mAP gain.
A reproduction that reports only mAP would have missed the actual cost.

The one delta clearing the noise floor is `der_enh_sevcond` — *our* E3 extension, which
conditions the gates on a measured severity score instead of hoping the router infers it.
Its dark-quartile result points the way the paper predicts:

| variant | q1 (darkest) | q2 | q3 | q4 |
|---|---|---|---|---|
| baseline | 0.5610 | 0.6359 | 0.5882 | 0.5710 |
| der_enh_sevcond | **0.5973** | 0.6550 | 0.5554 | 0.5774 |

**+0.036 on the darkest quartile.** One seed, 164 images per bin — a lead, not a result.

### 2.4 The noise floor, and why it is stated first

Seven variants trained on identical data and hyper-parameters, differing only in modules,
spanned **0.5794–0.5850 mAP50 on best epoch: a 0.56-point spread**. That is a direct
estimate of run-to-run noise, so `evaluate.py` hard-codes it and prints *(within noise)*
next to any smaller delta. The honesty rule is enforced in the tooling, not left to
whoever writes the paper.

### 2.5 Measured performance facts

| | |
|---|---|
| ExDark, 4,588 imgs @448 | ~6 s/epoch (765 img/s), 100 epochs in 9.9 min |
| BDD100K, 66,312 imgs @448 | 62.4 s/epoch (1,056 img/s), 50 epochs in 52 min |
| First epoch penalty | ~42 s one-time MIOpen kernel compilation |
| VRAM | 3.8 GB of 192 GB per process |
| 4-way concurrency | **Stalls.** ~104 dataloader workers on 20 vCPU, all contending on one MIOpen kernel cache; GPU read 100% busy while logs froze for minutes |

BDD is *faster per image* than ExDark because 1280×720 caches to 448×252 — fewer pixels
than ExDark's squarer images. The spec had budgeted ~$75 for the experiments; the real
cost of the full ExDark ablation was about **$2.30**.

### 2.6 A claim the data cannot support

BDD100K's test split has **13 foggy images**. The paper's fog claims are untestable here,
and reporting a fog number from 13 images would be indefensible. `evaluate.py` flags any
bin under 100 images as low-confidence with its count. Night, by contrast, has 3,929 test
images — that claim *can* be tested properly, which is why BDD was worth the effort.

---

## 3. Mistakes, grouped by the lesson rather than the date

Thirty-plus distinct failures. They sort into six lessons, and the grouping is the useful
part: most were instances of a small number of recurring mistakes.

### Lesson 1 — A test that cannot fail on the real environment is not a test

**The expensive one.** All six module variants died on the MI300X within 45 seconds:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
  der.py in route →  logits = self.router(z)
```

Ultralytics validates with `model.half()` (`BaseValidator.__call__` runs
`model.half() if self.args.half else model.float()`, and `args.half` defaults True on
CUDA/ROCm). Our parameters became Half while the `fp32_math` helper kept casting
activations to Float.

Every local test passed, because **AMP never engages on CPU**. 297 tests, full CPU smoke
runs of two variants, and none of them could have caught it. The baseline survived only
because it has no modules and therefore no fp32 region to clash with.

- **Fix:** `apply_in(module, x)` calls a submodule at its own parameter dtype and returns
  fp32 — a no-op in training, correct during halved validation. Nine call sites, plus
  `blur_kernel.float()` because `model.half()` halves buffers too.
- **The real lesson:** the reproduction needed no GPU. `model.half()` on CPU triggers the
  identical error. 16 tests now cover it, including an injected `DetectionModel`, halved,
  for both combine modes. I verified they have teeth by restoring the old call semantics
  at runtime and watching the exact error return.
- **Interview framing:** "I had good coverage and still shipped a bug that killed every
  run. The gap wasn't quantity of tests, it was that my test environment couldn't reach
  the failure mode. Once I asked 'what can CPU structurally not see?', mixed precision
  was the obvious answer — and it turned out to be reproducible on CPU after all."

### Lesson 2 — The printed number and the enforced number must be the same number

This happened **twice**, and the second time is the more instructive.

1. `MAX_RUNTIME_HOURS=1` never crossed into the container, so the watchdog that caps
   spend printed `6h × $1.99 = $11.94` while I had asked for one hour. Cause: `docker run
   -e` applies only when a container is *created*, and the wrapper deliberately reuses a
   running one.
2. I fixed that by adding a `FORWARD` list — and then wrote `run_batch.sh` with two *new*
   knobs, `RUN_MAX_HOURS` and `BATCH_MAX_HOURS`, which were not in it. The batch announced
   "per-run cap 0.4 h" against a 52-minute run: **every run would have been killed around
   epoch 23.** Caught by reading the plan the script printed, 40 seconds after launch.

- **Fix:** forward on `docker exec` as well as `docker run`, print what was forwarded, and
  add a test that *derives* the requirement instead of listing it — it scans the scripts
  for `${VAR:-default}` knobs and fails on any that is read inside the container but never
  forwarded. That test would have caught instance 2 the moment `run_batch.sh` was written.
- **Interview framing:** "I fixed a class of bug by enumerating its instances, which meant
  the fix went stale the moment I added a new one. The second fix derives the list from
  the code, so it can't."

### Lesson 3 — Verify the environment; never infer it

The spec described a machine that did not exist.

| Assumed | Actually |
|---|---|
| PyTorch on the host | **No torch on the host at all.** The AMD "PyTorch on ROCm" image ships a bare system Python; torch lives only inside two preloaded Docker images |
| `$2.59/GPU-hour` | **$1.99** — the credit buys ~50 hours, not ~38 |
| `--group-add render` (as ROCm examples show) | Docker resolves group names against the **image's** `/etc/group`, not the host's. `rocm/pytorch:latest` has no `render` group and Docker refuses the whole run rather than skipping the flag |
| `numpy>=1.26` is fine | Resolves to 2.2.6; this ROCm torch was compiled against NumPy 1.x, so every array conversion the dataloader makes fails. Pinned `<2` **on the Droplet only** — the local CPU wheel is fine with NumPy 2 |
| `from ultralytics.utils import settings` | Doesn't exist in 8.3.253; the singleton is package-level |
| `rsync` available | Not installed on the owner's Windows machine — and the pull-home is what hard rule 0.4 depends on |
| Validation split is `val/` | Roboflow calls it `valid/`. My check reported "val is empty" on a **perfectly healthy dataset** |
| 4-way concurrency will help | It stalls (see §2.5) |

The `valid/` one is worth dwelling on: a false alarm is expensive, because the natural
reaction to "dataset is empty" is to rebuild 6,554 images. The fix was to read the split
paths from `data.yaml` rather than guess their names.

- **Interview framing:** "The project rule was 'verify, don't assume'. Every single
  environment assumption I inherited from the spec turned out wrong, including the price.
  The ones that cost real time were the ones I didn't think to check because they seemed
  too basic to be wrong."

### Lesson 4 — A silent partial success is worse than a loud failure

- **`tar -I zstd` with no zstd installed** creates an empty file, exits non-zero, and
  looks exactly like a successful bundle in `ls -lh`. This is the failure that would have
  had the owner destroy a Droplet whose results never left it. Fixed with a gzip fallback
  **and** a size-and-readability check that names the recovery step.
- **Windows `tar.exe` is bsdtar**, whose `--exclude` matches *every* path component — so
  `--exclude=./data` also dropped `src/yolo_der/data`, and the upload reported success
  while the package was quietly missing a subpackage. It surfaced much later as
  `ModuleNotFoundError` on a billing machine. Fixed with a `REQUIRED` manifest check
  before sending, which catches the whole class under either tar.
- **`polars` was missing** and training died *after the last epoch*, at plotting time —
  the most expensive possible moment to discover a missing dependency.
- **MLflow marked crashed runs FINISHED** via its `atexit` hook, making a failure
  indistinguishable from a success in the UI. Added a `teardown` callback that tags
  `crashed=true` and ends the run FAILED.
- **`ent_loss` silently vanished** from the metrics: `DetectionTrainer.get_validator()`
  reassigns `loss_names` unconditionally, and `_setup_train` calls it *after*
  `set_model_attributes`, so the extra column ended up unnamed and every metric shifted
  one position left.

- **Interview framing:** "The bugs I now design against aren't crashes — they're the ones
  that return exit 0. Anywhere I compress, copy, or transfer, I verify the artifact rather
  than the exit code."

### Lesson 5 — When your own guard fires, suspect the guard

- **My weight-transfer check raised `RuntimeError: partial weight transfer: 319/355` on a
  completely healthy load.** I had treated every shape mismatch as fatal, but COCO's
  `nc=80` detection head legitimately cannot transfer into an `nc=12` model. Diagnosed by
  scripting the comparison: 0 keys absent by name, 36 shape-mismatched, all under
  `model.22.cv3.*`. The fix distinguishes the two: absent-by-name is always fatal (it means
  the inject/load order was wrong), shape mismatch is fatal only *outside* the
  nc-dependent class head.
- **I claimed `unzip` had silently truncated** the BDD archive — 1,160 of 70,000 images,
  exit 0 — and wrote a parallel Python extractor to work around it. **I was wrong.** The
  Kaggle mirror nests train images in `trainA/trainB/testA/testB`, and my `ls` had counted
  four directory entries plus 1,156 loose files. `37216 + 24750 + 4130 + 2748 + 1156 =
  70,000` exactly. I kept the extractor only because it verifies every member's size, which
  is the check that would have settled the question in one step.
- **Two of my own tests matched the prose in my own comments** rather than the code — a
  test asserting `"--group-add render" not in body` tripped over the comment explaining
  why not to use it. Added a `code_lines()` helper that strips comments.

- **Interview framing:** "Twice I diagnosed a tool's bug that was actually my own
  measurement being wrong. The habit I took from it: before blaming the tool, make the
  measurement unambiguous — count with `find`, not `ls`; compare state dicts, don't read
  a summary line."

### Lesson 6 — Safeguards have to survive contact with the actual setup

The account has no payment method, so DigitalOcean issues **no API token** — which removes
the trap-on-exit destroy and the reaper that hard rule 0.2 requires. The owner approved
manual console launch, and we wrote compensating safeguards. Then two of them turned out
to be wrong as specified:

- **"Banner, then `poweroff`" is actively harmful here.** With no object storage there is
  nowhere for the Droplet to push to, so the bundle sits on its own disk — and a powered-off
  Droplet is unreachable *and still billed*. Poweroff would hide the artifacts while the
  meter ran. `--poweroff` is now opt-in and **refuses unless the bundle reached a remote**.
- **The reaper gap is real and it cost money.** The batch finished in 78 minutes; the
  Droplet then sat idle for ~85 more because the session that was watching it ended and
  the watcher died with it, while the training — running under `nohup` — correctly carried
  on. That idle time cost about **$2.80**, as much as an entire earlier day.

- **Interview framing:** "The safeguard that failed wasn't the one that was hard to build.
  It was the one that depended on my own process staying alive. The training survived the
  session ending; the supervision didn't."

### Smaller ones, kept because they cost real minutes

| Mistake | Fix |
|---|---|
| `der_cfg` stashed on `self.args` → `SyntaxError: not a valid YOLO argument`. `DetectionValidator` re-validates a copy through `get_cfg` | Plain instance attribute with a class-level default, bypassing the config system |
| MLflow 3 broke `runs:/` model registration (resolves against LoggedModel now) | `create_model_version(source=f"{artifact_uri}/weights")` |
| MLflow artifacts landed in the repo's `mlruns/`, so the pull-home would have missed them | Pin `artifact_location` under `RUNS_DIR` |
| MLflow's legal metric-name charset is **platform-dependent** (Windows disallows `:`) | Sanitise to the Windows-safe intersection, so a run logged on Linux stays readable on the laptop |
| `prepare.py` wrote a relative `path:` into `data.yaml`; Ultralytics resolves it against cwd then `~/datasets` | Omit it; `train.py` pins it absolute at launch |
| `evaluate.py` resolved a relative `path:` against the yaml's own directory → `data/prepared/data/prepared` | Resolve against the repo root, which is what it is relative to |
| `.dvc/cache` holds a second full copy of the dataset — 96% of a 256 MB upload | Exclude it; the Droplet rebuilds data from `params.yaml` |
| GNU tar reads `C:/...` as a remote host spec | POSIX paths in Git Bash |
| `printf '%s\n'` with **zero arguments** still prints one empty line → `docker exec` got an empty arg | Guard with `((${#out[@]}))` |
| Bundles were 1.2 GB because `save_period: 1` writes a checkpoint every epoch — 8.4 GB for one ablation | Exclude `weights/epoch*.pt`; they are resume state, not results. 1.2 GB → 44 MB |
| Owner's laptop could not open its own results — no zstd on Windows | `scripts/unbundle.py` reads it via the `zstandard` wheel, verifying contents before extracting |
| Repo was private, so the Droplet could not clone it | `git ls-remote` probe, falling back to `send_repo.sh` over SSH |
| Mixed int/str in a parquet column → `ArrowInvalid` | Make the column consistently a string |
| A test asserted the executable bit; this repo commits infra `100644` on Windows | Assert the thing that actually matters: the docs invoke scripts via `bash` |

---

## 4. What the MLOps actually buys

Worth being concrete, because "MLOps" is otherwise a word.

- **Hardware provenance.** Every MLflow run carries `gpu=AMD Instinct MI300X VF` and
  `hip=6.4.43483`. `torch.version.hip` being non-empty is the evidence the run used AMD
  hardware — the hackathon deliverable — and it is attached to the run, not to a claim in
  a README.
- **Reproducibility.** Each run tags `git_sha` and a `data_hash` taken from the DVC lock,
  so a result points at an exact revision and an exact dataset.
- **Cost control under a hard constraint.** Watchdog, `EXIT`-trap bundling on every path
  (success, crash, timeout, Ctrl-C), bundle verification, and a printed worst-case cost
  before anything starts. Two of these caught real problems.
- **Honesty as a code path.** The noise floor and the minimum-bin-size rule live in
  `evaluate.py`, so a within-noise delta prints as within-noise whoever runs it.
- **297 tests**, with the infrastructure scripts covered by contract tests — they can only
  be exercised for real on a paid machine, so what is testable is that the safeguards are
  still present. Each assertion exists because its absence cost money or lost a run.

---

## 5. If someone asks "what would you do differently"

1. **Run one epoch on the real hardware before writing 300 tests against a simulator.**
   The AMP bug, the NumPy pin, and the no-torch-on-host discovery would all have surfaced
   on day one for about $0.50.
2. **Budget for seeds from the start.** A 0.56-point noise floor against deltas of 0.002–0.008
   means single-seed runs could never have answered the question. Three seeds of two
   variants costs ~$2 and 25 minutes; I should have planned that instead of seven
   single-seed variants.
3. **Measure the mechanism before scaling the experiment.** Routing entropy takes two
   minutes to compute on a trained model and would have shown H≈0.9998 after the *first*
   run — before spending anything on the other six. The expensive ablation answered "does
   it help?" when the cheap diagnostic answered "is it even doing the thing?".
4. **Derive invariants, don't enumerate them.** See Lesson 2.

---

## 6. Where to look in the repo

| | |
|---|---|
| Hard rules, current state, approved deviations | `CLAUDE.md` |
| Full brief | `docs/PROJECT_SPEC.md` |
| BDD100K scale-up plan | `docs/BDD100K_ADDENDUM.md` |
| Owner-facing launch procedure | `docs/MANUAL_LAUNCH.md` |
| The research code, hand-written per rule 0.8 | `src/yolo_der/models/{der,enhancenet,inject}.py` |
| Ultralytics integration | `src/yolo_der/trainer.py` |
| MLflow callbacks | `src/yolo_der/tracking.py` |
| Evaluation and the ablation table | `src/yolo_der/evaluate.py` |
| Routing/gate analysis | `src/yolo_der/analysis/record_gates.py` |
| Cloud path | `infra/{send_repo,in_container,bootstrap,run_job,run_batch,fetch_results}.sh` |
| Results | `reports/ablation.md`, `reports/eval/*.json`, `reports/gates/*.parquet` |
