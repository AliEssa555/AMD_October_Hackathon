# Addendum A — BDD100K scale-up (read after `PROJECT_SPEC.md`)

> **For Claude Code:** This addendum extends `docs/PROJECT_SPEC.md`. Where it conflicts with the spec, **this file wins for today's work**. All hard rules in Spec Section 0 still apply, with the clarifications in A.1.

## A.1 Situation (as of Sun 18 Oct 2026, ~10:15 PKT)

- The owner already has a **running MI300X GPU Droplet**: 1 GPU, 192 GB VRAM, 20 vCPU, 240 GB RAM. Work directly on it over SSH. Do **not** create another GPU Droplet.
- **Credits expire 19 Oct 2026.** Plan to have all training finished by **~20:00 PKT on 18 Oct** at the latest, leaving the evening for evaluation, write-up and teardown. If the remaining credit or time is less than a step needs, stop and tell the owner.
- **Measured throughput on ExDark: ~5 s/epoch** (~4.5k train images, ≈ 900 img/s). Loading with 20 vCPUs is the likely limit. Re-measure on BDD100K with a 1-epoch run before launching long runs.
- The ExDark ablation (Spec Section 9.3, batches A–C) is the main deliverable and **takes priority**. If it isn't finished, finish it first. BDD100K is an **additional** experiment. Never overwrite ExDark runs, data, or MLflow runs.
- Rules that change for this machine, because it was not launched by `launch_gpu.sh`:
  1. It has **no self-destroy trap and no watchdog**. Before starting any long job, ask the owner for the Droplet's ID, then set up a safety net: either schedule the destroy with `doctl compute droplet delete <id> --force` on a timer from the owner's laptop or a scheduled task, or run the job under `timeout` and tell the owner when it ends. **Tell the owner plainly when the GPU is idle, because it is billed while idle.** A powered-off Droplet is still billed (Spec 0.2).
  2. Nothing important lives only on this Droplet (Spec 0.4). Before the end of every job: checkpoints and metrics are in MLflow / Spaces.
  3. Never print or commit the Kaggle key, the Spaces keys, or the DO token. Pass them as environment variables for the session only.
- Still ⛔ **CONFIRM** with the owner before: starting any run longer than 15 minutes (show the plan, the expected hours and `hours × $2.59`), and before destroying the Droplet.

## A.2 Why BDD100K

ExDark (6.5k images) is too small for this GPU, and the paper's claims concern adverse conditions. BDD100K gives ~100k real driving images with per-image **`timeofday`** (daytime / night / dawn/dusk) and **`weather`** (clear / overcast / partly cloudy / rainy / snowy / foggy) labels. That gives us:

1. A 10× larger training set, so results are less seed-noise-dominated.
2. **Ground-truth conditions** to evaluate the paper's claim by condition: does DER/EnhanceNet help more at night / in rain / in fog?
3. A check on the paper's severity metrics: correlate `s_dark` with `timeofday` and `s_fog` with `weather = foggy` (it is expected to work poorly for night; report either way).

## A.3 The data

Source: Kaggle dataset `solesensei/solesensei_bdd100k` (a mirror of BDD100K). The owner confirmed it contains `bdd100k_labels_images_train.json` (1.45 GB) in the original BDD format: per image `name`, `attributes {weather, scene, timeofday}`, `timestamp`, `labels[]` (each with `category` and, for objects, `box2d {x1,y1,x2,y2}`). BDD100K's own terms apply to the data; credit Berkeley DeepDrive in the README.

1. **Check the layout first** (owner's key exported as `KAGGLE_USERNAME` / `KAGGLE_KEY`; `pip install kaggle`):
   ```bash
   kaggle datasets files solesensei/solesensei_bdd100k
   ```
   Confirm: (a) train and val label JSONs with `attributes`; (b) the image folders for train and val (test images aren't needed). If the layout differs from this section, stop and tell the owner.
2. **Check resources**: `df -h` (need ~40 GB free: archive + extracted images + converted copy), `nproc`, `free -g`.
3. **Download and unzip on the GPU Droplet** (not the 2 GB hub; it would run out of memory and disk) into `/work/data_raw/bdd100k/`. Time it, and tell the owner the elapsed time. Run `df -h` after unzipping, then delete the archive.
4. **Convert** with `convert_bdd100k.py` (delivered with this addendum; place it at `src/yolo_der/data/convert_bdd100k.py`, make it importable as `python -m yolo_der.data.convert_bdd100k`):
   ```bash
   pip install ijson
   python -m yolo_der.data.convert_bdd100k --root /work/data_raw/bdd100k --out /work/yolo-der/data/bdd100k --link
   ```
   What it does (read it; it's short):
   - streams the JSON with `ijson` (low memory);
   - keeps the 10 box classes (pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign) and ignores lane / drivable-area polygons;
   - drops degenerate boxes (< 1 px), keeps images with zero boxes as background;
   - **BDD val → our `test`**; 5% of BDD train (seeded) → our `val`; the rest → `train`;
   - writes `attributes.csv` (`split,image,weather,scene,timeofday,n_boxes`), `data.yaml`, and `bins/` — per-condition test lists (`tod_night.txt`, `wx_rainy.txt`, …) with matching `bins/data_<group>.yaml`.
5. **Sanity checks (show the owner the output):**
   - the printed split counts (expect roughly 66k / 3.5k / 10k) and the `eval bins:` counts (night should be several thousand test images; rare groups such as fog will have only a few dozen, so **mark any bin with fewer than ~100 images as low-confidence** in the results);
   - the `attributes.csv` distributions (`timeofday`, `weather`, including `undefined`);
   - draw 8 random images with their boxes to `reports/bdd_sample_check.png` and look at them (catches an x/y or class-order bug).
6. **Version it:** `dvc add data/bdd100k` (or a new stage), `dvc push` to Spaces if the credit and time allow (it's ~6+ GB; same-region traffic is free). If not, skip the push and say so; the data can be re-created from Kaggle.
7. Add a CSV join step so the existing `severity.py` can run on BDD100K (`--data data/bdd100k --out data/bdd100k/severity.csv`). It will take a while on 100k images at full resolution; run it with the 20 CPUs in parallel (`multiprocessing.Pool(18)`) and only on the **test split** plus a 10k random subset of train, unless the owner asks for more.

## A.4 Configs

Add `configs/bdd_base.yaml` (deep-merge on top of `configs/base.yaml`; same variants as ExDark just with these overrides):

```yaml
data: /work/yolo-der/data/bdd100k/data.yaml
train:
  epochs: 50            # fixed before first launch; resume restores it (Spec 9.4)
  imgsz: 448            # same as ExDark so results are comparable; see note below
  batch: 64
  cache: ram            # ~22 GB per process at 448; fine with 240 GB (check free -g if running 2+ processes)
  workers: 8            # 2 concurrent runs → 16 workers on 20 vCPUs; do not exceed ~18 in total
  save_period: 1
  val: true
```

Variants for BDD (names prefixed `bdd_`): `bdd_baseline`, `bdd_der_enh`. Add `bdd_der`, `bdd_enh` only if time and credit allow. Fixed seed 0 for the first pass.

Note on resolution: BDD images are 1280×720 and many objects (traffic lights / signs) are tiny, so 448 will give a low absolute mAP on those. This is **accepted**: the point is the relative comparison, and 448 is what the paper used and what ExDark used. Mention it in the write-up. Do not switch to 640 without asking (about 2× the cost).

## A.5 Run plan (⛔ CONFIRM before launching)

1. **Throughput test (~10 min):** `bdd_baseline` for 1 epoch with `--fraction 0.1`, then 1 epoch on the full train set, alone, then 2 concurrent processes. Record s/epoch for each. Print the table.
2. **Plan from measurements.** With ExDark's ~900 img/s the full 66k epoch is ~75 s alone, so 50 epochs ≈ 1.0–1.1 h alone; two concurrent runs share the CPU, so expect up to ~2× longer wall time each (≈ 2–2.5 h together), about $5–6.5 total. **Use the measured numbers, not these.** Show the owner the plan and cost; wait for "yes".
3. **Launch `bdd_baseline` and `bdd_der_enh` concurrently**, same seed, same epochs, logging to MLflow experiment `yolo-der-bdd100k` (a new experiment; don't mix with ExDark). Tag with `dataset=bdd100k`.
4. While they train: the owner (or you) checks MLflow every ~20 minutes and reports epoch time and mAP. If a run crashes, read the log, fix, and resume from `last.pt` (Spec 9.4); don't restart from scratch.
5. If time remains **before 17:00 PKT**, optionally launch `bdd_der` and `bdd_enh` (another ≈ 2 h), or a second seed of the first two. Don't start anything that cannot finish by 20:00 PKT.

## A.6 Evaluation (after training; this is the deliverable)

Extend `evaluate.py` with a `--dataset bdd100k` mode:

1. Overall test metrics (mAP50, mAP50-95, precision, recall, per-class AP50) on BDD val (= our `test`).
2. **Per-condition table:** for each `bins/data_<group>.yaml`, run `val(split="test")` and report mAP50 per group for each variant, plus Δ vs baseline. Groups: `tod_daytime`, `tod_night`, `tod_dawn_dusk`, `wx_clear`, `wx_overcast`, `wx_partly_cloudy`, `wx_rainy`, `wx_snowy`, `wx_foggy`. Ignore `undefined` groups. Mark bins with < 100 images as low-confidence.
3. Params, GFLOPs and latency as in Spec 10.1.
4. Routing analysis (Spec 10.3) on the test split: join gate weights with `attributes.csv`, and show mean gate weights per `timeofday` / `weather` group. Spearman ρ between `s_dark` and the gate weights, as before. With a single seed, say clearly that Δs smaller than the ExDark seed spread (use that spread as the noise estimate) are not conclusive.
5. **Severity-metric validation:** boxplots of `s_dark` by `timeofday`, and `s_fog` by `weather`. Report the AUC for night vs day using `s_dark`, and for foggy vs clear using `s_fog`.
6. Log all of it to MLflow under the `test/` prefix and write `reports/bdd/ablation.md`, `reports/bdd/by_condition.png` and `reports/bdd/severity_validation.png`.

## A.7 Cross-dataset check (optional, only if finished early)

Evaluate the **ExDark-trained** baseline and `der_enh` on BDD night images, mapping classes onto the overlap (ExDark's person/bicycle/car/bus/motorbike ⇄ BDD pedestrian-rider/bicycle/car/bus/motorcycle). Write the mapping in code and state it in the results. This takes minutes and needs no training (Spec E4).

## A.8 Teardown (⛔ CONFIRM before deleting)

Follow Spec Phase 8, with these specifics for this Droplet:
1. Confirm that MLflow has all runs, `best.pt` files and reports, and that the reports are committed. Copy `reports/` and the `attributes.csv` into the repo or Spaces.
2. Show the owner the final `reports/gpu_ledger.md` (add a line for this Droplet: hours and cost, from the Billing page).
3. Ask for a "yes", then destroy the Droplet (`doctl compute droplet delete <id>`) and verify with `doctl compute droplet list` that no GPU Droplet remains. **Do not use power-off.**

## A.9 Gotchas specific to this addendum

- `convert_bdd100k.py --link` creates **symlinks** to the extracted images. The bins' `.txt` files hold absolute paths. Don't move or delete `/work/data_raw` until training and evaluation finish (or drop `--link` to copy, which costs ~6 GB more disk).
- Ultralytics writes `labels/*.cache` next to the labels. If you change labels, delete the caches.
- Ultralytics' per-bin evaluation needs the label file to be found by replacing `/images/` with `/labels/` in each path; the converter's layout satisfies that. **Verify on one small bin** (e.g. `tod_dawn_dusk`) that it reports a nonzero instance count before running all bins.
- `cache: ram` with 2+ processes uses RAM per process; check `free -g` after the first epoch.
- `timeofday` and `weather` may contain `undefined`; keep them in `attributes.csv` but exclude them from the per-condition table.
- 100k-image datasets on a fresh Droplet: the first epoch is slow while caching. Judge s/epoch from epoch 2 onward.
