# YOLO-DER on AMD Instinct MI300X

Implementation of **"YOLO-DER: A Dynamic Enhancement Routing Framework for Adverse Weather
Vehicle Detection"** (Gao, Omar & Mahmuddin, *Electronics* 2025, 14, 4851,
[doi:10.3390/electronics14244851](https://doi.org/10.3390/electronics14244851)) on the
**ExDark** low-light dataset, trained on **AMD Instinct MI300X** GPUs via the AMD Developer
Cloud (DigitalOcean).

> **Status, 18 Oct 2026.** ExDark ablation complete; BDD100K ablation training. Full
> account of what was built, what it found and every mistake along the way:
> **[docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md)**.

## Headline result

**The router does not route.** Normalised routing entropy is **0.9997–0.9999** (1.0 =
perfectly uniform) in every variant, with per-path weights hugging 0.25 for every image.
In this reproduction the paper's "Dynamic Enhancement Routing" behaves as a fixed
equal-weight ensemble of four conv paths.

This is not a dead module — the ReZero scales move well off zero, so the modules do train
and contribute. And it is not a flaw in the pattern generally: EnhanceNet's branch gate
*is* non-uniform at H≈0.82. Enhancement gating discriminates; routing does not.

| variant | mAP50 | Δ vs baseline | +params % | ms/img |
|---|---|---|---|---|
| baseline | 0.5672 | — | 0.00 | **24.35** |
| der | 0.5688 | +0.0016 *(within noise)* | 7.20 | 30.23 |
| der_enh | 0.5657 | −0.0015 *(within noise)* | 9.52 | 34.38 |
| der_enh_entreg | 0.5721 | +0.0049 *(within noise)* | 9.52 | 34.97 |
| **der_enh_sevcond** | **0.5756** | **+0.0084** | 9.53 | 35.92 |

ExDark test split, 100 epochs, seed 0, 448 px. **Noise floor 0.0056 mAP50**, measured as
the spread across all seven variants under identical settings — deltas below it are not
evidence, and `evaluate.py` labels them so.

Two things a mAP-only table would hide:

- **Added parameters stay under the paper's <10% claim, but latency rises 41%** (24.35 →
  34.38 ms/img) for no mAP gain.
- The only delta above the noise floor belongs to `der_enh_sevcond`, an **extension of
  ours** (severity-conditioned gates), not to the paper's mechanism. Its gain concentrates
  in the darkest test quartile: 0.5610 → **0.5973**. One seed, 164 images — a lead worth
  three seeds, not a result.

### Caveats stated up front

- **One seed per variant.** Deltas of 0.002–0.008 against a 0.0056 noise floor cannot be
  settled this way. The honest reading of the ExDark table is "no clear difference".
- **`entropy_reg` was ~100× too small** (logged loss 0.01498, exactly
  `0.01 × mean(3·log4, 3·log5)`), so the paper's "entropy-aware" claim has not yet been
  tested, only measured.
- **BDD100K cannot test the fog claim**: 13 foggy test images. It can test night — 3,929.

## Hardware provenance

Every MLflow run carries `gpu=AMD Instinct MI300X VF` and `hip=6.4.43483`;
`torch.version.hip` being non-empty is the evidence the run used AMD hardware. Measured
throughput: **ExDark 765 img/s** (100 epochs in 9.9 min), **BDD100K 1,056 img/s**
(50 epochs in 52 min), 3.8 GB of 192 GB VRAM. The full ExDark ablation cost about **$2.30**
against the ~$75 originally budgeted.

## What it is

A stock **YOLOv8n** detector with two small, hand-written modules inserted after three
backbone stages:

- **DER** (Dynamic Enhancement Routing) — global-average-pooled features drive a softmax
  router over N lightweight enhancement paths; all paths run and are mixed softly.
  Paper Eq. 1–2.
- **EnhanceNet** — five differentiable branches (tone, contrast, detail, white balance,
  identity) blended by a CBAM-style gate. Paper Eq. 3–9.

Both are identity at initialisation (ReZero-style `λ = 0`), so injecting them into pretrained
YOLOv8n weights does not change the model's outputs at step 0.

## Deviations from the paper

| Paper | Here | Why |
|---|---|---|
| YOLOv12 | YOLOv8n | Mature Ultralytics + ROCm support and pretrained weights |
| `L_enh` Smooth-L1 vs "high-quality reference features" | `λ_enh = 0` | ExDark has no clean/degraded pairs; the term is not computable. No fake reference is invented. |
| Varifocal classification loss | Ultralytics default (BCE + CIoU + DFL) | Keeps the detector loss untouched across variants |
| 300 epochs × 3 seeds, batch 6 | 100 epochs × 2 seeds, batch 64 | ~30 MI300X-hour budget |
| `s_fog` transmission estimator unspecified | Dark Channel Prior (He et al. 2009) | The paper does not specify one |
| N enhancement paths undefined | 4 explicit residual paths (zero / 3×3 DW / dilated DW / SE) | Our interpretation, documented in `src/yolo_der/models/der.py` |

## Research extensions (the paper's gaps)

- **E1** — the paper asserts "entropy-aware" but never measures it. We measure routing and
  feature entropy, and try an explicit entropy regulariser.
- **E2** — routing behaviour is never analysed. We log per-image gate weights and correlate
  them with a measured darkness score `s_dark`.
- **E3** — gates are predicted from GAP features only. We additionally condition them on `s_dark`.

## Repository tour

```
configs/            training variants (one ablation row each)
src/yolo_der/       the package: modules, trainer, tracking, analysis
infra/              doctl scripts: hub, ephemeral GPU launch, reaper, status
api/                Node.js + TypeScript ONNX inference API
dashboard/          static demo UI served by the API
docs/PROJECT_SPEC.md   the full build spec
```

## Quickstart

```bash
# Windows (no make): ./tasks.ps1 <target> with the same target names.
make setup      # .venv + pip install -e ".[dev,train]" + pre-commit
make lint test  # ruff + fast unit tests
make data       # dvc repro: download -> prepare -> severity
make smoke      # 1 epoch on 2% of the data, CPU/T4
```

GPU training is launched manually and never from CI:

```bash
make launch ARGS="--configs 'baseline der enh der_enh' --seed 0 --hours 6"
```

It prints the plan and the worst-case cost and waits for you to type `yes`. A powered-off GPU
Droplet is still billed — only destroying it stops the meter, which is why every launch has a
self-destroy trap, a `timeout` watchdog, and a scheduled reaper behind it.

## Credits and licences

- **Dataset:** ExDark (Loh & Chan, 2019), via
  [Roboflow Universe](https://universe.roboflow.com/project-h68de/exdark-kd37x) — **CC BY 4.0**.
- **Detector:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — AGPL-3.0.
- **Paper:** Gao, Omar & Mahmuddin, *Electronics* 2025, 14, 4851.
- **Dark Channel Prior:** He, Sun & Tang, CVPR 2009.
- Project code: MIT.
