# YOLO-DER on AMD Instinct MI300X

Implementation of **"YOLO-DER: A Dynamic Enhancement Routing Framework for Adverse Weather
Vehicle Detection"** (Gao, Omar & Mahmuddin, *Electronics* 2025, 14, 4851,
[doi:10.3390/electronics14244851](https://doi.org/10.3390/electronics14244851)) on the
**ExDark** low-light dataset, trained on **AMD Instinct MI300X** GPUs via the AMD Developer
Cloud (DigitalOcean).

> Results, figures and costs are filled in at Phase 8. Right now this is a build in progress.

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
