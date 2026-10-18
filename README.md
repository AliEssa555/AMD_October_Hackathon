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

**The router does not route, and when forced to, the detector collapses.**

Normalised routing entropy is **0.9997–0.9999** (1.0 = perfectly uniform) across four
independent DER implementations, with per-path weights hugging 0.25 for every image. The
paper's "Dynamic Enhancement Routing" behaves here as a fixed equal-weight ensemble.

That is not a dead module — the ReZero scales train well away from zero — and not a flaw
in the pattern, because EnhanceNet's branch gate *is* selective at H≈0.82. In
`der_routes_enh`, where DER's router **is** EnhanceNet's gate, the same module
discriminates when used as a gate and does not when used as a router.

The paper's own remedy is an entropy penalty. It had never been tested: at the configured
0.01 the loss term is `0.01 × mean(3·log4, 3·log5) = 0.01498`, about 0.3% of a detection
loss of order 2–6. Running the sweep properly gives a cliff:

| `entropy_reg` | target | DER routing H | mAP50 |
|---|---|---|---|
| 0.01 | all | 0.9998 | 0.5850 |
| 0.1 | all | 0.9996 | 0.5858 |
| 0.5 | all | — | 0.5811 |
| 1.0 | all | 0.6835 | 0.5756 |
| **1.0** | **DER only** | — | **0.5877** |
| 5.0 | all | 0.0147 | 0.4405 |
| **5.0** | **DER only** | — | **0.3220** |

**The regulariser works** — at 5.0 one path takes 0.99 — and **a decisive router is
catastrophic**: −0.257 mAP50 against baseline. Targeting DER alone is *worse* than
targeting both modules, which is what makes the collapse attributable to routing rather
than to EnhanceNet's gate dying alongside it.

### The ablation, and why none of it is a win

| variant | seeds | mAP50 | Δ vs baseline | +params % | ms/img |
|---|---|---|---|---|---|
| baseline | 3 | 0.5792 | — | 0.00 | **24.35** |
| der | 1 | 0.5688 | within noise | 7.20 | 30.23 |
| enh | 1 | 0.5608 | within noise | 2.32 | 30.85 |
| der_enh | 1 | 0.5657 | within noise | 9.52 | 34.38 |
| der_enh_sevcond | 3 | 0.5882 | +0.0090, within noise | 9.53 | 35.92 |

**Measured noise floor: 0.0115 mAP50**, the spread across three seeds of the baseline
alone. Every delta in the table is smaller than that.

Our own extension (E3, severity-conditioned gates) looked like the one real gain at
+0.0084 on a single seed. **With three seeds it is +0.0090 against a 0.0115 floor, and the
ranges overlap.** We deleted our own best result rather than report it.

Cost is the part a mAP-only table hides: added parameters stay inside the paper's <10%
claim, but **latency rises 41%** (24.35 → 34.38 ms/img) for no gain at any setting.

### On BDD100K, 14× more data

| run | epochs | mAP50 | Δ |
|---|---|---|---|
| bdd_baseline | 50 | 0.3711 | — |
| bdd_der | 50 | 0.3748 | +0.0037 |
| bdd_enh | 50 | 0.3733 | +0.0022 |
| bdd_der_enh | 48 | 0.3732 | +0.0021 |

Still tiny, and the ordering differs from ExDark — `der` leads here and trailed there,
which is itself evidence these are noise. BDD was chosen for its ground-truth conditions:
`tod_night` has 3,929 test images so the night claim is testable, while `wx_foggy` has
**13**, so the fog claim is not, and `evaluate.py` flags any bin under 100.

### Caveats stated before the numbers

- Single seed for most variants; three for `baseline` and `der_enh_sevcond` only.
- `bdd_der_enh` is 48/50 epochs — the watchdog killed it two short, and it was still
  improving.
- 448 px throughout, matching the paper; BDD's traffic lights and signs are a few pixels
  across at that size, which is why its absolute mAP is far lower.

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
