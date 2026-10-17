# YOLO-DER on AMD — Project Build Spec

> **For Claude Code:** This is the complete brief for building this project. Read the whole file before writing any code. Sections 0–4 are rules and context, 5–12 are build specs, 13 is the phased task list with acceptance criteria, 14 lists known gotchas, 15 is the glossary for the owner. Work through Section 13 phase by phase and stop wherever it says **⛔ CONFIRM**.

---

## 0. Hard rules (read first)

These override anything else in this document.

1. **Never create a GPU Droplet without explicit user confirmation in chat.** GPU time is the scarce resource (~50 hours total at the verified $1.99/hr). Before every GPU launch, print the run plan and estimated cost (`hours × $1.99`) and wait for an explicit "yes".
2. **A powered-off GPU Droplet is still billed.** Only *destroying* it stops billing. Every GPU launch path must (a) destroy the Droplet when training ends, (b) have a max-runtime watchdog, and (c) be covered by the reaper (Section 8.5).
3. **Never commit secrets.** Tokens and keys live in `.env` (gitignored) locally and in GitHub Actions secrets in CI. The very first commit's `.gitignore` must include `.env`, `.env.*`, `*.pem`, `.dvc/config.local`, `mlruns/`, `runs/`, `*.db`.
4. **Nothing important lives on the GPU Droplet.** Datasets come from Spaces; checkpoints, metrics, and plots go to MLflow (artifacts in Spaces) before the Droplet is destroyed.
5. **Tag every cloud resource `yolo-der`.** Ephemeral GPU Droplets additionally get `ephemeral-gpu`.
6. **Verify, don't assume.** Where this spec says *verify* (Droplet size slugs, region availability, Ultralytics internals, Docker image tags, MLflow flags), check the live CLI output, installed source, or current docs before relying on it. Library internals change between versions; this spec was written against Ultralytics 8.3.x and MLflow 3.x.
7. **Credits expire 19 October 2026.** Today is 11 October 2026. Final teardown (Phase 8) must complete by 18 October.
8. **Hand-write the research code.** DER, EnhanceNet, severity metrics, and the extension losses are written by hand in plain PyTorch with comments explaining the math. Reuse Ultralytics only for the detector backbone/neck/head, loss assignment, data loading, and NMS.
9. **Small, reviewable commits** with conventional-commit messages (`feat:`, `fix:`, `chore:`, `docs:`, `test:`). Each phase ends with a commit and a short summary to the user.

---

## 1. Project summary

Implement, train, evaluate, and serve a computer-vision model based on **"YOLO-DER: A Dynamic Enhancement Routing Framework for Adverse Weather Vehicle Detection"** (Gao, Omar & Mahmuddin, *Electronics* 2025, 14, 4851, doi:10.3390/electronics14244851) on the **ExDark** low-light dataset, training on **AMD Instinct MI300X** GPUs via the AMD Developer Cloud (DigitalOcean-powered). The paper PDF is available to the owner; ask for it if a detail in Section 2 is unclear.

Goals, in priority order:

1. **Learning.** The owner has basic PyTorch and neural-network knowledge and is learning training and fine-tuning. Code must be readable and commented where non-obvious. Prefer clarity over cleverness.
2. **Industry-grade MLOps.** Versioned data (DVC on S3-compatible storage), tracked experiments and a model registry (MLflow), containerized ROCm training (Docker), disposable cloud GPUs launched from scripts (`doctl`, optionally Terraform), CI (GitHub Actions), and model serving (ONNX + Node.js).
3. **Hackathon deliverable** (lablab.ai AMD challenge): a working demo, an ablation table, routing-analysis plots, and a write-up showing training ran on AMD hardware.

---

## 2. The paper (what we implement)

YOLO-DER is a standard YOLO detector (YOLOv12 in the paper; **YOLOv8n in our build**, because it has mature Ultralytics + ROCm support and pretrained weights) with two small modules inserted after several backbone stages. Neck, heads, and NMS are unchanged.

### 2.1 DER — Dynamic Enhancement Routing (paper Eq. 1–2, Fig. 2)
- `Z = GAP(F)`: global average pool the feature map `F ∈ R^{B×C×H×W}` to `Z ∈ R^{B×C}`. The paper claims Z encodes brightness, contrast, and haze cues.
- `ô = softmax(W2 · ReLU(W1·Z + b1) + b2)`, with `W1 ∈ R^{d×C}`, `W2 ∈ R^{N×d}`: routing weights over N enhancement paths.
- Output: weighted sum of the N paths applied to F (soft routing, every path runs).
- Fig. 2 also shows an SE block, a conv, an FC, and BatchNorm before routing. Treat those as optional; implement the Eq. 1–2 core first.
- Inserted at 3 backbone stages in the paper (after an early C3k2, after the first A2C2F, after C2PSA, per Fig. 1).

### 2.2 EnhanceNet (paper Eq. 3–9, Fig. 3)
Five differentiable branches on a feature map, blended by a soft gate:

| Branch | Formula | Predicted params |
|---|---|---|
| Tone | `Norm(F)^γ_c` | γ_c per channel |
| Contrast | `α_c (F − μ_c)/(σ_c + ε) + μ_c + b_c` | α_c, b_c |
| Detail | `F + κ (F − G_σ * F)` | κ (scalar) |
| White balance | `s_c · F` | s_c |
| Identity | `F` | — |

- Parameters come from lightweight MLPs on `GAP(F)`.
- Gate (CBAM-inspired): `β = softmax(W2 σ(W1 GAP(F)))` over the 5 branches.
- `F_enh = Σ_i β_i F_i`, `F_final = F + λ_l F_enh`, λ_l learnable.
- Paper claims < 10% extra params, ~3 ms inference overhead.

### 2.3 Severity metrics (Eq. 10–11)
- `s_dark = max(0, (τ − μ_Y)/τ)`, μ_Y = mean of the Y (luma) channel scaled to [0, 1], τ = 0.3.
- `s_fog = 1 − E_x[t(x)]`, where t(x) = atmospheric transmission. The paper does not specify the estimator; **we use the Dark Channel Prior** transmission estimate (He, Sun & Tang 2009: patch 15×15, ω = 0.95, atmospheric light from the top 0.1% brightest dark-channel pixels).
- Thresholds: foggy if `s_fog ≥ 0.5`, dark if `μ_Y ≤ 0.3`.
- Paper validated these against human ratings (Pearson 0.84 / 0.81). We only use them as analysis signals and optional conditioning.

### 2.4 Loss (Eq. 12–13)
`L = λ_cls·L_vfl + λ_box·L_ciou + λ_enh·L_enh`, with weights 1.0 / 5.0 / 0.1 in the paper. `L_enh` is Smooth-L1 between enhanced features and "high-quality reference features".

**ExDark has no clean/degraded image pairs, so `L_enh` cannot be computed as described.** We keep Ultralytics' default detection loss (it uses BCE + CIoU + DFL rather than Varifocal; do not swap it in this project) and set `λ_enh = 0`. Document this deviation in the write-up. Do not invent a fake reference.

### 2.5 Paper training setup (for reference, not to copy)
PyTorch 2.2, RTX 4090, input 448×448, 300 epochs, batch 6, SGD momentum 0.937, weight decay 5e-4, cosine LR, 3 warm-up epochs, Xavier init for DER/EnhanceNet, 3 repeats averaged. **We will not reproduce 300 epochs × 3 repeats.** See Section 9.3 for our run budget.

### 2.6 Paper numbers to sanity-check direction
ExDark ablation (Fig. 6, approximate): YOLOv12 ≈ 81.7 mAP50 → +DER ≈ 82.0 → +EnhanceNet ≈ 82.6 → both ≈ 84.6 (Table 4 reports 85.6 for the full model). Our absolute numbers will differ (YOLOv8n, 12 classes, fewer epochs). **The goal is to test whether the ablation direction holds**, and to report honestly if it doesn't.

### 2.7 Gaps in the paper = our contribution
1. **"Entropy-aware" is asserted, never measured or optimized (Sec. 3.4).** → Task E1: measure feature entropy before/after the modules, and try an explicit entropy regularizer.
2. **Routing behaviour is never analysed.** → Task E2: log DER/EnhanceNet gate weights per image and correlate with s_dark. Does the network route differently on darker images?
3. **Gate parameters are predicted from GAP features only.** → Task E3 (optional): condition the gates on the measured `s_dark` scalar and compare.
4. **No cross-dataset evaluation.** → Task E4 (optional, only if GPU hours remain).

---

## 3. Budget and timeline

### 3.1 Prices (from DigitalOcean docs, verified Oct 2026)
| Resource | Price | Notes |
|---|---|---|
| AMD MI300X GPU Droplet, 1 GPU (192 GB HBM) | ~~$2.59/hr~~ **$1.99/hr** | Per-second billing, 60 s minimum. **Billed while powered off.** |

> **Price correction, 17 Oct 2026.** The live `devcloud.amd.com` create form quotes
> **$1.99/GPU/hr** for `gpu-mi300x1-192gb-devcloud`, not the $2.59 assumed when this spec
> was written. The $100 credit therefore buys ~50 GPU-hours, not ~38. Every cost print and
> budget figure below that still says $2.59 is an over-estimate; `GPU_HOURLY_USD` in
> `.env` is the single source of truth for scripts.

| AMD MI300X, 8 GPUs | $20.72/hr | Out of scope unless the owner explicitly asks. |
| Spaces Object Storage | $5/month, prorated hourly | 250 GiB storage + 1 TiB egress included. Spaces → Droplet traffic in the same region is free. |
| CPU Droplet (~1 vCPU / 2 GB) | ~$12/month, per second | Hosts MLflow + API. Verify the slug and price with `doctl compute size list`. |
| Container Registry, Starter | Free | 1 repo, 500 MiB — enough for the slim inference image only. |
| Snapshots | $0.06/GB/month | Not needed; cloud-init rebuilds the GPU box. |

### 3.2 Allocation of the $100 credit
| Item | Estimate |
|---|---|
| Spaces, 8 days | ~$1.50 |
| CPU Droplet, 8 days | ~$3.50 |
| Safety buffer | ~$10 |
| **GPU training** | **~$80 ≈ 30 MI300X-hours** |

The owner should set a **spend alert at $85** in Billing → Spend alerts before Phase 4. The billing page shows no saved payment method; DigitalOcean may require one before creating GPU Droplets, and any usage beyond the credit is charged to it.

### 3.3 Timeline
| Day | Date | Phases | GPU hrs |
|---|---|---|---|
| 1 | Sun 11 Oct | 0, 1, 2 (repo, data, cloud base) | 0 |
| 2 | Mon 12 Oct | 3 (modules, tests, Colab validation) | 0 |
| 3 | Tue 13 Oct | 4 (GPU smoke test) | ~2 |
| 4–6 | 14–16 Oct | 5 (ablation + extensions) | ~20–24 |
| 7 | Sat 17 Oct | 6, 7 (ONNX export, API, dashboard) | 0–1 |
| 8 | Sun 18 Oct | 8 (write-up, teardown) | 0 |

---

## 4. Architecture

```
                 ┌──────────────────────────────────────┐
                 │ GitHub repo + GitHub Actions          │
                 │ lint/test · build images · launch     │
                 │ scripts · scheduled reaper            │
                 └───────────┬───────────────┬──────────┘
          launch (⛔ CONFIRM) │               │ build + deploy inference image
                             ▼               ▼
┌────────────────────────────────┐   ┌───────────────────────────────────┐
│ MI300X GPU Droplet (ephemeral) │   │ CPU Droplet "yolo-der-hub"        │
│ tags: yolo-der, ephemeral-gpu  │   │ (always on until Phase 8)         │
│ image: gpu-amd-base (ROCm)     │   │ docker compose:                   │
│ cloud-init:                    │VPC│  • mlflow  :5000  (VPC only)      │
│ 1 install docker, clone repo   │──▶│  • api     :3000  (Node.js)       │
│ 2 dvc pull (Spaces)            │   │  • caddy   :80/:443 (public)      │
│ 3 train in rocm/pytorch        │   └─────────────────┬─────────────────┘
│   container, `timeout` watchdog│                     │
│ 4 log to MLflow over VPC       │                     │
│ 5 self-destroy via DO API      │                     │
└───────────────┬────────────────┘                     │
                ▼                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │ Spaces bucket "yolo-der-<suffix>" (same region)      │
        │  dvc/     → DVC remote (raw + prepared data, CSVs)   │
        │  mlflow/  → MLflow artifacts (ckpts, plots, ONNX)    │
        │  logs/    → raw training logs uploaded on exit       │
        └─────────────────────────────────────────────────────┘
All Droplets share one VPC. Cloud Firewall on the hub: 22 from owner IP,
80/443 from anywhere, 5000 from the VPC CIDR only.
```

**Local development:** the owner's machine and Google Colab (free T4). Colab reaches MLflow through an SSH tunnel (`ssh -L 5000:<hub-private-ip>:5000 root@<hub-public-ip>`, or via the hub's public IP with a temporary firewall rule for the owner's IP) and pulls data from Spaces with DVC. **All code is debugged on CPU/T4 first; MI300X time is only for real runs.**

### 4.1 Why each component (for the owner)
| Component | Industry skill |
|---|---|
| Spaces (S3 API) | Object storage. Same API as AWS S3; DVC, MLflow, boto3 all speak it. |
| Ephemeral GPU + cloud-init | Compute as cattle, not pets. Real training jobs run this way. |
| Docker (`rocm/pytorch`) | Reproducible environments that run identically anywhere. |
| MLflow server | Central experiment tracking + model registry, decoupled from compute. |
| DVC | Data versioning: every result is tied to an exact dataset hash. |
| ONNX + onnxruntime-node | Framework-independent model serving on cheap CPUs. |
| VPC + Cloud Firewall | Private networking and least-exposure security. |
| doctl / Terraform | Infrastructure as code: reproducible, reviewable cloud setup. |
| GitHub Actions | CI/CD: tests, image builds, scheduled housekeeping. |

Deliberately **out of scope** (cost/time): Kubernetes, managed databases, App Platform services, multi-GPU DDP (unless the owner asks; see E5).


---

## 5. Repository layout

Create exactly this structure in Phase 0 (empty files with a docstring/TODO are fine until their phase):

```
yolo-der/
├── CLAUDE.md                      # this spec (or a short pointer to docs/PROJECT_SPEC.md)
├── README.md                      # owner-facing: what, how to run, results table (Phase 8)
├── pyproject.toml                 # package "yolo_der", Python 3.11+, ruff + pytest config
├── params.yaml                    # DVC params (data + eval settings)
├── dvc.yaml / dvc.lock            # data pipeline (Section 6)
├── .env.example                   # every env var name, no values
├── .gitignore  .dockerignore  .pre-commit-config.yaml
├── configs/
│   ├── base.yaml                  # shared training hyper-parameters
│   ├── baseline.yaml              # YOLOv8n, no modules
│   ├── der.yaml                   # + DER only
│   ├── enh.yaml                   # + EnhanceNet only
│   ├── der_enh.yaml               # + both (sequential)
│   ├── der_routes_enh.yaml        # + both, DER routes the EnhanceNet branches (optional)
│   ├── der_enh_entreg.yaml        # E1: + entropy regulariser
│   └── der_enh_sevcond.yaml       # E3: + severity-conditioned gates
├── src/yolo_der/
│   ├── __init__.py
│   ├── config.py                  # load base.yaml + variant yaml → dataclass
│   ├── data/
│   │   ├── download.py            # Roboflow → data/raw
│   │   ├── prepare.py             # data/raw → data/prepared (+ data.yaml, stats)
│   │   └── severity.py            # per-image s_dark / s_fog CSV + dark bins
│   ├── models/
│   │   ├── der.py                 # DER module (hand-written)
│   │   ├── enhancenet.py          # EnhanceNet module (hand-written)
│   │   ├── severity_torch.py      # differentiable s_dark on a batch (for E3)
│   │   ├── inject.py              # wrap Ultralytics layers with our modules
│   │   └── yolo_der_model.py      # DetectionModel subclass (+ extra loss terms)
│   ├── losses.py                  # entropy + load-balancing regularisers (E1)
│   ├── trainer.py                 # DetectionTrainer subclass
│   ├── tracking.py                # MLflow callbacks, run-id persistence, S3 checkpoint sync
│   ├── train.py                   # CLI entry: python -m yolo_der.train --config ...
│   ├── evaluate.py                # test-set eval, per-dark-bin eval, latency, params/GFLOPs
│   ├── analysis/
│   │   ├── record_gates.py        # forward hooks → routing/gate weights per image (parquet)
│   │   ├── entropy.py             # E1 measurements
│   │   └── plots.py               # E2 figures
│   └── export.py                  # ONNX export + parity check + upload to Spaces
├── tests/
│   ├── test_der.py  test_enhancenet.py  test_inject.py
│   ├── test_severity.py  test_losses.py  test_checkpoint_roundtrip.py
│   └── test_export.py             # marked slow
├── docker/
│   ├── train.Dockerfile           # FROM rocm/pytorch:<pinned tag>
│   └── api.Dockerfile             # node:22-bookworm-slim, multi-stage
├── infra/
│   ├── cloud-init.yaml.tmpl       # GPU Droplet user-data template
│   ├── bootstrap.sh               # runs on the GPU Droplet (watchdog, train, upload, self-destroy)
│   ├── launch_gpu.sh              # ⛔ prints plan + cost, asks for "yes", creates the Droplet
│   ├── reaper.sh                  # destroys ephemeral-gpu Droplets older than N hours
│   ├── status.sh                  # lists all yolo-der resources + balance
│   ├── hub/
│   │   ├── docker-compose.yml     # mlflow, api, caddy
│   │   ├── mlflow.Dockerfile      # python slim + mlflow + boto3
│   │   ├── Caddyfile
│   │   └── backup_mlflow.sh       # sqlite .backup → Spaces
│   └── terraform/                 # optional (Phase 2b): VPC, firewall, hub, bucket
├── api/                           # Node.js + TypeScript inference API
│   ├── package.json  tsconfig.json
│   ├── src/{server.ts, model.ts, preprocess.ts, postprocess.ts, mlflow.ts, config.ts}
│   └── test/{nms.test.ts, letterbox.test.ts, golden.test.ts}
├── dashboard/                     # static HTML/JS served by the API
├── notebooks/
│   └── colab_dev.ipynb            # Colab: clone repo, dvc pull, tunnel to MLflow, short runs
├── reports/                       # generated tables/figures (small files only; committed)
└── .github/workflows/
    ├── ci.yml  deploy-api.yml  reaper.yml
```

Conventions:
- Python: `ruff` (lint + format), type hints on public functions, docstrings that cite the paper equation (e.g. `"""Paper Eq. 4: contrast branch."""`).
- Everything runnable as `python -m yolo_der.<module>`; no notebooks in the critical path.
- All paths come from config or env vars; never hard-code `/content/...` or `/root/...`.
- Inside containers the repo lives at `/work/yolo-der`, data at `/work/yolo-der/data`, runs at `/work/runs`. Ultralytics' resume stores absolute paths, so these must be identical on every GPU Droplet (Section 9.4).

### 5.1 Environment variables (`.env.example`)
```
# DigitalOcean
DIGITALOCEAN_ACCESS_TOKEN=        # owner's full token, laptop + GitHub Actions only
DO_SELF_DESTROY_TOKEN=            # custom-scoped token: droplet read + delete only (verify scope names)
DO_REGION=                        # region where MI300X is available (verify)
DO_VPC_UUID=
DO_SSH_KEY_FINGERPRINT=
GPU_SIZE_SLUG=                    # verify with `doctl compute size list` (expected like gpu-mi300x1-192gb)
GPU_IMAGE_SLUG=gpu-amd-base       # verify with `doctl compute image list --public`
MAX_RUNTIME_HOURS=6
# Spaces (S3-compatible)
SPACES_BUCKET=yolo-der-<suffix>
SPACES_ENDPOINT=https://<region>.digitaloceanspaces.com
AWS_ACCESS_KEY_ID=                # Spaces access key (boto3/DVC/MLflow read these names)
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1      # dummy value; boto3 requires one
MLFLOW_S3_ENDPOINT_URL=${SPACES_ENDPOINT}
# MLflow
MLFLOW_TRACKING_URI=http://<hub-private-ip>:5000   # http://localhost:5000 via SSH tunnel from laptop/Colab
MLFLOW_EXPERIMENT_NAME=yolo-der-exdark
# Data
ROBOFLOW_API_KEY=
```

---

## 6. Data and DVC

### 6.1 Source
- Roboflow Universe: workspace `project-h68de`, project `exdark-kd37x`, **version 2**, export format `yolov8`. 6,554 images, 12 classes, CC BY 4.0 (credit Roboflow and the original ExDark authors, Loh & Chan 2019, in the README).
- Download with the `roboflow` package: `Roboflow(api_key=...).workspace("project-h68de").project("exdark-kd37x").version(2).download("yolov8", location="data/raw")`.
- **Never hard-code class order.** Read `names` from the downloaded `data.yaml`; the API's `labels.json` is generated from it at export time.

### 6.2 `prepare.py`
1. Validate: every image has a label file (empty label files are allowed for background images); every label line has 5 floats in [0, 1]; class ids are < nc. Log and drop corrupt files; write a list of dropped files to `data/prepared/dropped.txt`.
2. Keep Roboflow's train/valid/test split (record the counts). If Roboflow's version has no test split, carve 10% of train with a fixed seed, stratified by the image's majority class.
3. Copy (or hard-link) to `data/prepared/{train,valid,test}/{images,labels}`.
4. Write `data/prepared/data.yaml` with a **relative** `path:` field, `train: train/images`, `val: valid/images`, `test: test/images`, `nc`, `names`.
5. Write `reports/data_stats.json` (images and instances per split and class) and `reports/class_hist.png`.

### 6.3 `severity.py`
For every image in every split, compute with NumPy/OpenCV (read with `cv2.imread`, convert BGR→RGB, scale to [0, 1]):
- `mu_Y` = mean of `Y = 0.299R + 0.587G + 0.114B`.
- `s_dark = max(0, (0.3 − mu_Y) / 0.3)`.
- `s_fog = 1 − mean(t)`, where `t = 1 − ω · dark_channel(I / A)`, with patch 15, ω = 0.95, A = mean RGB of the brightest 0.1% dark-channel pixels. Clip t to [0.1, 1]. Resize the long side to 640 before DCP for speed; document this.
- `dark_bin`: quartile bin (`q1` darkest … `q4` brightest) computed from **test-split** `mu_Y` quantiles, then applied to all splits.

Output: `data/severity.csv` with columns `split,image,mu_Y,s_dark,s_fog,dark_bin`, plus `reports/severity_hist.png`. Also write `data/prepared/test_bins/q{1..4}.txt` (image path lists) and `data/prepared/data_test_q{1..4}.yaml` so Ultralytics can evaluate per bin (Ultralytics accepts a `.txt` list of image paths as a split; verify).

Expected: ExDark is mostly dark, so most images will have `s_dark > 0` and `s_fog` will be noisy (DCP misreads darkness as haze). That is a finding to report, not a bug.

### 6.4 `params.yaml`
```yaml
data:
  roboflow: {workspace: project-h68de, project: exdark-kd37x, version: 2, format: yolov8}
  test_fraction_if_missing: 0.1
  seed: 0
severity:
  tau: 0.3
  dcp_patch: 15
  dcp_omega: 0.95
  dcp_top_fraction: 0.001
  dcp_resize_long_side: 640
```

### 6.5 `dvc.yaml`
```yaml
stages:
  download:
    cmd: python -m yolo_der.data.download --out data/raw
    deps: [src/yolo_der/data/download.py]
    params: [data.roboflow]
    outs: [data/raw]
  prepare:
    cmd: python -m yolo_der.data.prepare --raw data/raw --out data/prepared
    deps: [src/yolo_der/data/prepare.py, data/raw]
    params: [data.test_fraction_if_missing, data.seed]
    outs: [data/prepared]
    metrics: [reports/data_stats.json: {cache: false}]
    plots: [reports/class_hist.png: {cache: false}]
  severity:
    cmd: python -m yolo_der.data.severity --data data/prepared --out data/severity.csv
    deps: [src/yolo_der/data/severity.py, data/prepared]
    params: [severity]
    outs: [data/severity.csv]
    plots: [reports/severity_hist.png: {cache: false}]
```
Note: `severity.py` also writes `test_bins/` and the per-bin yamls inside `data/prepared`, which is already the `prepare` stage's output. **DVC forbids two stages writing the same output.** Write them to `data/bins/` instead, declare `data/bins` as an output of `severity`, and use paths relative to `data/prepared` inside the yamls. Verify with `dvc repro` and `dvc status`.

Training is **not** a DVC stage. It runs on ephemeral GPUs and is tracked in MLflow. Each MLflow run logs the `dvc.lock` md5 of `data/prepared` as the `data_hash` tag, which links results to the exact dataset.

### 6.6 Remote
Phase 1 starts with a local remote (`dvc remote add -d local /tmp/dvc-store`). Phase 2 adds Spaces and makes it the default:
```bash
dvc remote add -d spaces s3://$SPACES_BUCKET/dvc
dvc remote modify spaces endpointurl $SPACES_ENDPOINT
dvc remote modify --local spaces access_key_id $AWS_ACCESS_KEY_ID      # goes to .dvc/config.local (gitignored)
dvc remote modify --local spaces secret_access_key $AWS_SECRET_ACCESS_KEY
dvc push
```
Install with `pip install "dvc[s3]"`. Acceptance: a fresh clone + `dvc pull` reproduces `data/prepared` byte-for-byte (`dvc status` clean).

---

## 7. Model modules (hand-written PyTorch)

General rules for every module:
- Input and output shape are identical: `(B, C, H, W) → (B, C, H, W)`.
- **Identity at initialisation.** The residual scale `λ` starts at 0 (ReZero-style), so injecting the modules into a pretrained YOLOv8n does not change its outputs at step 0. A unit test asserts this exactly.
- Every predicted parameter is passed through a bounded function (below), so training cannot explode.
- **Run the module math in fp32.** Wrap the forward in `torch.autocast(device_type=x.device.type, enabled=False)` and use `x.float()`, then cast the output back to `x.dtype`. `pow`, `logit`, and the division by σ are unstable in fp16 under AMP.
- Lightweight MLP hidden width: `d = max(C // 8, 16)`.
- Initialisation: Xavier-uniform for the hidden Linear layers (paper Sec. 4.1). **Zero-init the last Linear layer of every parameter head and gate**, so all predicted parameters start at their neutral value and gates start uniform.
- Recording: each module has `self.record: bool = False`. When True, the forward stores detached tensors on `self.last` (a dict: routing/gate weights `(B, N)` and mean predicted parameters). Analysis scripts enable this; training leaves it off. Recording must never change the output, and no code path may branch on tensor values (keeps ONNX export traceable).

### 7.1 `der.py` — DER
Paper Eq. 1–2 define the router; the paper does not define the N "enhancement paths". **Our interpretation (document it in the code and README):** N = 4 lightweight residual paths:

| # | Path `P_n(F)` | Intent |
|---|---|---|
| 0 | `0` (zero / identity path) | "do nothing" option for already-good features |
| 1 | depthwise 3×3 conv → BN → SiLU → pointwise 1×1 conv | local refinement |
| 2 | depthwise 3×3 conv, dilation 2 → BN → SiLU → pointwise 1×1 | wider context |
| 3 | SE channel re-weighting: `F · sigmoid(MLP(GAP(F))) − F` | global channel recalibration |

```
Z      = GAP(F)                                  # (B, C)
logits = W2 · ReLU(W1 · Z + b1) + b2             # (B, N)          Eq. 1
ô      = softmax(logits / T)                     # T = 1 default   Eq. 2
F_der  = Σ_n ô_n · P_n(F)                        # soft routing, all paths run
out    = F + λ_der · F_der                       # λ_der learnable scalar, init 0
```
- Constructor: `DER(channels, num_paths=4, hidden=None, temperature=1.0, cond_dim=0)`. `cond_dim > 0` concatenates a conditioning vector (E3) to Z before W1; `forward(x, cond=None)`.
- Expose `self.last["routing"] = ô` when recording, and `self.routing_for_loss = ô` (not detached) when an extension loss needs it. Clear it every forward so stale tensors don't hold the graph.
- The Fig. 2 extras (SE + conv + BN before routing) are **not** implemented in v1. Add them only if time allows, as `DER(..., pre_block=True)`.

### 7.2 `enhancenet.py` — EnhanceNet
Shared trunk `h = ReLU(W_t · GAP(F) + b_t)` with width d, then separate zero-initialised heads. `μ_c` and `σ_c` are the per-sample, per-channel spatial mean and standard deviation of F; `ε = 1e-5`.

| Branch | Head output | Bounded parameter (neutral at zero-init) | Branch output `F_i` |
|---|---|---|---|
| Tone (Eq. 3) | `a_γ ∈ R^C` | `γ = 3^{tanh(a_γ)} ∈ [1/3, 3]` (γ = 1) | `N = sigmoid((F−μ)/(σ+ε))`; `N' = clamp(N^γ, 1e-4, 1−1e-4)`; `F_tone = μ + (σ+ε)·logit(N')` |
| Contrast (Eq. 4) | `a_α, a_b ∈ R^C` | `α = softplus(a_α)/ln 2` (α = 1), `b = 0.1·tanh(a_b)·σ` (b = 0) | `α·(F−μ)/(σ+ε) + μ + b` |
| Detail (Eq. 5) | `a_κ ∈ R^1` | `κ = 2·sigmoid(a_κ) ∈ (0, 2)` (κ = 1) | `F + κ·(F − G_σ * F)`; `G_σ` = fixed 5×5 Gaussian, σ = 1, depthwise conv, registered buffer, reflect padding |
| White balance (Eq. 6) | `a_s ∈ R^C` | `s = 1 + 0.5·tanh(a_s) ∈ [0.5, 1.5]` (s = 1) | `s · F` |
| Identity (Eq. 7) | — | — | `F` |

Notes:
- Why the sigmoid/logit dance for tone: features are unbounded and `x^γ` needs inputs in (0, 1). Normalising per channel, applying the curve, and mapping back keeps the operation a pure tone remap of each channel's distribution. Write this explanation as a comment.
- Gate (Eq. 8, CBAM-inspired): `β = softmax(W_g2 · ReLU(W_g1 · GAP(F)))` over 5 branches, so β starts uniform at 0.2 each. The paper writes a generic σ; we use ReLU and document it.
- Fusion (Eq. 9): `F_enh = Σ_i β_i F_i`; `out = F + λ_enh · F_enh`, with `λ_enh` a learnable scalar initialised to 0. Implement as in the paper, even though `F_enh` already contains F through the identity branch.
- Constructor: `EnhanceNet(channels, hidden=None, cond_dim=0, external_gate=False)`. With `external_gate=True`, `forward(x, gate=β)` uses the supplied weights instead of its own gate (used by the `der_routes_enh` combination).
- Record: `β (B,5)`, mean γ, mean α, κ, mean s, per image.

### 7.3 Combining DER and EnhanceNet
Config `modules.combine`:
- `sequential` (default, main ablation): at each insertion point `out = EnhanceNet(DER(F))`.
- `der_routes_enh` (optional variant): DER with `num_paths=5` produces `ô`, which is passed to `EnhanceNet(external_gate=True)` as β. This is one plausible reading of Fig. 1–3 ("DER routes features to enhancement paths"). Report it as a variant, not as "the paper's method".

The owner should check Figures 1–3 in the PDF and say which reading is closer. Ask once in Phase 3; default to `sequential`.

### 7.4 `inject.py` — inserting into YOLOv8n
YOLOv8n backbone (`model.model[i]`, verify against the installed Ultralytics):

| i | Layer | Out channels (n-scale) | Stride |
|---|---|---|---|
| 2 | C2f | 32 | 4 |
| 4 | C2f | 64 | 8 (P3) |
| 6 | C2f | 128 | 16 (P4) |
| 8 | C2f | 256 | 32 |
| 9 | SPPF | 256 | 32 (P5) |

Default insertion points `insert_at: [4, 6, 9]`: early (P3), middle (P4), and after the last backbone block. This mirrors the paper's "early C3k2, first A2C2F, after C2PSA" as closely as the YOLOv8 architecture allows.

Mechanism:
```python
class EnhancedLayer(nn.Module):
    """Wraps an Ultralytics layer: y = enhance(layer(x)). Keeps Ultralytics' routing metadata."""
    def __init__(self, layer: nn.Module, enhance: nn.Module):
        super().__init__()
        self.layer, self.enhance = layer, enhance
        # Ultralytics' DetectionModel._predict_once uses these attributes:
        self.f, self.i, self.type = layer.f, layer.i, f"yolo_der.{type(enhance).__name__}"
        self.np = sum(p.numel() for p in self.parameters())
    def forward(self, x):
        return self.enhance(self.layer(x))
```
- `inject(model: DetectionModel, cfg) -> DetectionModel` replaces `model.model[i]` in place for each `i` in `insert_at`. Indices do not shift, so the neck's skip connections (`save` list) still work.
- **Find channels at runtime** with a dummy forward + hooks (`torch.zeros(1, 3, 448, 448)`). Don't hard-code the table above.
- **State-dict keys change.** Wrapping renames `model.4.cv1.conv.weight` to `model.4.layer.cv1.conv.weight`. Therefore:
  - **Fresh run from `yolov8n.pt`:** build → `model.load(pretrained)` → `inject()`.
  - **Resume or load our own checkpoint:** build → `inject()` → `model.load(our_ckpt)`.
  - `get_model` decides by checking whether the incoming weights already contain `EnhancedLayer` modules.
  - Test both paths. Ultralytics prints "Transferred X/Y items"; assert X == Y in the resume test.
- Checkpoints are pickled whole models, so `yolo_der` must be importable (installed with `pip install -e .`) anywhere a `.pt` is loaded. Keep class names and module paths stable after Phase 3.

### 7.5 `yolo_der_model.py` — model subclass
- `class YoloDERModel(DetectionModel)` takes `(cfg, nc, der_cfg)` and calls `inject` at the right time (above).
- **Severity conditioning (E3):** if `severity_cond` is on, wrap layer 0 in a `SeverityProbe` that computes `s_dark` per image from the raw input batch (`x` is RGB in [0, 1]; verify channel order in Ultralytics' dataloader) and stores it in a shared context object that the DER/EnhanceNet modules read as `cond` (shape `(B, 1)`). Computing it inside the model means it survives ONNX export.
- **Extra loss terms (E1):** override `loss(batch, preds=None)`. Call `super().loss(...)`, then add `losses.entropy_regulariser(...)` collected from the modules' `routing_for_loss`. Ultralytics returns `(loss_sum_scaled_by_batch, loss_items)`; scale the extra term the same way and append it to `loss_items`. Update `trainer.loss_names` to add `"ent_loss"`. **Verify** the exact return contract in `ultralytics/utils/loss.py` (`v8DetectionLoss.__call__`) and `DetectionModel.loss` in the installed version first.

### 7.6 `losses.py`
- `routing_entropy(p) = −Σ p log(p + 1e-8)`, averaged over the batch. Applies to DER's `ô` and EnhanceNet's `β`.
- `entropy_regulariser = w_ent · mean_layers(routing_entropy)`. A positive `w_ent` sharpens routing (confident choices); default 0.01.
- `load_balance = w_lb · KL(mean_batch(p) ‖ uniform)`. This stops all images collapsing onto one path (the Mixture-of-Experts trick); default 0.01.
- Both are 0 unless the config enables them.

### 7.7 Unit tests (`tests/`, run on CPU in CI, < 60 s total)
1. **Shape:** for C ∈ {32, 64, 256} and odd H/W (e.g. 13×17), output shape == input shape, for DER, EnhanceNet, and both.
2. **Identity at init:** `torch.equal(module(x), x)` for each module with λ = 0.
3. **Gradient flow:** set λ = 0.1, run forward + `.sum().backward()`; every parameter has a finite, non-None grad, and the detail branch's Gaussian buffer has no grad.
4. **Bounds:** with random head weights (std 10), γ ∈ [1/3, 3], α > 0, κ ∈ (0, 2), s ∈ [0.5, 1.5], and the outputs contain no NaN/Inf.
5. **Softmax:** routing and gate weights sum to 1 (atol 1e-6).
6. **AMP:** under `torch.autocast("cpu", dtype=torch.bfloat16)`, the output has no NaN and the input dtype is preserved.
7. **Injection parity:** `DetectionModel("yolov8n.yaml")` with random weights, forward a 1×3×448×448 tensor, inject with λ = 0; the outputs are identical. After injection `len(model.model)` is unchanged and the `save` indices still work.
8. **Checkpoint round-trip:** `torch.save({"model": model}, f)`, reload, build a fresh injected model, `load`, and all keys transfer.
9. **Severity:** an all-black image gives s_dark = 1; a uniform 0.5-grey image gives s_dark = 0; a synthetic hazy image (`J·t + A·(1−t)` with t = 0.3) gives s_fog ≈ 0.7 ± 0.1; the torch and NumPy s_dark implementations agree (atol 1e-5).
10. **Losses:** entropy of a uniform distribution over N paths equals log N; the load-balance KL is 0 for uniform.

---

## 8. Cloud infrastructure (DigitalOcean / AMD Developer Cloud)

### 8.1 Accounts, tokens, keys (owner does these by hand; Claude Code writes the checklist)
1. **Confirm which account holds the $100 credit.** AMD Developer Cloud runs on DigitalOcean, but the credit may sit in a separate team or console. Create the API token **in the team that shows the credit** on its Billing page, then run `doctl auth init --context yolo-der` and `doctl balance get` and confirm the credit appears. Verify before creating anything.
2. Tokens:
   - `DIGITALOCEAN_ACCESS_TOKEN`: full access, laptop and GitHub Actions only.
   - `DO_SELF_DESTROY_TOKEN`: custom-scoped with droplet read + delete only (verify the scope names in the token UI). This is the only DO token that ever reaches a GPU Droplet.
3. A Spaces access key. If the UI offers per-bucket ("limited") access keys, scope the key to the project bucket only (verify). The API's runtime key should be read-only if possible.
4. Upload an SSH key: `doctl compute ssh-key import yolo-der --public-key-file ~/.ssh/id_ed25519.pub`.
5. Set a spend alert at $85. Check whether a payment method is required for GPU Droplets.

### 8.2 Region, VPC, firewall
- **Region:** the one where `doctl compute size list` shows an MI300X slug **and** Spaces is offered. If Spaces is not in the GPU region, use the nearest Spaces region; the included 1 TiB egress covers our traffic. Record the choice in README.
- **VPC:** `doctl vpcs create --name yolo-der-vpc --region $DO_REGION`. Every Droplet joins it (`--vpc-uuid`).
- **Cloud Firewall `yolo-der-hub-fw`** (attach by tag `hub`):
  - inbound 22/tcp from the owner's IP only;
  - inbound 80, 443/tcp from anywhere;
  - inbound 5000/tcp from the VPC CIDR only;
  - all outbound allowed.
- GPU Droplets get a separate firewall `yolo-der-gpu-fw` (tag `ephemeral-gpu`): inbound 22 from the owner's IP only, all outbound allowed.

### 8.3 Spaces bucket
- `yolo-der-<random-6-chars>` in the chosen region, private, with the prefixes `dvc/`, `mlflow/`, `logs/`, `checkpoints/`, `onnx/`, `reports/`, `backups/`.
- Create it from the UI, or with any S3 client: `aws s3 mb s3://... --endpoint-url $SPACES_ENDPOINT`.

### 8.4 Hub Droplet `yolo-der-hub`
- Size: the smallest that runs MLflow and the API comfortably. Start with the 1 vCPU / 2 GB tier, verify the slug and price, and add 1 GB of swap.
- Image: `ubuntu-24-04-x64`. Tags: `yolo-der,hub`. VPC: yes.
- Create it with `doctl compute droplet create`, with a cloud-init that installs Docker Engine + the compose plugin from Docker's apt repo, creates `/opt/hub`, and enables unattended-upgrades.

`infra/hub/docker-compose.yml`:
```yaml
services:
  mlflow:
    build: {context: ., dockerfile: mlflow.Dockerfile}   # python:3.12-slim + pip install "mlflow==3.*" boto3
    restart: unless-stopped
    env_file: /opt/hub/.env          # AWS_* keys, MLFLOW_S3_ENDPOINT_URL, SPACES_BUCKET
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////data/mlflow.db
      --artifacts-destination s3://${SPACES_BUCKET}/mlflow
      --serve-artifacts
    volumes: [mlflow-data:/data]
    ports: ["${HUB_PRIVATE_IP}:5000:5000"]    # bind to the VPC interface only, never 0.0.0.0
  api:
    image: ${API_IMAGE}                       # ghcr.io/<owner>/yolo-der-api:<sha> (Section 11.4)
    restart: unless-stopped
    env_file: /opt/hub/.env
    environment: {MLFLOW_TRACKING_URI: "http://mlflow:5000", PORT: "3000"}
    depends_on: [mlflow]
    expose: ["3000"]
  caddy:
    image: caddy:2
    restart: unless-stopped
    ports: ["80:80", "443:443"]
    volumes: [./Caddyfile:/etc/caddy/Caddyfile:ro, caddy-data:/data]
    environment: {SITE_ADDRESS: "${SITE_ADDRESS}"}
volumes: {mlflow-data: {}, caddy-data: {}}
```
Notes:
- **MLflow file store is deprecated** (it raised `MlflowException` on Colab). Always use the SQLite (or another database) backend.
- With `--serve-artifacts`, clients upload artifacts **through** the MLflow server, so GPU Droplets don't need Spaces keys for MLflow. They still need them for DVC and checkpoint sync.
- MLflow 3.x has host-header / CORS protection flags. If requests to `http://<private-ip>:5000` return 403 "Invalid Host header", set the allowed-hosts option. Verify the flag name with `mlflow server --help`.
- No auth on MLflow: it is reachable only on the VPC and through an SSH tunnel. Don't expose it via Caddy.
- `SITE_ADDRESS`: without a domain, use `<ip-with-dashes>.sslip.io` (e.g. `203-0-113-7.sslip.io`). Caddy gets a real Let's Encrypt cert for it automatically.
- `Caddyfile`:
  ```
  {$SITE_ADDRESS} {
      encode gzip
      reverse_proxy api:3000
  }
  ```
- `backup_mlflow.sh`: `sqlite3 /data/mlflow.db ".backup /tmp/mlflow-$(date +%F).db"`, then upload to `s3://$SPACES_BUCKET/backups/`. Run it via cron every 6 h and before teardown.

Access from the laptop or Colab:
```bash
ssh -N -L 5000:<hub-private-ip>:5000 root@<hub-public-ip>
export MLFLOW_TRACKING_URI=http://localhost:5000
```
Colab: run the tunnel with a private key stored in Colab secrets, or temporarily add the Colab egress IP to the firewall's port 22 rule. Document both options in `notebooks/colab_dev.ipynb`.

### 8.5 Ephemeral GPU training Droplet

**`infra/launch_gpu.sh`** (run from the laptop only):
1. Inputs: `--configs "baseline der enh der_enh"` (run concurrently on one GPU), `--seed`, `--hours` (default `MAX_RUNTIME_HOURS`), `--resume-from s3://...` (optional), `--git-ref` (default the current commit, which must be pushed).
2. Pre-flight checks:
   - the working tree is clean and pushed;
   - `doctl compute droplet list --tag-name ephemeral-gpu` is empty (one GPU Droplet at a time unless the owner says otherwise);
   - `doctl balance get` is printed;
   - the hub's MLflow answers via its private IP (check from the hub over SSH).
3. Print the plan: configs, seed, max hours, **max cost = hours × $2.59**, and the credit remaining after the worst case. Then require the user to type `yes`. (In Claude Code, also wait for the owner's "yes" in chat: rule 0.1.)
4. Render `cloud-init.yaml.tmpl` with `envsubst` into a temp file (mode 600; delete it after create). It contains:
   - `write_files`: `/opt/yolo-der/run.env` (mode 600) holding `GIT_REF`, `CONFIGS`, `SEED`, `MAX_RUNTIME_HOURS`, `RESUME_FROM`, `MLFLOW_TRACKING_URI=http://<hub-private-ip>:5000`, the Spaces keys, `DO_SELF_DESTROY_TOKEN`, and the repo URL. If the repo is private, add a read-only deploy key or a fine-grained PAT with read access.
   - `/opt/yolo-der/bootstrap.sh`, copied from `infra/bootstrap.sh`.
   - `runcmd: [[bash, -lc, "/opt/yolo-der/bootstrap.sh > /var/log/yolo-der.log 2>&1"]]`.
5. Create the Droplet:
   ```bash
   doctl compute droplet create "yolo-der-gpu-$(date +%m%d-%H%M)" \
     --region $DO_REGION --size $GPU_SIZE_SLUG --image $GPU_IMAGE_SLUG \
     --vpc-uuid $DO_VPC_UUID --ssh-keys $DO_SSH_KEY_FINGERPRINT \
     --tag-names yolo-der,ephemeral-gpu --user-data-file "$RENDERED" --wait
   ```
6. Print the Droplet ID, its IP, and how to follow the log: `ssh root@IP tail -f /var/log/yolo-der.log`.

User-data is readable from inside the Droplet through the metadata service. That is acceptable only because every secret in it is narrowly scoped: the self-destroy token can only delete Droplets, and the Spaces key should ideally be bucket-scoped. **Never put the full-access DO token in user-data.**

**`infra/bootstrap.sh`** (runs as root on the GPU Droplet):
```bash
set -euo pipefail
source /opt/yolo-der/run.env
DROPLET_ID=$(curl -s http://169.254.169.254/metadata/v1/id)
cleanup() {                                   # ALWAYS runs: success, failure, or timeout
  docker run --rm -v /var/log:/logs -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
    amazon/aws-cli s3 cp /logs/yolo-der.log "s3://$SPACES_BUCKET/logs/$DROPLET_ID.log" \
    --endpoint-url "$SPACES_ENDPOINT" || true
  curl -s -X DELETE -H "Authorization: Bearer $DO_SELF_DESTROY_TOKEN" \
    "https://api.digitalocean.com/v2/droplets/$DROPLET_ID"
}
trap cleanup EXIT
```
Then, in order:
1. Install Docker if missing (the `gpu-amd-base` image may already ship it; verify).
2. Run `rocm-smi`; fail fast if no GPU is visible.
3. Clone the repo at `$GIT_REF` into `/work/yolo-der`.
4. `docker build -f docker/train.Dockerfile -t yolo-der-train .`
5. Run the job inside the container, under the watchdog:
   ```bash
   timeout --signal=TERM --kill-after=5m "${MAX_RUNTIME_HOURS}h" \
     docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
       --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
       --env-file /opt/yolo-der/run.env -v /work:/work -w /work/yolo-der \
       yolo-der-train bash infra/run_job.sh
   ```
   `infra/run_job.sh` runs `dvc pull`, then launches one `python -m yolo_der.train --config configs/<c>.yaml --seed $SEED` process **per config in parallel** (`&` + `wait`), then runs `python -m yolo_der.evaluate` for each finished run.
6. Exit. The trap uploads the log and destroys the Droplet.

`--device=/dev/kfd --device=/dev/dri --group-add video` is the standard way to give a container ROCm GPU access. Verify against the current ROCm Docker docs.

**`docker/train.Dockerfile`:**
```dockerfile
FROM rocm/pytorch:<PINNED_TAG>      # verify: latest tag compatible with the host ROCm driver (host ships ROCm 7.x)
WORKDIR /work/yolo-der
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . "ultralytics==8.3.*" "mlflow==3.*" "dvc[s3]" boto3 pandas pyarrow \
 && python -c "import torch,sys; assert torch.version.hip, 'pip replaced ROCm torch!'; print(torch.__version__, torch.version.hip)"
```
The assertion is mandatory: if any dependency pulls a CUDA/CPU torch wheel from PyPI, the build must fail. If that happens, install `ultralytics` with `--no-deps` and add its remaining deps explicitly.

**Reaper (`infra/reaper.sh`):** list Droplets tagged `ephemeral-gpu`, and destroy any whose `created_at` is older than `REAPER_MAX_AGE_HOURS` (default `MAX_RUNTIME_HOURS + 1`). Log each action. Run it:
- from GitHub Actions on a 30-minute cron (Section 12);
- manually via `make reap`.

That gives four independent layers: normal exit → trap, the `timeout` watchdog → trap, the reaper, and the spend alert.

**`infra/status.sh`:** prints all resources tagged `yolo-der` (Droplets, firewalls, volumes), the bucket size, `doctl balance get`, and month-to-date usage. Run it at the start and end of every phase.

### 8.6 Terraform (optional Phase 2b, learning stretch)
Model the long-lived resources only: VPC, firewalls, hub Droplet, Spaces bucket, and a DO Project grouping them. **GPU Droplets are not in Terraform**; they are created and destroyed by script.
- State: local `terraform.tfstate` (gitignored), or the S3 backend on Spaces (`skip_credentials_validation`, `skip_region_validation`, `skip_requesting_account_id`, `skip_s3_checksum` set to true; verify the current flags).
- Do this only after the doctl version works, then `terraform import` the existing resources. It teaches import/plan/apply without risking the schedule.

---

## 9. Training

### 9.1 Configs
`configs/base.yaml`:
```yaml
model: yolov8n.pt
data: data/prepared/data.yaml
train:
  epochs: 100          # FIXED before first launch: Ultralytics resume restores epochs from the checkpoint
  imgsz: 448           # paper's input size
  batch: 64            # same for every variant; paper used 6 on a 4090 (note the deviation)
  optimizer: SGD       # don't let 'auto' pick AdamW
  lr0: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  cos_lr: true
  warmup_epochs: 3
  patience: 0          # no early stopping; every variant gets the same epochs
  save_period: 1
  cache: ram           # MI300X Droplets have lots of RAM; Colab: false
  workers: 8           # per process; with 4 concurrent runs keep total ≤ vCPU count
  amp: true
  deterministic: false
  plots: true
modules:
  der: false
  enhancenet: false
  insert_at: [4, 6, 9]
  combine: sequential
  der_paths: 4
  severity_cond: false
  entropy_reg: 0.0
  load_balance: 0.0
```
Each variant file only overrides `modules.*` (and `name`). `config.py` deep-merges `base.yaml` with the variant and validates the result with a dataclass. Unknown keys are an error.

### 9.2 `trainer.py` and `tracking.py`
- `class YoloDERTrainer(DetectionTrainer)`:
  - `get_model(cfg=None, weights=None, verbose=True)` returns `YoloDERModel`, using the load-then-inject / inject-then-load rule (7.4). **Verify the signature** in the installed `ultralytics/models/yolo/detect/train.py`.
  - If extension losses are on, extend `self.loss_names`.
  - **Optimizer groups:** Ultralytics' `build_optimizer` applies weight decay to all non-bias, non-BN weights, which would decay our `λ` scalars toward 0. Override it (or post-process `self.optimizer.param_groups`) so `λ_der`, `λ_enh`, and every gate/head bias get `weight_decay=0`. Verify by printing the groups once.
- **Disable Ultralytics' built-in MLflow callback** (`from ultralytics import settings; settings.update({"mlflow": False})`). Otherwise it creates a duplicate run.
- `tracking.py` registers our own callbacks via `trainer.add_callback`:
  - `on_pretrain_routine_end`: start a run, or reattach to the stored run ID on resume. Set the run name to `f"{variant}-s{seed}"`. Log the flattened config as params, and tags:
    - `variant`, `seed`, `git_sha`, `data_hash` (from `dvc.lock`);
    - `torch`, `hip` (`torch.version.hip`), `gpu` (`torch.cuda.get_device_name()`), `ultralytics`;
    - `host` (`colab-t4` / `do-mi300x`), and parameter counts (base vs added).
  - `on_fit_epoch_end`: log `trainer.metrics` and the losses with `step=epoch`, the learning rates, and epoch time. **Sanitise metric names**: MLflow rejects `(` and `)`, so `metrics/mAP50(B)` becomes `metrics/mAP50B`.
  - `on_model_save`: upload `last.pt` and `mlflow_run_id.txt` to `s3://$SPACES_BUCKET/checkpoints/<variant>-s<seed>/` (overwrite). This makes resume possible on a new Droplet.
  - `on_train_end`: log `best.pt`, `results.csv`, and all plots as artifacts. Register `best.pt` as a version of the registered model `yolo-der-exdark` with tags `variant` and `seed`.
- `train.py` CLI: `--config`, `--seed`, `--resume s3://...|path`, `--fraction` (data subset for smoke tests), `--epochs-override` (refuses to run when resuming). It writes everything under `/work/runs/<variant>-s<seed>` (Colab: `/content/drive/MyDrive/yolo-der/runs/...`, from env `RUNS_DIR`).

### 9.3 Run plan and GPU-hour budget
**Measure first.** The smoke test (Phase 4) records seconds per epoch for 1 process and for 4 concurrent processes on one MI300X. Recompute this table from those numbers before launching Phase 5, and show it to the owner.

Planning assumptions: 100 epochs of YOLOv8n at 448 px on ~4.5k training images ≈ 0.5–1.5 h per run alone. 4 concurrent runs on one MI300X are likely CPU/dataloader-bound, so wall time grows less than 4×.

| Batch | Runs (concurrent on 1 GPU) | Est. wall hours | Est. cost |
|---|---|---|---|
| Smoke (Phase 4) | baseline, 3 epochs, 1 & 4 processes | 1 | $2.59 |
| A | baseline, der, enh, der_enh, seed 0 | 4–6 | $10–16 |
| B | the same four, seed 1 | 4–6 | $10–16 |
| C | der_enh_entreg, der_enh_sevcond, der_routes_enh, seed 0 | 4–6 | $10–16 |
| D (optional) | extensions, seed 1, or E4 | 4–6 | $10–16 |
| Reserve | reruns/failures | ≥ 4 | ≥ $10 |

Each batch also runs its evaluation before self-destroying. **Hard ceiling: ~28 GPU-hours total.** Track the running total in `reports/gpu_ledger.md`: date, Droplet ID, hours, cost, purpose.

### 9.4 Checkpoint and resume rules
- Ultralytics saves `last.pt` every epoch (`save_period: 1`). `tracking.py` also mirrors it to Spaces.
- **Same machine:** `python -m yolo_der.train --resume /work/runs/<run>/weights/last.pt`.
- **New Droplet:** `--resume s3://.../last.pt`. The CLI downloads it to the **same absolute path** recorded inside the checkpoint (`/work/runs/<run>/weights/last.pt`), downloads `mlflow_run_id.txt`, then resumes so MLflow continues the same run.
- Resume restores every training arg from the checkpoint, including `epochs`. To train longer you must start a new run from `best.pt` as weights. The CLI enforces this.
- On Colab the same logic applies with Drive paths (this matches the existing starter notebook's `run_state.json` pattern; port it into `tracking.py`).

### 9.5 What "done" looks like for a run
MLflow shows:
- metrics for every epoch;
- the tags `hip` (non-empty on MI300X) and `gpu` ("AMD Instinct MI300X");
- `best.pt` registered in the model registry;
- test metrics from `evaluate.py`.

`checkpoints/<run>/last.pt` exists in Spaces, and the Droplet is gone.

---

## 10. Evaluation and research extensions

### 10.1 `evaluate.py` (core metrics)
For each finished run (`--run-id` or `--registered-version`):
1. `model.val(data=..., split="test", imgsz=448, batch=32, conf=0.001, iou=0.7)`. Record:
   - mAP50 and mAP50-95;
   - precision and recall;
   - per-class AP50.
2. Per dark bin: the same evaluation on `data_test_q{1..4}.yaml`, giving mAP50 per brightness quartile. **This is the key table:** if the modules help, the gain should be largest in q1 (darkest).
3. Cost:
   - total params and added params (absolute and %; the paper claims < 10%);
   - GFLOPs at 448 (`thop` via Ultralytics' `model.info()`);
   - latency in ms/img at batch 1 on the training GPU: 50 warm-up and 200 timed iterations, `torch.cuda.synchronize()`, fp16.
4. Log everything to the run in MLflow under the `test/` prefix, and write `reports/eval/<variant>-s<seed>.json`.

`python -m yolo_der.evaluate --aggregate` then builds:
- `reports/ablation.csv` and `reports/ablation.md`: one row per variant, mean and min–max over seeds for each metric, plus Δ vs baseline;
- `reports/dark_bins.png`: grouped bars of mAP50 by quartile per variant.

**Honesty rule:** with 2 seeds, a Δ smaller than the seed spread is reported as "no clear difference". Never cherry-pick the better seed.

### 10.2 E1 — Is it really "entropy-aware"? (paper Sec. 3.4 gap)
Measurement (`analysis/entropy.py`, no training needed; runs on any finished model):
- **Routing entropy:** H(ô) and H(β) per image per insertion point, normalised by log N to [0, 1].
- **Feature entropy:** for feature maps before and after each module, compute per-channel histogram entropy (64 bins over the channel's [p1, p99] range), averaged over channels. Compare the mean change (after − before) by dark bin.
- Questions to answer in the write-up:
  - Do the modules change feature entropy at all?
  - More on darker images?
  - Is routing entropy low (decisive) or near 1 (the gate is just averaging)?

Intervention (training, batch C): `der_enh_entreg` uses `entropy_reg: 0.01` and `load_balance: 0.01`. Compare against `der_enh` on mAP, on the dark-bin gain, and on routing entropy.

### 10.3 E2 — Does routing follow image severity?
`analysis/record_gates.py`:
- Load a model, set `record=True` on all modules, and register forward hooks.
- Run the test set at batch 1, without augmentation.
- Save `reports/gates/<variant>-s<seed>.parquet` with columns: `image`, `layer`, `module`, `path_or_branch`, `weight`, plus the mean predicted γ / α / κ / s.
- Join with `data/severity.csv`.

`analysis/plots.py` produces:
- scatter plots of s_dark vs each routing weight, per layer, with a LOWESS line;
- Spearman ρ with 95% bootstrap CIs (1,000 resamples) per (layer, path), as a heatmap;
- mean gate weights per dark bin (stacked bars);
- **hypothesis test:** the tone branch's mean γ should be < 1 (brightening) more often on darker images. Report the Spearman ρ between s_dark and mean γ.

These figures are the most original part of the write-up. Keep them clean and labelled.

### 10.4 E3 — Severity-conditioned gates
Config `der_enh_sevcond` (`severity_cond: true`) passes in-model s_dark (`cond_dim=1`) to every DER and EnhanceNet MLP (7.5). Compare with `der_enh`:
- overall mAP;
- q1 (darkest) mAP;
- whether routing correlates more strongly with s_dark (it should, trivially; the interesting question is whether that helps mAP).

### 10.5 E4 — Cross-dataset generalisation (optional)
Only if ≥ 6 GPU-hours remain after batch C, and the owner agrees.
- Evaluate the ExDark-trained baseline and `der_enh` zero-shot on another low-light detection set with overlapping classes (e.g. person, car, bicycle, motorbike, dog).
- The owner picks the dataset. Claude Code checks the licence, writes the class mapping, and evaluates only on the shared classes.
- This needs no training, so it can run on the hub CPU via ONNX, or at the end of a GPU batch.

### 10.6 E5 — Multi-GPU DDP (only if the owner explicitly asks)
An 8× MI300X Droplet costs $20.72/hr. Ultralytics supports `device=0,1,...,7` (DDP).

The custom trainer must stay picklable, because Ultralytics DDP re-launches a generated script that re-imports the trainer. Verify that `YoloDERTrainer` works under DDP with a 2-epoch run before any real use.

This is out of scope by default.

---

## 11. Export and serving

### 11.1 `export.py`
1. Load `best.pt` of the chosen run, by default the best `der_enh` variant by test mAP50-95. Set the registered model alias `champion` on it in MLflow.
2. Turn off recording on all modules. Then run `model.export(format="onnx", imgsz=448, opset=17, simplify=True, dynamic=False, half=False)`.
3. **Parity check:** onnxruntime (CPU) vs PyTorch on 20 test images. Assert the max absolute difference of the raw output is < 1e-3 and the post-NMS detections match (IoU > 0.99 for paired boxes).
4. Expected output shape: `(1, 4 + nc, 4116)`, where 4116 = 56² + 28² + 14² anchors at 448 px with strides 8/16/32, and the layout is `[cx, cy, w, h, class scores…]` with **no objectness**. Verify against the exported model's output metadata.
5. Write `labels.json` (from `data.yaml` names, in order) and `model_meta.json` (imgsz, nc, variant, MLflow run ID, mAP, export date, git sha).
6. Upload to `s3://$SPACES_BUCKET/onnx/<version>/{model.onnx,labels.json,model_meta.json}` and write `onnx/champion.json` = `{"version": "<version>"}`. Also log these files as MLflow artifacts.
7. Optional stretch: dynamic INT8 quantisation (`onnxruntime.quantization.quantize_dynamic`). Report the size, the CPU latency, and the mAP change.

### 11.2 Node.js API (`api/`, TypeScript, Node 22 LTS)
Dependencies:
- `express`, `multer` (memory storage, 10 MB limit, jpeg/png/webp only);
- `onnxruntime-node`, `sharp`, `zod` (query validation), `pino` (JSON logs);
- `helmet`, `express-rate-limit` (30 req/min/IP on `/api/infer`);
- `@aws-sdk/client-s3` (model download);
- `prom-client` (optional `/metrics`).

Dev dependencies: `typescript`, `tsx`, `vitest`, `eslint`.

Startup:
1. Read `onnx/champion.json` from Spaces.
2. Download the model, `labels.json`, and `model_meta.json` to `/tmp/model`.
3. Create an `InferenceSession` with `graphOptimizationLevel: "all"` and `intraOpNumThreads` = CPU count.
4. Run one warm-up inference.
5. If the download fails, fall back to a model baked into the image at `/app/model` (CI copies it in at build time if available).

Endpoints:

| Method | Path | Behaviour |
|---|---|---|
| GET | `/api/health` | `{status, model_version, variant, uptime_s}` |
| POST | `/api/infer?conf=0.25&iou=0.45` | multipart field `image` → `{detections: [{class_id, class_name, score, box: [x1, y1, x2, y2]}], severity: {mu_Y, s_dark}, image: {width, height}, timing_ms: {pre, infer, post}, model_version}` |
| GET | `/api/runs` | proxies MLflow `POST /api/2.0/mlflow/runs/search` (experiment `yolo-der-exdark`) → slim list `{run_id, variant, seed, status, test_map50, test_map50_95, q1_map50}`; cached 60 s |
| GET | `/api/ablation` | `reports/ablation.json` from Spaces (uploaded by `evaluate --aggregate`) |
| GET | `/api/figures/:name` | streams whitelisted PNGs from `s3://…/reports/` |
| GET | `/` | the static dashboard |

`preprocess.ts`:
1. `sharp(buf).rotate()` (applies EXIF orientation), then `.removeAlpha().raw()` for RGB.
2. Letterbox to 448×448: keep the aspect ratio, pad with (114, 114, 114), centre the image, and keep `scale`, `padX`, `padY`.
3. Convert HWC uint8 to a CHW Float32Array divided by 255. The tensor is `[1, 3, 448, 448]`.
4. Compute `mu_Y` and `s_dark` from the original pixels (same formula as Python).

`postprocess.ts`:
1. The output is `[1, 4+nc, 4116]`; iterate over anchors. For each, take the best class score; keep it if ≥ conf.
2. Convert cxcywh → xyxy.
3. Apply class-wise greedy NMS at `iou`, keeping at most 100 detections.
4. Undo the letterbox: subtract the padding, divide by the scale, and clip to the image size.
5. Write NMS as a pure function with unit tests.

Tests (vitest):
- NMS on hand-made boxes;
- letterbox maths for both portrait and landscape images;
- a **golden test**: 3 fixture images whose expected detections are produced by Python (`export.py --write-golden api/test/fixtures/`). Boxes must match within 2 px, scores within 0.01.

### 11.3 Dashboard (`dashboard/`)
Plain HTML, CSS, and JS, with no framework and no build step. The API serves it as static files. Three tabs:
1. **Try it:** upload an image or pick one of 6 bundled dark sample images (from the test split; CC BY 4.0 credit shown). Draw the boxes on a canvas, and show the class/score list, s_dark as a gauge, and latency.
2. **Ablation:** the table from `/api/ablation`, the dark-bin chart, and a "trained on AMD Instinct MI300X" badge using the `gpu` tag from MLflow.
3. **Routing analysis:** the E1/E2 figures, with a 2–3 sentence caption each, written by the owner.

### 11.4 `docker/api.Dockerfile`
- Multi-stage: build on `node:22-bookworm-slim` (`npm ci`, `tsc`), runtime on `node:22-bookworm-slim` with production dependencies only.
- Don't use Alpine: `sharp` and `onnxruntime-node` ship glibc binaries.
- Run as non-root `node`, with `HEALTHCHECK` on `/api/health`.
- Registry: **GHCR** (`ghcr.io/<owner>/yolo-der-api`) by default. DOCR Starter (500 MiB, free) is an alternative only if the image fits; check `docker image ls`, since onnxruntime-node is large. The ROCm training image is never pushed anywhere; it is built on the GPU Droplet.

---

## 12. CI/CD (GitHub Actions)

`ci.yml`, on push and pull request:
- **python:** ubuntu-latest, Python 3.11.
  - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then `pip install -e ".[dev]"`.
  - `ruff check`, `ruff format --check`, `pytest -m "not slow"`.
  - Cache pip.
- **node:** `npm ci`, `npm run lint`, `npm test`, `npm run build` in `api/`.
- **docker-api:** build `docker/api.Dockerfile`, no push.

`deploy-api.yml`, on push to `main` touching `api/**`, `dashboard/**`, `docker/api.Dockerfile`, or on manual dispatch:
1. Build and push `ghcr.io/<owner>/yolo-der-api:<sha>` and `:latest` (`permissions: packages: write`).
2. SSH to the hub (secrets `HUB_HOST`, `HUB_SSH_KEY`) and run `cd /opt/hub && API_IMAGE=...:<sha> docker compose pull api && docker compose up -d api`.
3. Curl `https://$SITE_ADDRESS/api/health` and fail the job if it is not 200 within 60 s.

`reaper.yml`: `schedule: cron "*/30 * * * *"` plus `workflow_dispatch`. It installs doctl (`digitalocean/action-doctl@v2`) and runs `infra/reaper.sh`. Two caveats:
- scheduled workflows run only on the default branch and can be delayed by several minutes;
- GitHub disables schedules on repos with 60 days of inactivity (irrelevant here).

GPU launches stay **manual from the laptop**: no CI workflow creates GPU Droplets.

Secrets:
- `DIGITALOCEAN_ACCESS_TOKEN`
- `HUB_HOST`, `HUB_SSH_KEY`
- `SPACES_KEY`, `SPACES_SECRET` (only if CI uploads anything)
- `GITHUB_TOKEN` (built in; used for GHCR)

---

## 13. Phased task list

Work strictly in order. At the end of every phase:
1. Run the acceptance checks.
2. Commit.
3. Run `infra/status.sh` (from Phase 2 onward).
4. Post a short summary to the owner: what was built, what to look at, what was learned, the next phase.

**⛔ CONFIRM** means: stop, show the plan and cost, and wait for an explicit "yes".

### Phase 0 — Repo bootstrap (Day 1)
- `git init`. Create the Section 5 tree, `.gitignore` (rule 0.3), `.env.example`, `pyproject.toml` with extras `[dev]` (pytest, ruff, pre-commit) and `[train]` (ultralytics 8.3.*, mlflow 3.*, dvc[s3], boto3, opencv-python-headless, pandas, pyarrow, scipy, matplotlib, roboflow), and pre-commit (ruff, end-of-file, detect-private-key).
- `Makefile` targets:
  - `setup`, `lint`, `test`
  - `data` (`dvc repro`)
  - `tunnel` (SSH tunnel to MLflow)
  - `smoke` (1-epoch CPU/T4 run with `--fraction 0.02`)
  - `launch` (`infra/launch_gpu.sh`), `reap`, `status`
  - `export`, `api-dev`
- Place this spec at `docs/PROJECT_SPEC.md`, and a short `CLAUDE.md` that points to it and repeats Section 0.
- ✅ `make lint test` passes (with placeholder tests). `git log` shows the first commit with no secrets. The owner creates the GitHub repo and pushes.

### Phase 1 — Data and DVC (Day 1)
- Implement `download.py`, `prepare.py`, `severity.py`, `params.yaml`, `dvc.yaml`. `dvc init`, with a local remote first.
- ✅ `dvc repro` completes. `reports/data_stats.json` shows 12 classes and all three splits. The severity histogram looks sane (most ExDark images dark). `dvc status` is clean. The data stats and the severity histogram are shown to the owner.

### Phase 2 — Cloud base (Day 1)
- Write the owner's manual checklist for 8.1 (credit team, tokens, SSH key, spend alert, payment method) and wait for them to finish it.
- Script `infra/setup_base.sh` (idempotent, doctl). It creates the VPC, both firewalls, and the hub Droplet, and documents bucket creation.
- Deploy `infra/hub/` (MLflow + Caddy; the api service is commented out until Phase 7). Run `backup_mlflow.sh` via cron.
- Add the Spaces DVC remote and `dvc push`.
- ✅ Through the tunnel, `mlflow.set_experiment(...)` and logging a dummy metric plus a 1 KB artifact works, and the artifact appears in Spaces under `mlflow/`. `dvc pull` into a fresh clone works. The cost so far is ≈ $0 GPU.
- 2b (optional, only if ahead of schedule): Terraform (8.6).

### Phase 3 — Modules, trainer, tests, Colab validation (Day 2)
- Implement Sections 7.1–7.7, `trainer.py`, `tracking.py`, `train.py`, the configs, and the `notebooks/colab_dev.ipynb` driver. The notebook clones the repo, installs `.[train]`, sets up the tunnel, and runs `dvc pull` and `make smoke`.
- Ask the owner once about the DER/EnhanceNet combination reading (7.3).
- The owner reads `der.py` and `enhancenet.py` and asks questions before moving on. This is the main learning step. Walk through each equation ↔ line of code.
- ✅ All unit tests pass in CI.
- ✅ On Colab T4 or CPU, each of the 4 core variants runs 2 epochs at `--fraction 0.1` and logs to the hub's MLflow with correct tags.
- ✅ Added params are < 10% of YOLOv8n (print them).
- ✅ Kill test: interrupt a run at epoch 2, resume from the Drive `last.pt`, and confirm the same MLflow run continues at epoch 3.
- ✅ The checkpoint round-trip test passes with "Transferred N/N".

### Phase 4 — GPU smoke test (Day 3) ⛔ CONFIRM
- Implement `launch_gpu.sh`, `cloud-init.yaml.tmpl`, `bootstrap.sh`, `run_job.sh`, `reaper.sh`, `status.sh`, `reaper.yml`, and `train.Dockerfile`. Dry-run first: `launch_gpu.sh --dry-run` prints the rendered cloud-init with secrets masked.
- Test the reaper safely: create a **cheap CPU Droplet** tagged `ephemeral-gpu`, run `reaper.sh` with max age 0, and confirm it is destroyed.
- ⛔ CONFIRM, then launch: baseline, 3 epochs, `MAX_RUNTIME_HOURS=1`. In the same session, also run 4 concurrent 1-epoch processes to measure throughput.
- ✅ The log shows `rocm-smi` with an MI300X, and the Docker build assertion printed a HIP version.
- ✅ MLflow tag `gpu` = AMD Instinct MI300X.
- ✅ The Droplet destroyed itself; `doctl compute droplet list --tag-name ephemeral-gpu` is empty.
- ✅ The log is in `logs/`.
- ✅ The GPU ledger is updated, and the 9.3 table is recomputed with measured s/epoch.

### Phase 5 — Experiments (Days 4–6) ⛔ CONFIRM for each batch
- Batch A, then B, then C (then D only if budget allows), per 9.3. Before each: show the updated ledger and remaining credit.
- While a batch runs, the owner works on analysis code (Phase 6) locally against finished runs.
- If a run fails: read the log from Spaces, fix it, write a test reproducing the bug if possible, and resume from the S3 checkpoint. Don't restart from scratch unless the checkpoint is corrupt.
- ✅ Every planned run has test metrics in MLflow. The total GPU spend is ≤ ~$75. No GPU Droplets remain.

### Phase 6 — Evaluation and analysis (Days 5–7, overlaps Phase 5)
- `evaluate --aggregate`, `record_gates`, `entropy`, `plots`. The gate recording can run on CPU (hub or laptop) because it is inference only on ~650 test images with YOLOv8n.
- ✅ `reports/ablation.md`, `reports/dark_bins.png`, the routing-correlation heatmap, the γ vs s_dark plot, and an entropy table exist and are committed. The owner writes the captions.

### Phase 7 — Export, API, dashboard, deploy (Day 7)
- `export.py` with the parity check, the Node API, the dashboard, `api.Dockerfile`, and `deploy-api.yml`. Enable the api service on the hub.
- ✅ The golden tests pass. `https://<site>/api/health` is 200. Uploading a dark sample in the dashboard shows boxes. `/api/runs` lists the experiments. CPU latency on the hub is shown on the page (expect tens to low hundreds of ms on 1 vCPU).

### Phase 8 — Write-up and teardown (Day 8, by 18 Oct) ⛔ CONFIRM before deleting anything
- README:
  - the problem and the paper;
  - our deviations (YOLOv8n, no `L_enh`, DCP for s_fog, the DER path interpretation, batch size, epochs);
  - architecture diagram, ablation table, dark-bin figure, routing findings;
  - how to reproduce, costs (from the ledger);
  - credits and licences.
- `reports/writeup.md`: a 2–3 page hackathon write-up. Help the owner prepare a 2-minute demo video script (they record it).
- Back up: `backup_mlflow.sh`, then `aws s3 sync s3://$SPACES_BUCKET ./final_backup --endpoint-url ...` to the laptop (excluding `dvc/` if large), and an MLflow export of the registered model.
- **Decide with the owner:** the credit expires 19 Oct, but judging may come later. Options:
  - (a) keep the hub running on their own card (~$12/month);
  - (b) move the API to a free host that runs a Docker container with the ONNX model baked in;
  - (c) record the demo and tear everything down.
- ⛔ CONFIRM, then tear down in order:
  1. any GPU Droplets (there should be none);
  2. the hub Droplet;
  3. firewalls;
  4. the VPC;
  5. the registry repo;
  6. empty and delete the bucket;
  7. check whether Spaces must also be cancelled in settings to stop the subscription (verify);
  8. revoke the DO tokens and Spaces keys.
- ✅ `infra/status.sh` shows no resources. The Billing page shows the final usage. Screenshot it for the write-up ("trained on AMD for $X").

---

## 14. Known gotchas (read before debugging)

1. **MLflow file store is deprecated.** `mlruns/` file tracking raises `MlflowException` in MLflow 3.x. Use `sqlite:///...` locally/Colab and the hub server everywhere else.
2. **Ultralytics resume restores `epochs` and every other arg from the checkpoint.** Choose `epochs` before the first launch. Resume also needs the checkpoint at the same absolute path, and the `data` yaml at the path recorded in it.
3. **Injected modules rename state-dict keys.** Load pretrained weights before injecting, and load our own checkpoints after injecting (7.4). A silent "Transferred 300/355 items" means the modules started from scratch.
4. **Pickled checkpoints need `yolo_der` importable**, on Colab, the GPU container, the export machine, and in CI.
5. **pip can silently replace ROCm torch** with a CUDA/CPU wheel. The Dockerfile assertion on `torch.version.hip` catches it. On ROCm, `torch.cuda.*` APIs work and the `device="cuda"`/`0` strings are correct (HIP is exposed through the CUDA API).
6. **Ultralytics' built-in MLflow callback** auto-enables when mlflow is installed and creates duplicate runs. Disable it in settings.
7. **MLflow metric names** can't contain `(` or `)`, so sanitise Ultralytics keys.
8. **AMP + custom math:** keep DER/EnhanceNet in fp32 internally (7). Ultralytics' AMP check downloads a small model on first run; that needs internet on the Droplet, which it has.
9. **GPU Droplets bill while powered off.** Only destroy stops billing. The CLI has `droplet-action power-off`; never use it for this project.
10. **The ROCm container image is ~20+ GB.** First pull takes minutes of GPU time; that is accepted. Don't push it to DOCR (500 MiB Starter limit).
11. **Spaces needs the endpoint everywhere:** DVC (`endpointurl`), boto3/MLflow (`MLFLOW_S3_ENDPOINT_URL`), aws-cli (`--endpoint-url`). Also set a dummy `AWS_DEFAULT_REGION`.
12. **Same-region Spaces ↔ Droplet traffic is free.** Cross-region counts toward the 1 TiB allowance (fine).
13. **Concurrent runs on one GPU contend for CPU, not GPU.** Split `workers` so the total ≤ vCPUs. With `cache: ram`, each process holds its own copy (~a few GB for ExDark at 448; fine).
14. **Ultralytics' dataloader** may need a larger `--shm-size` in Docker (we set 32 GB). If you see "bus error" or dataloader crashes, that's why.
15. **Colab:** sessions die without warning, so mirror `last.pt` to Drive every epoch. The SSH tunnel dies with the session; restart it in the first cell.
16. **DCP s_fog on night images is unreliable:** dark regions look like haze to the dark channel. Report it, don't "fix" it.
17. **Credits expire 19 Oct 2026.** Usage after that, or beyond $100, bills the payment method.
18. **Class order:** always from `data.yaml`. The Node API reads `labels.json`; never type the class list by hand.
19. **ONNX export:** no Python branching on tensor values, no in-place ops on inputs, and recording off. If export fails on `logit`, use `log(p) − log1p(−p)`.
20. **`doctl` context:** make sure the active context is the AMD/credit account (`doctl auth list`) before every create.

---

## 15. Glossary (for the owner)

- **Backbone / neck / head:** feature extractor / multi-scale feature fusion / box + class predictor. Our modules live in the backbone.
- **GAP:** global average pooling. Averages each channel over H×W, giving one number per channel.
- **Soft routing:** all paths run, and their outputs are mixed by softmax weights. (Hard routing would pick one.)
- **ReZero / identity init:** a residual branch scaled by a learnable λ starting at 0. The new module starts as a no-op, so pretrained behaviour is preserved and it learns to contribute gradually.
- **mAP50 / mAP50-95:** mean average precision at IoU 0.5 / averaged over IoU 0.5–0.95. The second is stricter about box quality.
- **Ablation:** adding or removing one component at a time to measure its contribution.
- **Seed spread:** the variation between runs that differ only in the random seed. Any claimed improvement must be larger than this.
- **Dark Channel Prior (DCP):** in haze-free outdoor images, most patches have some pixel near 0 in at least one colour channel. Haze lifts this "dark channel", which lets you estimate transmission.
- **Spearman ρ:** rank correlation (−1 to 1). Robust to non-linear but monotonic relationships.
- **ROCm / HIP:** AMD's GPU compute stack / its CUDA-like programming interface. PyTorch on ROCm reuses the `torch.cuda` API.
- **Cloud-init / user-data:** a script DigitalOcean runs on a Droplet's first boot. It is how an ephemeral machine configures itself.
- **VPC:** a private network between your Droplets. Traffic on it isn't exposed to the internet.
- **DVC:** Git for data. Git stores small `.dvc`/`dvc.lock` pointer files, and the data itself lives in a remote (Spaces).
- **MLflow tracking / registry:** a database of runs (params, metrics, artifacts) / named, versioned models with aliases like `champion`.
- **ONNX:** a framework-neutral model format. onnxruntime runs it on CPU/GPU from Python, Node, C++, etc.
- **Letterbox:** resizing with a preserved aspect ratio plus padding to a square input. Boxes must be un-letterboxed afterwards.
- **NMS:** non-maximum suppression. Removes duplicate overlapping boxes, keeping the highest-scoring one.
- **Reaper:** a scheduled job that deletes resources that outlived their purpose. Here, a safety net against a forgotten GPU.

---

*End of spec. First action for Claude Code: read Sections 0–4 again, then start Phase 0 and post a short plan for Phases 0–2 to the owner.*
