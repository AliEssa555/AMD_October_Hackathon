"""Test-set evaluation, per-bin mAP, cost metrics, and the ablation table (spec 10.1).

Two modes:

    python -m yolo_der.evaluate --run runs/der_enh-s0            # one run
    python -m yolo_der.evaluate --aggregate --reports reports/eval

The per-bin table is the point. Overall mAP tells you whether a variant is better on
average; the bins tell you *where*, which is the paper's actual claim -- that routing and
enhancement help most when the image is degraded. ExDark bins by darkness quartile
(q1 darkest), BDD100K by ground-truth `timeofday` / `weather`.

**Honesty rule (spec 10.1).** With one seed there is no variance estimate, so a delta
smaller than the spread we measured across the ExDark ablation is reported as "no clear
difference", not as an improvement. :data:`NOISE_FLOOR_MAP50` holds that spread and where
it came from. Never report a bin with too few images as if it were a measurement.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Observed spread of best-epoch mAP50 across the seven ExDark variants on one seed
#: (0.5794-0.5850). Every variant used identical data and hyper-parameters bar the
#: modules, so this is a direct estimate of run-to-run noise: any delta below it is not
#: evidence. Measured 18 Oct 2026; see CLAUDE.md.
NOISE_FLOOR_MAP50 = 0.0056

#: A bin with fewer images than this is reported but flagged; BDD's `wx_foggy` has 13
#: test images, which is an anecdote rather than a measurement (Addendum A.6.2).
MIN_BIN_IMAGES = 100

LATENCY_WARMUP = 50
LATENCY_ITERS = 200


@dataclass
class RunResult:
    """Everything measured for one finished run."""

    variant: str
    seed: int
    weights: str
    overall: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, float] = field(default_factory=dict)
    bins: dict[str, dict[str, float]] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialise for ``reports/eval/<variant>-s<seed>.json``."""
        return {
            "variant": self.variant,
            "seed": self.seed,
            "weights": self.weights,
            "overall": self.overall,
            "per_class": self.per_class,
            "bins": self.bins,
            "cost": self.cost,
            "environment": self.environment,
        }


def materialise(src: Path, dest_dir: Path) -> Path:
    """Copy a dataset yaml with ``path`` pinned to an absolute directory.

    Ultralytics resolves a relative ``path:`` against the current working directory and
    then against ``~/datasets``, so a yaml saying ``path: data/prepared`` only works when
    run from the repo root. Pinning it absolute makes evaluation cwd-independent, exactly
    as ``train.py`` does for training.

    The subtlety is what a *relative* ``path`` is relative to. It is the repo root, not
    the yaml's own directory -- which matters for the bin yamls, since
    ``data/bins/data_test_q1.yaml`` says ``path: data/prepared`` and then
    ``test: ../bins/test_bins/q1.txt``. Joining against the yaml's directory instead
    produced ``data/prepared/data/prepared`` and a "images not found" failure.
    """
    from yolo_der.data.paths import repo_root

    doc = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    declared = doc.get("path")
    if declared is None:
        # No `path:` at all -- prepare.py omits it deliberately -- so the yaml's own
        # directory is the dataset root.
        base = src.parent
    else:
        base = Path(str(declared))
        if not base.is_absolute():
            base = repo_root() / base
    doc["path"] = str(base.resolve())
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    dest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return dest


def discover_bins(data_yaml: Path) -> dict[str, Path]:
    """Find the per-condition dataset yamls that sit alongside ``data_yaml``.

    ExDark writes them to ``data/bins/data_test_q*.yaml``; the BDD converter writes
    ``data/bdd100k/bins/data_<group>.yaml``. Both are searched, so neither dataset needs
    a special case at the call site.
    """
    found: dict[str, Path] = {}
    candidates = [data_yaml.parent / "bins", data_yaml.parent.parent / "bins"]
    for d in candidates:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("data_*.yaml")):
            name = p.stem.removeprefix("data_")
            found.setdefault(name, p)
    return found


def bin_image_count(bin_yaml: Path) -> int:
    """Number of images a bin actually contains, for the low-confidence flag."""
    from yolo_der.data.paths import repo_root

    doc = yaml.safe_load(bin_yaml.read_text(encoding="utf-8")) or {}
    declared = doc.get("path")
    if declared is None:
        base = bin_yaml.parent
    else:
        base = Path(str(declared))
        if not base.is_absolute():
            base = repo_root() / base
    target = doc.get("test") or doc.get("val")
    if not target:
        return 0
    listing = (base / str(target)).resolve()
    if listing.is_file():
        return sum(1 for line in listing.read_text(encoding="utf-8").splitlines() if line.strip())
    if listing.is_dir():
        return sum(1 for _ in listing.iterdir())
    return 0


def _metrics_of(results: Any) -> dict[str, float]:
    """Pull the four headline numbers out of an Ultralytics ``DetMetrics``."""
    box = results.box
    return {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }


def _per_class_ap50(results: Any, names: dict[int, str]) -> dict[str, float]:
    """AP50 keyed by class name, for the per-class column of the report."""
    out: dict[str, float] = {}
    with contextlib.suppress(Exception):
        for idx, ap in zip(results.box.ap_class_index, results.box.ap50, strict=False):
            out[str(names.get(int(idx), int(idx)))] = float(ap)
    return out


def measure_latency(model: Any, imgsz: int, device: str) -> dict[str, float]:
    """Milliseconds per image at batch 1 (spec 10.1 item 3).

    fp16 on an accelerator, fp32 on CPU. Reported as median and p90 rather than a mean:
    the first few iterations after warm-up are still noisy and a mean hides that.
    """
    import torch

    net = model.model if hasattr(model, "model") else model
    net = net.to(device).eval()
    half = device != "cpu"
    if half:
        net = net.half()
    x = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.half if half else torch.float)

    def sync() -> None:
        if device != "cpu":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            net(x)
        sync()
        samples = []
        for _ in range(LATENCY_ITERS):
            t0 = time.perf_counter()
            net(x)
            sync()
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return {
        "latency_ms_median": statistics.median(samples),
        "latency_ms_p90": samples[int(0.9 * len(samples)) - 1],
        "latency_batch": 1,
        "latency_half": float(half),
    }


def cost_metrics(model: Any, imgsz: int, device: str) -> dict[str, float]:
    """Params, added params, GFLOPs and latency."""
    from yolo_der.models.inject import added_parameter_report

    net = model.model if hasattr(model, "model") else model
    out: dict[str, float] = {}

    with contextlib.suppress(Exception):
        report = added_parameter_report(net)
        out.update(
            {
                "params_total": float(report["params_base"] + report["params_added"]),
                "params_base": float(report["params_base"]),
                "params_added": float(report["params_added"]),
                "params_added_pct": float(report["params_added_pct"]),
            }
        )
    if "params_total" not in out:
        out["params_total"] = float(sum(p.numel() for p in net.parameters()))
        out["params_added"] = 0.0
        out["params_added_pct"] = 0.0

    with contextlib.suppress(Exception):
        from ultralytics.utils.torch_utils import get_flops

        out["gflops"] = float(get_flops(net, imgsz=imgsz))

    with contextlib.suppress(Exception):
        out.update(measure_latency(model, imgsz, device))
    return out


def evaluate_run(
    run_dir: Path,
    *,
    data: Path | None = None,
    imgsz: int = 448,
    batch: int = 32,
    device: str = "cpu",
    with_bins: bool = True,
    scratch: Path | None = None,
) -> RunResult:
    """Evaluate one finished run on the test split and on every bin."""
    import torch
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"no best.pt in {run_dir}")

    variant, _, seed_part = run_dir.name.rpartition("-s")
    seed = int(seed_part) if seed_part.isdigit() else 0

    # The dataset the run was trained on, unless overridden. args.yaml is written by
    # Ultralytics into the run directory, so a run carries its own provenance.
    if data is None:
        args_file = run_dir / "args.yaml"
        if not args_file.is_file():
            raise FileNotFoundError(f"pass --data: no args.yaml in {run_dir}")
        data = Path(str((yaml.safe_load(args_file.read_text(encoding="utf-8")) or {})["data"]))
    data = data.resolve()

    scratch = scratch or run_dir / "eval-cfg"
    model = YOLO(str(weights))
    result = RunResult(variant=variant or run_dir.name, seed=seed, weights=str(weights))

    overall = model.val(
        data=str(materialise(data, scratch)),
        split="test",
        imgsz=imgsz,
        batch=batch,
        conf=0.001,
        iou=0.7,
        device=device,
        verbose=False,
        plots=False,
    )
    result.overall = _metrics_of(overall)
    result.per_class = _per_class_ap50(overall, getattr(overall, "names", {}) or {})

    if with_bins:
        for name, bin_yaml in discover_bins(data).items():
            count = bin_image_count(bin_yaml)
            try:
                res = model.val(
                    data=str(materialise(bin_yaml, scratch)),
                    split="test",
                    imgsz=imgsz,
                    batch=batch,
                    conf=0.001,
                    iou=0.7,
                    device=device,
                    verbose=False,
                    plots=False,
                )
            except Exception as exc:  # noqa: BLE001 - one bad bin must not lose the rest
                result.bins[name] = {"error": str(exc)[:200], "images": count}
                continue
            entry = _metrics_of(res)
            entry["images"] = float(count)
            entry["low_confidence"] = float(count < MIN_BIN_IMAGES)
            result.bins[name] = entry

    result.cost = cost_metrics(model, imgsz, device)
    result.environment = {
        "torch": torch.__version__,
        "hip": str(torch.version.hip),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
        "imgsz": str(imgsz),
    }
    return result


def log_to_mlflow(result: RunResult, run_dir: Path) -> str | None:
    """Attach the evaluation to the training run in MLflow, under ``test/``."""
    run_id_file = run_dir / "weights" / "mlflow_run_id.txt"
    if not run_id_file.is_file():
        return None
    try:
        import mlflow

        from yolo_der.tracking import sanitise_key
    except ModuleNotFoundError:
        return None

    run_id = run_id_file.read_text(encoding="utf-8").strip()
    if not run_id:
        return None

    flat: dict[str, float] = {f"test/{k}": v for k, v in result.overall.items()}
    for cls, ap in result.per_class.items():
        flat[f"test/ap50/{cls}"] = ap
    for name, entry in result.bins.items():
        for k, v in entry.items():
            if isinstance(v, (int, float)):
                flat[f"test/bin/{name}/{k}"] = float(v)
    for k, v in result.cost.items():
        flat[f"test/cost/{k}"] = float(v)

    with contextlib.suppress(Exception):
        client = mlflow.tracking.MlflowClient()
        for key, value in flat.items():
            client.log_metric(run_id, sanitise_key(key), value)
        return run_id
    return None


# ----------------------------------------------------------------- aggregation ----


def aggregate(reports: Path, out_dir: Path, baseline: str = "baseline") -> Path:
    """Build ``ablation.csv`` and ``ablation.md`` from the per-run JSON files."""
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(reports.glob("*.json"))]
    if not results:
        raise FileNotFoundError(f"no evaluation JSON in {reports}")

    # Group by variant so multiple seeds collapse into mean and min-max.
    by_variant: dict[str, list[dict]] = {}
    for r in results:
        by_variant.setdefault(r["variant"], []).append(r)

    def agg(vals: list[float]) -> tuple[float, float, float]:
        return statistics.fmean(vals), min(vals), max(vals)

    base_rows = by_variant.get(baseline)
    base_map50 = statistics.fmean([r["overall"]["mAP50"] for r in base_rows]) if base_rows else None

    bin_names: list[str] = []
    for r in results:
        for b in r["bins"]:
            if b not in bin_names:
                bin_names.append(b)
    bin_names.sort()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation.csv"
    md_path = out_dir / "ablation.md"

    header = ["variant", "seeds", "mAP50", "mAP50_min", "mAP50_max", "mAP50-95", "delta_mAP50"]
    header += ["params_added_pct", "gflops", "latency_ms_median"]
    header += [f"bin_{b}" for b in bin_names]

    lines_csv = [",".join(header)]
    rows_md = []
    for variant in sorted(by_variant):
        runs = by_variant[variant]
        m50, lo, hi = agg([r["overall"]["mAP50"] for r in runs])
        m95 = statistics.fmean([r["overall"]["mAP50-95"] for r in runs])
        delta = "" if base_map50 is None else f"{m50 - base_map50:+.4f}"
        cost = runs[0].get("cost", {})
        row = [
            variant,
            str(len(runs)),
            f"{m50:.4f}",
            f"{lo:.4f}",
            f"{hi:.4f}",
            f"{m95:.4f}",
            delta,
            f"{cost.get('params_added_pct', 0.0):.2f}",
            f"{cost.get('gflops', float('nan')):.2f}",
            f"{cost.get('latency_ms_median', float('nan')):.2f}",
        ]
        for b in bin_names:
            vals = [
                r["bins"][b]["mAP50"] for r in runs if b in r["bins"] and "mAP50" in r["bins"][b]
            ]
            row.append(f"{statistics.fmean(vals):.4f}" if vals else "")
        lines_csv.append(",".join(row))
        rows_md.append((variant, len(runs), m50, lo, hi, m95, delta, cost, runs))

    csv_path.write_text("\n".join(lines_csv) + "\n", encoding="utf-8")

    md = [
        "# Ablation",
        "",
        f"Baseline: `{baseline}`. Deltas are mAP50 against it.",
        "",
        f"**Noise floor: {NOISE_FLOOR_MAP50:.4f} mAP50.** That is the spread measured across "
        "all seven ExDark variants on a single seed, with identical data and "
        "hyper-parameters. A delta smaller than it is not evidence of anything, and is "
        "marked *(within noise)* below.",
        "",
        "| variant | seeds | mAP50 | min-max | mAP50-95 | Δ mAP50 | +params % | GFLOPs | ms/img |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, n, m50, lo, hi, m95, delta, cost, _ in rows_md:
        note = ""
        if delta and abs(float(delta)) < NOISE_FLOOR_MAP50 and variant != baseline:
            note = " *(within noise)*"
        md.append(
            f"| `{variant}` | {n} | {m50:.4f} | {lo:.4f}-{hi:.4f} | {m95:.4f} | "
            f"{delta}{note} | {cost.get('params_added_pct', 0.0):.2f} | "
            f"{cost.get('gflops', float('nan')):.2f} | "
            f"{cost.get('latency_ms_median', float('nan')):.2f} |"
        )

    if bin_names:
        md += ["", "## mAP50 by bin", ""]
        md.append("| variant | " + " | ".join(bin_names) + " |")
        md.append("|---" * (len(bin_names) + 1) + "|")
        low_conf: set[str] = set()
        for variant, _n, _m50, _lo, _hi, _m95, _d, _c, runs in rows_md:
            cells = []
            for b in bin_names:
                vals = [
                    r["bins"][b]["mAP50"]
                    for r in runs
                    if b in r["bins"] and "mAP50" in r["bins"][b]
                ]
                if any(r["bins"].get(b, {}).get("low_confidence") for r in runs):
                    low_conf.add(b)
                cells.append(f"{statistics.fmean(vals):.4f}" if vals else "-")
            md.append(f"| `{variant}` | " + " | ".join(cells) + " |")
        if low_conf:
            counts = {}
            for r in results:
                for b in sorted(low_conf):
                    if b in r["bins"]:
                        counts[b] = int(r["bins"][b].get("images", 0))
            detail = ", ".join(f"`{b}` ({counts.get(b, 0)} images)" for b in sorted(low_conf))
            md += [
                "",
                f"**Low-confidence bins** (fewer than {MIN_BIN_IMAGES} test images): {detail}. "
                "These are reported for completeness and should not be used to support a "
                "claim.",
            ]

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return md_path


# ------------------------------------------------------------------------- CLI ----


def build_parser() -> argparse.ArgumentParser:
    """CLI of spec 10.1."""
    p = argparse.ArgumentParser(prog="python -m yolo_der.evaluate", description=__doc__)
    p.add_argument("--run", type=Path, action="append", help="a run directory; repeatable")
    p.add_argument("--runs-dir", type=Path, help="evaluate every */weights/best.pt under here")
    p.add_argument("--data", type=Path, help="override the dataset yaml")
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu", help="cpu, 0, cuda:0 ...")
    p.add_argument("--no-bins", action="store_true", help="skip the per-bin evaluation")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--reports", type=Path, default=Path("reports/eval"))
    p.add_argument("--aggregate", action="store_true", help="build the ablation table and exit")
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--baseline", default="baseline", help="variant to take deltas against")
    return p


def main(argv: list[str] | None = None) -> int:
    """Evaluate runs, or aggregate what has already been evaluated."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.aggregate:
        path = aggregate(args.reports, args.out, baseline=args.baseline)
        print(f"wrote {path} and {path.with_name('ablation.csv')}")
        return 0

    runs: list[Path] = list(args.run or [])
    if args.runs_dir:
        runs += [p.parent.parent for p in sorted(args.runs_dir.glob("*/weights/best.pt"))]
    if not runs:
        parser.error("pass --run, --runs-dir, or --aggregate")

    args.reports.mkdir(parents=True, exist_ok=True)
    failures = 0
    for run_dir in runs:
        print(f"\n=== {run_dir.name} ===")
        try:
            result = evaluate_run(
                run_dir,
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                with_bins=not args.no_bins,
            )
        except Exception as exc:  # noqa: BLE001 - one bad run must not lose the others
            print(f"  FAILED: {exc}")
            failures += 1
            continue

        out = args.reports / f"{result.variant}-s{result.seed}.json"
        out.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
        print(
            f"  mAP50={result.overall['mAP50']:.4f} "
            f"mAP50-95={result.overall['mAP50-95']:.4f} "
            f"bins={len(result.bins)} -> {out}"
        )
        if not args.no_mlflow and log_to_mlflow(result, run_dir):
            print("  logged to MLflow under test/")

    return 1 if failures and failures == len(runs) else 0


if __name__ == "__main__":
    sys.exit(main())
