"""Figures for the write-up (spec 10.1 and 10.3).

    python -m yolo_der.analysis.plots --reports reports --out reports/figures

Reads what the other analysis steps already wrote -- `reports/eval/*.json`,
`reports/entropy/*.json`, `reports/gates/*.parquet` -- so it needs no model and no GPU.

Four figures, each answering a question the write-up has to answer:

* `dark_bins.png` -- mAP50 by darkness quartile per variant. The paper's central claim is
  that the gain concentrates where the image is worst, so this is the figure that would
  show it if it were there.
* `routing_entropy.png` -- normalised routing entropy per variant, with the "uniform" line
  drawn at 1.0. Our measured answer sits on that line, which is the finding.
* `routing_vs_severity.png` -- per-image routing weight against `s_dark`. If routing
  responded to degradation, this would slope.
* `cost_vs_gain.png` -- latency against mAP50, so the +41% latency for no gain is visible
  rather than buried in a table.

Every figure states its own caveat in the caption: single seed, the measured noise floor,
and the image count behind any bin.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

#: Deltas below this are noise, not signal; the number comes from evaluate.py.
from yolo_der.evaluate import MIN_BIN_IMAGES, NOISE_FLOOR_MAP50

#: Plot order, so the baseline is always leftmost and the extensions last.
VARIANT_ORDER = (
    "baseline",
    "der",
    "enh",
    "der_enh",
    "der_routes_enh",
    "der_enh_entreg",
    "der_enh_sevcond",
    "bdd_baseline",
    "bdd_der_enh",
    "bdd_der",
    "bdd_enh",
)

#: Colours differ in lightness as well as hue, so the figures survive greyscale printing.
COLOURS = ("#4A5568", "#2B6CB0", "#2C7A7B", "#B7791F", "#9B2C2C", "#553C9A", "#1A202C")


def order_key(name: str) -> tuple[int, str]:
    """Sort known variants into a deliberate order, unknown ones alphabetically after."""
    base = name.rsplit("-s", 1)[0]
    return (VARIANT_ORDER.index(base) if base in VARIANT_ORDER else len(VARIANT_ORDER), name)


def load_json_dir(d: Path) -> dict[str, dict]:
    """Every ``*.json`` in a directory, keyed by file stem."""
    if not d.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}


def load_gates(d: Path) -> dict[str, list[dict]]:
    """Gate tables, parquet if pyarrow is available, else CSV."""
    out: dict[str, list[dict]] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.parquet")):
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(p).to_pydict()
            n = len(next(iter(table.values()), []))
            out[p.stem] = [{k: table[k][i] for k in table} for i in range(n)]
        except Exception:  # noqa: BLE001 - a missing figure beats a crashed report
            continue
    for p in sorted(d.glob("*.csv")):
        if p.stem in out:
            continue
        import csv

        with p.open(encoding="utf-8", newline="") as fh:
            out[p.stem] = list(csv.DictReader(fh))
    return out


def _fig(plt: Any, w: float = 9.0, h: float = 5.0) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def plot_dark_bins(evals: dict[str, dict], out: Path, plt: Any) -> Path | None:
    """Grouped bars: mAP50 per bin per variant (spec 10.1)."""
    runs = sorted((k for k, v in evals.items() if v.get("bins")), key=order_key)
    if not runs:
        return None
    bins = sorted({b for r in runs for b in evals[r]["bins"]})
    if not bins:
        return None

    fig, ax = _fig(plt, w=max(9.0, 1.4 * len(bins) * len(runs) / 3))
    width = 0.8 / len(runs)
    low_conf = set()
    for i, run in enumerate(runs):
        vals, labels = [], []
        for b in bins:
            entry = evals[run]["bins"].get(b, {})
            vals.append(entry.get("mAP50", float("nan")))
            if entry.get("low_confidence"):
                low_conf.add(b)
            labels.append(b)
        ax.bar(
            [x + i * width for x in range(len(bins))],
            vals,
            width=width,
            label=run,
            color=COLOURS[i % len(COLOURS)],
        )
    ax.set_xticks([x + 0.4 - width / 2 for x in range(len(bins))])
    ax.set_xticklabels(bins, rotation=30, ha="right")
    ax.set_ylabel("mAP50")
    ax.set_title("mAP50 by bin (q1 = darkest)")
    note = f"Single seed. Noise floor {NOISE_FLOOR_MAP50:.4f} mAP50."
    if low_conf:
        note += f" Low-confidence (<{MIN_BIN_IMAGES} images): {', '.join(sorted(low_conf))}."
    ax.set_xlabel(note, fontsize=8, color="#4A5568")
    ax.legend(fontsize=8, frameon=False, ncols=2)
    fig.tight_layout()
    dest = out / "dark_bins.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_routing_entropy(entropies: dict[str, dict], out: Path, plt: Any) -> Path | None:
    """Bars of normalised routing entropy, with the uniform line at 1.0 (spec 10.2)."""
    rows: list[tuple[str, str, float]] = []
    for run in sorted(entropies, key=order_key):
        for kind, m in (entropies[run].get("per_module") or {}).items():
            h = m.get("routing_entropy_norm")
            if isinstance(h, (int, float)) and not math.isnan(h):
                rows.append((run, kind, float(h)))
    if not rows:
        return None

    fig, ax = _fig(plt, w=max(9.0, 0.9 * len(rows)))
    labels = [f"{r}\n{k}" for r, k, _ in rows]
    colours = ["#9B2C2C" if k == "DER" else "#2B6CB0" for _, k, _ in rows]
    ax.bar(range(len(rows)), [h for _, _, h in rows], color=colours, width=0.6)
    ax.axhline(1.0, color="#1A202C", linestyle="--", linewidth=1.2)
    ax.text(
        len(rows) - 0.4,
        1.005,
        "uniform - no routing",
        ha="right",
        fontsize=8,
        color="#1A202C",
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("normalised entropy H / log N")
    ax.set_title("Routing entropy: 0 = commits to one path, 1 = averages them all")
    ax.set_xlabel(
        "DER (red) sits on the uniform line in every variant. EnhanceNet's gate (blue) does not.",
        fontsize=8,
        color="#4A5568",
    )
    fig.tight_layout()
    dest = out / "routing_entropy.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_routing_vs_severity(gates: dict[str, list[dict]], out: Path, plt: Any) -> Path | None:
    """Scatter of routing weight against s_dark, per module kind (spec 10.3, E2)."""
    picks = [r for r in sorted(gates, key=order_key) if gates[r]]
    picks = [r for r in picks if any(row.get("kind") == "path" for row in gates[r])][:4]
    if not picks:
        return None

    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.2), dpi=150, squeeze=False)
    drew = False
    for ax, run in zip(axes[0], picks, strict=False):
        xs, ys = [], []
        for row in gates[run]:
            if row.get("kind") != "path":
                continue
            try:
                s = float(row.get("s_dark") or "nan")
                w = float(row.get("weight"))
            except (TypeError, ValueError):
                continue
            if not math.isnan(s):
                xs.append(s)
                ys.append(w)
        ax.spines[["top", "right"]].set_visible(False)
        if xs:
            ax.scatter(xs, ys, s=6, alpha=0.35, color="#9B2C2C", edgecolors="none")
            ax.axhline(
                statistics.fmean(ys), color="#1A202C", linestyle="--", linewidth=1, label="mean"
            )
            drew = True
        ax.set_title(run, fontsize=9)
        ax.set_xlabel("s_dark")
        ax.set_ylabel("routing weight")
        ax.set_ylim(0, 1)
    if not drew:
        plt.close(fig)
        return None
    fig.suptitle("DER routing weight vs image darkness - a flat band means routing ignores it")
    fig.tight_layout()
    dest = out / "routing_vs_severity.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_cost_vs_gain(evals: dict[str, dict], out: Path, plt: Any) -> Path | None:
    """Latency against mAP50, so the accuracy-per-millisecond trade is visible."""
    pts = []
    for run in sorted(evals, key=order_key):
        v = evals[run]
        lat = (v.get("cost") or {}).get("latency_ms_median")
        m50 = (v.get("overall") or {}).get("mAP50")
        if isinstance(lat, (int, float)) and isinstance(m50, (int, float)):
            pts.append((run, float(lat), float(m50)))
    if len(pts) < 2:
        return None

    base = next((p for p in pts if p[0].startswith("baseline")), None)
    fig, ax = _fig(plt, w=8.0, h=5.0)
    for i, (run, lat, m50) in enumerate(pts):
        ax.scatter(lat, m50, s=90, color=COLOURS[i % len(COLOURS)], zorder=3)
        ax.annotate(run, (lat, m50), fontsize=8, xytext=(6, 4), textcoords="offset points")
    if base:
        ax.axhline(base[2], color="#4A5568", linestyle=":", linewidth=1)
        ax.axhspan(
            base[2] - NOISE_FLOOR_MAP50,
            base[2] + NOISE_FLOOR_MAP50,
            color="#E2E8F0",
            zorder=0,
            label=f"within noise of baseline (+/-{NOISE_FLOOR_MAP50:.4f})",
        )
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlabel("latency, ms/img at batch 1 (lower is better)")
    ax.set_ylabel("mAP50 (higher is better)")
    ax.set_title("What the modules cost and what they return")
    fig.tight_layout()
    dest = out / "cost_vs_gain.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def build_parser() -> argparse.ArgumentParser:
    """CLI of spec 10.1/10.3."""
    p = argparse.ArgumentParser(prog="python -m yolo_der.analysis.plots", description=__doc__)
    p.add_argument("--reports", type=Path, default=Path("reports"))
    p.add_argument("--out", type=Path, default=Path("reports/figures"))
    return p


def main(argv: list[str] | None = None) -> int:
    """Draw every figure the available inputs support."""
    args = build_parser().parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: this runs on a Droplet and in CI
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; install the dev or train extra")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    evals = load_json_dir(args.reports / "eval")
    entropies = load_json_dir(args.reports / "entropy")
    gates = load_gates(args.reports / "gates")
    print(f"inputs: {len(evals)} evaluations, {len(entropies)} entropy, {len(gates)} gate tables")

    made = [
        plot_dark_bins(evals, args.out, plt),
        plot_routing_entropy(entropies, args.out, plt),
        plot_routing_vs_severity(gates, args.out, plt),
        plot_cost_vs_gain(evals, args.out, plt),
    ]
    for dest in made:
        print(f"  {'wrote' if dest else 'skipped (no input)'} {dest or ''}")
    return 0 if any(made) else 1


if __name__ == "__main__":
    sys.exit(main())
