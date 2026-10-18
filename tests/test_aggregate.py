"""The ablation table must not mislead (spec 10.1).

Every test here pins a bug that was actually printed in a table I was about to put in
front of someone: a 3-epoch smoke test averaged in as a seed, deltas taken across two
datasets whose absolute mAP differs by 0.2, a hard-coded noise floor used when real
repeated seeds existed, and failed measurements printed as "0.00" and "nan".
"""

import pytest

from yolo_der.evaluate import (
    NOISE_FLOOR_MAP50,
    aggregate,
    dataset_of,
    drop_short_runs,
    measured_noise_floor,
)


def run(variant, seed=0, map50=0.5, epochs=100, cost=None, bins=None):
    return {
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "weights": "x.pt",
        "overall": {"mAP50": map50, "mAP50-95": map50 / 2, "precision": 0.7, "recall": 0.5},
        "per_class": {},
        "bins": bins or {},
        "cost": cost or {},
        "environment": {},
    }


class TestDatasetSplit:
    def test_bdd_variants_are_their_own_dataset(self):
        assert dataset_of("bdd_der_enh") == "bdd100k"
        assert dataset_of("bdd_baseline") == "bdd100k"

    def test_exdark_is_the_default(self):
        for v in ("baseline", "der", "der_enh_sevcond", "der_enh_deronly50"):
            assert dataset_of(v) == "exdark"


class TestSmokeRunExclusion:
    def test_a_three_epoch_run_is_not_a_seed(self):
        """bdd_baseline merged a 50-epoch run with a 3-epoch throughput test."""
        runs = [
            run("bdd_baseline", 0, 0.3696, epochs=50),
            run("bdd_baseline", 99, 0.2427, epochs=3),
        ]
        kept, dropped = drop_short_runs(runs)
        assert len(kept) == 1
        assert kept[0]["seed"] == 0
        assert "bdd_baseline-s99 (3 epochs)" in dropped

    def test_genuine_seeds_are_all_kept(self):
        runs = [run("baseline", s, 0.57, epochs=100) for s in (0, 1, 2)]
        kept, dropped = drop_short_runs(runs)
        assert len(kept) == 3 and not dropped

    def test_a_slightly_short_run_is_still_a_seed(self):
        """bdd_der_enh hit the watchdog at 48/50; that is a seed, not a smoke test."""
        runs = [run("bdd_der_enh", 0, 0.37, epochs=48), run("bdd_der_enh", 1, 0.37, epochs=50)]
        kept, dropped = drop_short_runs(runs)
        assert len(kept) == 2 and not dropped

    def test_never_drops_everything(self):
        """If epochs are unknown (0), keep the runs rather than emptying the table."""
        runs = [run("x", 0, 0.5, epochs=0), run("x", 1, 0.5, epochs=0)]
        kept, _ = drop_short_runs(runs)
        assert len(kept) == 2


class TestNoiseFloor:
    def test_measured_from_repeated_seeds_when_available(self):
        by_variant = {
            "baseline": [run("baseline", s, m) for s, m in ((0, 0.5803), (1, 0.5844), (2, 0.5729))],
        }
        floor, note = measured_noise_floor(by_variant)
        assert floor == pytest.approx(0.0115, abs=1e-4)
        assert "measured" in note and "3 seeds" in note

    def test_falls_back_to_the_constant_with_one_seed_each(self):
        by_variant = {"a": [run("a")], "b": [run("b")]}
        floor, note = measured_noise_floor(by_variant)
        assert floor == NOISE_FLOOR_MAP50
        assert "assumed" in note

    def test_takes_the_widest_spread_not_the_narrowest(self):
        """The floor has to cover the worst observed replication, not the best."""
        by_variant = {
            "tight": [run("tight", 0, 0.50), run("tight", 1, 0.501)],
            "loose": [run("loose", 0, 0.50), run("loose", 1, 0.52)],
        }
        floor, _ = measured_noise_floor(by_variant)
        assert floor == pytest.approx(0.02, abs=1e-6)


class TestTable:
    def test_deltas_are_never_taken_across_datasets(self, tmp_path):
        """`bdd_der` was once reported as -0.1925 against the ExDark baseline."""
        reports = tmp_path / "eval"
        reports.mkdir()
        import json

        for r in (
            run("baseline", 0, 0.5672),
            run("der", 0, 0.5688),
            run("bdd_baseline", 0, 0.3711),
            run("bdd_der", 0, 0.3748),
        ):
            (reports / f"{r['variant']}-s{r['seed']}.json").write_text(json.dumps(r))

        md = aggregate(reports, tmp_path).read_text(encoding="utf-8")
        assert "## exdark" in md and "## bdd100k" in md
        # bdd_der is +0.0037 against its own baseline, not -0.19 against ExDark's.
        bdd_line = next(ln for ln in md.splitlines() if ln.startswith("| `bdd_der`"))
        assert "+0.0037" in bdd_line, bdd_line
        assert "-0.19" not in md

    def test_failed_measurements_are_named_not_printed_as_zero(self, tmp_path):
        import json

        reports = tmp_path / "eval"
        reports.mkdir()
        r = run("baseline", 0, 0.5, cost={"gflops_error": "thop missing"})
        (reports / "baseline-s0.json").write_text(json.dumps(r))
        md = aggregate(reports, tmp_path).read_text(encoding="utf-8")
        assert "Measurements that failed" in md
        assert "thop missing" in md
        # and the column shows a dash, not a fake 0.00
        assert "| 0.00 |" not in md
