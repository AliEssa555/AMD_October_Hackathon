"""Tracking tests (spec 9.2).

The point of most of these is to check our sanitiser against MLflow's **own** validator
rather than against our reading of its docs, so a future MLflow bump that tightens the
rules fails here instead of two hours into a GPU run.
"""

import posixpath

import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow.utils.validation import (  # noqa: E402
    MAX_ENTITY_KEY_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
    _validate_metric_name,
    validate_param_and_metric_name,
)

from yolo_der.tracking import (  # noqa: E402
    MAX_KEY_LEN,
    PARAMS_PER_BATCH,
    MlflowTracker,
    data_hash,
    environment_tags,
    git_sha,
    sanitise_key,
)

# The names Ultralytics actually produces, plus deliberately hostile ones.
ULTRALYTICS_KEYS = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "train/box_loss",
    "val/cls_loss",
    "lr/pg0",
    "fitness",
]
HOSTILE_KEYS = [
    "a:b",  # legal on POSIX, ILLEGAL on Windows
    "a//b",
    "/leading",
    "trailing/",
    "..",
    "./x",
    "a/../b",
    "",
    "   ",
    "weird" + chr(92) + "slash",
    "emoji\U0001f600name",
    "x" * 400,
]


@pytest.mark.parametrize("key", ULTRALYTICS_KEYS + HOSTILE_KEYS)
def test_sanitised_keys_pass_mlflows_own_validator(key):
    """The contract: whatever we hand MLflow, MLflow accepts."""
    cleaned = sanitise_key(key)
    _validate_metric_name(cleaned)  # raises MlflowException if we got it wrong


@pytest.mark.parametrize("key", ULTRALYTICS_KEYS + HOSTILE_KEYS)
def test_sanitised_keys_are_valid_on_both_platforms(key):
    """We target the Windows-safe intersection, so colons must be gone regardless of host.

    MLflow's charset is platform dependent; a name produced on the Linux Droplet has to be
    loggable from the owner's Windows laptop too, or cross-machine resume breaks.
    """
    cleaned = sanitise_key(key)
    assert validate_param_and_metric_name(cleaned), cleaned
    assert ":" not in cleaned
    assert posixpath.normpath(cleaned) == cleaned


def test_parens_are_stripped_not_replaced():
    """Spec 9.2 names the exact transformation."""
    assert sanitise_key("metrics/mAP50(B)") == "metrics/mAP50B"
    assert sanitise_key("metrics/mAP50-95(B)") == "metrics/mAP50-95B"


def test_slashes_and_dashes_and_dots_survive():
    """These are meaningful structure in Ultralytics' names; we must not flatten them."""
    assert sanitise_key("train/box_loss") == "train/box_loss"
    assert sanitise_key("metrics/mAP50-95") == "metrics/mAP50-95"
    assert sanitise_key("a.b/c") == "a.b/c"


def test_empty_and_whitespace_keys_get_a_usable_name():
    assert sanitise_key("") == "unnamed"
    assert sanitise_key("   ") == "unnamed"
    assert sanitise_key("((((") == "unnamed"


def test_long_keys_are_truncated_to_mlflows_limit():
    assert MAX_KEY_LEN == MAX_ENTITY_KEY_LENGTH
    assert len(sanitise_key("x" * 400)) == MAX_ENTITY_KEY_LENGTH


def test_sanitise_is_idempotent():
    """Re-logging a key must not slowly mangle it."""
    for key in ULTRALYTICS_KEYS + HOSTILE_KEYS:
        once = sanitise_key(key)
        assert sanitise_key(once) == once, key


def test_params_batch_size_matches_mlflows_limit():
    """Ultralytics' flattened args exceed 100 keys, so this constant must stay in step."""
    assert PARAMS_PER_BATCH == MAX_PARAMS_TAGS_PER_BATCH


def test_environment_tags_carry_the_rocm_evidence():
    """The `hip` tag is how the write-up proves the run used AMD hardware."""
    tags = environment_tags()
    assert {"torch", "ultralytics", "host", "hip", "gpu"} <= set(tags)
    assert all(isinstance(v, str) for v in tags.values())
    # torch.version.hip is None on a CPU/CUDA build; the key must still exist, as "".
    assert "hip" in tags


def test_git_sha_is_reported_for_this_repo():
    sha = git_sha()
    assert sha != "unknown"
    assert len(sha.split("-")[0]) >= 7


def test_data_hash_is_unknown_without_a_lock_file(tmp_path):
    assert data_hash(tmp_path) == "unknown"


def test_data_hash_reads_the_prepared_stage_md5(tmp_path):
    (tmp_path / "dvc.lock").write_text(
        "stages:\n"
        "  prepare:\n"
        "    outs:\n"
        "    - path: data/prepared\n"
        "      hash: md5\n"
        "      md5: 0123456789abcdef0123456789abcdef.dir\n",
        encoding="utf-8",
    )
    assert data_hash(tmp_path) == "0123456789abcdef0123456789abcdef.dir"


def test_stored_run_id_round_trips(tmp_path):
    assert MlflowTracker.stored_run_id(tmp_path) is None
    (tmp_path / "mlflow_run_id.txt").write_text("abc123\n", encoding="utf-8")
    assert MlflowTracker.stored_run_id(tmp_path) == "abc123"


def test_disabled_tracker_registers_no_callbacks():
    """--no-mlflow must leave the trainer completely untouched."""
    calls = []

    class FakeTrainer:
        def add_callback(self, event, cb):
            calls.append(event)

    MlflowTracker(variant="baseline", seed=0, enabled=False).attach(FakeTrainer())
    assert calls == []


def test_enabled_tracker_registers_every_callback():
    calls = []

    class FakeTrainer:
        def add_callback(self, event, cb):
            calls.append(event)

    MlflowTracker(variant="der_enh", seed=1).attach(FakeTrainer())
    assert calls == [
        "on_pretrain_routine_end",
        "on_fit_epoch_end",
        "on_model_save",
        "on_train_end",
        # Not in the spec's list of four, but required: MLflow's own atexit hook closes a
        # dangling run as FINISHED, so a crashed run would be indistinguishable from a
        # good one in the UI. `teardown` marks it FAILED and tags it crashed=true.
        "teardown",
    ]


def test_run_name_follows_the_spec_convention():
    assert MlflowTracker(variant="der_enh", seed=1).run_name == "der_enh-s1"


class TestCrashIsRecordedAsFailure:
    """A crashed run must not read as a successful one.

    Ultralytics calls `run_callbacks("teardown")` only at the end of the happy path
    (engine/trainer.py:527, after on_train_end, not in a finally), so an exception skips
    it and MLflow's atexit hook closes the dangling run as FINISHED. Seven AMP-crashed
    runs sat in the database labelled FINISHED with zero metrics before this was fixed.
    """

    def test_fail_exists_and_is_not_the_ultralytics_hook(self):
        assert hasattr(MlflowTracker, "fail")
        assert hasattr(MlflowTracker, "teardown")

    def test_fail_is_a_no_op_once_completed(self):
        """on_train_end sets completed; fail must not then overwrite a good run."""
        t = MlflowTracker(variant="baseline", seed=0)
        t.completed = True
        t.fail(reason="should be ignored")  # must not raise, must not touch MLflow

    def test_fail_is_a_no_op_when_disabled(self):
        t = MlflowTracker(variant="baseline", seed=0, enabled=False)
        t.fail(reason="should be ignored")

    def test_fail_never_raises_even_with_mlflow_broken(self, monkeypatch):
        """It runs on paths that are already failing; it must never mask the traceback."""
        import yolo_der.tracking as tracking

        def boom(*a, **k):
            raise RuntimeError("tracking server gone")

        monkeypatch.setattr(tracking.mlflow, "active_run", boom)
        t = MlflowTracker(variant="baseline", seed=0)
        t.fail(reason="x", exc=ValueError("original"))
        assert t.completed, "a failed fail() must still stop later attempts"

    def test_train_cli_marks_the_run_failed_on_exception(self):
        """The explicit belt: train.py must call fail() before re-raising."""
        from pathlib import Path

        src = Path("src/yolo_der/train.py").read_text(encoding="utf-8")
        assert "tracker.fail(" in src
        i, j = src.index("trainer.train()"), src.index("tracker.fail(")
        assert i < j, "fail() must be in the except block after train()"
        assert "raise" in src[j : j + 200], "the original exception must be re-raised"

    def test_atexit_backstop_is_registered(self):
        """For hard exits that never unwind -- e.g. SIGTERM from the watchdog."""
        from pathlib import Path

        src = Path("src/yolo_der/tracking.py").read_text(encoding="utf-8")
        assert "atexit.register(self.fail" in src
        assert "_atexit_registered" in src
