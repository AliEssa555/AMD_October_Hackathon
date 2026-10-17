"""CLI tests for ``python -m yolo_der.train`` (spec 9.2, 9.4)."""

import pytest
import yaml

pytest.importorskip("ultralytics")

from yolo_der.config import load_config  # noqa: E402
from yolo_der.train import (  # noqa: E402
    TRAIN_ARG_NAMES,
    VARIANTS,
    build_overrides,
    build_parser,
    materialise_data_yaml,
    resolve_runs_dir,
)


def parse(*argv):
    return build_parser().parse_args(list(argv))


def test_every_variant_name_has_a_config_file():
    """A typo in VARIANTS would only surface as a confusing load error at launch."""
    for variant in VARIANTS:
        assert load_config(variant).name == variant


def test_epochs_override_is_refused_with_resume(capsys):
    """Resume restores epochs from the checkpoint, so the flag would be a silent lie."""
    from yolo_der.train import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--config", "baseline", "--resume", "x.pt", "--epochs-override", "5"])
    assert excinfo.value.code != 0
    assert "--epochs-override cannot be combined with --resume" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "-1", "1.5"])
def test_fraction_outside_zero_to_one_is_refused(bad, capsys):
    from yolo_der.train import main

    with pytest.raises(SystemExit):
        main(["--config", "baseline", "--fraction", bad])
    assert "--fraction must be in (0, 1]" in capsys.readouterr().err


def test_epochs_override_alone_is_accepted():
    args = parse("--config", "baseline", "--epochs-override", "3")
    assert args.epochs_override == 3 and args.resume is None


def test_materialise_data_yaml_pins_an_absolute_path(tmp_path):
    """Ultralytics resolves a relative `path:` against the cwd and then ~/datasets.

    Without this, a run launched from anywhere but the repo root fails pointing at a
    directory nobody chose (ultralytics/data/utils.py::check_det_dataset).
    """
    src_dir = tmp_path / "prepared"
    src_dir.mkdir()
    src = src_dir / "data.yaml"
    src.write_text(
        yaml.safe_dump(
            {"path": "data/prepared", "train": "train/images", "nc": 2, "names": ["a", "b"]}
        ),
        encoding="utf-8",
    )

    dest = materialise_data_yaml(src, tmp_path / "run")
    doc = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert doc["path"] == str(src_dir.resolve())
    assert doc["train"] == "train/images"  # split keys untouched
    assert doc["names"] == ["a", "b"]
    assert src.read_text(encoding="utf-8").count("data/prepared") == 1  # source untouched


def test_materialise_data_yaml_adds_path_when_absent(tmp_path):
    """prepare.py now omits `path:` deliberately; we must still pin it."""
    src_dir = tmp_path / "prepared"
    src_dir.mkdir()
    src = src_dir / "data.yaml"
    src.write_text(yaml.safe_dump({"train": "train/images", "nc": 1, "names": ["a"]}), "utf-8")
    doc = yaml.safe_load(materialise_data_yaml(src, tmp_path / "run").read_text("utf-8"))
    assert doc["path"] == str(src_dir.resolve())


def test_every_mapped_train_arg_is_a_real_yolo_argument():
    """get_cfg aborts on an unknown key, so this mapping is a hard contract."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    unknown = [n for n in TRAIN_ARG_NAMES if n not in DEFAULT_CFG_DICT]
    assert not unknown, f"not YOLO arguments: {unknown}"


def test_train_config_fields_are_all_forwarded():
    """A field added to TrainConfig but not to TRAIN_ARG_NAMES would be silently ignored."""
    from dataclasses import fields

    from yolo_der.config import TrainConfig

    missing = {f.name for f in fields(TrainConfig)} - set(TRAIN_ARG_NAMES)
    assert not missing, f"TrainConfig fields never reach Ultralytics: {sorted(missing)}"


def test_overrides_place_the_run_under_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    cfg = load_config("der_enh", seed=3)
    args = parse("--config", "der_enh", "--seed", "3")

    data_dir = tmp_path / "data" / "prepared"
    data_dir.mkdir(parents=True)
    (data_dir / "data.yaml").write_text(
        yaml.safe_dump({"train": "train/images", "nc": 1, "names": ["a"]}), "utf-8"
    )

    overrides = build_overrides(cfg, args, tmp_path)
    assert overrides["project"] == str(tmp_path)
    assert overrides["name"] == "der_enh-s3"
    assert overrides["seed"] == 3
    assert overrides["exist_ok"] is True
    assert "der_cfg" not in overrides  # must never reach get_cfg


def test_cli_flags_win_over_the_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    data_dir = tmp_path / "data" / "prepared"
    data_dir.mkdir(parents=True)
    (data_dir / "data.yaml").write_text(
        yaml.safe_dump({"train": "train/images", "nc": 1, "names": ["a"]}), "utf-8"
    )
    cfg = load_config("baseline")
    args = parse(
        "--config",
        "baseline",
        "--batch",
        "8",
        "--workers",
        "2",
        "--cache",
        "false",
        "--device",
        "cpu",
        "--epochs-override",
        "7",
    )
    overrides = build_overrides(cfg, args, tmp_path)
    assert overrides["batch"] == 8
    assert overrides["workers"] == 2
    assert overrides["cache"] is False  # the string "false" must become a bool
    assert overrides["device"] == "cpu"
    assert overrides["epochs"] == 7


def test_runs_dir_defaults_and_honours_the_env(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNS_DIR", raising=False)
    assert resolve_runs_dir().name == "runs"
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "elsewhere"))
    assert resolve_runs_dir() == (tmp_path / "elsewhere").resolve()
