"""Phase 0 acceptance: the tree, the configs, and the secret-hygiene rules hold."""

import re
import subprocess

import pytest
import yaml

# Hard rule 0.3: these patterns must be in .gitignore from the very first commit.
REQUIRED_IGNORES = [".env", ".env.*", "*.pem", ".dvc/config.local", "mlruns/", "runs/", "*.db"]

EXPECTED_DIRS = [
    "configs",
    "docker",
    "docs",
    "infra/hub",
    "src/yolo_der/analysis",
    "src/yolo_der/data",
    "src/yolo_der/models",
    "tests",
]

VARIANTS = [
    "baseline",
    "der",
    "enh",
    "der_enh",
    "der_routes_enh",
    "der_enh_entreg",
    "der_enh_sevcond",
]


def test_package_imports():
    import yolo_der

    assert yolo_der.__version__


@pytest.mark.parametrize("rel", EXPECTED_DIRS)
def test_expected_dirs_exist(repo, rel):
    assert (repo / rel).is_dir(), f"missing directory: {rel}"


@pytest.mark.parametrize("pattern", REQUIRED_IGNORES)
def test_gitignore_covers_secrets_and_artefacts(repo, pattern):
    lines = {ln.strip() for ln in (repo / ".gitignore").read_text().splitlines()}
    assert pattern in lines, f".gitignore must ignore {pattern!r} (hard rule 0.3)"


def test_source_packages_are_not_gitignored(repo):
    """Regression: a bare ``data/`` ignore rule also hides ``src/yolo_der/data``.

    The rule must be anchored (``/data/``) so only the dataset directory is ignored.
    """
    must_be_tracked = [
        "src/yolo_der/data/prepare.py",
        "src/yolo_der/data/severity.py",
        "src/yolo_der/data/download.py",
    ]
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.split()
    )
    missing = [p for p in must_be_tracked if p not in tracked]
    assert not missing, f"not tracked by git (ignore rule too broad?): {missing}"


def test_dataset_directory_is_still_gitignored(repo):
    """The other half of the rule: raw/prepared data must never enter git."""
    result = subprocess.run(
        ["git", "check-ignore", "data/raw/anything.jpg"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "data/ must stay ignored -- it is DVC-tracked"


def test_env_example_has_no_values(repo):
    """.env.example documents names only -- a filled-in value would be a leaked secret."""
    secretish = re.compile(r"(TOKEN|SECRET|KEY)\s*=\s*(\S+)")
    for ln in (repo / ".env.example").read_text().splitlines():
        if ln.lstrip().startswith("#"):
            continue
        m = secretish.search(ln)
        if m:
            # Only placeholder-style values are allowed (e.g. the dummy AWS region).
            assert m.group(2).startswith(("<", "$", "ghcr.io/")), f"value in .env.example: {ln}"


def test_base_config_shape(repo):
    cfg = yaml.safe_load((repo / "configs/base.yaml").read_text())
    assert cfg["train"]["imgsz"] == 448, "paper input size"
    assert cfg["train"]["optimizer"] == "SGD", "'auto' would silently pick AdamW"
    assert cfg["train"]["patience"] == 0, "no early stopping: equal epochs per variant"
    assert cfg["train"]["save_period"] == 1, "resume depends on last.pt every epoch"
    assert cfg["modules"]["insert_at"] == [4, 6, 9]


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_configs_only_override_allowed_keys(repo, variant):
    """Variant files carry `name` plus `modules.*` and nothing else (spec 9.1)."""
    base = yaml.safe_load((repo / "configs/base.yaml").read_text())
    cfg = yaml.safe_load((repo / f"configs/{variant}.yaml").read_text())
    assert cfg["name"] == variant
    assert set(cfg) <= {"name", "modules", "train"}
    assert set(cfg.get("modules", {})) <= set(base["modules"])


def test_params_yaml_matches_paper_constants(repo):
    p = yaml.safe_load((repo / "params.yaml").read_text())
    assert p["severity"]["tau"] == 0.3  # Eq. 10 darkness threshold
    assert p["severity"]["dcp_patch"] == 15  # He et al. 2009
    assert p["severity"]["dcp_omega"] == 0.95
