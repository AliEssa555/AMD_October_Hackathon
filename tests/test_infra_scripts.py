"""Contract tests for the manual-launch scripts (CLAUDE.md "Approved deviation from 0.2").

These scripts cannot be unit-tested the way a function can: they only do their real work
on a $1.99/hr machine. What *is* testable is that the safeguards are still in them. Each
one here exists because its absence costs money or loses a run, so a refactor that drops
one should fail here rather than on the Droplet.
"""

from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[1] / "infra"
SCRIPTS = ("bootstrap.sh", "run_job.sh", "fetch_results.sh")


def read(name: str) -> str:
    return (INFRA / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_exists_and_is_a_bash_script(name):
    path = INFRA / name
    assert path.is_file(), f"{name} is missing"
    assert read(name).startswith("#!/usr/bin/env bash")


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_docs_invoke_the_script_through_bash(name):
    """The exec bit is not tracked by this repo, so `bash <script>` is the contract.

    Every infra script is committed 100644 because the owner works on Windows, which
    means the Droplet gets a non-executable file. Invoking them through bash is what
    makes that harmless, so the docs saying so is the thing worth pinning -- not the
    mode on disk, which is why the obvious st_mode assertion here was wrong.
    file. Invoking them as `bash <script>` is what makes that harmless, which means the
    docs saying so is the actual contract -- not the mode on disk.
    """
    doc = (INFRA.parent / "docs" / "MANUAL_LAUNCH.md").read_text(encoding="utf-8")
    assert f"bash {name}" in doc or f"bash infra/{name}" in doc


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_use_lf_endings(name):
    r"""A CRLF shebang makes bash fail with a bare `\r: command not found` on the Droplet."""
    assert b"\r\n" not in (INFRA / name).read_bytes()


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_secret_is_baked_into_a_script(name):
    """Hard rule 0.3. These files are committed; secrets reach them from the environment."""
    body = read(name)
    for token in ("dop_v1_", "DO_SELF_DESTROY_TOKEN=dop", "AWS_SECRET_ACCESS_KEY="):
        assert token not in body, f"{name} looks like it contains a secret ({token})"
    # The Roboflow key may be *referenced*, never assigned a literal.
    assert "ROBOFLOW_API_KEY=" not in body.replace("ROBOFLOW_API_KEY=...", "")


class TestRunJob:
    """run_job.sh is the only script that spends money, so it carries the safeguards."""

    body = property(lambda self: read("run_job.sh"))

    def test_training_runs_under_a_timeout_watchdog(self):
        """Safeguard 1. Without this a hung dataloader bills until someone notices."""
        assert "timeout --signal=TERM --kill-after=120s" in self.body
        assert '"${TIMEOUT_SECONDS}s"' in self.body

    def test_the_watchdog_is_computed_without_python(self):
        """It must be computable before we trust the interpreter, and dry-runnable on Windows."""
        assert "awk -v h=" in self.body
        assert "python3 -c" not in self.body.split("# ------")[1]

    def test_an_exit_trap_bundles_artifacts_on_every_path(self):
        """Safeguard 2 / hard rule 0.4: success, crash, timeout and Ctrl-C all bundle."""
        assert "trap on_exit EXIT" in self.body
        # Registered before training starts, or a crash in between loses the logs.
        assert self.body.index("trap on_exit EXIT") < self.body.index("timeout --signal")

    def test_the_bundle_size_is_verified(self):
        """`tar -I zstd` without zstd leaves a 0-byte file that looks like success."""
        assert "bundle_bytes=" in self.body
        assert "DO NOT destroy the" in self.body

    def test_it_refuses_to_start_without_a_rocm_gpu(self):
        """A paid run that silently fell back to CPU is the worst outcome available."""
        assert "refusing to start a paid run" in self.body
        assert "hip=None" in self.body

    def test_poweroff_is_refused_unless_the_bundle_was_pushed(self):
        """Powering off hides the artifacts and does not stop billing."""
        assert "--poweroff ignored" in self.body

    def test_the_plan_prints_the_worst_case_cost(self):
        """Hard rule 0.1: no launch without hours x rate in front of the owner."""
        assert "worst-case" in self.body
        assert "GPU_HOURLY_USD:-1.99" in self.body

    def test_a_dry_run_disarms_the_trap(self):
        """--dry-run must not write a bundle or a done marker."""
        i = self.body.index("--dry-run: nothing was launched")
        assert "trap - EXIT" in self.body[i : i + 200]

    def test_the_banner_tells_the_owner_destroying_is_what_stops_billing(self):
        assert "STILL BILLED" in self.body


class TestBootstrap:
    body = property(lambda self: read("bootstrap.sh"))

    def test_the_gpu_is_checked_before_anything_is_installed(self):
        """Discovering a bad Droplet in minute one is the cheapest possible failure."""
        b = self.body
        assert b.index('bold "GPU"') < b.index('bold "Python dependencies"')

    def test_it_refuses_a_non_rocm_torch(self):
        assert "this torch is not a ROCm build -- refusing" in self.body

    def test_it_verifies_pip_did_not_replace_torch(self):
        """A resolver that pulls the CUDA wheel turns a GPU run into a silent CPU run."""
        b = self.body
        assert "TORCH_BEFORE" in b and "TORCH_AFTER" in b
        assert "pip replaced torch" in b

    def test_ultralytics_is_installed_with_no_deps(self):
        """Otherwise its opencv-python and torch pins fight the image (see pyproject)."""
        assert '--no-deps "ultralytics==8.3.*"' in self.body

    def test_polars_is_installed(self):
        """Missing polars kills training at plotting time, after the last epoch."""
        assert "polars" in self.body

    def test_the_fixed_paths_match_the_conventions(self):
        """Ultralytics writes absolute paths into last.pt, so resume needs these identical."""
        b = self.body
        assert 'REPO="$WORK/yolo-der"' in b and 'RUNS="$WORK/runs"' in b

    def test_it_does_not_start_training(self):
        """Hard rule 0.1: bootstrap is free setup; spending is a separate, explicit step."""
        assert "yolo_der.train" not in self.body


class TestFetchResults:
    body = property(lambda self: read("fetch_results.sh"))

    def test_it_never_destroys_anything(self):
        """Destroying is the owner's deliberate console action, not a script's."""
        for forbidden in ("doctl compute droplet delete", "poweroff", "shutdown"):
            assert forbidden not in self.body

    def test_it_refuses_to_bless_an_empty_pull(self):
        """The failure mode is destroying a Droplet whose results never arrived."""
        assert "do NOT destroy the Droplet yet" in self.body
        assert "no checkpoints and no results.csv arrived" in self.body

    def test_it_verifies_each_bundle_reads_back(self):
        assert "tar -tf" in self.body
        assert "CORRUPT" in self.body

    def test_it_handles_both_bundle_compressions(self):
        """run_job.sh falls back to gzip when zstd is absent."""
        assert "*.tar.zst" in self.body and "*.tar.gz" in self.body


def test_the_token_dependent_scripts_are_gone_not_stubbed():
    """They cannot work without an API token; a stub would invite someone to try."""
    for name in ("launch_gpu.sh", "reaper.sh", "cloud-init.yaml.tmpl"):
        assert not (INFRA / name).exists()
        assert not (INFRA / f"{name}.todo").exists()


def test_the_manual_launch_doc_exists_and_quotes_the_verified_price():
    doc = (INFRA.parent / "docs" / "MANUAL_LAUNCH.md").read_text(encoding="utf-8")
    assert "$1.99" in doc
    assert "2.59" in doc  # the wrong price is named as wrong, so nobody reintroduces it
    assert "Destroy" in doc
