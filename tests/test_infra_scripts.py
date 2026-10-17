"""Contract tests for the manual-launch scripts (CLAUDE.md "Approved deviation from 0.2").

These scripts cannot be unit-tested the way a function can: they only do their real work
on a $1.99/hr machine. What *is* testable is that the safeguards are still in them. Each
one here exists because its absence costs money or loses a run, so a refactor that drops
one should fail here rather than on the Droplet.
"""

from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parents[1] / "infra"
SCRIPTS = (
    "bootstrap.sh",
    "run_job.sh",
    "fetch_results.sh",
    "send_repo.sh",
    "in_container.sh",
)


def read(name: str) -> str:
    return (INFRA / name).read_text(encoding="utf-8")


def code_lines(name: str) -> str:
    """The script with comment lines stripped.

    Needed because these scripts explain their own pitfalls in comments, so a naive
    `"--group-add render" not in body` trips over the comment saying not to use it.
    """
    return chr(10).join(
        line for line in read(name).splitlines() if not line.lstrip().startswith("#")
    )


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
    # The Roboflow key may be referenced, and shown as a placeholder in a usage line,
    # but never assigned a real value. A real key has no angle brackets or dots in it.
    import re

    for match in re.findall(r"ROBOFLOW_API_KEY=(\S*)", body):
        assert match in ("", "<key>", "...") or match.startswith("$"), (
            f"{name} may contain a real Roboflow key: {match!r}"
        )


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


class TestTransferWithoutRsync:
    """The owner has ssh, scp and tar but no rsync, which was checked, not assumed."""

    def test_send_repo_streams_a_tarball_over_ssh(self):
        body = read("send_repo.sh")
        assert "tar -cz" in body and "tar -xz" in body

    def test_send_repo_excludes_the_dvc_cache(self):
        """DVC keeps a second full copy of the dataset: 96% of the tarball."""
        body = read("send_repo.sh")
        for path in ("./.dvc/cache", "./data", "./.venv", "./runs", "./mlruns"):
            assert f"--exclude={path}" in body

    def test_send_repo_excludes_use_a_leading_dot_slash(self):
        """`--exclude=data` silently matches nothing here; that is a 256 MB upload.

        Checked against the array only. The first version of this test scanned the whole
        file and tripped over the comment explaining the bug.
        """
        body = read("send_repo.sh")
        array = body.split("EXCLUDES=(")[1].split(")")[0]
        for line in array.splitlines():
            line = line.strip()
            if not line.startswith("--exclude="):
                continue
            assert line.startswith("--exclude=./"), line

    def test_send_repo_verifies_the_commit_matches(self):
        """A truncated stream can still unpack into something that looks plausible."""
        body = read("send_repo.sh")
        assert "rev-parse HEAD" in body
        assert "the remote checkout does not match this one" in body

    def test_fetch_results_does_not_require_rsync(self):
        """Hard rule 0.4 cannot depend on software the owner does not have."""
        body = read("fetch_results.sh")
        assert "rsync not found" not in body
        assert "command -v rsync" in body
        assert "tar -cz -C" in body  # the fallback that actually runs here

    def test_send_repo_does_not_train_or_destroy(self):
        body = read("send_repo.sh")
        for forbidden in ("yolo_der.train", "poweroff", "droplet delete"):
            assert forbidden not in body


class TestContainerWrapper:
    """The AMD image has no torch on the host; PyTorch is only inside the containers.

    Every assertion here corresponds to something that actually failed on the live
    Droplet during the Phase 4 run.
    """

    body = property(lambda self: read("in_container.sh"))

    def test_it_does_not_pass_group_add_render(self):
        """Docker resolves group names inside the image, which has no `render` group.

        It then refuses the whole run rather than skipping the flag. Probing the host
        for the group was the wrong fix, for the same reason.
        """
        code = code_lines("in_container.sh")
        assert "--group-add render" not in code
        assert "--group-add video" in code

    def test_it_requests_the_rocm_devices(self):
        for flag in ("--device=/dev/kfd", "--device=/dev/dri"):
            assert flag in self.body

    def test_it_raises_shm_for_dataloader_workers(self):
        """The 64 MB docker default makes workers die with an opaque Bus error."""
        assert "--ipc=host" in self.body
        assert "--shm-size" in self.body

    def test_it_bind_mounts_work_so_paths_match_host_and_container(self):
        """Ultralytics writes absolute paths into last.pt (gotcha 7)."""
        assert '-v "$WORK:$WORK"' in self.body

    def test_the_container_state_probe_strips_whitespace(self):
        """`docker inspect` output carried a newline, so neither case arm matched.

        The script then tried to remove a container that did not exist.
        """
        assert "tr -d '[:space:]'" in self.body
        assert "state=absent" in self.body

    def test_it_verifies_the_gpu_through_the_container(self):
        """The host seeing an MI300X does not prove the container can use it."""
        assert "torch.cuda.is_available()" in self.body
        assert "torch.version.hip" in self.body

    def test_it_keeps_stdin_attached_for_the_exit_trap(self):
        """Ctrl-C must reach the process so run_job.sh still bundles artifacts."""
        assert "docker exec -i" in self.body


class TestSendRepoManifest:
    """A bad exclude removes something load-bearing and only fails later, remotely."""

    body = property(lambda self: read("send_repo.sh"))

    def test_it_checks_required_paths_before_sending(self):
        assert "REQUIRED=(" in self.body
        assert "Checking the archive manifest" in self.body

    def test_the_required_list_covers_the_data_subpackage(self):
        """`--exclude=./data` under bsdtar also dropped src/yolo_der/data.

        On the Droplet that surfaced as ModuleNotFoundError: No module named
        'yolo_der.data', long after the upload reported success.
        """
        for path in (
            "./src/yolo_der/data/prepare.py",
            "./src/yolo_der/data/download.py",
            "./src/yolo_der/data/severity.py",
        ):
            assert path in self.body

    def test_every_required_path_exists_in_this_repo(self):
        """A stale entry would make the guard fail on a healthy archive."""
        array = self.body.split("REQUIRED=(")[1].split(")")[0]
        root = INFRA.parent
        for line in array.splitlines():
            line = line.strip()
            if not line.startswith("./"):
                continue
            assert (root / line[2:]).exists(), f"REQUIRED names a missing path: {line}"

    def test_it_warns_about_windows_bsdtar(self):
        assert "bsdtar" in self.body


class TestWatchdogCrossesTheContainerBoundary:
    """The printed cost and the enforced cap must be the same number.

    `docker run -e` applies only when a container is created, and in_container.sh reuses
    a running one, so a per-run `MAX_RUNTIME_HOURS=1` never reached run_job.sh: it printed
    "6h x $1.99 = $11.94" while the owner had asked for one hour. Caught on the live
    Droplet, which is the expensive place to find it.
    """

    body = property(lambda self: read("in_container.sh"))

    def test_max_runtime_hours_is_forwarded(self):
        assert "MAX_RUNTIME_HOURS" in self.body.split("FORWARD=(")[1].split(")")[0]

    def test_the_env_is_forwarded_on_exec_not_only_on_run(self):
        code = code_lines("in_container.sh")
        exec_line = next(line for line in code.splitlines() if line.startswith("exec docker exec"))
        assert "EXEC_ENV" in exec_line, exec_line

    def test_every_cost_shaping_variable_is_forwarded(self):
        """Anything run_job.sh reads from the environment to size or price a run."""
        forwarded = self.body.split("FORWARD=(")[1].split(")")[0]
        for name in ("MAX_RUNTIME_HOURS", "GPU_HOURLY_USD", "RUNS_DIR", "BUNDLE_DIR"):
            assert name in forwarded, name

    def test_unset_variables_are_not_forwarded_as_empty(self):
        """An empty -e would override run_job.sh's own default with the empty string."""
        assert '[[ -n "${!name-}" ]]' in self.body

    def test_it_reports_what_it_forwarded(self):
        """So the owner can see the cap in force, not just the one they typed."""
        assert "forwarding:" in self.body
