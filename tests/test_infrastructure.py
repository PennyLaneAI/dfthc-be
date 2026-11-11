"""Repository scaffold, environment and test-runner checks (plan Task 0.1)."""

import ctypes
import os
import re
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

REPO = Path(__file__).resolve().parents[1]


def test_package_imports():
    import dfthc

    assert dfthc.__version__ == "0.1.0"


def test_warning_variable_exported_before_python_started():
    assert os.environ.get("CATALYST_SILENCE_RULE_LOWERING_WARNINGS") == "0", (
        "run the tests through scripts/test.sh, which exports the variable before Python starts"
    )


def test_runner_defaults_to_two_workers():
    requested = os.environ.get("DFTHC_TEST_WORKERS") or "2"
    if requested.isdigit():
        assert os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0") == requested, (
            "scripts/test.sh runs 2 xdist workers unless DFTHC_TEST_WORKERS says otherwise"
        )


def test_runner_lowers_the_priority():
    qos_utility = 0x11
    niceness = os.getpriority(os.PRIO_PROCESS, 0)
    qos_class = ctypes.CDLL("/usr/lib/libSystem.dylib").qos_class_self()
    assert niceness >= 10 or 0 < qos_class <= qos_utility, (
        f"niceness {niceness}, QoS class {qos_class:#x}: run the tests through scripts/test.sh, "
        "which lowers the priority with nice or, in the agent sandbox, with taskpolicy"
    )


def test_rule_lowering_warnings_are_recorded():
    with warnings.catch_warnings(record=True) as records:
        warnings.warn("probe", RuleLoweringWarning, stacklevel=1)
    assert [str(r.message) for r in records] == ["probe"]


@pytest.mark.parametrize("name", ["pennylane", "pennylane_catalyst", "pennylane_lightning"])
def test_lock_file_pins_installed_versions(name):
    lock = (REPO / "requirements-nightly.lock").read_text()
    pinned = re.search(rf"^{name.replace('_', '[-_]')}==(\S+)$", lock, re.MULTILINE | re.IGNORECASE)
    assert pinned is not None, f"{name} missing from requirements-nightly.lock"
    assert pinned.group(1) == version(name)


@pytest.mark.parametrize("script", ["scripts/test.sh", "scripts/setup_env.sh"])
def test_scripts_are_executable(script):
    assert os.access(REPO / script, os.X_OK)


@pytest.mark.release
def test_release_marked_probe():
    """Collected only by the release tier; the next test checks that `-m "not slow"` drops it."""


@pytest.mark.subprocess
def test_release_marker_implies_slow():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:xdist", "-m", "not slow",
         f"{Path(__file__).name}::test_release_marked_probe"],
        cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
    )
    assert "test_release_marked_probe" not in result.stdout
    assert "1 deselected" in result.stdout, result.stdout + result.stderr


def _fake_executable(path, body):
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


@pytest.mark.subprocess
def test_setup_env_targets_main_checkout_from_a_worktree(tmp_path):
    """From a linked worktree the script must look at the main checkout's .venv, not its own.

    A fake python3.12 first on PATH records any call and fails, so no environment is ever built.
    """
    calls = tmp_path / "python3.12.calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_executable(bin_dir / "python3.12", f'echo "$@" >> "{calls}"\nexit 1')
    common_git_dir = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    main_venv = Path(common_git_dir).parent / ".venv"
    assert main_venv.exists(), "the canonical environment must exist for this test"

    result = subprocess.run(
        [str(REPO / "scripts/setup_env.sh")], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}, check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert f"{main_venv} exists" in result.stderr, result.stderr
    assert not calls.exists(), f"setup_env.sh tried to build an environment: {calls.read_text()}"


@pytest.mark.subprocess
def test_runner_fails_closed_without_lowered_priority(tmp_path):
    """Without a working nice or taskpolicy, test.sh exits 2 before pytest starts (plan 12.9)."""
    marker = tmp_path / "pytest.started"
    bin_dir = tmp_path / "bin"
    (tmp_path / "venv/bin").mkdir(parents=True)
    bin_dir.mkdir()
    _fake_executable(tmp_path / "venv/bin/python", f'touch "{marker}"')
    _fake_executable(bin_dir / "nice", 'echo "nice: setpriority: Operation not permitted" >&2')
    _fake_executable(bin_dir / "taskpolicy", "exit 1")

    result = subprocess.run(
        [str(REPO / "scripts/test.sh"), "-q"], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "DFTHC_VENV": str(tmp_path / "venv"),
             "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "cannot lower the priority" in result.stderr, result.stderr
    assert not marker.exists(), "pytest was started at normal priority"
