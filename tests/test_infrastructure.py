"""Repository scaffold, environment and test-runner checks (plan Task 0.1)."""

import ctypes
import os
import re
import subprocess
import sys
import warnings
from importlib.metadata import distributions
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


def normalized(name: str) -> str:
    """A distribution name in PEP 503's normal form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_differences(lock: str, installed: dict[str, str]) -> list[str]:
    """How the ``name==version`` lines of ``lock`` differ from the ``installed`` distributions,
    names normalized: a pin missing from the environment or at another version, and an installed
    distribution without a pin, except pip, which ``pip freeze`` omits on Python 3.12."""
    pins = dict(line.split("==", 1) for line in lock.split() if not line.startswith("#"))
    pinned = {normalized(name): pin for name, pin in pins.items()}
    current = {normalized(name): ver for name, ver in installed.items()
               if normalized(name) != "pip"}
    differences = []
    for name, pin in sorted(pinned.items()):
        if name not in current:
            differences.append(f"{name}=={pin} is pinned but not installed")
        elif current[name] != pin:
            differences.append(f"{name}=={pin} is pinned but {current[name]} is installed")
    return differences + [f"{name}=={current[name]} is installed but not pinned"
                          for name in sorted(current.keys() - pinned.keys())]


def test_lock_file_pins_exactly_the_installed_distributions():
    """requirements-nightly.lock is the running environment's pip freeze: every installed
    distribution at its version and nothing else (final review m-13), so a toolchain change
    without a refreshed lock, and hence a new code hash, fails here."""
    installed = {dist.metadata["Name"]: dist.version for dist in distributions()}
    assert lock_differences((REPO / "requirements-nightly.lock").read_text(), installed) == []
    assert installed and {"pennylane", "pennylane-catalyst", "pennylane-lightning"} <= {
        normalized(name) for name in installed}


def test_lock_differences_sees_a_missing_and_an_extra_distribution():
    """The whole-lock check fails on a pin the environment lacks and on a distribution the lock
    lacks, which a check of three named packages would pass."""
    lock = "pennylane==1.0\nnumpy==2.0\nPennyLane_Lightning==1.1\n"
    installed = {"PennyLane": "1.0", "pennylane-lightning": "1.1", "pip": "26.3"}
    assert lock_differences(lock, installed | {"numpy": "2.0"}) == []
    assert lock_differences(lock, installed | {"jax": "0.7"}) == [
        "numpy==2.0 is pinned but not installed", "jax==0.7 is installed but not pinned"]
    assert lock_differences(lock, installed | {"numpy": "2.1"}) == [
        "numpy==2.0 is pinned but 2.1 is installed"]


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
@pytest.mark.parametrize(("workers", "expected"), [(None, "2"), ("", "2"), ("0", "0"), ("3", "3")],
                         ids=["unset", "empty", "zero", "three"])
def test_runner_passes_the_requested_worker_count(tmp_path, workers, expected):
    """test.sh hands pytest ``-n 2`` with DFTHC_TEST_WORKERS unset or empty, and its value
    otherwise (final review m-17): a fake DFTHC_VENV python prints its arguments, so a test.sh
    that hard-codes the default fails here."""
    (tmp_path / "env/bin").mkdir(parents=True)
    _fake_executable(tmp_path / "env/bin/python", 'echo "ARGS $*"')
    env = {key: value for key, value in os.environ.items() if key != "DFTHC_TEST_WORKERS"}
    env["DFTHC_VENV"] = str(tmp_path / "env")
    if workers is not None:
        env["DFTHC_TEST_WORKERS"] = workers
    result = subprocess.run([str(REPO / "scripts/test.sh"), "-q"], cwd=REPO, capture_output=True,
                            text=True, env=env, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == f"ARGS -m pytest -n {expected} -q", result.stdout


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
