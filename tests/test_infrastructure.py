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
