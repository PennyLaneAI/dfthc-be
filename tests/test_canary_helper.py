"""Tests for dfthc.testing.canary, the subprocess runner of the canary suite (plan Task 0.2)."""

import sys

import pytest

from dfthc.testing.canary import SRC, outcome, run_snippet, snippet_command

pytestmark = pytest.mark.subprocess


def test_run_snippet_reports_ok_and_error_lines():
    result = run_snippet(
        """
        report("SUM", lambda: 1 + 1)
        report("FAIL", lambda: 1 / 0)
        report("GATES", lambda: sorted(GATE_SET)[:2])
        """
    )
    assert outcome(result, "SUM") == "OK 2"
    assert outcome(result, "FAIL") == "ERROR ZeroDivisionError division by zero"
    assert outcome(result, "GATES") == 'OK ["Adjoint(S)", "Adjoint(TemporaryAND)"]'


def test_run_snippet_exports_or_unsets_the_warning_variable():
    body = """
        import os
        report("VAR", lambda: os.environ.get("CATALYST_SILENCE_RULE_LOWERING_WARNINGS"))
        """
    assert outcome(run_snippet(body), "VAR") == 'OK "0"'
    assert outcome(run_snippet(body, export_warning_variable=False), "VAR") == "OK null"


def test_outcome_explains_a_missing_report_line():
    result = run_snippet('print("nothing to see")')
    with pytest.raises(AssertionError, match="no report line for 'MISSING'"):
        outcome(result, "MISSING")


def test_snippet_command_drops_hostile_parent_variables(monkeypatch):
    for name, value in {
        "PYTHONWARNINGS": "error",
        "PYTHONDEVMODE": "1",
        "JAX_ENABLE_X64": "0",
        "JAX_PLATFORMS": "bogus",
        "XLA_FLAGS": "--bogus",
        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "1",
        "CATALYST_FOO": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("DFTHC_KEEP_ME", "yes")

    command, env = snippet_command("pass", export_warning_variable=False)
    assert command[:2] == [sys.executable, "-c"]
    removed = ("PYTHONWARNINGS", "PYTHONDEVMODE", "JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS",
               "CATALYST_SILENCE_RULE_LOWERING_WARNINGS", "CATALYST_FOO")
    assert [name for name in removed if name in env] == []
    assert env["DFTHC_KEEP_ME"] == "yes"
    assert env["PYTHONPATH"] == str(SRC)

    _, exported = snippet_command("pass", export_warning_variable=True)
    assert exported["CATALYST_SILENCE_RULE_LOWERING_WARNINGS"] == "0"
    assert not [name for name in exported if name.startswith("JAX_") or name == "CATALYST_FOO"]
