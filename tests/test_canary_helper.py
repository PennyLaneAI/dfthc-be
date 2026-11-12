"""Tests for dfthc.testing.canary, the subprocess runner of the canary suite (plan Task 0.2)."""

import pytest

from dfthc.testing.canary import outcome, run_snippet

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
