"""Canaries L-18 and L-20: rule-lowering warnings and undecomposed operators (spec section 14)."""

import pytest

from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled, pytest.mark.subprocess]

_COUNT_RULE_WARNINGS = """
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

@qp.qjit(capture=True, enable_aot_compilation=False)
@qp.decompose(gate_set=GATE_SET)
@qp.qnode(qp.device("null.qubit", wires=3))
def circuit():
    qp.Toffoli([0, 1, 2])
    qp.CZ([0, 1])
    return qp.probs(wires=[0])

def rule_warnings():
    with warnings.catch_warnings(record=True) as records:
        qp.analyze(circuit, level="user")()
    return sum(issubclass(r.category, RuleLoweringWarning) for r in records)

report("RULE_WARNINGS", rule_warnings)
"""


def test_warnings_silenced_without_env():
    """L-18 (catalyst#3190): without the variable, Catalyst ignores RuleLoweringWarning."""
    result = run_snippet(_COUNT_RULE_WARNINGS, export_warning_variable=False)
    assert outcome(result, "RULE_WARNINGS") == "OK 0"


def test_warnings_visible_with_env():
    """L-18: with CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 exported, the warnings appear."""
    result = run_snippet(_COUNT_RULE_WARNINGS, export_warning_variable=True)
    count = outcome(result, "RULE_WARNINGS")
    assert count.startswith("OK ") and int(count.split()[1]) > 0


def test_unreachable_op_reports_missing_rule():
    """L-18 (catalyst#2767): no path gives the misleading "could not be parsed" error.

    The real cause, "Decomposition rule not found for operator '<op>'", is in the same message,
    next to the misleading text, which this canary records so that a fix of catalyst#2767 flips
    it. The text alone proves nothing: a successful verbose compile prints it too (spike T6). The
    evidence of the failure is the CompileError and the named operator.
    """
    result = run_snippet(
        """
        @qp.qjit(capture=True, enable_aot_compilation=False)
        @qp.decompose(gate_set=GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            qp.ctrl(qp.adjoint(qp.TemporaryAND([0, 1, 2])), control=3)
            return qp.probs(wires=[0])

        report("UNREACHABLE", lambda: counts(circuit))
        """
    )
    unreachable = outcome(result, "UNREACHABLE")
    assert unreachable.startswith("ERROR CompileError")
    assert "the rules file could not be parsed" in unreachable
    assert "Decomposition rule not found for operator 'C(Adjoint(TemporaryAND))" in unreachable


def test_undecomposed_operator_op_reaches_runtime():
    """L-20 (catalyst#3430): an Operator2 listed in the gate set and left undecomposed.

    On 2026-11-11 it does not reach runtime as a bare named gate: with a rule, the device pass
    decomposes it and the result is correct; without a rule, compiling for execution fails.
    """
    result = run_snippet(
        """
        class L20WithRule(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

        @qp.register_resources({qp.X: 1})
        def _l20_rule(wires):
            qp.X(wires[0])

        qp.add_decomps(L20WithRule, _l20_rule)

        class L20NoRule(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

        def build(op_class):
            @qp.qjit(capture=True)
            @qp.decompose(gate_set=GATE_SET | {op_class.__name__: 1})
            @qp.qnode(qp.device("lightning.qubit", wires=1))
            def circuit():
                op_class([0])
                return qp.probs(wires=[0])
            return circuit

        report("WITH_RULE_USER", lambda: counts(build(L20WithRule)))
        report("WITH_RULE_DEVICE", lambda: counts(build(L20WithRule), level="device"))
        report("WITH_RULE_RUN", lambda: np.asarray(build(L20WithRule)()).tolist())
        report("NO_RULE_RUN", lambda: np.asarray(build(L20NoRule)()).tolist())
        """
    )
    assert outcome(result, "WITH_RULE_USER") == 'OK {"L20WithRule": "1"}'
    assert outcome(result, "WITH_RULE_DEVICE") == 'OK {"PauliX": "1"}'
    assert outcome(result, "WITH_RULE_RUN") == "OK [0.0, 1.0]"
    assert outcome(result, "NO_RULE_RUN").startswith("ERROR CompileError")
