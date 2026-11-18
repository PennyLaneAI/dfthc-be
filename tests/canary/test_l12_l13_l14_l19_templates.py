"""Canaries L-12, L-13, L-14 and L-19: PennyLane templates (spec section 14)."""

import json

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import ANALYZE_NAMES
from dfthc.testing.canary import TOFFOLI_GATE_SET, outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]

_BITS = np.random.default_rng(3).integers(0, 2, size=(16, 1))


def _dirty_qrom():
    qp.QROM(_BITS, control_wires=[0, 1, 2, 3], target_wires=[4], work_wires=[5, 6, 7], clean=False)


def test_select_swap_qrom_twice_is_not_identity():
    """L-12: the select-swap QROM (chosen for this dirty QROM) is not self-inverse."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=TOFFOLI_GATE_SET)
    @qp.qnode(qp.device("lightning.qubit", wires=8))
    def circuit():
        for wire in range(4):
            qp.H(wire)
        _dirty_qrom()
        _dirty_qrom()
        return qp.state()

    state = np.asarray(circuit())
    initial = np.zeros(256, dtype=complex)
    initial[[k << 4 for k in range(16)]] = 0.25
    assert abs(np.vdot(initial, state) - 0.8125) < 1e-9


def test_select_swap_qrom_counts_are_symbolic():
    """L-12 and L-22: qp.analyze counts of the select-swap QROM are symbolic."""

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.decompose(gate_set=TOFFOLI_GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=8))
    def circuit():
        _dirty_qrom()
        return qp.probs(wires=[0])

    assert qp.analyze(circuit, level="user")().resources.is_symbolic


@pytest.mark.subprocess
def test_select_with_shared_target_crashes():
    """L-13: Select whose operators share a target wire crashes the compiler."""
    result = run_snippet(
        """
        @qp.qjit(capture=True, enable_aot_compilation=False)
        @qp.decompose(gate_set=TOFFOLI_GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=5))
        def circuit():
            qp.Select([qp.X(4), qp.Z(4), qp.Y(4), qp.H(4)], control=[0, 1], work_wires=[2, 3])
            return qp.probs(wires=[0])

        report("SELECT", lambda: counts(circuit))
        """
    )
    select = outcome(result, "SELECT")
    assert select.startswith("ERROR CompileError")
    assert "qubit operands must refer to distinct qubits" in select


@pytest.mark.subprocess
def test_mcx_without_work_wires_has_no_path():
    """L-14: MultiControlledX with four controls and no work wires has no path."""
    result = run_snippet(
        """
        @qp.qjit(capture=True, enable_aot_compilation=False)
        @qp.decompose(gate_set=TOFFOLI_GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=5))
        def circuit():
            qp.MultiControlledX(wires=[0, 1, 2, 3, 4])
            return qp.probs(wires=[0])

        report("MCX", lambda: counts(circuit))
        """
    )
    mcx = outcome(result, "MCX")
    assert mcx.startswith("ERROR CompileError")
    assert "Decomposition rule not found for operator 'MultiControlledX" in mcx


@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("label", "statement"),
    [
        ("UNIFORM_PREP", "qp.UniformPrep(5, target_wires=[0, 1, 2], work_wires=[3, 4, 5])"),
        ("ALIAS_SAMPLING", "qp.AliasSampling((0.1, 0.2, 0.3, 0.2, 0.2), 3, target_wires=[0, 1, 2], "
                           "temp_wires=list(range(3, 15)), work_wires=list(range(15, 21)))"),
    ],
)
def test_template_has_no_toffoli_level_path(label, statement):
    """L-19: for a non-power-of-two size L = 5, both need arbitrary rotations, so no path exists.

    The power-of-two case is ``test_power_of_two_alias_sampling_compiles``.
    """
    result = run_snippet(
        f"""
        @qp.qjit(capture=True, enable_aot_compilation=False)
        @qp.decompose(gate_set=TOFFOLI_GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=21))
        def circuit():
            {statement}
            return qp.probs(wires=[0])

        report("{label}", lambda: counts(circuit))
        """
    )
    template = outcome(result, label)
    assert template.startswith("ERROR CompileError")
    assert "Decomposition rule not found for operator" in template


@pytest.mark.subprocess
def test_power_of_two_alias_sampling_compiles():
    """L-19, changed: for a power-of-two size (L = 4) AliasSampling compiles to the gate set
    (observed 2026-11-11), because its uniform preparation is then Hadamards only."""
    result = run_snippet(
        """
        @qp.qjit(capture=True, enable_aot_compilation=False)
        @qp.decompose(gate_set=TOFFOLI_GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=19))
        def circuit():
            qp.AliasSampling((0.1, 0.2, 0.3, 0.4), 3, target_wires=[0, 1],
                             temp_wires=list(range(2, 13)), work_wires=list(range(13, 19)))
            return qp.probs(wires=[0])

        report("ALIAS_SAMPLING_4", lambda: counts(circuit))
        """
    )
    alias = outcome(result, "ALIAS_SAMPLING_4")
    assert alias.startswith("OK ")
    assert set(json.loads(alias.removeprefix("OK "))) <= ANALYZE_NAMES | {"Toffoli"}
