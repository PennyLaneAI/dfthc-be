"""Canaries L-15 to L-17: measurement bugs, device-level decomposition, qp.analyze behaviour.

Each test pins today's behaviour (spec section 14). Two L-15 claims behave differently on
2026-11-11 than when the spec was written; the docstrings and docs/limitations.md say how.
"""

import warnings

import numpy as np
import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc.compile import GATE_SET
from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]


@pytest.mark.subprocess
def test_hw_like_postselection_wrong_under_capture():
    """L-15 (pennylane#8379): hardware-like one-shot postselection under capture.

    On 2026-11-11 qjit(capture=True) rejects postselect_mode="hw-like" outright.
    """
    result = run_snippet(
        """
        @qp.qjit(capture=True)
        @qp.set_shots(100)
        @qp.qnode(qp.device("lightning.qubit", wires=2), mcm_method="one-shot",
                  postselect_mode="hw-like")
        def circuit():
            qp.H(0)
            qp.measure(0, postselect=1)
            return qp.sample(wires=[0])

        report("HW_LIKE", lambda: np.asarray(circuit()).tolist())
        """
    )
    hw_like = outcome(result, "HW_LIKE")
    assert hw_like.startswith("ERROR CompileError")
    assert "hw-like is not supported" in hw_like


def test_mcm_inside_cond_missed():
    """L-15 (catalyst#2382): a measurement nested in qp.cond under one-shot execution.

    On 2026-11-11 the nested measurement is executed on every shot (mean about 1/2), so this
    reproducer no longer shows the bug; the spec still avoids measurements inside qp.cond.
    """

    @qp.qjit(capture=True, seed=2026)
    @qp.set_shots(400)
    @qp.qnode(qp.device("lightning.qubit", wires=3), mcm_method="one-shot")
    def circuit():
        qp.X(0)
        first = qp.measure(0)

        def nested():
            qp.H(1)
            second = qp.measure(1)
            qp.cond(second, qp.X)(2)

        qp.cond(first, nested)()
        return qp.sample(wires=[2])

    mean = float(np.mean(np.asarray(circuit())))
    assert abs(mean - 0.5) < 0.125  # five standard deviations at 400 shots


def test_pauli_measure_postselect_ignored():
    """L-15 (catalyst#3338): qp.pauli_measure(postselect=...) is ignored."""

    @qp.qjit(capture=True, seed=2026)
    @qp.set_shots(400)
    @qp.qnode(qp.device("lightning.qubit", wires=1), mcm_method="one-shot")
    def circuit():
        qp.H(0)
        qp.pauli_measure("Z", wires=[0], postselect=1)
        return qp.sample(wires=[0])

    mean = float(np.mean(np.asarray(circuit())))
    assert abs(mean - 0.5) < 0.125  # an honoured postselection would give 1.0


def test_device_pass_runs():
    """L-16 (catalyst#3275): executing on lightning.qubit decomposes again to device gates."""

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    def circuit():
        qp.TemporaryAND([0, 1, 2])
        return qp.probs(wires=[0])

    user = dict(qp.analyze(circuit, level="user")().resources.counts)
    device = dict(qp.analyze(circuit, level="device")().resources.counts)
    assert user == {"TemporaryAND": 1}
    assert "TemporaryAND" not in device
    assert device.get("Toffoli") == 1


class L16Op(Operator2):
    wire_argnames = ("wires",)

    def __init__(self, wires):
        super().__init__(wires)


@qp.register_resources({qp.X: 1, qp.CNOT: 1})
def _l16_rule(wires):
    qp.X(wires[0])
    qp.CNOT([wires[0], wires[1]])


qp.add_decomps(L16Op, _l16_rule)


def test_device_level_has_no_operator_op():
    """L-16: at level="device" a decomposed Operator2 leaves no named operation behind."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        L16Op([0, 1])
        return qp.probs(wires=[1])

    device = dict(qp.analyze(circuit, level="device")().resources.counts)
    assert "L16Op" not in device
    assert np.allclose(np.asarray(circuit()), [0.0, 1.0])


def test_analyze_requires_qjit():
    """L-17: qp.analyze accepts only qjit-compiled QNodes."""

    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        qp.X(0)
        return qp.probs(wires=[0])

    with pytest.raises(ValueError, match="can only be applied to a qjit'd QNode"):
        qp.analyze(circuit)()


def _branch_circuit(hinted):
    def true_fn():
        qp.X(0)
        qp.X(0)

    def false_fn():
        qp.Y(0)

    if hinted:
        true_fn = qp.hint({"branch-prob": 0.25})(true_fn)
        false_fn = qp.hint({"branch-prob": 0.75})(false_fn)

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(pred: bool):
        qp.cond(pred, true_fn, false_fn)()
        return qp.probs(wires=[0])

    return circuit


def test_unhinted_cond_counts_max_branch_with_warning():
    """L-17: an unhinted conditional counts its most expensive branch and warns."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        resources = qp.analyze(_branch_circuit(hinted=False))(True).resources
    assert dict(resources.counts) == {"PauliX": 2, "PauliY": 1}
    assert any("unable to determine the branch" in str(r.message) for r in records)


def test_branch_prob_hint_gives_expected_counts():
    """L-17 (pennylane#10230): hinted conditionals count their expected gates.

    The expected count of each gate type is summed over the program and rounded up once:
    four one-X conditionals at probability 1/2 count 2 PauliX, and the 1/4-versus-3/4 branch
    pair counts ceil(0.5) = 1 PauliX and ceil(0.75) = 1 PauliY.
    """
    pair = qp.analyze(_branch_circuit(hinted=True))(True).resources
    assert dict(pair.counts) == {"PauliX": 1, "PauliY": 1}

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def four_conditionals(first: bool, second: bool):
        for pred, wire in ((first, 0), (second, 1), (first, 0), (second, 1)):

            @qp.hint({"branch-prob": 0.5})
            def flip(wire=wire):
                qp.X(wire)

            qp.cond(pred, flip)()
        return qp.probs(wires=[0])

    four = qp.analyze(four_conditionals)(True, False).resources
    assert dict(four.counts) == {"PauliX": 2}


def _loop_circuit(hinted):
    def body(_):
        qp.Z(0)

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(n: int):
        loop = qp.for_loop(n)(body)
        if hinted:
            loop = qp.hint({"num-iters": 7})(loop)
        loop()
        return qp.probs(wires=[0])

    return circuit


def test_dynamic_loop_counts_are_symbolic():
    """L-17 and L-22: a loop with a runtime bound gives symbolic counts."""
    assert qp.analyze(_loop_circuit(hinted=False))(3).resources.is_symbolic


def test_num_iters_hint_makes_loop_concrete():
    """L-17: a num-iters hint makes the loop count concrete."""
    resources = qp.analyze(_loop_circuit(hinted=True))(3).resources
    assert not resources.is_symbolic
    assert dict(resources.counts) == {"PauliZ": 7}
