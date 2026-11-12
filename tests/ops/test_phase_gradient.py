"""PhaseGradientRotation (spec 9.1) and the phase-gradient sign convention (spec 6.5)."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.phase_gradient import PhaseGradientRotation, _phase_gradient_rotation


def _gradient(b):
    """PhaseGradientStatePrep's state on b wires: sum_y exp(-2 pi i y / 2^b) |y> / sqrt(2^b)."""
    return np.exp(-2j * np.pi * np.arange(2**b) / 2**b) / np.sqrt(2**b)


def _lightning_state(num_wires, body):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return qp.state()

    return np.asarray(circuit())


@pytest.mark.compiled
def test_semiadder_adding_one_kicks_back_the_positive_phase():
    """Spec 6.5: adding 1 into PennyLane's phase gradient multiplies it by exp(+2 pi i / 2^b)."""
    b = 4
    grad, x = [0, 1, 2, 3], [4, 5, 6, 7]

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        qp.X(x[-1])
        qp.SemiAdder(x, grad, [8, 9, 10])

    one = np.zeros(2**4)
    one[1] = 1.0
    expected = np.exp(2j * np.pi / 2**b) * np.kron(np.kron(_gradient(b), one), np.eye(8)[0])
    assert np.allclose(_lightning_state(11, body), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_rotation_applies_its_phase(m):
    """T3: k = 1 and k = 2^m - 1 on two |+> targets, against an (m+1)-wire phase gradient whose
    top m wires the rotation uses; exact state, so the sign of spec 6.5 is pinned."""
    b = m + 1
    t1, t2 = 0, 1
    grad = list(range(2, 2 + b))
    scratch = list(range(2 + b, 2 + b + m))
    work = list(range(2 + b + m, 2 + b + 2 * m - 1))
    num_wires = 2 + b + 2 * m - 1

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        qp.H(t1)
        qp.H(t2)
        PhaseGradientRotation(1, m, [t1], grad, scratch, work)
        PhaseGradientRotation(2**m - 1, m, [t2], grad, scratch, work)

    omega = np.exp(2j * np.pi / 2**m)
    targets = np.kron([1, omega], [1, omega ** (2**m - 1)]) / 2
    expected = np.kron(np.kron(targets, _gradient(b)), np.eye(2 ** (2 * m - 1))[0])
    assert np.allclose(_lightning_state(num_wires, body), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
def test_adjoint_applies_the_conjugate_phase():
    """The auto-derived adjoint subtracts: exp(-2 pi i k / 2^m) on |1>."""
    k, m = 3, 3
    grad, scratch, work = [1, 2, 3], [4, 5, 6], [7, 8]

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        qp.H(0)
        qp.adjoint(PhaseGradientRotation(k, m, [0], grad, scratch, work))

    target = np.array([1, np.exp(-2j * np.pi * k / 2**m)]) / np.sqrt(2)
    expected = np.kron(np.kron(target, _gradient(m)), np.eye(2**5)[0])
    assert np.allclose(_lightning_state(9, body), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("k", "m", "adjoint"),
                         [(1, 1, False), (5, 3, False), (5, 3, True), (31, 5, False),
                          (0, 4, False)])
def test_toffoli_count_matches_the_closed_form(k, m, adjoint):
    """T4: T = (m - 1)[k != 0] for the rotation and its adjoint; M = 0."""
    b = m

    def program():
        op = PhaseGradientRotation(k, m, [0], list(range(1, 1 + b)),
                                   list(range(1 + b, 1 + b + m)),
                                   list(range(1 + b + m, 2 * m + b)))
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=2 * m + b)
    expected = costs.phase_gradient_rotation(k, m)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    """T5 at level="device" (L-20)."""

    def program():
        PhaseGradientRotation(5, 3, [0], [1, 2, 3], [4, 5, 6], [7, 8])

    compiled_resources(program, num_wires=9, level="device")


@pytest.mark.parametrize(("k", "m"), [(1, 1), (6, 3)])
def test_rule_is_valid(k, m):
    """T2 at two sizes (tape mode)."""
    op = PhaseGradientRotation(k, m, [0], list(range(1, 1 + m)), list(range(1 + m, 1 + 2 * m)),
                               list(range(1 + 2 * m, 3 * m)))
    qp.testing.assert_valid_decomp_rule(op, _phase_gradient_rotation, check_matrix=False)


def test_required_wires():
    assert PhaseGradientRotation.required_wires(5, 3) == {
        "target": 1, "phase_grad": 3, "scratch": 3, "work": 2}


@pytest.mark.parametrize(
    ("k", "m", "grad", "scratch", "match"),
    [(8, 3, 3, 3, r"k must lie in \[0, 7\]"), (1, 0, 3, 0, r"m must lie in \[1, None\]"),
     (1, 3, 2, 3, "'phase_grad' has 2 wires; expected at least 3"),
     (1, 3, 3, 2, "'scratch' has 2 wires; expected 3"), (1.0, 3, 3, 3, "k must be an int")],
)
def test_constructor_validates(k, m, grad, scratch, match):
    with pytest.raises(ValueError, match=match):
        PhaseGradientRotation(k, m, [0], list(range(1, 1 + grad)),
                              list(range(10, 10 + scratch)), list(range(20, 20 + max(m - 1, 0))))
