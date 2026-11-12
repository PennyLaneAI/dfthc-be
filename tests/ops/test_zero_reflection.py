"""ZeroReflection (spec 9.6): exact matrices with the global sign, controlled version, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.reflections import (
    ZeroReflection,
    _adjoint_zero_reflection,
    _c_zero_reflection,
    _zero_reflection,
)


def split(n_w, target_value):
    """Pattern wires 0 .. n_w - 1: zeros, then the last wire with ``target_value``."""
    wires = list(range(n_w))
    return (wires, []) if target_value == 0 else (wires[:-1], wires[-1:])


def choi_matrix(num_system, apply, num_work):
    """The matrix of ``apply`` on wires 0 .. num_system - 1, from one compiled lightning run.

    Each system wire is entangled with a reference wire (H, CNOT), so the output amplitude of
    |y>_system |x>_reference |0>_work is U[y, x] / sqrt(2^n); work must come back to |0>.
    """
    system = list(range(num_system))
    reference = list(range(num_system, 2 * num_system))
    work = list(range(2 * num_system, 2 * num_system + num_work))

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=2 * num_system + num_work))
    def circuit():
        for s, r in zip(system, reference, strict=True):
            qp.H(r)
            qp.CNOT([r, s])
        apply(system, work)
        return qp.state()

    state = np.asarray(circuit()).reshape(2**num_system, 2**num_system, 2**num_work)
    assert np.allclose(state[:, :, 1:], 0, rtol=0, atol=1e-10)
    return state[:, :, 0] * np.sqrt(2**num_system)


def reflection_diagonal(n_w, target_value):
    pattern = 1 if target_value == 1 else 0  # the integer of the pattern, wire 0 most significant
    return np.where(np.arange(2**n_w) == pattern, 1.0, -1.0)


@pytest.mark.compiled
@pytest.mark.parametrize("target_value", [0, 1])
@pytest.mark.parametrize("n_w", [1, 2, 3, 6])
def test_exact_matrix_with_global_sign(n_w, target_value):
    """T3: the compiled matrix equals 2 Pi - I exactly, global sign included."""
    zero, one = split(n_w, target_value)
    work = ZeroReflection.required_wires(len(zero), len(one))["work"]

    def apply(system, work_wires):
        ZeroReflection([system[w] for w in zero], [system[w] for w in one], work_wires)

    matrix = choi_matrix(n_w, apply, work)
    assert np.allclose(matrix, np.diag(reflection_diagonal(n_w, target_value)), rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize("target_value", [0, 1])
@pytest.mark.parametrize("n_w", [1, 2, 3, 6])
def test_exact_controlled_matrix(n_w, target_value):
    """T3: C(ZeroReflection) equals |0><0| (x) I + |1><1| (x) (2 Pi - I); a wrong global sign of
    the uncontrolled reflection would show here as a relative phase."""
    zero, one = split(n_w, target_value)
    work = ZeroReflection.required_wires(len(zero), len(one), controlled=True)["work"]

    def apply(system, work_wires):
        control, pattern = system[0], system[1:]
        qp.ctrl(ZeroReflection([pattern[w] for w in zero], [pattern[w] for w in one],
                               work_wires), control=control)

    matrix = choi_matrix(n_w + 1, apply, work)
    expected = np.concatenate([np.ones(2**n_w), reflection_diagonal(n_w, target_value)])
    assert np.allclose(matrix, np.diag(expected), rtol=0, atol=1e-10)


@pytest.mark.compiled
def test_adjoint_is_the_reflection_itself():
    zero, one = split(3, 1)

    def apply(system, work_wires):
        qp.adjoint(ZeroReflection([system[w] for w in zero], [system[w] for w in one],
                                  work_wires))

    assert np.allclose(choi_matrix(3, apply, 0), np.diag(reflection_diagonal(3, 1)), rtol=0,
                       atol=1e-10)


COUNT_CASES = [pytest.param(n_w, controlled, id=f"{n_w}-{'controlled' if controlled else 'plain'}")
               for n_w in (1, 2, 3, 6, 46) for controlled in (False, True)
               if (n_w, controlled) != (46, True)]
COUNT_CASES.append(pytest.param(46, True, id="46-controlled", marks=pytest.mark.slow))


@pytest.mark.compiled
@pytest.mark.parametrize(("n_w", "controlled"), COUNT_CASES)
def test_toffoli_count_matches_the_closed_form(n_w, controlled):
    """T4: T = max(n_w - 2, 0), controlled n_w - 1, for both pattern values of the target wire
    (one program applies both, so it counts twice); n_w = 46 is FeMoco's RefAB (table-free).
    The controlled 46-wire compile takes 8 to 18 s, so it runs in the nightly tier."""
    work = list(range(n_w, n_w + ZeroReflection.required_wires(n_w, 0, controlled)["work"]))
    control = n_w + len(work)

    def program():
        for target_value in (0, 1):
            zero, one = split(n_w, target_value)
            if controlled:
                qp.ctrl(ZeroReflection(zero, one, work), control=control)
            else:
                ZeroReflection(zero, one, work)

    resources = compiled_resources(program, num_wires=control + 1)
    expected = 2 * costs.zero_reflection(n_w, controlled)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.counts.get("Toffoli", 0) == 2 * int(n_w + controlled >= 3)


@pytest.mark.compiled
@pytest.mark.parametrize("controlled", [False, True], ids=["plain", "controlled"])
def test_no_dfthc_operator_survives_at_device_level(controlled):
    def program():
        op = ZeroReflection([0, 1, 2], [3], [4, 5])
        if controlled:
            qp.ctrl(op, control=6)

    compiled_resources(program, num_wires=7, level="device")


@pytest.mark.parametrize(("zero", "one", "work"), [([0], [], []), ([0, 1, 2], [3], [4])])
def test_rules_are_valid(zero, one, work):
    """T2 for the base, the adjoint and the controlled rule (tape mode)."""
    op = ZeroReflection(zero, one, work + [9])
    qp.testing.assert_valid_decomp_rule(op, _zero_reflection, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_zero_reflection,
                                        check_matrix=False)
    qp.testing.assert_valid_decomp_rule(qp.ctrl(op, control=10), _c_zero_reflection,
                                        check_matrix=False)


def test_controlled_rule_needs_one_more_work_wire():
    op = ZeroReflection([0, 1, 2], [3], [4])
    with pytest.raises(ValueError, match="C\\(ZeroReflection\\) needs 2 work wires, got 1"):
        qp.testing.assert_valid_decomp_rule(qp.ctrl(op, control=10), _c_zero_reflection,
                                            check_matrix=False)


def test_required_wires_and_validation():
    assert ZeroReflection.required_wires(46, 0) == {"zero_wires": 46, "one_wires": 0, "work": 43}
    assert ZeroReflection.required_wires(5, 1, controlled=True)["work"] == 4
    assert ZeroReflection.required_wires(0, 1)["work"] == 0
    with pytest.raises(ValueError, match="n_w >= 1"):
        ZeroReflection([], [], [])
    with pytest.raises(ValueError, match="'work' has 1 wires; expected at least 2"):
        ZeroReflection([0, 1, 2, 3], [4], [5])
