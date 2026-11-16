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


def reflection_diagonal(n_w, one):
    """Diagonal of 2 Pi - I for the pattern with ones on the wires ``one`` (wire 0 is the most
    significant bit of the pattern's integer): +1 at the pattern, -1 elsewhere."""
    pattern = sum(2 ** (n_w - 1 - w) for w in one)
    return np.where(np.arange(2**n_w) == pattern, 1.0, -1.0)


# (zero wires, one wires) over system wires 0 .. n_w - 1, for n_w = 1, 2, 3, 6. Most cases are
# all zeros ("zeros") or all zeros with a one on the last wire, the MCZ target ("last-one");
# "mixed-011" has a one-valued wire (1) among the MCZ controls next to a zero-valued control (0).
PATTERN_CASES = [
    pytest.param(1, [0], [], id="1-zeros"),
    pytest.param(1, [], [0], id="1-last-one"),
    pytest.param(2, [0, 1], [], id="2-zeros"),
    pytest.param(2, [0], [1], id="2-last-one"),
    pytest.param(3, [0, 1, 2], [], id="3-zeros"),
    pytest.param(3, [0], [1, 2], id="3-mixed-011"),
    pytest.param(6, [0, 1, 2, 3, 4, 5], [], id="6-zeros"),
    pytest.param(6, [0, 1, 2, 3, 4], [5], id="6-last-one"),
]


@pytest.mark.compiled
@pytest.mark.parametrize(("n_w", "zero", "one"), PATTERN_CASES)
def test_exact_matrix_with_global_sign(n_w, zero, one):
    """T3: the compiled matrix equals 2 Pi - I exactly, global sign included, for patterns with
    the target wire 0 or 1 and, at n_w = 3, a one-valued wire among the MCZ controls."""
    work = ZeroReflection.required_wires(len(zero), len(one))["work"]

    def apply(system, work_wires):
        ZeroReflection([system[w] for w in zero], [system[w] for w in one], work_wires)

    matrix = choi_matrix(n_w, apply, work)
    assert np.allclose(matrix, np.diag(reflection_diagonal(n_w, one)), rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("n_w", "zero", "one"), PATTERN_CASES)
def test_exact_controlled_matrix(n_w, zero, one):
    """T3: C(ZeroReflection) equals |0><0| (x) I + |1><1| (x) (2 Pi - I) on the patterns of the
    plain test; a wrong global sign of the uncontrolled reflection would show here as a relative
    phase."""
    work = ZeroReflection.required_wires(len(zero), len(one), controlled=True)["work"]

    def apply(system, work_wires):
        control, pattern = system[0], system[1:]
        qp.ctrl(ZeroReflection([pattern[w] for w in zero], [pattern[w] for w in one],
                               work_wires), control=control)

    matrix = choi_matrix(n_w + 1, apply, work)
    expected = np.concatenate([np.ones(2**n_w), reflection_diagonal(n_w, one)])
    assert np.allclose(matrix, np.diag(expected), rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("n_w", "zero", "one"), [
    pytest.param(1, [0], [], id="1-zeros"), pytest.param(1, [], [0], id="1-last-one"),
    pytest.param(3, [0], [1, 2], id="3-mixed-011"), pytest.param(6, [0, 1, 2, 3, 4], [5],
                                                                 id="6-last-one")])
def test_exact_controlled_matrix_for_control_value_0(n_w, zero, one):
    """T3 (final review I-1): C(ZeroReflection) with control value 0 equals
    |0><0| (x) (2 Pi - I) + |1><1| (x) I, global sign included, on the compiled path, where the
    rule sees the control value only as a dynamic argument; control value 1 is the test above."""
    work = ZeroReflection.required_wires(len(zero), len(one), controlled=True)["work"]

    def apply(system, work_wires):
        control, pattern = system[0], system[1:]
        qp.ctrl(ZeroReflection([pattern[w] for w in zero], [pattern[w] for w in one],
                               work_wires), control=control, control_values=[0])

    matrix = choi_matrix(n_w + 1, apply, work)
    expected = np.concatenate([reflection_diagonal(n_w, one), np.ones(2**n_w)])
    assert np.allclose(matrix, np.diag(expected), rtol=0, atol=1e-10)


@pytest.mark.compiled
def test_adjoint_is_the_reflection_itself():
    zero, one = [0], [1, 2]  # the mixed pattern 011

    def apply(system, work_wires):
        qp.adjoint(ZeroReflection([system[w] for w in zero], [system[w] for w in one],
                                  work_wires))

    assert np.allclose(choi_matrix(3, apply, 0), np.diag(reflection_diagonal(3, one)), rtol=0,
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
@pytest.mark.parametrize("n_w", [1, 2, 3, 6])
def test_toffoli_count_for_control_value_0(n_w):
    """T4 (final review I-1): control value 0 costs what control value 1 costs, T = n_w - 1 and
    M = 0, for both pattern values of the target wire; it adds only Clifford and phase gates."""
    work = list(range(n_w, n_w + ZeroReflection.required_wires(n_w, 0, True)["work"]))
    control = n_w + len(work)

    def program():
        for target_value in (0, 1):
            zero, one = split(n_w, target_value)
            qp.ctrl(ZeroReflection(zero, one, work), control=control, control_values=[0])

    resources = compiled_resources(program, num_wires=control + 1)
    expected = 2 * costs.zero_reflection(n_w, True)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli - 2 * int(n_w >= 2)


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
    """T2 for the base, the adjoint and the controlled rule, for both control values (tape
    mode)."""
    op = ZeroReflection(zero, one, work + [9])
    qp.testing.assert_valid_decomp_rule(op, _zero_reflection, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_zero_reflection,
                                        check_matrix=False)
    for value in (1, 0):
        qp.testing.assert_valid_decomp_rule(qp.ctrl(op, control=10, control_values=[value]),
                                            _c_zero_reflection, check_matrix=False)


def test_controlled_rule_needs_one_more_work_wire():
    op = ZeroReflection([0, 1, 2], [3], [4])
    with pytest.raises(ValueError, match="C\\(ZeroReflection\\) needs 2 work wires, got 1"):
        qp.testing.assert_valid_decomp_rule(qp.ctrl(op, control=10), _c_zero_reflection,
                                            check_matrix=False)


def test_controlled_rule_refuses_two_controls():
    """The pinned C(ZeroReflection) rule builds one control wire, of either control value; two
    controls would compile silently to another operation (final review I-1), so it raises."""
    op = ZeroReflection([0, 1, 2], [3], [4, 5])
    with pytest.raises(ValueError, match="exactly one control wire; got 2"):
        qp.testing.assert_valid_decomp_rule(qp.ctrl(op, control=[10, 11]), _c_zero_reflection,
                                            check_matrix=False)


def test_required_wires_and_validation():
    assert ZeroReflection.required_wires(46, 0) == {"zero_wires": 46, "one_wires": 0, "work": 43}
    assert ZeroReflection.required_wires(5, 1, controlled=True)["work"] == 4
    assert ZeroReflection.required_wires(0, 1)["work"] == 0
    with pytest.raises(ValueError, match="n_w >= 1"):
        ZeroReflection([], [], [])
    with pytest.raises(ValueError, match="'work' has 1 wires; expected at least 2"):
        ZeroReflection([0, 1, 2, 3], [4], [5])
