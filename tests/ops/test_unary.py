"""Exact unary iteration (spec 9.3 and 9.24): flags on every address, counts, work.

``UnaryIterationProbe`` is a test operator whose rule copies every leaf flag of an exact
iteration onto its own wire, so the flags can be checked on every basis address.
"""

import itertools
import math

import numpy as np
import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources, register_rule
from dfthc.ops.unary import Emitter, as_resources, iterate, iterate_nested, work_required
from dfthc.testing.sparse_sim import SparseSimulator, SparseState


class UnaryIterationProbe(Operator2):
    """Copies the flag of leaf ``i`` (in iteration order) onto ``flags[i]``."""

    compilable_argnames = ("widths", "ranges", "control_value")
    wire_argnames = ("address", "flags", "work", "control")

    def __init__(self, widths, ranges, control_value, address, flags, work, control):
        super().__init__(tuple(widths), tuple(tuple(r) for r in ranges), control_value, address,
                         flags, work, control)


def _probe_body(emit, widths, ranges, control_value, address, flags, work, control):
    registers, start = [], 0
    for n, (lo, hi) in zip(widths, ranges, strict=True):
        registers.append((address[start:start + n], lo, hi))
        start += n
    leaves = iter(range(len(flags)))

    def leaf(flag, values):
        target = flags[next(leaves)]
        if flag is None:
            emit.x(target)
        else:
            emit.cnot(flag, target)

    ctrl = None if control_value is None else control[0]
    iterate_nested(registers, work, leaf, ctrl, 1 if control_value is None else control_value,
                   emit=emit)


def _probe_resources(widths, ranges, control_value, address, flags, work, control):
    counter = Emitter(counting=True)
    _probe_body(counter, widths, ranges, control_value, range(len(address)), range(len(flags)),
                range(len(work)), range(len(control)))
    return as_resources(counter.counts)


@qp.register_resources(_probe_resources)
def _unary_iteration_probe(widths, ranges, control_value, address, flags, work, control):
    _probe_body(Emitter(), widths, ranges, control_value, address, flags, work, control)


register_rule(UnaryIterationProbe, _unary_iteration_probe)


def _layout(widths, ranges, control_value):
    """Wire registers of a probe: address, flags, work, then the control."""
    num_flags = math.prod(hi - lo for lo, hi in ranges)
    sizes = [sum(widths), num_flags, work_required(widths, control_value is not None),
             int(control_value is not None)]
    starts = np.cumsum([0] + sizes)
    return [list(range(starts[i], starts[i + 1])) for i in range(4)], int(starts[-1])


def _expected_flags(widths, ranges, control_value, address_value, control_bit):
    """Index of the raised flag for one basis input, or None."""
    if control_value is not None and control_bit != control_value:
        return None
    values, shift = [], sum(widths)
    for n in widths:
        shift -= n
        values.append((address_value >> shift) & (2**n - 1))
    if not all(lo <= v < hi for v, (lo, hi) in zip(values, ranges, strict=True)):
        return None
    index = 0
    for v, (lo, hi) in zip(values, ranges, strict=True):
        index = index * (hi - lo) + (v - lo)
    return index


def _expected_state(widths, ranges, control_value):
    """The probe's output on H|0> on the address and the control, as a dense vector."""
    (address, flags, _, control), num_wires = _layout(widths, ranges, control_value)
    num_address = sum(widths)
    state = np.zeros(2**num_wires, dtype=complex)
    amplitude = 1 / np.sqrt(2 ** (num_address + len(control)))
    for a, c in itertools.product(range(2**num_address), range(2 ** len(control))):
        ones = [address[i] for i in range(num_address) if (a >> (num_address - 1 - i)) & 1]
        ones += control if c else []
        raised = _expected_flags(widths, ranges, control_value, a, c)
        if raised is not None:
            ones.append(flags[raised])
        state[sum(1 << (num_wires - 1 - w) for w in ones)] += amplitude
    return state


def _probe_program(widths, ranges, control_value):
    (address, flags, work, control), num_wires = _layout(widths, ranges, control_value)

    def program():
        for wire in address + control:
            qp.H(wire)
        UnaryIterationProbe(widths, ranges, control_value, address, flags, work, control)

    return program, num_wires


CASES = [
    ((3,), ((1, 7),), None),
    ((4,), ((5, 6),), 1),
    ((2, 2), ((1, 4), (0, 3)), 0),
]


def _simulate_iterate(n, lo, hi, control_value):
    """Record iterate() directly with integer wires and run it on H|0> in the sparse simulator."""
    address = list(range(n))
    flags = list(range(n, n + hi - lo))
    control = [] if control_value is None else [n + hi - lo]
    work = list(range(n + hi - lo + len(control),
                      n + hi - lo + len(control) + work_required([n], bool(control))))

    def leaf(flag, value):
        if flag is None:
            qp.X(flags[value - lo])
        else:
            qp.CNOT([flag, flags[value - lo]])

    with qp.queuing.AnnotatedQueue() as queue:
        for wire in address + control:
            qp.H(wire)
        iterate(address, lo, hi, work, leaf, control[0] if control else None,
                1 if control_value is None else control_value)
    num_wires = n + hi - lo + len(control) + len(work)
    final = SparseSimulator(num_wires).run(queue.queue, SparseState.basis(num_wires, 0))
    return final, address, flags, control, work


@pytest.mark.parametrize("control_value", [None, 0, 1])
def test_every_basis_address_raises_exactly_its_flag(control_value):
    """T3, exhaustive for n <= 4: every range [lo, hi), every address, sparse simulator."""
    for n in range(5):
        for lo, hi in itertools.combinations(range(2**n + 1), 2):
            final, address, flags, control, work = _simulate_iterate(n, lo, hi, control_value)
            assert len(final.amps) == 2 ** (n + len(control))
            assert np.allclose(final.amps, 1 / np.sqrt(len(final.amps)), rtol=0, atol=1e-12)
            for row in range(len(final.amps)):
                a = sum(int(final.bits(w)[row]) << (n - 1 - i) for i, w in enumerate(address))
                c = int(final.bits(control[0])[row]) if control else 0
                raised = [i for i, w in enumerate(flags) if final.bits(w)[row]]
                expected = _expected_flags((n,), ((lo, hi),), control_value, a, c)
                assert raised == ([] if expected is None else [expected]), (n, lo, hi, a, c)
                assert not any(final.bits(w)[row] for w in work), (n, lo, hi, a)


def test_counts_equal_the_closed_form_on_the_grid():
    """T4 without compiling: every range with n <= 6, controlled and not, against costs.unary."""
    for n in range(7):
        for (lo, hi), controlled in itertools.product(
                itertools.combinations(range(2**n + 1), 2), (False, True)):
            counter = Emitter(counting=True)
            iterate(range(n), lo, hi, range(n), lambda flag, value: None,
                    control=-1 if controlled else None, emit=counter)
            expected = costs.unary(lo, hi, n, controlled)
            assert counter.counts["TemporaryAND"] == expected, (lo, hi, n, controlled)
            assert counter.counts["Adjoint(TemporaryAND)"] == expected


def test_nested_counts_equal_the_closed_form():
    rng = np.random.default_rng(93)
    for _ in range(300):
        n1, n2 = int(rng.integers(1, 5)), int(rng.integers(0, 5))
        lo1, hi1 = sorted(int(v) for v in rng.choice(2**n1 + 1, 2, replace=False))
        lo2, hi2 = sorted(int(v) for v in rng.choice(2**n2 + 1, 2, replace=False))
        controlled = bool(rng.integers(2))
        counter = Emitter(counting=True)
        iterate_nested([(range(n1), lo1, hi1), (range(n1, n1 + n2), lo2, hi2)], range(12),
                       lambda flag, values: None, control=-1 if controlled else None,
                       emit=counter)
        expected = costs.unary_nested((lo1, hi1, n1), (lo2, hi2, n2), controlled)
        assert counter.counts["TemporaryAND"] == expected


def test_work_required():
    assert [work_required([n], False) for n in range(5)] == [0, 0, 1, 2, 3]
    assert [work_required([n], True) for n in range(5)] == [0, 1, 2, 3, 4]
    assert work_required([9, 1], False) == 8 + 1  # the FeMoco inner read: rows, then blocks
    assert work_required([0, 3], False) == 2 and work_required([0, 3], True) == 3
    assert work_required([4, 0], True) == 4


def test_invalid_arguments_are_rejected():
    def leaf(flag, value):
        return None

    with pytest.raises(ValueError, match="nonempty range"):
        iterate(range(3), 4, 4, range(3), leaf)
    with pytest.raises(ValueError, match="nonempty range"):
        iterate(range(3), 0, 9, range(3), leaf)
    with pytest.raises(ValueError, match="needs 3 work wires, got 2"):
        iterate(range(3), 0, 8, range(2), leaf, control=-1)
    with pytest.raises(ValueError, match="control_value must be 0 or 1"):
        iterate(range(3), 0, 8, range(3), leaf, control=-1, control_value=2)


@pytest.mark.compiled
@pytest.mark.parametrize(("widths", "ranges", "control_value"), CASES)
def test_compiled_flags_on_lightning(widths, ranges, control_value):
    """T3 on the compiled path: the probe's exact output state, including amplitudes."""
    program, num_wires = _probe_program(widths, ranges, control_value)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        program()
        return qp.state()

    expected = _expected_state(widths, ranges, control_value)
    assert np.allclose(np.asarray(circuit()), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("widths", "ranges", "control_value"), CASES)
def test_compiled_counts_equal_the_closed_form(widths, ranges, control_value):
    """T4: compiled T equals U (single) or U_2 (nested), and every AND is uncomputed measured."""
    program, num_wires = _probe_program(widths, ranges, control_value)
    resources = compiled_resources(program, num_wires=num_wires)
    triples = [(lo, hi, n) for n, (lo, hi) in zip(widths, ranges, strict=True)]
    controlled = control_value is not None
    expected = (costs.unary(*triples[0], controlled) if len(triples) == 1
                else costs.unary_nested(triples[0], triples[1], controlled))
    assert resources.toffoli == resources.measured_and_uncomputes == expected
    assert resources.measurements == 0


@pytest.mark.parametrize(("widths", "ranges", "control_value"), [CASES[0], CASES[2]])
def test_rule_is_valid(widths, ranges, control_value):
    """T2: declared resources equal the emitted operations (tape mode)."""
    (address, flags, work, control), _ = _layout(widths, ranges, control_value)
    op = UnaryIterationProbe(widths, ranges, control_value, address, flags, work, control)
    qp.testing.assert_valid_decomp_rule(op, _unary_iteration_probe, check_matrix=False)
