"""PhaseFixup and emit_phase_fixup (spec 9.4): exact phases on the covered set (risk K-2)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.qroam import PhaseFixup, QROAMRead, _phase_fixup, emit_phase_fixup
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator
from dfthc.testing.tape import decompose_for_simulation

# Covered sets of real reads (QROAMRead.loaded_ranges): the outer read X_o = 5, k = 1, and the
# inner read with rows [0, 2) of 3 bits and (B + 1, k2) = (3, 1).
OUTER = ((3,), QROAMRead.loaded_ranges((3,), ((0, 5),), 1))
INNER = ((3, 2), QROAMRead.loaded_ranges((3, 2), ((0, 2), (0, 3)), 1))


def layout(widths, covered, ob, control_value):
    sizes = PhaseFixup.required_wires(widths, covered, ob, control_value)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def make_fixup(parities, widths, covered, ob, control_value, wires):
    return PhaseFixup(parities, widths, covered, ob, control_value, wires["address"],
                      wires["onehot"], wires["work"], wires["control"])


def expected_state(widths, covered, control_value, parities, num_wires):
    """(-1)^f(a) on covered addresses with the enable on, over H|0> on address and control;
    the address wires come first, then the control wire."""
    order = PhaseFixup.address_order(widths, covered)
    phase = dict(zip(order, (-1) ** np.asarray(parities, dtype=int), strict=True))
    n, c = sum(widths), int(control_value is not None)
    state = np.zeros(2**num_wires, dtype=complex)
    for a in range(2**n):
        for bit in range(2**c):
            on = control_value is None or bit == control_value
            index = (a << (num_wires - n)) | (bit << (num_wires - n - 1))
            state[index] = (phase.get(a, 1) if on else 1) / np.sqrt(2 ** (n + c))
    return state


def tables(size, seed=4):
    """Random, all-ones and single-bit parity tables."""
    rng = np.random.default_rng(seed)
    single = np.zeros(size, dtype=bool)
    single[size // 2] = True
    return [rng.integers(0, 2, size).astype(bool), np.ones(size, dtype=bool), single]


CONFIGS = [(OUTER, 1, None), (OUTER, 2, 1), (OUTER, 0, None), (INNER, 3, None), (INNER, 2, 0),
           (INNER, 0, None)]
IDS = ["outer-ob1", "outer-ob2-enable1", "outer-ob0", "inner-ob3", "inner-ob2-enable0",
       "inner-ob0"]


def _move_control_first(wires):
    """Put the control wire right after the address, so expected_state can index it."""
    order = wires["address"] + wires["control"] + wires["onehot"] + wires["work"]
    relabel = {old: new for new, old in enumerate(order)}
    return {name: [relabel[w] for w in register] for name, register in wires.items()}


@pytest.mark.compiled
@pytest.mark.parametrize(("shape", "ob", "control_value"), CONFIGS, ids=IDS)
def test_fixup_applies_exactly_the_parity_table(shape, ob, control_value):
    """T3 (risk K-2): random, all-ones and single-bit tables give exactly (-1)^f(a) on the
    covered set and nothing elsewhere, on lightning.qubit; parities are a runtime argument, so
    one compile serves all three tables. The ob = 0 rows are the fallback of spec 9.4."""
    widths, covered = shape
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, ob, control_value)
    wires = _move_control_first(wires)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(parities):
        for wire in wires["address"] + wires["control"]:
            qp.H(wire)
        make_fixup(parities, widths, covered, ob, control_value, wires)
        return qp.state()

    for table in tables(size):
        state = np.asarray(circuit(jnp.asarray(table)))
        expected = expected_state(widths, covered, control_value, table, num_wires)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), table


def test_tape_path_with_measured_parities():
    """The tape path (sparse simulator): parities from deterministic measurements m0 = 1, m1 = 0,
    on the covered set x in [1, 3), b in [1, 4) (nonzero lower bounds, partly covered leaves).
    Three fixups on one address: ob = 3 signs the even positions of address_order; the whole
    address low (ob = 5) without an enable, whose leaf has no flag (a conditional Z on the one-hot
    wire), signs the positions with i % 4 == 1; the same with an enable of value 0, whose leaf
    flag is the control, signs the positions with i % 4 == 3 where the control is 0. A direct
    call of emit_phase_fixup with ob = 6, one bit beyond the address, raises."""
    widths, covered = (3, 2), ((1, 3), (1, 4))
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, 5, 0)
    work, source = [num_wires], [num_wires + 1, num_wires + 2]
    assert PhaseFixup.required_wires(widths, covered, 3, None)["work"] == 1 and not wires["work"]

    def qfunc():
        for wire in wires["address"] + wires["control"]:
            qp.H(wire)
        qp.X(source[0])
        measured = [qp.measure(source[0]), qp.measure(source[1])]
        make_fixup([measured[i % 2] for i in range(size)], widths, covered, 3, None,
                   dict(wires, onehot=wires["onehot"][:8], work=work, control=[]))
        make_fixup([measured[int(i % 4 != 1)] for i in range(size)], widths, covered, 5, None,
                   dict(wires, control=[]))
        make_fixup([measured[int(i % 4 != 3)] for i in range(size)], widths, covered, 5, 0, wires)

    tape = decompose_for_simulation(qfunc)
    simulator = SparseSimulator(num_wires + 3, policy=ForcedPolicy([1, 0]), strict=False)
    state = simulator.run(tape.operations)
    position = {a: i for i, a in enumerate(PhaseFixup.address_order(widths, covered))}
    assert len(state.amps) == 2 ** (sum(widths) + 1)
    for row, amp in enumerate(state.amps):
        a = sum(int(state.bits(w)[row]) << (4 - i) for i, w in enumerate(wires["address"]))
        c = int(state.bits(wires["control"][0])[row])
        i = position.get(a)
        flips = 0 if i is None else (i % 2 == 0) + (i % 4 == 1) + (i % 4 == 3 and c == 0)
        assert abs(amp - (-1) ** flips / np.sqrt(64)) < 1e-12, (a, c)
        assert not any(state.bits(w)[row] for w in wires["onehot"] + work)
    with pytest.raises(ValueError, match="onehot_bits must be in"):
        emit_phase_fixup([True] * size, address=wires["address"], address_widths=widths,
                         covered=covered, onehot=list(range(100, 164)), onehot_bits=6, work=work)


def erase_shapes(p):
    """The covered sets and one-hot bits of the outer and inner erases of preset p (spec 9.24)."""
    outer = ((p.w_x,), QROAMRead.loaded_ranges((p.w_x,), ((0, p.X_o),), p.k1), p.ke1)
    inner_widths = (p.w_x, p.w_b)
    inner = (inner_widths, QROAMRead.loaded_ranges(inner_widths, ((0, p.RC), (0, p.B + 1)),
                                                   p.k2), p.ke2 + p.w_b)
    return [outer, inner]


@pytest.mark.compiled
@pytest.mark.parametrize(("shape", "control_value"),
                         [(s, None) for s in erase_shapes(H2_STO3G_TEST) + erase_shapes(
                             H2_631G_TEST)] + [(erase_shapes(H2_631G_TEST)[1], 1)],
                         ids=["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner",
                              "631g-inner-enable"])
def test_toffoli_count_matches_the_closed_form(shape, control_value):
    """T4: T = U(covered high ranges, c) + O(ob); every AND is uncomputed by measurement."""
    _assert_compiled_counts(shape, control_value)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("shape", erase_shapes(MID_NAPHTHALENE), ids=["outer", "inner"])
def test_toffoli_count_at_mid_naphthalene(shape):
    """T4 at MID_NAPHTHALENE (nightly tier); compiled_resources rejects symbolic counts."""
    _assert_compiled_counts(shape, None)


def _assert_compiled_counts(shape, control_value):
    widths, covered, ob = shape
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, ob, control_value)
    resources = compiled_resources(
        lambda: make_fixup(np.ones(size, dtype=bool), widths, covered, ob, control_value, wires),
        num_wires=num_wires)
    expected = costs.phase_fixup(PhaseFixup.high_ranges(widths, covered, ob), ob,
                                 control_value is not None)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


def test_femoco_counts_on_the_closed_form():
    outer, inner = erase_shapes(FEMOCO54)
    assert PhaseFixup.high_ranges(*outer) == [(0, 21, 5)]
    assert PhaseFixup.high_ranges(*inner) == [(0, 68, 7)]
    assert costs.phase_fixup([(0, 21, 5)], 4, False).toffoli == 22 + 14
    assert costs.phase_fixup([(0, 68, 7)], 7, False).toffoli == 70 + 126


def _compile_recording(widths, covered, ob):
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, ob, None)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        resources = compiled_resources(
            lambda: make_fixup(np.ones(size, dtype=bool), widths, covered, ob, None, wires),
            num_wires=num_wires)
    unresolved = [r for r in records if "unable to determine the branch" in str(r.message)]
    return size, resources, unresolved


@pytest.mark.compiled
@pytest.mark.parametrize(("shape", "ob", "gate"), [(INNER, 3, "CZ"), (INNER, 0, "PauliZ")])
def test_erase_conditionals_are_hinted(shape, ob, gate):
    """Every conditional carries branch-prob 0.5, so the n conditional gates count n / 2 (L-17:
    expected counts, summed and rounded up once). With n even, a single unhinted conditional
    would count n / 2 + 1. qp.analyze's "unable to determine the branch" warning cannot serve
    here: it fires once per compile if any function of the module branches unhinted, and the
    rules of TemporaryAND that Catalyst lowers into every module with an AND do."""
    size, resources, _ = _compile_recording(*shape, ob)
    assert size % 2 == 0
    assert resources.counts[gate] == size // 2


@pytest.mark.compiled
def test_and_free_fixup_compiles_without_the_unresolved_branch_warning():
    """A fixup without any TemporaryAND (one high bit, uncontrolled, ob = 1): no warning."""
    size, resources, unresolved = _compile_recording((2,), ((0, 4),), 1)
    assert resources.toffoli == 0 and resources.counts["CZ"] == size // 2
    assert not unresolved


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    widths, covered = INNER
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, 3, None)
    compiled_resources(lambda: make_fixup(np.ones(size, dtype=bool), widths, covered, 3, None,
                                          wires), num_wires=num_wires, level="device")


def _measured(count):
    with qp.queuing.AnnotatedQueue():
        measured = [qp.measure(w) for w in range(100, 103)]
    return [measured[i % 3] for i in range(count)]


@pytest.mark.parametrize(("shape", "ob", "control_value"),
                         [(OUTER, 1, None), (INNER, 3, 0), (INNER, 0, None)])
def test_rule_is_valid_in_tape_mode(shape, ob, control_value):
    """T2 in tape mode, with MeasurementValue parities."""
    widths, covered = shape
    size = len(PhaseFixup.address_order(widths, covered))
    wires, _ = layout(widths, covered, ob, control_value)
    op = make_fixup(_measured(size), widths, covered, ob, control_value, wires)
    qp.testing.assert_valid_decomp_rule(op, _phase_fixup, check_matrix=False)


def test_tape_mode_requires_measurement_values():
    widths, covered = OUTER
    wires, _ = layout(widths, covered, 1, None)
    with pytest.raises(ValueError, match="every parity must be a MeasurementValue; entry 0"):
        make_fixup(np.ones(6, dtype=bool), widths, covered, 1, None, wires)


def test_address_order_and_high_ranges():
    assert PhaseFixup.address_order((3, 2), ((1, 3), (0, 2))) == [4, 5, 8, 9]
    assert PhaseFixup.high_ranges((3, 2), ((1, 3), (0, 2)), 1) == [(1, 3, 3), (0, 1, 1)]
    assert PhaseFixup.high_ranges((3, 2), ((1, 3), (0, 2)), 3) == [(0, 2, 2)]
    assert PhaseFixup.high_ranges((3,), ((0, 6),), 3) == [(0, 1, 0)]
    assert PhaseFixup.required_wires((3, 2), ((0, 2), (0, 4)), 3, 1) == {
        "address": 5, "onehot": 8, "work": 2, "control": 1}


@pytest.mark.parametrize(("change", "match"), [
    ({"parities": np.ones(5, dtype=bool)}, r"one parity per covered address \(6\), got 5"),
    ({"onehot_bits": 4}, r"onehot_bits must lie in \[0, 3\]"),
    ({"onehot": [3]}, "'onehot' has 1 wires; expected 2"),
])
def test_constructor_validates(change, match):
    args = {"parities": _measured(6), "address_widths": (3,), "covered": ((0, 6),),
            "onehot_bits": 1, "control_value": None, "address": [0, 1, 2], "onehot": [3, 4],
            "work": [5], "control": []} | change
    with pytest.raises(ValueError, match=match):
        PhaseFixup(**args)
