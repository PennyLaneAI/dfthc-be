# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""QROAMRead (spec 9.3): every computational address, no phase, counts, helpers."""

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from pennylane.templates.subroutines.qrom import _qrom_unary_iteration
from pennylane.typing import AbstractArray

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, PENNYLANE_PINS, compiled_resources
from dfthc.ops.qroam import QROAMRead, _qroam_read
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, lg
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation


def random_table(widths, live_ranges, width, seed=7):
    """An entry of ``width`` random bits at every live address, as 32-bit words."""
    rng = np.random.default_rng(seed)
    (lo1, hi1), *rest = live_ranges
    rows = [(v,) for v in range(lo1, hi1)]
    if rest:
        rows = [(v, u) for v in range(lo1, hi1) for u in range(*rest[0])]
    table = []
    for values in rows:
        address = values[0] if len(values) == 1 else (values[0] << widths[1]) | values[1]
        value = int(rng.integers(0, 2**width))
        table.append((address, tuple((value >> (32 * w)) & (2**32 - 1)
                                     for w in range(-(-width // 32)))))
    return tuple(table)


def outer_shape(X_o, k, width):
    """The outer read: x_o over [0, X_o) on lg(X_o) bits."""
    return (lg(X_o),), ((0, X_o),), width, k


def inner_shape(RC, X_o, B1, k, width):
    """The inner read: rows x_o in [0, RC) of lg(X_o) bits, then b in [0, B + 1)."""
    return (lg(X_o), lg(B1)), ((0, RC), (0, B1)), width, k


def layout(widths, live_ranges, width, k, control_value=None):
    sizes = QROAMRead.required_wires(widths, live_ranges, width, k, control_value)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def columns(table, widths, live_ranges, width, k):
    return QROAMRead.columns_from_table(table, address_widths=widths, live_ranges=live_ranges,
                                        width=width, block_bits=k)


def make_read(table, widths, live_ranges, width, k, control_value, wires):
    return QROAMRead(columns(table, widths, live_ranges, width, k), widths, live_ranges,
                     width, k, control_value, wires["address"], wires["target"], wires["junk"],
                     wires["work"], wires["control"])


def value_of(state, register, row):
    return sum(int(state.bits(w)[row]) << (len(register) - 1 - i) for i, w in enumerate(register))


# Exhaustive failed-address shapes (spec 12.1, T3 item 1): X_o = 5 (failed x_o = 5..7), RC = 2
# (one-body rows 2..4), and (B + 1, k2) as listed; widths w_b + 4 as d2 with b_k2 = 2, w_r = 0.
# (5, 1) and (7, 0) leave high values of b unvisited, so the flattened select x_o ‖ b_top has
# zero columns between the rows.
INNER = [inner_shape(2, 5, B1, k, lg(B1) + 4) for B1, k in [(3, 1), (5, 1), (6, 2), (7, 0),
                                                            (28, 4)]]
# Outer shapes: x_o over [2, 6) with k = 0, a nonzero lower live bound, so addresses 0 and 1 load
# nothing (zero columns 0 and 1 of the select); then x_o over [0, 5) with k = 1, with k = 2 and
# 37-bit entries, and with k = 3 (no select control wires).
OUTER = [((3,), ((2, 6),), 5, 0)] + [outer_shape(5, k, 37 if k == 2 else 5) for k in (1, 2, 3)]
OUTER_IDS = ["outer-nonzero-lo-k0", "outer-k1", "outer-k2-wide", "outer-k3"]


@pytest.mark.parametrize("shape", INNER + OUTER,
                         ids=[f"inner-{s[1][1][1]}-{s[3]}" for s in INNER] + OUTER_IDS)
def test_every_address_loads_loaded_bits(shape):
    """T3, exhaustive (sparse simulator): on H|0> over the whole address register, every
    computational address ends with target ‖ junk == loaded_bits, work in |0>, amplitude +1/sqrt N.
    Includes failed x_o >= X_o, failed b >= B + 1, one-body rows, dead block positions, a nonzero
    lower live bound (outer-nonzero-lo-k0), entries wider than 32 bits (outer-k2-wide), and
    (outer-k3) a fully low address, whose select has no control wires."""
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)

    def qfunc():
        for wire in wires["address"]:
            qp.H(wire)
        make_read(table, widths, live, width, k, None, wires)

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    size = 2 ** sum(widths)
    assert len(state.amps) == size
    assert np.allclose(state.amps, 1 / np.sqrt(size), rtol=0, atol=1e-12)
    contents = wires["target"] + wires["junk"]
    seen = set()
    for row in range(size):
        a = value_of(state, wires["address"], row)
        seen.add(a)
        expected = QROAMRead.loaded_bits(table, a, address_widths=widths, live_ranges=live,
                                         width=width, block_bits=k)
        assert value_of(state, contents, row) == expected, a
        assert value_of(state, wires["work"], row) == 0
    assert seen == set(range(size))


@pytest.mark.parametrize("control_value", [0, 1])
def test_enable_gates_the_read(control_value):
    """With the enable wire in |+>, only its control_value branch loads anything."""
    widths, live, width, k = INNER[2]
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k, control_value)

    def qfunc():
        for wire in wires["address"] + wires["control"]:
            qp.H(wire)
        make_read(table, widths, live, width, k, control_value, wires)

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    assert len(state.amps) == 2 ** (sum(widths) + 1)
    for row in range(len(state.amps)):
        a = value_of(state, wires["address"], row)
        on = value_of(state, wires["control"], row) == control_value
        expected = QROAMRead.loaded_bits(table, a, address_widths=widths, live_ranges=live,
                                         width=width, block_bits=k) if on else 0
        assert value_of(state, wires["target"] + wires["junk"], row) == expected
        assert value_of(state, wires["work"], row) == 0


def _lightning_state(num_wires, body):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps={**FIXED_DECOMPS, **PENNYLANE_PINS})
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return qp.state()

    return np.asarray(circuit())


def _loaded_state(table, widths, live, width, k, wires, num_wires):
    """sum_a |a>|loaded_bits(a)>|0> / sqrt(N) as a dense vector."""
    state = np.zeros(2**num_wires, dtype=complex)
    n = sum(widths)
    rest = num_wires - n
    contents = len(wires["target"]) + len(wires["junk"])
    for a in range(2**n):
        loaded = QROAMRead.loaded_bits(table, a, address_widths=widths, live_ranges=live,
                                       width=width, block_bits=k)
        state[(a << rest) | (loaded << (rest - contents))] = 1 / np.sqrt(2**n)
    return state


@pytest.mark.compiled
@pytest.mark.parametrize("k", [1, 2, 3])
def test_compiled_read_on_lightning(k):
    """T3 on the compiled path: the exact output state, so the read adds no phase. At k = 3 the
    whole 3-bit address is low: the select has no control wire and loads its one column with
    conditional X gates; there T4 also holds, with the analyzed X count equal to QROM's declared
    upper bound 2**k * width plus four per swap (the conditional control flips of each swap's
    MultiControlledX, which qp.analyze counts as taken)."""
    widths, live, width, _ = outer_shape(5, k, 2)
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)

    def body():
        for wire in wires["address"]:
            qp.H(wire)
        make_read(table, widths, live, width, k, None, wires)

    expected = _loaded_state(table, widths, live, width, k, wires, num_wires)
    assert np.allclose(_lightning_state(num_wires, body), expected, rtol=0, atol=1e-10)
    if k == 3:
        resources = _assert_compiled_counts((widths, live, width, k))
        assert resources.counts["PauliX"] == 2**k * width + 4 * (2**k - 1) * width


# Runtime columns: three columns of K = 4 positions (dead positions 11, unvisited 12 to 15).
RUNTIME = ((4,), ((0, 11),), 3, 2)


@pytest.mark.compiled
def test_compiled_read_with_runtime_columns():
    """One lightning qjit, the address and the columns as runtime arguments, every address of
    the domain: target ‖ junk equals loaded_bits. A bit-order and a position-order mutant of the
    columns both fail, so the column layout is the one the docstring states."""
    widths, live, width, k = RUNTIME
    table = random_table(widths, live, width, seed=40)
    data = columns(table, widths, live, width, k)
    assert data.shape == (3, 12) and data.dtype == bool
    wires, num_wires = layout(widths, live, width, k)
    n = sum(widths)
    contents = wires["target"] + wires["junk"]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps={**FIXED_DECOMPS, **PENNYLANE_PINS})
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(a, array):
        for position, wire in enumerate(wires["address"]):
            qp.cond(((a >> (n - 1 - position)) & 1) == 1, qp.X)(wire)
        QROAMRead(array, widths, live, width, k, None, wires["address"], wires["target"],
                  wires["junk"], wires["work"], [])
        return qp.probs(wires=contents)

    def outcomes(array):
        return [int(np.argmax(circuit(a, jnp.asarray(array)))) for a in range(2**n)]

    expected = [QROAMRead.loaded_bits(table, a, address_widths=widths, live_ranges=live,
                                      width=width, block_bits=k) for a in range(2**n)]
    assert outcomes(data) == expected
    positions_reversed = data.reshape(3, 2**k, width)[:, ::-1].reshape(3, -1)
    for mutant in (data[:, ::-1], positions_reversed):
        assert outcomes(mutant) != expected


def preset_shapes(p):
    return [outer_shape(p.X_o, p.k1, p.d1), inner_shape(p.RC, p.X_o, p.B + 1, p.k2, p.d2)]


def _assert_compiled_counts(shape, control_value=None):
    """T, M and A exact against the closed form: QROM's ANDs over the select's columns plus one
    TemporaryAND per CSWAP (no Toffoli), every AND measured. Returns the analyzed resources."""
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k, control_value)
    resources = compiled_resources(
        lambda: make_read(table, widths, live, width, k, control_value, wires),
        num_wires=num_wires)
    num_columns, num_controls = QROAMRead.select_size(widths, live, k, control_value)
    expected = costs.qroam_read(num_columns, num_controls, k, width)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli
    assert (resources.counts.get("Toffoli", 0), resources.counts.get("TemporaryAND", 0)) == (
        0, expected.toffoli)
    return resources


@pytest.mark.compiled
@pytest.mark.parametrize("shape", preset_shapes(H2_STO3G_TEST) + preset_shapes(H2_631G_TEST),
                         ids=["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner"])
def test_toffoli_count_at_the_test_presets(shape):
    """T4 at H2_STO3G_TEST and H2_631G_TEST: T = N(c, columns) + (2^k - 1) width; A = N."""
    _assert_compiled_counts(shape)


@pytest.mark.compiled
@pytest.mark.parametrize(("shape", "control_value"), [
    (INNER[1], None), (INNER[3], None), (OUTER[0], None), (OUTER[3], None), (INNER[2], 0),
    (INNER[2], 1)], ids=["inner-5-1-gaps", "inner-7-0-gaps", "outer-nonzero-lo", "outer-k3",
                         "enable0", "enable1"])
def test_toffoli_count_on_padded_selects_and_with_enable(shape, control_value):
    """T4 where the select has zero columns (unvisited b values between rows, a nonzero lower
    bound), no control wire, or an enable as its extra most significant control wire."""
    _assert_compiled_counts(shape, control_value)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("shape", preset_shapes(MID_NAPHTHALENE), ids=["outer", "inner"])
def test_toffoli_count_at_mid_naphthalene(shape):
    """T4 at MID_NAPHTHALENE (nightly tier)."""
    _assert_compiled_counts(shape)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 is asserted on the closed forms; no FeMoco table is compiled (12.9). The outer
    select runs over 81 columns of 4 x 24 bits on 7 control wires, the inner one over the 540
    columns of x_o ‖ b_top (contiguous: K2 = 16 >= 2**(w_b - 1)) of 16 x 26 bits on 10 wires,
    one AND each fewer than nested exact iteration (84 and 273 + 270 = 543). The loaded ranges
    of the outer and inner reads are the covered sets of their erases (Tasks 1C.4 and 1C.5); the
    outer read's end 324 is block-aligned."""
    p = FEMOCO54
    outer, inner = preset_shapes(p)
    assert QROAMRead.select_size(*outer[:2], outer[3], None) == (81, 7)
    assert QROAMRead.select_size(*inner[:2], inner[3], None) == (540, 10)
    assert QROAMRead.columns_shape(*outer) == (81, 96)
    assert QROAMRead.columns_shape(*inner) == (540, 416)
    assert costs.qroam_read(81, 7, p.k1, p.d1).toffoli == 83 + 72 == 155
    assert costs.qroam_read(540, 10, p.k2, p.d2).toffoli == 542 + 390 == 932
    assert QROAMRead.required_wires(*outer[:2], p.d1, p.k1, None) == {
        "address": 9, "target": 24, "junk": 72, "work": 6, "control": 0}
    assert QROAMRead.required_wires(*inner[:2], p.d2, p.k2, None) == {
        "address": 14, "target": 26, "junk": 390, "work": 9, "control": 0}
    assert QROAMRead.loaded_ranges(*outer[:2], outer[3]) == ((0, 324),)
    assert QROAMRead.loaded_ranges(*inner[:2], inner[3]) == ((0, 270), (0, 32))


@pytest.mark.compiled
def test_select_is_pinned_to_the_unary_rule(capfd):
    """T6: with one-bit entries the select's c - 1 = 3 work wires meet the condition of QROM's
    select-swap rule (depth 4), so without a pin the solver would choose by cost; the pin in
    PENNYLANE_PINS fixes the unary-iteration rule, whose counts are concrete (L-12)."""
    shape = ((4,), ((0, 16),), 1, 0)
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)
    assert PENNYLANE_PINS == {"QROM": _qrom_unary_iteration}
    qp.analyze(verbose_program(lambda: make_read(table, widths, live, width, k, None, wires),
                               num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_qroam_read", "_qrom_unary_iteration"} <= set(
        chosen_rules(captured.out + captured.err))
    assert "_select_swap" not in chosen_rules(captured.out + captured.err)
    _assert_compiled_counts(shape)


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    widths, live, width, k = INNER[0]
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)
    compiled_resources(lambda: make_read(table, widths, live, width, k, None, wires),
                       num_wires=num_wires, level="device")


@pytest.mark.parametrize(("shape", "control_value"), [(OUTER[1], None), (INNER[2], 0),
                                                    (INNER[2], 1)])
def test_rule_is_valid(shape, control_value):
    """T2: the rule emits exactly its declared QROM, CSWAPs and enable X gates."""
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, _ = layout(widths, live, width, k, control_value)
    qp.testing.assert_valid_decomp_rule(make_read(table, widths, live, width, k, control_value,
                                                  wires), _qroam_read, check_matrix=False)


def test_junk_layout():
    assert [QROAMRead.junk_layout(2, low) for low in range(4)] == [
        [1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]]
    assert QROAMRead.junk_layout(0, 0) == []


def test_loaded_bits_by_hand():
    """X_o = 5, k = 1, width 2: blocks {0,1}, {2,3}, {4,5 (dead)}; addresses 6, 7 not visited."""
    table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
    args = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
    loaded = [QROAMRead.loaded_bits(table, a, **args) for a in range(8)]
    assert loaded == [(1 << 2) | 2, (2 << 2) | 1, (3 << 2) | 1, (1 << 2) | 3,
                      (2 << 2) | 0, (0 << 2) | 2, 0, 0]


def test_columns_by_hand():
    """Column h holds block h's positions in order, each entry big-endian (spec 6.7 on the
    wires: bit i of a position is entry bit width - 1 - i); dead positions and unvisited columns
    are zero. X_o = 5, k = 1, width 2: entries 1, 2, 3, 1, 2; address 5 is dead. Two registers
    with b over [0, 3) on 2 bits, k = 1: column (x_o << 1) | b_top, and column 3 (x_o = 1,
    b_top = 1) holds only the live address 6."""
    table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
    args = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
    data = QROAMRead.columns_from_table(table, **args)
    assert data.dtype == bool
    assert data.astype(int).tolist() == [[0, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 0]]
    two = {"address_widths": (1, 2), "live_ranges": ((0, 2), (0, 3)), "width": 1,
           "block_bits": 1}
    table = tuple((a, (1,)) for a in (0, 1, 2, 4, 5, 6))
    assert QROAMRead.columns_from_table(table, **two).astype(int).tolist() == [
        [1, 1], [1, 0], [1, 1], [1, 0]]


def test_select_sizes():
    """The select runs over the high address bits (plus the enable) and over every column up to
    the last visited block: zero columns fill a nonzero lower bound and unvisited high values of
    an inner register."""
    assert QROAMRead.select_size((3,), ((0, 5),), 1, None) == (3, 2)
    assert QROAMRead.select_size((3,), ((0, 5),), 1, 0) == (3, 3)
    assert QROAMRead.select_size((3,), ((0, 5),), 3, None) == (1, 0)
    assert QROAMRead.select_size((3,), ((2, 6),), 0, None) == (6, 3)
    assert QROAMRead.select_size((3, 3), ((0, 2), (0, 5)), 3, None) == (2, 3)
    assert QROAMRead.select_size((3, 3), ((0, 2), (0, 5)), 1, None) == (7, 5)
    assert QROAMRead.select_size((3, 3), ((1, 3), (0, 5)), 2, None) == (6, 4)
    assert QROAMRead.columns_shape((3, 3), ((1, 3), (0, 5)), 33, 2) == (6, 132)


def test_loaded_sets():
    assert QROAMRead.loaded_ranges((3, 3), ((0, 2), (0, 5)), 2) == ((0, 2), (0, 8))
    assert QROAMRead.loaded_set((3,), ((0, 5),), 1) == [0, 1, 2, 3, 4, 5]
    assert QROAMRead.loaded_set((2, 2), ((1, 2), (0, 3)), 1) == [4, 5, 6, 7]
    # A split register with a nonzero lower bound, and a block-aligned upper end.
    assert QROAMRead.loaded_ranges((3,), ((3, 7),), 1) == ((2, 8),)
    assert QROAMRead.loaded_set((2,), ((0, 4),), 1) == [0, 1, 2, 3]


@pytest.mark.parametrize(("table", "match"), [
    (((5, (1,)),), "outside the live ranges"),
    (((0, (4,)),), "does not fit in 2 bits"),
    (((0, (1, 0)),), "must be 1 words"),
    (((0, (1,)), (0, (2,))), "appears twice"),
])
def test_columns_from_table_validates(table, match):
    with pytest.raises(ValueError, match=match):
        QROAMRead.columns_from_table(table, address_widths=(3,), live_ranges=((0, 5),),
                                     width=2, block_bits=1)


def _columns(*ones):
    """Valid columns of the X_o = 5, k = 1, width-2 read, zero but at the (column, bit) ones."""
    data = np.zeros((3, 4), dtype=bool)
    for column, bit in ones:
        data[column, bit] = True
    return data


def _read_args(**changes):
    """Valid QROAMRead arguments of the X_o = 5, k = 1, width-2 read, with ``changes``."""
    return {"columns": _columns(), "address_widths": (3,), "live_ranges": ((0, 5),), "width": 2,
            "block_bits": 1, "control_value": None, "address": [0, 1, 2], "target": [3, 4],
            "junk": [5, 6], "work": [7], "control": []} | changes


@pytest.mark.parametrize(("change", "match"), [
    ({"block_bits": 4}, r"block_bits must lie in \[0, 3\]"),
    ({"address_widths": (0, 3), "live_ranges": ((0, 1), (0, 5))}, "nonempty first register"),
    ({"live_ranges": ((0, 9),)}, "not a nonempty range of 3-bit values"),
    ({"control_value": 2}, r"control_value must lie in \[0, 1\]"),
    ({"junk": [9]}, "'junk' has 1 wires; expected 2"),
    ({"columns": np.zeros((2, 4), dtype=bool)}, r"shape \(3, 4\)"),
    ({"columns": AbstractArray((3, 2), bool)}, r"shape \(3, 4\)"),
    ({"columns": np.zeros((3, 4), dtype=np.int64)}, "bool array"),
    ({"columns": AbstractArray((3, 4), np.int64)}, "bool array"),
    ({"columns": _columns((2, 3))}, "dead position"),
])
def test_constructor_validates(change, match):
    with pytest.raises(ValueError, match=match):
        QROAMRead(**_read_args(**change))
