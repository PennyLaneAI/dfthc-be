"""QROAMRead (spec 9.3): every computational address, no phase, counts, helpers."""

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from pennylane.typing import AbstractArray

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.qroam import QROAMEmitter, QROAMRead, _qroam_read, _read_body
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, lg
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


def leaf_words(table, widths, live_ranges, width, k):
    return QROAMRead.leaf_words_from_table(table, address_widths=widths, live_ranges=live_ranges,
                                           width=width, block_bits=k)


def make_read(table, widths, live_ranges, width, k, control_value, wires):
    return QROAMRead(leaf_words(table, widths, live_ranges, width, k), widths, live_ranges,
                     width, k, control_value, wires["address"], wires["target"], wires["junk"],
                     wires["work"], wires["control"])


def value_of(state, register, row):
    return sum(int(state.bits(w)[row]) << (len(register) - 1 - i) for i, w in enumerate(register))


# Exhaustive failed-address shapes (spec 12.1, T3 item 1): X_o = 5 (failed x_o = 5..7), RC = 2
# (one-body rows 2..4), and (B + 1, k2) as listed; widths w_b + 4 as d2 with b_k2 = 2, w_r = 0.
INNER = [inner_shape(2, 5, B1, k, lg(B1) + 4) for B1, k in [(3, 1), (5, 1), (6, 2), (7, 0),
                                                            (28, 4)]]
OUTER = [outer_shape(5, k, 5) for k in (0, 1, 2, 3)]


@pytest.mark.parametrize("shape", INNER + OUTER,
                         ids=[f"inner-{s[1][1][1]}-{s[3]}" for s in INNER]
                         + [f"outer-k{s[3]}" for s in OUTER])
def test_every_address_loads_loaded_bits(shape):
    """T3, exhaustive (sparse simulator): on H|0> over the whole address register, every
    computational address ends with target ‖ junk == loaded_bits, work in |0>, amplitude +1/sqrt N.
    Includes failed x_o >= X_o, failed b >= B + 1, one-body rows, dead block positions, and
    (outer-k3) the flagless leaf of a fully low address."""
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
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
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
    whole 3-bit address is low: one flagless leaf, whose loads are conditional X gates; there T4
    also holds, with the analyzed X count equal to the declared upper bound 2**k * width."""
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
        resources, declared = _assert_compiled_counts((widths, live, width, k))
        assert resources.counts["PauliX"] == declared["PauliX"] == 2**k * width


# The compiled read check at FeMoco's inner block size and with two-word entries (Task 0.9d,
# Opus M-2): K = 16 positions of width 40 > 32, so every load loop crosses a word boundary.
# Its 640 target and junk wires are aliased onto ALIAS lightning wires: virtual wire j is
# physical wire j mod ALIAS, so each physical wire ends with the XOR of its virtual wires' bits,
# then permuted by the swap network. 7 divides no 2**b * 40, so every CSWAP acts on distinct
# wires. The rule is called directly, because an operator rejects repeated wires.
MULTIWORD = ((6,), ((0, 40),), 40, 4)
ALIAS = 7


def _aliased_expected(table, address):
    """The physical register after the read at ``address``, as a big-endian integer: the
    spec's loads (wire i of position p holds entry bit width - 1 - i) and swap network, with
    every virtual wire folded onto its physical one."""
    widths, live, width, k = MULTIWORD
    entries = {a: sum(w << (32 * i) for i, w in enumerate(words)) for a, words in table}
    bits = [0] * ALIAS
    block, low = address >> k, address & (2**k - 1)
    if block in QROAMRead.leaf_blocks(widths, live, k):
        for p in range(2**k):
            value = entries.get((block << k) | p, 0)
            for i in range(width):
                bits[(p * width + i) % ALIAS] ^= (value >> (width - 1 - i)) & 1
    for b in reversed(range(k)):
        if (low >> b) & 1:
            for r in range(2**b):
                for i in range(width):
                    x, y = (r * width + i) % ALIAS, ((r + 2**b) * width + i) % ALIAS
                    bits[x], bits[y] = bits[y], bits[x]
    return sum(bit << (ALIAS - 1 - j) for j, bit in enumerate(bits))


@pytest.mark.compiled
def test_compiled_multiword_read_with_sixteen_positions():
    """One lightning qjit, the address and leaf_words as runtime arguments, every address of
    the domain (three leaves, dead positions 40 to 47, unvisited 48 to 63): the aliased register
    equals the spec's loads. A word-order and a bit-order mutant of the array both fail."""
    widths, live, width, k = MULTIWORD
    table = random_table(widths, live, width, seed=40)
    words = leaf_words(table, widths, live, width, k)
    assert words.shape == (3, 16, 2)
    n = sum(widths)
    address = list(range(n))
    work = [n]
    physical = list(range(n + 1, n + 1 + ALIAS))
    register = [physical[j % ALIAS] for j in range(2**k * width)]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=n + 1 + ALIAS))
    def circuit(a, array):
        for position, wire in enumerate(address):
            qp.cond(((a >> (n - 1 - position)) & 1) == 1, qp.X)(wire)
        _qroam_read(array, widths, live, width, k, None, address, register[:width],
                    register[width:], work, [])
        return qp.probs(wires=physical)

    def outcomes(array):
        return [np.asarray(circuit(a, jnp.asarray(array))) for a in range(2**n)]

    probabilities = outcomes(words)
    assert all(abs(p.max() - 1) < 1e-9 for p in probabilities)
    assert [int(np.argmax(p)) for p in probabilities] == [
        _aliased_expected(table, a) for a in range(2**n)]
    natural = np.zeros_like(words)
    for leaf, block in enumerate(QROAMRead.leaf_blocks(widths, live, k)):
        for p in range(2**k):
            entry = dict(table).get((block << k) | p, (0, 0))
            natural[leaf, p] = entry
    for mutant in (words[:, :, ::-1], natural):
        assert [int(np.argmax(p)) for p in outcomes(mutant)] != [
            _aliased_expected(table, a) for a in range(2**n)]


def preset_shapes(p):
    return [outer_shape(p.X_o, p.k1, p.d1), inner_shape(p.RC, p.X_o, p.B + 1, p.k2, p.d2)]


def _declared_counts(widths, live, width, k, control_value):
    """The rule's declared gate counts by name (its counting twin)."""
    counter = QROAMEmitter(counting=True)
    wires, _ = layout(widths, live, width, k, control_value)
    _read_body(counter, None, widths, live, width, k, control_value, **wires)
    return counter.counts


def _assert_compiled_counts(shape, control_value=None):
    """T, M and A exact; CNOT equal to the declared upper bound plus two per CSWAP, because
    qp.analyze counts every unhinted load conditional. Returns the analyzed resources and the
    declared counts."""
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k, control_value)
    resources = compiled_resources(
        lambda: make_read(table, widths, live, width, k, control_value, wires),
        num_wires=num_wires)
    ranges = QROAMRead.visited_ranges(widths, live, k)
    expected = costs.qroam_read(ranges, k, width, control_value is not None)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    swaps = (2**k - 1) * width
    assert resources.measured_and_uncomputes == expected.toffoli - swaps
    declared = _declared_counts(widths, live, width, k, control_value)
    assert declared["CSWAP"] == swaps
    assert resources.counts["CNOT"] == declared["CNOT"] + 2 * swaps
    return resources, declared


@pytest.mark.compiled
@pytest.mark.parametrize("shape", preset_shapes(H2_STO3G_TEST) + preset_shapes(H2_631G_TEST),
                         ids=["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner"])
def test_toffoli_count_at_the_test_presets(shape):
    """T4 at H2_STO3G_TEST and H2_631G_TEST: T = U(visited) + (2^k - 1) width; A = U."""
    _assert_compiled_counts(shape)


@pytest.mark.compiled
def test_toffoli_count_with_enable():
    _assert_compiled_counts(INNER[2], control_value=0)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("shape", preset_shapes(MID_NAPHTHALENE), ids=["outer", "inner"])
def test_toffoli_count_at_mid_naphthalene(shape):
    """T4 at MID_NAPHTHALENE (nightly tier)."""
    _assert_compiled_counts(shape)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 is asserted on the closed forms; no FeMoco table is compiled (12.9). The four
    reads of a walk have 81 + 540 + 54 + 270 = 945 leaves (Task 0.9d)."""
    p = FEMOCO54
    outer, inner = preset_shapes(p)
    assert QROAMRead.visited_ranges(*outer[:2], outer[3]) == [(0, 81, 7)]
    assert QROAMRead.visited_ranges(*inner[:2], inner[3]) == [(0, 270, 9), (0, 2, 1)]
    assert costs.qroam_read([(0, 81, 7)], p.k1, p.d1, False).toffoli == 84 + 72 == 156
    assert costs.qroam_read([(0, 270, 9), (0, 2, 1)], p.k2, p.d2, False).toffoli == 933
    assert QROAMRead.required_wires(*outer[:2], p.d1, p.k1, None) == {
        "address": 9, "target": 24, "junk": 72, "work": 6, "control": 0}
    assert QROAMRead.required_wires(*inner[:2], p.d2, p.k2, None) == {
        "address": 14, "target": 26, "junk": 390, "work": 9, "control": 0}
    tail = ((p.w_x,), ((p.RC, p.X_o),), (p.N - 1) * p.b_rot + 2, 0)
    sf = ((p.w_r, p.w_b), ((0, p.R), (0, p.B)), (p.N - 1) * p.b_rot + 1, 0)
    shapes = [QROAMRead.leaf_words_shape(*s) for s in (outer, inner, tail, sf)]
    assert shapes == [(81, 4, 1), (540, 16, 1), (54, 1, 25), (270, 1, 25)]
    assert sum(s[0] for s in shapes) == 945


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    widths, live, width, k = INNER[0]
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)
    compiled_resources(lambda: make_read(table, widths, live, width, k, None, wires),
                       num_wires=num_wires, level="device")


@pytest.mark.parametrize(("shape", "control_value"), [(OUTER[1], None), (INNER[2], 0)])
def test_rule_is_valid(shape, control_value):
    """T2: with exact=False, every emitted gate must be declared."""
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


def test_leaf_words_by_hand():
    """Wire order (spec 6.7 on the wires): bit i of the word stream is entry bit width - 1 - i.
    X_o = 5, k = 1, width 2: entries 1 = 0b01 -> 0b10, 2 -> 0b01, 3 -> 0b11; address 5 is dead.
    Width 33: the top entry bit lands in word 0, bit 0, and bit 0 in word 1, bit 0."""
    table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
    args = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
    words = QROAMRead.leaf_words_from_table(table, **args)
    assert words.dtype == np.int64
    assert words.tolist() == [[[2], [1]], [[3], [2]], [[1], [0]]]
    wide = {"address_widths": (1,), "live_ranges": ((0, 2),), "width": 33, "block_bits": 1}
    words = QROAMRead.leaf_words_from_table(((0, (0, 1)), (1, (1, 0))), **wide)
    assert words.tolist() == [[[1, 0], [0, 1]]]


def test_loaded_and_visited_sets():
    assert QROAMRead.visited_ranges((3,), ((0, 5),), 1) == [(0, 3, 2)]
    assert QROAMRead.visited_ranges((3, 3), ((0, 2), (0, 5)), 3) == [(0, 2, 3)]
    assert QROAMRead.loaded_ranges((3, 3), ((0, 2), (0, 5)), 2) == ((0, 2), (0, 8))
    assert QROAMRead.loaded_set((3,), ((0, 5),), 1) == [0, 1, 2, 3, 4, 5]
    assert QROAMRead.loaded_set((2, 2), ((1, 2), (0, 3)), 1) == [4, 5, 6, 7]
    assert QROAMRead.leaf_blocks((3, 3), ((1, 3), (0, 5)), 2) == [2, 3, 4, 5]
    assert QROAMRead.leaf_blocks((3,), ((0, 5),), 3) == [0]
    assert QROAMRead.leaf_words_shape((3, 3), ((1, 3), (0, 5)), 33, 2) == (4, 4, 2)


@pytest.mark.parametrize(("table", "match"), [
    (((5, (1,)),), "outside the live ranges"),
    (((0, (4,)),), "does not fit in 2 bits"),
    (((0, (1, 0)),), "must be 1 words"),
    (((0, (1,)), (0, (2,))), "appears twice"),
])
def test_leaf_words_from_table_validates(table, match):
    with pytest.raises(ValueError, match=match):
        QROAMRead.leaf_words_from_table(table, address_widths=(3,), live_ranges=((0, 5),),
                                        width=2, block_bits=1)


def _words(**changes):
    """Valid leaf_words of the X_o = 5, k = 1, width-2 read, with entries changed."""
    words = np.zeros((3, 2, 1), dtype=object)
    for (leaf, position), word in changes.get("set", {}).items():
        words[leaf, position, 0] = word
    return words


@pytest.mark.parametrize(("change", "match"), [
    ({"block_bits": 4}, r"block_bits must lie in \[0, 3\]"),
    ({"address_widths": (0, 3), "live_ranges": ((0, 1), (0, 5))}, "nonempty first register"),
    ({"live_ranges": ((0, 9),)}, "not a nonempty range of 3-bit values"),
    ({"control_value": 2}, r"control_value must lie in \[0, 1\]"),
    ({"junk": [9]}, "'junk' has 1 wires; expected 2"),
    ({"leaf_words": np.zeros((2, 2, 1), dtype=np.int64)}, r"shape \(3, 2, 1\)"),
    ({"leaf_words": AbstractArray((3, 2, 2), np.int64)}, r"shape \(3, 2, 1\)"),
    ({"leaf_words": _words(set={(0, 0): 2**32 + 1})}, r"integers in \[0, 2\*\*32\)"),
    ({"leaf_words": _words(set={(0, 0): True})}, r"integers in \[0, 2\*\*32\)"),
    ({"leaf_words": _words(set={(0, 0): 1.0})}, r"integers in \[0, 2\*\*32\)"),
    ({"leaf_words": _words(set={(2, 1): 1})}, "padding or dead position"),
    ({"leaf_words": _words(set={(0, 0): 4})}, "bits beyond width 2"),
])
def test_constructor_validates(change, match):
    args = {"leaf_words": np.zeros((3, 2, 1), dtype=np.int64), "address_widths": (3,),
            "live_ranges": ((0, 5),), "width": 2, "block_bits": 1, "control_value": None,
            "address": [0, 1, 2], "target": [3, 4], "junk": [5, 6], "work": [7],
            "control": []} | change
    with pytest.raises(ValueError, match=match):
        QROAMRead(**args)
