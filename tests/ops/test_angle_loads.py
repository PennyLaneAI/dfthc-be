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

"""TailAngleLoad and SFAngleLoad (spec 9.10, 9.11): every address, exclusive branches, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import (
    SFAngleLoad,
    TailAngleLoad,
    _rows,
    _sf_angle_load,
    _sf_angle_load_resources,
    _tail_angle_load,
    _tail_angle_load_resources,
    angle_entry,
)
from dfthc.ops.qroam import dummy_wires
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation


def random_tables(p, seed=3):
    """Random angle tables of p's shape, every angle below 2**(b_rot - 1)."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.N))
    sf = tuple(tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.B))
               for _ in range(p.R))
    return tail, sf


def register_value(angles, b_rot):
    """The angle registers' big-endian integer when register k, of b_rot - 1 bits, holds
    angles[k] (spec 6.7), written out bit by bit, independently of the module's angle_entry."""
    return int("".join(format(angle, f"0{b_rot - 1}b") for angle in angles), 2)


def vendored_tables(system, p):
    tables = WalkTables.from_factors(load_factors(system), p)
    return tables.tail_angles, tables.sf_angles


# R = 3 leaves r = 3 unvisited, X_o = 6 leaves x_o = 6, 7 failed, B = 3 makes b = 3 the identity
# term; R = 1 gives an SF address of b alone (r has no wires); at N = 2 and B = 2 the select's
# one-hot needs b_rot = 4 (spec 8.3).
RANDOM_R3 = WalkParams(N=3, R=3, B=3, C=1, n_plus=1, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
SHAPES = {
    "sto3g": (H2_STO3G_TEST, vendored_tables("h2_sto3g_R2_B2_C1", H2_STO3G_TEST)),
    "631g": (H2_631G_TEST, vendored_tables("h2_631g_R2_B2_C2", H2_631G_TEST)),
    "random-R3": (RANDOM_R3, random_tables(RANDOM_R3)),
    "random-R1": (RANDOM_R1, random_tables(RANDOM_R1, seed=0)),
}


def layout(p, tail, sf):
    """Wires: x_o, r, b, angles, G0, G1, b_neq_B, then work for the larger load."""
    sizes = {"x_o": p.w_x, "r": p.w_r, "b": p.w_b, "angles": (p.N - 1) * (p.b_rot - 1), "G0": 1,
             "G1": 1, "b_neq_B": 1}
    tail_work = TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"]
    sf_work = SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"]
    sizes["work"] = max(tail_work, sf_work)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, tail_work, sf_work


def tail_load(p, tail, wires, work):
    TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                  wires["G0"], wires["G1"], wires["work"][:work])


def sf_load(p, sf, wires, work):
    SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"], wires["angles"],
                wires["b_neq_B"], wires["work"][:work])


def value_of(state, register, row):
    return sum(int(state.bits(w)[row]) << (len(register) - 1 - i) for i, w in enumerate(register))


def simulate(num_wires, inputs, body):
    """H on every input wire, then ``body``, on the sparse simulator; every basis value of the
    inputs must occur once, with amplitude +1/sqrt(2**len(inputs)) (no phase)."""

    def qfunc():
        for wire in inputs:
            qp.H(wire)
        body()

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    assert len(state.amps) == 2 ** len(inputs)
    assert np.allclose(state.amps, 2 ** (-len(inputs) / 2), rtol=0, atol=1e-12)
    return state


@pytest.mark.parametrize("name", SHAPES)
def test_tail_load_on_every_address(name):
    """T3 (sparse simulator, H on x_o): tail rows x_o in [RC, X_o) get their angles, G0 = 1 and
    G1 = [x_o - RC >= N+]; SF rows and failed x_o >= X_o are untouched; b_neq_B, r and b are not
    the tail load's wires and stay |0>."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, tail_work, _ = layout(p, tail, sf)
    state = simulate(num_wires, wires["x_o"], lambda: tail_load(p, tail, wires, tail_work))
    for row in range(len(state.amps)):
        x_o = value_of(state, wires["x_o"], row)
        is_tail = p.RC <= x_o < p.X_o
        expected = (register_value(tail[x_o - p.RC], p.b_rot) if is_tail else 0,
                    int(is_tail), int(is_tail and x_o - p.RC >= p.n_plus), 0, 0)
        actual = tuple(value_of(state, wires[k], row)
                       for k in ("angles", "G0", "G1", "b_neq_B", "work"))
        assert actual == expected, x_o


@pytest.mark.parametrize("name", SHAPES)
def test_sf_load_on_every_address(name):
    """T3 (sparse simulator, H on r, b and the enable G0): where G0 = 0 and (r, b) lies in
    [0, R) x [0, B), the angles of (r, b) and b_neq_B = 1; nothing on G0 = 1, b = B, failed b or
    r >= R; G1 is not the SF load's wire and stays |0>."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, _, sf_work = layout(p, tail, sf)
    state = simulate(num_wires, wires["r"] + wires["b"] + wires["G0"],
                     lambda: sf_load(p, sf, wires, sf_work))
    for row in range(len(state.amps)):
        r, b, g0 = (value_of(state, wires[k], row) for k in ("r", "b", "G0"))
        on = g0 == 0 and r < p.R and b < p.B
        expected = (register_value(sf[r][b], p.b_rot) if on else 0, int(on), 0, 0)
        actual = tuple(value_of(state, wires[k], row) for k in ("angles", "b_neq_B", "G1", "work"))
        assert actual == expected, (r, b, g0)


@pytest.mark.parametrize("name", SHAPES)
def test_loads_write_exclusive_branches(name):
    """T3 (sparse simulator, H on x_o, r and b): after both loads every branch holds one
    family's data: a tail row its angles, G0 = 1, G1 and b_neq_B = 0 whatever (r, b) is (the SF
    load leaves it untouched); any other x_o the SF entry of (r, b) and b_neq_B = 1 if (r, b) is in
    range, else zeros, with G0 = G1 = 0 (the tail load left it untouched)."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)

    def body():
        tail_load(p, tail, wires, tail_work)
        sf_load(p, sf, wires, sf_work)

    state = simulate(num_wires, wires["x_o"] + wires["r"] + wires["b"], body)
    for row in range(len(state.amps)):
        x_o, r, b = (value_of(state, wires[k], row) for k in ("x_o", "r", "b"))
        if p.RC <= x_o < p.X_o:
            expected = (register_value(tail[x_o - p.RC], p.b_rot), 1,
                        int(x_o - p.RC >= p.n_plus), 0)
        else:
            on = r < p.R and b < p.B
            expected = (register_value(sf[r][b], p.b_rot) if on else 0, 0, 0, int(on))
        actual = tuple(value_of(state, wires[k], row) for k in ("angles", "G0", "G1", "b_neq_B"))
        assert actual == expected, (x_o, r, b)
        assert value_of(state, wires["work"], row) == 0


def compiled_loads_state(name):
    """Both loads on H over x_o, r and b, compiled and run on lightning, and the state the
    exclusive-branch table predicts."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)
    inputs = wires["x_o"] + wires["r"] + wires["b"]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        for wire in inputs:
            qp.H(wire)
        tail_load(p, tail, wires, tail_work)
        sf_load(p, sf, wires, sf_work)
        return qp.state()

    expected = np.zeros(2**num_wires)
    for value in range(2 ** len(inputs)):
        bits = dict(zip(inputs, ((value >> (len(inputs) - 1 - i)) & 1
                                 for i in range(len(inputs))), strict=True))
        x_o = sum(bits[w] << (p.w_x - 1 - i) for i, w in enumerate(wires["x_o"]))
        r = sum(bits[w] << (p.w_r - 1 - i) for i, w in enumerate(wires["r"]))
        b = sum(bits[w] << (p.w_b - 1 - i) for i, w in enumerate(wires["b"]))
        if p.RC <= x_o < p.X_o:
            registers = (register_value(tail[x_o - p.RC], p.b_rot), 1,
                         int(x_o - p.RC >= p.n_plus), 0)
        else:
            on = r < p.R and b < p.B
            registers = (register_value(sf[r][b], p.b_rot) if on else 0, 0, 0, int(on))
        index = value
        for register, content in zip(("angles", "G0", "G1", "b_neq_B"), registers, strict=True):
            index = (index << len(wires[register])) | content
        expected[index << len(wires["work"])] = 2 ** (-len(inputs) / 2)
    return np.asarray(circuit()), expected


@pytest.mark.compiled
def test_compiled_loads_on_lightning():
    """T3 on the compiled path at H2_STO3G_TEST (14 wires) and at random-R1 (12 wires; the SF
    address is b alone, and b = B = 2 and the failed b = 3 must stay untouched): both loads give
    exactly the state of the exclusive-branch table, so the rules' leaf words are built and
    loaded correctly under capture, without phases, and the one-register SF address lowers. The
    R1 SF entries are distinct and nonzero, so a wrong b shows."""
    _, (_, sf) = SHAPES["random-R1"]
    assert len(set(sf[0])) == len(sf[0]) and all(any(entry) for entry in sf[0])
    for name in ("sto3g", "random-R1"):
        state, expected = compiled_loads_state(name)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), name


def declared_cnots(p, tail, sf, which):
    """The rule's declared CNOT upper bound (its resource function): load CNOTs for every bit of
    every row plus the iteration's CNOTs."""
    if which == "tail":
        args = (tail, p.n_plus, p.RC, p.X_o, p.b_rot)
        sizes, resources = TailAngleLoad.required_wires(*args), _tail_angle_load_resources
    else:
        args = (sf, p.R, p.B, p.b_rot)
        sizes, resources = SFAngleLoad.required_wires(*args), _sf_angle_load_resources
    return resources(*args, **dummy_wires(sizes))[qp.CNOT]


def assert_load_counts(p, tail, sf, which):
    """T4: T, M and A exact against dfthc.costs; the CNOT count, an upper bound for the data
    (the rules have exact=False), equals the declared bound."""
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)
    if which == "tail":
        resources = compiled_resources(lambda: tail_load(p, tail, wires, tail_work), num_wires)
        expected = costs.tail_angle_load(p)
    else:
        resources = compiled_resources(lambda: sf_load(p, sf, wires, sf_work), num_wires)
        expected = costs.sf_angle_load(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli
    assert resources.counts["CNOT"] == declared_cnots(p, tail, sf, which)


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "which"), [("sto3g", "tail"), ("sto3g", "sf"), ("631g", "tail"),
                                             ("631g", "sf"), ("random-R1", "sf")])
def test_toffoli_count_compiled(name, which):
    """T4 at H2_STO3G_TEST and H2_631G_TEST with the vendored tables, and the SF load at
    random-R1 (the one-register address)."""
    p, (tail, sf) = SHAPES[name]
    assert_load_counts(p, tail, sf, which)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("which", ["tail", "sf"])
def test_toffoli_count_at_mid_naphthalene(which):
    """T4 at MID_NAPHTHALENE with the naphthalene tables (nightly tier): tail 13, SF 55."""
    assert_load_counts(MID_NAPHTHALENE, *vendored_tables("naphthalene", MID_NAPHTHALENE), which)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms (no FeMoco table is compiled): tail 61 and
    SF 11 + 10 * 28 = 291; the loads have 54 and 270 rows of 744 and 743 bits."""
    p = FEMOCO54
    assert costs.tail_angle_load(p) == costs.Cost(61) == costs.Cost(costs.unary(270, 324, 9, False))
    assert costs.sf_angle_load(p) == costs.Cost(11 + 10 * 28) == costs.Cost(291)
    tail = TailAngleLoad.load_shape(p.N, p.RC, p.X_o, p.b_rot)
    sf = SFAngleLoad.load_shape(p.N, p.R, p.B, p.b_rot)
    assert tail == {"address_widths": (9,), "live_ranges": ((270, 324),), "width": 744}
    assert sf == {"address_widths": (4, 5), "live_ranges": ((0, 10), (0, 27)), "width": 743}
    tail_angles, sf_angles = random_tables(p)
    assert TailAngleLoad.required_wires(tail_angles, p.n_plus, p.RC, p.X_o, p.b_rot)["work"] == 8
    assert SFAngleLoad.required_wires(sf_angles, p.R, p.B, p.b_rot)["work"] == 9


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names both pinned rules; T5 at level="device"."""
    p, (tail, sf) = SHAPES["sto3g"]
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)

    def program():
        tail_load(p, tail, wires, tail_work)
        sf_load(p, sf, wires, sf_work)

    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_tail_angle_load", "_sf_angle_load"} <= set(
        chosen_rules(captured.out + captured.err))
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize("name", ["sto3g", "random-R3", "random-R1"])
def test_rules_are_valid(name):
    """T2 (tape mode) for both rules at three sizes, one with a one-register SF address; each
    pinned rule is the only rule registered under its class name (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(TailAngleLoad)) == [_tail_angle_load]
    assert list(qp.list_decomps(SFAngleLoad)) == [_sf_angle_load]
    p, (tail, sf) = SHAPES[name]
    wires, _, tail_work, sf_work = layout(p, tail, sf)
    qp.testing.assert_valid_decomp_rule(
        TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                      wires["G0"], wires["G1"], wires["work"][:tail_work]),
        _tail_angle_load, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(
        SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"],
                    wires["angles"], wires["b_neq_B"], wires["work"][:sf_work]),
        _sf_angle_load, check_matrix=False)


def test_entries_and_rows_by_hand():
    """angle_entry puts register 0 first, b_rot - 1 bits each (spec 6.7); a load's rows hold its
    entries big-endian, bit i of a row for target wire i, so wider entries need no word packing."""
    assert angle_entry((1, 2, 3), 3) == 0b01_10_11
    rows = _rows([0b001_1_0, 0b010_1_0, 0b011_1_1], 5)
    assert rows.dtype == bool
    assert rows.astype(int).tolist() == [[0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 1]]
    # 47 bits: ((16383 << 30) | 1) << 2 | 0b10 = (16383 << 32) | 6.
    wide = _rows([(2**14 - 1 << 32) | 6], 47)[0]
    assert not wide[0] and wide[1:15].all() and not wide[15:44].any()
    assert wide[44:].tolist() == [True, True, False]


def test_load_shapes_and_required_wires():
    p, (tail, sf) = SHAPES["random-R1"]
    assert SFAngleLoad.load_shape(p.N, p.R, p.B, p.b_rot) == {
        "address_widths": (2,), "live_ranges": ((0, 2),), "width": 4}
    assert SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot) == {
        "r": 0, "b": 2, "G0": 1, "angles": 3, "b_neq_B": 1, "work": 2}
    p, (tail, sf) = SHAPES["631g"]
    assert TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot) == {
        "x_o": 3, "angles": 9, "G0": 1, "G1": 1, "work": 2}


TAIL = ((1,), (2,))
SF = (((1,), (2,)), ((3,), (0,)))


def tail_args(**changes):
    """Valid TailAngleLoad arguments (N = 2, RC = 2, b_rot = 3), with ``changes``."""
    return {"tail_angles": TAIL, "n_plus": 1, "RC": 2, "X_o": 4, "b_rot": 3, "x_o": [0, 1],
            "angles": [2, 3], "G0": [5], "G1": [6], "work": [7]} | changes


def sf_args(**changes):
    """Valid SFAngleLoad arguments (R = B = 2, N = 2, b_rot = 3), with ``changes``."""
    return {"sf_angles": SF, "R": 2, "B": 2, "b_rot": 3, "r": [0], "b": [1, 2], "G0": [3],
            "angles": [4, 5], "b_neq_B": [7], "work": [8, 9, 10]} | changes


@pytest.mark.parametrize(("build", "match"), [
    (lambda: TailAngleLoad(**tail_args(tail_angles=((4,), (2,)))),
     r"angles must lie in \[0, 2\*\*\(b_rot - 1\)\) = \[0, 4\), got 4"),
    (lambda: TailAngleLoad(**tail_args(tail_angles=((1, 2), (2, 1)))), r"shape \(2, 1\)"),
    (lambda: TailAngleLoad(**tail_args(tail_angles=3)),
     r"tail_angles must be a table of ints of shape \(N, N - 1\)"),
    (lambda: TailAngleLoad(**tail_args(tail_angles=(1, 2))),
     r"tail_angles must be a table of ints of shape \(2, 1\)"),
    (lambda: TailAngleLoad(**tail_args(X_o=5)), r"X_o must lie in \[4, 4\]"),
    (lambda: TailAngleLoad(**tail_args(n_plus=3)), r"n_plus must lie in \[0, 2\]"),
    (lambda: TailAngleLoad(**tail_args(work=[])), "'work' has 0 wires; expected 1"),
    (lambda: SFAngleLoad(**sf_args(R=3)), r"shape \(3, 2, 1\)"),
    (lambda: SFAngleLoad(**sf_args(sf_angles=3)), r"sf_angles must be a table of shape \(R, B,"),
    (lambda: SFAngleLoad(**sf_args(b=[1])), "'b' has 1 wires; expected 2"),
    (lambda: SFAngleLoad(**sf_args(b_rot=1)), r"b_rot must lie in \[2,"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
