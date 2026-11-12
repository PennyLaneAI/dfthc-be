"""Outcome certificates of spec 12.5 for TailAngleLoad, SFAngleLoad -> AngleErase (nightly tier).

The erase X-measures ``M = (N - 1) b_rot + 3`` qubits. The phase it leaves on a branch is
``(-1)^(xi . g)`` with ``g`` the branch's loaded content, which the load tests establish as a
function of ``(x_o, r, b)``, and its two fix-ups apply ``(-1)^f_xi`` with ``f`` from
``mcm_parities``, GF(2)-linear (test_angle_erase checks it on the erase's masks). So the residual
is linear in ``xi``, and exact restoration under the zero vector and every unit vector, M + 1
forced runs, certifies all 2**M outcomes. The conclusion also needs every conditional of the
certified tape to be diagonal with a GF(2)-linear predicate, which ``nonlinear_conditionals``
checks on the tape itself. Each run starts from a random complex amplitude on every value of
x_o, r and b (failed x_o, b = B, failed b and r >= R included) and must end in exactly that state
with every ancilla in |0>; strict mode also asserts that every erase measurement has
probability 1/2.

The suite includes partly covered SF fix-up leaves (a leaf is one value of r and the high bits
of b, with a one-hot over the low ob bits of b), as at FeMoco-54: every case with ob > 0 has a
leaf holding both a covered b < B and the never covered b = B or a failed b; the ob = 0 case
exercises the fallback, which has no one-hot leaf. A zero angle entry occurs in the STO-3G SF
table and is set in one tail row and one SF entry of the zero-block case; a zero mask cannot
occur, because every SF mask holds the b_neq_B bit and every tail mask the G0 bit (spec 9.17).
test_certificate_cases_cover_the_branches asserts these premises of the case design.
"""

import numpy as np
import pytest
from pennylane.ops import Conditional

from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import AngleErase, SFAngleLoad, TailAngleLoad
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation

pytestmark = pytest.mark.slow


def random_tables(p, seed, zero_tail=(), zero_sf=()):
    """Random angle tables of p's shape (angles below 2**(b_rot - 1)), with zero entries at the
    tail rows ``zero_tail`` and the SF addresses ``zero_sf``."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(0 if j in zero_tail else int(v) for v in rng.integers(0, bound, p.N - 1))
                 for j in range(p.N))
    sf = tuple(tuple(tuple(0 if (r, b) in zero_sf else int(v)
                           for v in rng.integers(0, bound, p.N - 1)) for b in range(p.B))
               for r in range(p.R))
    return tail, sf


def vendored_tables(system, p):
    tables = WalkTables.from_factors(load_factors(system), p)
    return tables.tail_angles, tables.sf_angles


RANDOM_R3 = WalkParams(N=3, R=3, B=3, C=1, n_plus=1, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
# (params, tables, SF one-hot bits): both test presets with their vendored tables (ob = w_b, the
# walk's); R = 3 (r = 3 unvisited, failed x_o = 6, 7) with a zero tail row and a zero SF entry, on
# the ob = 0 fallback and with a split one-hot (ob = 1: r and the high bit of b iterated); R = 1
# (the SF address is b alone).
CASES = {
    "sto3g": (H2_STO3G_TEST, vendored_tables("h2_sto3g_R2_B2_C1", H2_STO3G_TEST), 2),
    "631g": (H2_631G_TEST, vendored_tables("h2_631g_R2_B2_C2", H2_631G_TEST), 2),
    "R3-zero-blocks": (RANDOM_R3, random_tables(RANDOM_R3, 5, (1,), ((2, 0),)), 2),
    "R3-ob0": (RANDOM_R3, random_tables(RANDOM_R3, 6), 0),
    "R3-ob1": (RANDOM_R3, random_tables(RANDOM_R3, 7), 1),
    "R1": (RANDOM_R1, random_tables(RANDOM_R1, 9), 2),
}


def layout(p, tail, sf, ob):
    sizes = {"x_o": p.w_x, "r": p.w_r, "b": p.w_b, "angles": (p.N - 1) * p.b_rot, "G0": 1,
             "G1": 1, "b_neq_B": 1}
    works = (TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"],
             SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"],
             AngleErase.required_wires(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot,
                                       ob)["work"])
    sizes["work"] = max(works)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, works


def random_input_state(num_wires, inputs, seed=11):
    """A random complex amplitude on every basis value of ``inputs``, every other wire |0>."""
    rng = np.random.default_rng(seed)
    size = 2 ** len(inputs)
    amps = rng.normal(size=size) + 1j * rng.normal(size=size)
    keys = []
    for v in range(size):
        value = sum(((v >> (len(inputs) - 1 - i)) & 1) << (num_wires - 1 - wire)
                    for i, wire in enumerate(inputs))
        keys.append(SparseState.basis(num_wires, value).keys[0])
    return SparseState(num_wires, np.array(keys), amps / np.linalg.norm(amps))


def sorted_terms(state):
    order = np.lexsort(state.keys.T[::-1])
    return state.keys[order], state.amps[order]


@pytest.mark.parametrize("name", CASES)
def test_loads_erase_round_trip_certificate(name):
    p, (tail, sf), ob = CASES[name]
    wires, num_wires, works = layout(p, tail, sf, ob)

    def round_trip():
        TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                      wires["G0"], wires["G1"], wires["work"][:works[0]])
        SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"], wires["angles"],
                    wires["b_neq_B"], wires["work"][:works[1]])
        AngleErase(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot, ob, wires["angles"],
                   wires["b_neq_B"], wires["G0"], wires["G1"], wires["x_o"], wires["r"],
                   wires["b"], wires["work"][:works[2]])

    operations = decompose_for_simulation(round_trip).operations
    conditionals = [op for op in operations if isinstance(op, Conditional)]
    assert conditionals and {op.base.name for op in conditionals} <= {"PauliZ", "CZ"}
    assert not nonlinear_conditionals(operations)
    initial = random_input_state(num_wires, wires["x_o"] + wires["r"] + wires["b"])
    expected_keys, expected_amps = sorted_terms(initial)
    num_measured = (p.N - 1) * p.b_rot + 3
    schedules = [[0] * num_measured] + [
        [int(i == j) for j in range(num_measured)] for i in range(num_measured)]
    for schedule in schedules:
        simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
        keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
        assert simulator.policy.position == num_measured
        assert np.array_equal(keys, expected_keys), schedule
        assert np.allclose(amps, expected_amps, rtol=0, atol=1e-12), schedule


def test_certificate_cases_cover_the_branches():
    """The cases reach every branch type the erase must restore: tail rows of both families
    (D1 and Q1, except R1, which is all D1), SF rows with b < B and b = B, failed x_o and failed b
    where the registers allow them, and r >= R (R3). The zero-block case has an all-zero tail row
    and SF entry; every case with ob > 0 has a partly covered SF leaf (a value of b >> ob whose
    low positions hold both a covered b < B and an uncovered b >= B); R3-ob0 is the fallback."""
    for name, (p, _, _) in CASES.items():
        assert 0 < p.RC < p.X_o
        assert p.n_plus >= 1 and (p.n_plus < p.N or name == "R1")
    assert any(2**p.w_x > p.X_o for p, _, _ in CASES.values())
    assert any(2**p.w_b > p.B + 1 for p, _, _ in CASES.values())
    assert 2**RANDOM_R3.w_r > RANDOM_R3.R
    _, (tail, sf), _ = CASES["R3-zero-blocks"]
    assert any(not any(row) for row in tail)
    assert any(not any(entry) for plane in sf for entry in plane)
    for name, (p, _, ob) in CASES.items():
        if ob:
            leaves = {}
            for b in range(2**p.w_b):
                leaves.setdefault(b >> ob, set()).add(b < p.B)
            assert {True, False} in leaves.values(), name
    assert CASES["R3-ob0"][2] == 0
