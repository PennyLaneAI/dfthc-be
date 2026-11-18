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

"""Outcome certificates of spec 12.5 for QROAMRead -> QROAMErase round trips (nightly tier).

The phase an erase leaves on address ``a`` is ``(-1)^(xi . g(a))`` and its fixup applies
``(-1)^f_xi(a)``, both GF(2)-linear in the outcome vector ``xi`` (test_qroam_erase checks
``mcm_parities``). So the residual is linear, and exact restoration under the zero vector and
under every unit vector, M + 1 forced runs, certifies all 2**M outcomes. That conclusion also
needs every conditional of the certified tape to be diagonal with a GF(2)-linear predicate,
which ``nonlinear_conditionals`` checks on the tape itself. Each run starts from a random
complex amplitude on every value of the address and control wires (so any permutation or
relative phase of the addresses is caught) and must end in exactly that state with every
ancilla in |0>; strict mode also asserts that every erase measurement has probability 1/2.
"""

import numpy as np
import pytest
from pennylane.ops import Conditional

from dfthc.ops.qroam import QROAMErase, QROAMRead
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation

pytestmark = pytest.mark.slow


def random_table(widths, live_ranges, width, seed=5, zero=()):
    rng = np.random.default_rng(seed)
    (lo1, hi1), *rest = live_ranges
    addresses = ([(v << widths[1]) | u for v in range(lo1, hi1) for u in range(*rest[0])]
                 if rest else list(range(lo1, hi1)))
    return tuple((a, (int(rng.integers(0, 2**width)) * (a not in zero),)) for a in addresses)


def erase_shapes(p):
    return [((p.w_x,), ((0, p.X_o),), p.d1, p.k1, p.ke1),
            ((p.w_x, p.w_b), ((0, p.RC), (0, p.B + 1)), p.d2, p.k2, p.ke2 + p.w_b)]


# The outer and inner shapes of both test presets (k = 1); X_o = 5 with k = 2 (dead positions)
# on the ob = 0 fallback; rows [0, 3) of 3 bits with (B + 1, k2) = (6, 2) (dead positions, row 1
# zero so its masks are zero, the last fixup leaf partly covered), enabled with value 1 (ob = 4)
# and with value 0 (ob = 5: a 32-wire one-hot, 12 of them extra_onehot).
INNER_6_2 = ((3, 3), ((0, 3), (0, 6)), 5, 2)
SHAPES = erase_shapes(H2_STO3G_TEST) + erase_shapes(H2_631G_TEST) + [
    ((3,), ((0, 5),), 3, 2, 0), (*INNER_6_2, 4), (*INNER_6_2, 5)]
CASES = [(shape, None, ()) for shape in SHAPES[:5]] + [
    (SHAPES[5], 1, range(8, 16)), (SHAPES[6], 0, range(8, 16))]
IDS = ["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner", "outer-k2-ob0",
       "inner-6-2-enable1", "inner-6-2-ob5-enable0"]


def layout(shape, control_value):
    widths, live, width, k, ob = shape
    erase = QROAMErase.required_wires((), widths, live, width, k, control_value, ob)
    read = QROAMRead.required_wires(widths, live, width, k, control_value)
    sizes = dict(erase, work=max(erase["work"], read["work"]))
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, read["work"], erase["work"]


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


@pytest.mark.parametrize(("shape", "control_value", "zero"), CASES, ids=IDS)
def test_read_erase_round_trip_certificate(shape, control_value, zero):
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width, zero=zero)
    wires, num_wires, read_work, erase_work = layout(shape, control_value)
    columns = QROAMRead.columns_from_table(table, address_widths=widths, live_ranges=live,
                                           width=width, block_bits=k)

    def round_trip():
        QROAMRead(columns, widths, live, width, k, control_value, wires["address"],
                  wires["target"], wires["junk"], wires["work"][:read_work], wires["control"])
        QROAMErase(table, widths, live, width, k, control_value, ob, wires["address"],
                   wires["target"], wires["junk"], wires["work"][:erase_work], wires["control"],
                   wires["extra_onehot"])

    operations = decompose_for_simulation(round_trip).operations
    assert any(isinstance(op, Conditional) for op in operations)
    assert not nonlinear_conditionals(operations)
    initial = random_input_state(num_wires, wires["address"] + wires["control"])
    expected_keys, expected_amps = sorted_terms(initial)
    num_measured = 2**k * width
    schedules = [[0] * num_measured] + [
        [int(i == j) for j in range(num_measured)] for i in range(num_measured)]
    for schedule in schedules:
        simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
        keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
        assert simulator.policy.position == num_measured
        assert np.array_equal(keys, expected_keys), schedule
        assert np.allclose(amps, expected_amps, rtol=0, atol=1e-12), schedule
