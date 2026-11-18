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

"""WalkTables: everything the circuit loads, as compilable tuples (spec 8.4)."""

import dataclasses

import numpy as np
import pytest

from dfthc.data.alias import alias_tables, inner_alias_row
from dfthc.data.angles import angles_from_vector, discretize_angles, vector_from_angles
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.sos import inner_weights, outer_weights
from dfthc.data.tables import WalkTables
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE

PRESET_SYSTEMS = [(H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"), (H2_631G_TEST, "h2_631g_R2_B2_C2"),
                  (MID_NAPHTHALENE, "naphthalene")]


def _ints(values):
    if isinstance(values, tuple):
        for value in values:
            yield from _ints(value)
    else:
        yield values


@pytest.mark.parametrize(("params", "system"), PRESET_SYSTEMS)
def test_preset_sizes_equal_the_factor_shapes(params, system):
    """Tables accept the WalkParams presets."""
    f = load_factors(system)
    assert (params.N, params.R, params.B, params.C, params.n_plus) == (
        f.N, f.R, f.B, f.C, f.n_plus)


@pytest.mark.parametrize(("params", "system"), PRESET_SYSTEMS)
def test_tables_are_the_alias_and_angle_tables_of_the_factors(params, system):
    f = load_factors(system)
    tables = WalkTables.from_factors(f, params)
    alt, keep = alias_tables(outer_weights(f), params.b_k1)
    assert tables.outer_alt == tuple(alt.tolist()) and tables.outer_keep == tuple(keep.tolist())
    rows = [inner_alias_row(row, params.b_k2) for row in inner_weights(f)]
    for column, name in enumerate(("inner_alt", "inner_keep", "inner_sgn", "inner_d_sgn")):
        assert getattr(tables, name) == tuple(tuple(row[column].tolist()) for row in rows)
    tail = np.concatenate([f.u_plus, f.u_minus])
    for j, u in enumerate(tail):
        assert tables.tail_angles[j] == tuple(discretize_angles(angles_from_vector(u),
                                                                params.b_rot).tolist())
    for r, b in np.ndindex(f.R, f.B):
        reconstructed = vector_from_angles(tables.sf_angles[r][b], params.b_rot)
        bound = (f.N - 1) * np.pi / 2 ** (params.b_rot - 1)
        assert np.linalg.norm(reconstructed - f.U[r, b]) <= bound


def test_tables_are_hashable_and_round_trip_through_compilable():
    tables = WalkTables.from_factors(load_factors("h2_631g_R2_B2_C2"), H2_631G_TEST)
    values = tables.as_compilable()
    assert len(values) == 8 and isinstance(values, tuple)
    assert WalkTables.from_compilable(values) == tables
    assert hash(WalkTables.from_compilable(values)) == hash(tables)
    assert all(type(v) is int and 0 <= v < 2**31 for v in _ints(values))
    with pytest.raises(dataclasses.FrozenInstanceError):
        tables.outer_alt = ()


def test_table_shapes_at_femoco():
    tables = WalkTables.from_factors(random_factors(54, 10, 27, 27, n_plus=27, seed=0), FEMOCO54)
    assert np.shape(tables.outer_alt) == np.shape(tables.outer_keep) == (324,)
    for name in ("inner_alt", "inner_keep", "inner_sgn", "inner_d_sgn"):
        assert np.shape(getattr(tables, name)) == (270, 28)
    assert np.shape(tables.tail_angles) == (54, 53)
    assert np.shape(tables.sf_angles) == (10, 27, 53)
    assert max(_ints(tables.as_compilable())) < 2**31


def test_mismatched_params_are_rejected():
    with pytest.raises(ValueError, match="do not match the factors"):
        WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), H2_631G_TEST)


def test_entries_of_31_bits_or_more_are_rejected():
    """Compilable integers stay below 2**31 (L-24): 32-bit keep values are refused."""
    params = dataclasses.replace(H2_STO3G_TEST, b_k1=32)
    with pytest.raises(ValueError, match="below 2\\*\\*31"):
        WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), params)
    with pytest.raises(ValueError, match="below 2\\*\\*31"):
        WalkTables.from_compilable(((2**31,),) + ((),) * 7)
