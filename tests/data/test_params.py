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

"""WalkParams, its derived sizes and the presets (spec sections 3.1 and 8.3)."""

import dataclasses
import re

import pytest

from dfthc.params import (
    FEMOCO54,
    H2_631G_TEST,
    H2_STO3G_TEST,
    MID_NAPHTHALENE,
    WalkParams,
    lg,
    odd_part,
)


def test_lg_is_the_exact_ceiling_of_log2():
    assert [lg(y) for y in (1, 2, 3, 4, 5, 8, 9, 28, 270, 324)] == [0, 1, 2, 2, 3, 3, 4, 5, 9, 9]
    with pytest.raises(ValueError, match="positive"):
        lg(0)


def test_odd_part():
    assert [odd_part(y) for y in (1, 2, 3, 4, 12, 28, 324)] == [1, 1, 3, 1, 3, 7, 81]
    with pytest.raises(ValueError, match="positive"):
        odd_part(0)


def test_presets_hold_the_spec_values():
    assert FEMOCO54.as_compilable() == (54, 10, 27, 27, 27, 15, 15, 15, 7, 2, 4, 4, 2)
    assert H2_STO3G_TEST.as_compilable() == (2, 2, 2, 1, 1, 4, 2, 2, 3, 1, 1, 1, 1)
    assert H2_631G_TEST.as_compilable() == (4, 2, 2, 2, 3, 4, 2, 2, 3, 1, 1, 1, 1)
    assert MID_NAPHTHALENE.as_compilable() == (10, 8, 6, 5, 5, 15, 15, 15, 7, 1, 2, 3, 1)


def test_derived_values_at_femoco():
    p = FEMOCO54
    assert (p.X_o, p.RC, p.n_minus, p.w_x, p.w_b, p.w_r) == (324, 270, 27, 9, 5, 4)
    assert (p.d1, p.d2, p.b_grad, p.K1, p.K2, p.M1, p.M2) == (24, 26, 15, 4, 16, 81, 540)


def test_derived_values_at_h2_sto3g():
    p = H2_STO3G_TEST
    assert (p.X_o, p.RC, p.n_minus, p.w_x, p.w_b, p.w_r) == (4, 2, 1, 2, 2, 1)
    assert (p.d1, p.d2, p.b_grad, p.K1, p.K2, p.M1, p.M2) == (4, 7, 4, 2, 2, 2, 4)


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        ({"N": 1, "n_plus": 1}, "N >= 2"),
        ({"R": 0}, "R, B, C >= 1"),
        ({"B": 0}, "R, B, C >= 1"),
        ({"C": 0}, "R, B, C >= 1"),
        ({"n_plus": 3}, "0 <= n_plus <= N"),
        ({"n_plus": -1}, "0 <= n_plus <= N"),
        ({"b_rot": 2}, "b_rot >= 3"),
        ({"b_k1": 0}, "b_k1, b_k2 >= 1"),
        ({"b_k2": 0}, "b_k1, b_k2 >= 1"),
        ({"s_u": 2}, "3 <= s_u <= b_grad"),
        ({"s_u": 5}, "3 <= s_u <= b_grad"),
        ({"k1": 3}, "0 <= k1 <= w_x"),
        ({"k1": -1}, "0 <= k1 <= w_x"),
        ({"k2": 3}, "0 <= k2 <= w_b"),
        ({"ke1": 3}, "0 <= ke1 <= w_x"),
        ({"ke2": 2}, "0 <= ke2 <= lg(RC)"),
        ({"ke2": -1}, "0 <= ke2 <= lg(RC)"),
        ({"B": 4}, "2**w_b <= (N - 1)(b_rot - 1) + 1"),
    ],
)
def test_constraint_violations_raise(changes, constraint):
    """Spec 8.3; the last case is the select's one-hot (final review m-1): at N = 2, b_rot = 4
    and B = 4 it needs 2**3 = 8 wires, more than the (N - 1)(b_rot - 1) + 1 = 4 it resets."""
    with pytest.raises(ValueError, match=re.escape(constraint)):
        dataclasses.replace(H2_STO3G_TEST, **changes)


def test_fields_must_be_integers():
    with pytest.raises(ValueError, match="b_rot must be an int"):
        dataclasses.replace(H2_STO3G_TEST, b_rot=4.0)
    with pytest.raises(ValueError, match="k1 must be an int"):
        dataclasses.replace(H2_STO3G_TEST, k1=True)


def test_compilable_round_trip():
    for preset in (FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE):
        values = preset.as_compilable()
        assert len(values) == 13 and all(type(v) is int for v in values)
        assert WalkParams.from_compilable(values) == preset
    with pytest.raises(ValueError, match="13 values"):
        WalkParams.from_compilable(FEMOCO54.as_compilable()[:-1])


def test_params_are_hashable_and_frozen():
    assert len({FEMOCO54, dataclasses.replace(FEMOCO54), H2_STO3G_TEST}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        FEMOCO54.N = 3
