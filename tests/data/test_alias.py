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

"""Alias tables and their probabilities (spec 6.4 and 8.4)."""

import itertools

import numpy as np
import pytest

from dfthc.data.alias import alias_probabilities, alias_tables, inner_alias_row


def test_a_hand_computed_table():
    """Weights 1:2:3:4 at 2 bits discretize to 2, 3, 5, 6 sixteenths (Qualtran's rounding)."""
    alt, keep = alias_tables([1.0, 2.0, 3.0, 4.0], bits=2)
    assert alt.tolist() == [2, 3, 3, 3] and keep.tolist() == [2, 3, 3, 0]
    # Sixteenths are exact in binary, so the comparison is exact.
    assert np.array_equal(alias_probabilities(alt, keep, 2), np.array([2, 3, 5, 6]) / 16)


@pytest.mark.parametrize(("length", "bits"), [(3, 1), (5, 2), (28, 4), (81, 7), (324, 15)])
def test_probabilities_reproduce_the_weights(length, bits):
    """Each probability is within 1 / (L 2**bits) of w / sum(w), so within 2**-bits."""
    rng = np.random.default_rng(length)
    weights = rng.random(length) ** 3
    alt, keep = alias_tables(weights, bits)
    assert alt.shape == keep.shape == (length,)
    assert keep.min() >= 0 and keep.max() < 2**bits and 0 <= alt.min() and alt.max() < length
    error = np.abs(alias_probabilities(alt, keep, bits) - weights / weights.sum())
    assert error.max() <= 1 / (length * 2**bits) + 1e-15
    assert error.max() <= 2.0**-bits


@pytest.mark.parametrize(("length", "bits"), [(3, 2), (7, 3), (12, 2)])
def test_formula_equals_the_sampling_process(length, bits):
    """Spec 6.4: draw l uniform over [0, L) and m uniform over [0, 2**bits); the comparator sets
    less_than = [keep <= m], which swaps in alt. Enumerating every (l, m) gives the formula."""
    rng = np.random.default_rng(bits)
    alt, keep = alias_tables(rng.random(length), bits)
    counts = np.zeros(length)
    for index, m in itertools.product(range(length), range(2**bits)):
        less_than = keep[index] <= m
        counts[alt[index] if less_than else index] += 1
    assert np.array_equal(counts / (length * 2**bits), alias_probabilities(alt, keep, bits))


def test_an_all_zero_row_loads_nothing():
    alt, keep = alias_tables(np.zeros(5), bits=3)
    assert alt.tolist() == keep.tolist() == [0] * 5


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="non-negative"):
        alias_tables([1.0, -1.0], bits=2)
    with pytest.raises(ValueError, match="bits"):
        alias_tables([1.0, 1.0], bits=0)


def test_inner_row_signs_follow_the_surviving_index():
    """sgn = [w < 0] and d_sgn = sgn XOR sgn[alt]: Z(sgn) and CZ(less_than, d_sgn) leave every
    branch (l, m) with the sign of the index that survives the swap (spec 6.1)."""
    row = np.array([0.3, -1.2, 0.5, -0.1, 0.9, 0.0, -0.4])
    alt, keep, sgn, d_sgn = inner_alias_row(row, bits=3)
    assert sgn.tolist() == [0, 1, 0, 1, 0, 0, 1]
    assert np.array_equal(d_sgn, sgn ^ sgn[alt])
    assert np.array_equal((alt, keep), alias_tables(np.abs(row), 3))
    for index, m in itertools.product(range(len(row)), range(8)):
        less_than = keep[index] <= m
        survivor = alt[index] if less_than else index
        applied = (-1) ** sgn[index] * (-1) ** (less_than * d_sgn[index])
        assert applied == (-1) ** sgn[survivor]
