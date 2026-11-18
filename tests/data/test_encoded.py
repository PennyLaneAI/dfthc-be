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

"""The encoded-Hamiltonian model and the failure signs (spec 8.5; spec 12.1 T1 and T3 item 3)."""

import dataclasses

import numpy as np
import pytest

from dfthc.data.alias import alias_probabilities
from dfthc.data.angles import vector_from_angles
from dfthc.data.encoded import (
    dfthcbe_failure_signs,
    encoded_block,
    encoded_factors,
    our_failure_signs,
    uniform_failure_probability,
)
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.fermion import annihilation
from dfthc.data.sos import h_sa_dense, normalization
from dfthc.data.tables import WalkTables
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams, lg

TEST_PRESETS = [(H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"), (H2_631G_TEST, "h2_631g_R2_B2_C2")]
X_O_FAILS = WalkParams(N=2, R=3, B=1, C=1, n_plus=1, b_rot=6, b_k1=3, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
"""X_o = 5, so the uniform preparation of x_o fails too (dfthcbe's "x_o_fails" shape). At
b_rot = 6 the loaded directions are generic (at 4 bits every tail angle is a multiple of pi/2),
and b_k1 != b_k2 tells the outer keep bits from the inner ones."""

FAILURE_SIGN_MARGIN = 1e-2
"""The documented difference between dfthcbe's failure signs and ours exceeds this. The maximal
differences are 1/6 at (B+1, k2) = (5, 1) and 2 at (7, 0). With b_k2 = 2 keep bits an entry sign
differs from +1 by a multiple of 1/2, so every nonzero difference is a multiple of 1/6 at (5, 1),
an average over three failed b, and of 1/2 at (7, 0)."""


def _setup(params, factors):
    return params, WalkTables.from_factors(factors, params), factors


def _lcu_block(params, tables, factors):
    """<0|BE|0> from the circuit's structure, without walk_integrals (spec 3.3 and 8.5):
    (1 - P_x) sum_x p_x avg_sigma (2 A_x^dag A_x) - I, with A_SF = a_B + sum_b a_b M_b,
    M_b = (i/2) sum_sigma gamma_0 gamma_1 along u~_b, a_b the failure-weighted signed inner
    probabilities, and A = a_u (D1) or a_u^dag (Q1) on one spin. A failed b acts as +I: our
    exact read loads zeros there."""
    p, t, n = params, tables, params.N
    ann = np.array([[annihilation(j, s, n) for s in (0, 1)] for j in range(n)])
    eye = np.eye(4**n)

    def a_u(u, spin):
        return np.tensordot(u, ann[:, spin], axes=1)

    def pair(u):
        terms = [a_u(u, s) for s in (0, 1)]
        return sum(0.5j * (a + a.conj().T) @ (-1j * (a - a.conj().T)) for a in terms)

    outer = alias_probabilities(t.outer_alt, t.outer_keep, p.b_k1)
    fail_b = uniform_failure_probability(p.B + 1, p.s_u)
    block = 0
    for x in range(p.X_o):
        if x < p.RC:
            q = alias_probabilities(t.inner_alt[x], t.inner_keep[x], p.b_k2)
            amps = (1 - fail_b) * (1 - 2 * np.array(t.inner_sgn[x])) * q
            amps[p.B] += fail_b
            u = [vector_from_angles(t.sf_angles[x // p.C][b], p.b_rot) for b in range(p.B)]
            a = amps[p.B] * eye + sum(amps[b] * pair(u[b]) for b in range(p.B))
            block = block + outer[x] * 2 * a.conj().T @ a
        else:
            j = x - p.RC
            u = vector_from_angles(t.tail_angles[j], p.b_rot)
            ops = [a_u(u, s) if j < p.n_plus else a_u(u, s).conj().T for s in (0, 1)]
            block = block + outer[x] * sum(o.conj().T @ o for o in ops)
    return (1 - uniform_failure_probability(p.X_o, p.s_u)) * block - eye


@pytest.mark.parametrize(("params", "system"), TEST_PRESETS)
def test_the_two_constructions_agree(params, system):
    """Spec 8.6 check 2 for the encoded factors: H_walk - E_SoS^walk equals the generators."""
    p, t, f = _setup(params, load_factors(system))
    h_tilde, _ = encoded_block(p, t, f)
    assert np.allclose(h_tilde, h_sa_dense(encoded_factors(p, t, f)), rtol=0, atol=1e-10)


LCU_CASES = {
    "h2_sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "h2_631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "x_o_fails": lambda: (X_O_FAILS, random_factors(2, 3, 1, 1, n_plus=1, seed=3)),
}
FAILS = {"h2_sto3g": (False, True), "h2_631g": (False, True), "x_o_fails": (True, False)}
"""Whether the uniform preparation of x_o and of b fails: B + 1 = 3 fails, X_o = 5 fails."""


@pytest.mark.parametrize("case", list(LCU_CASES))
def test_block_equals_the_lcu_structure(case):
    p, t, f = _setup(*LCU_CASES[case]())
    h_tilde, lambda_tilde = encoded_block(p, t, f)
    failures = [uniform_failure_probability(p.X_o, p.s_u),
                uniform_failure_probability(p.B + 1, p.s_u)]
    assert tuple(failure > 0 for failure in failures) == FAILS[case]
    assert lambda_tilde == pytest.approx(normalization(f) / (1 - failures[0]), rel=1e-14, abs=0)
    assert np.allclose(h_tilde / lambda_tilde - np.eye(len(h_tilde)), _lcu_block(p, t, f),
                       rtol=0, atol=1e-10)


@pytest.mark.parametrize(("params", "system"), TEST_PRESETS)
def test_encoded_hamiltonian_approaches_h_sa(params, system):
    f = load_factors(system)
    errors = {}
    for bits in (4, 12):
        p = dataclasses.replace(params, b_rot=bits, b_k1=bits, b_k2=bits, s_u=bits)
        h_tilde, _ = encoded_block(p, WalkTables.from_factors(f, p), f)
        errors[bits] = np.linalg.norm(h_tilde - h_sa_dense(f), 2)
    assert errors[12] < 1e-2 < errors[4]


def test_encoded_factors_are_valid_factors_of_the_same_shape():
    p, t, f = _setup(H2_631G_TEST, load_factors("h2_631g_R2_B2_C2"))
    enc = encoded_factors(p, t, f)
    enc.validate()
    assert (enc.N, enc.R, enc.B, enc.C, enc.n_plus) == (f.N, f.R, f.B, f.C, f.n_plus)
    assert enc.eta == f.eta


def test_uniform_failure_probability_reproduces_dfthcbe():
    """Values from dfthcbe's UniformViaPhaseGradient.failure_probability at the pinned commit
    (Qualtran 0.7.0)."""
    recorded = {(3, 3): 0.07858495705504472, (5, 3): 0.1506780981594029,
                (324, 7): 0.0006445392838034328, (28, 7): 0.0007020808150612868,
                (7, 4): 0.029699337423164093, (25, 12): 9.387453336647742e-07}
    for (L, s), value in recorded.items():
        assert uniform_failure_probability(L, s) == pytest.approx(value, rel=1e-12, abs=0)
    assert uniform_failure_probability(4, 3) == uniform_failure_probability(64, 7) == 0.0


def _signs_case(B, k2, seed):
    """N = 2 with the smallest b_rot from 4 up whose select one-hot fits, 2**w_b <= b_rot
    (spec 8.3); the signs read only the inner tables."""
    b_rot = max(4, 2 ** lg(B + 1))
    params = WalkParams(N=2, R=2, B=B, C=1, n_plus=1, b_rot=b_rot, b_k1=2, b_k2=2, s_u=3, k1=1,
                        k2=k2, ke1=1, ke2=1)
    factors = random_factors(2, 2, B, 1, n_plus=1, seed=seed)
    return params, WalkTables.from_factors(factors, params), factors


@pytest.mark.parametrize(("B", "k2"), [(2, 1), (5, 2), (27, 4)])
def test_failure_signs_agree_when_the_last_block_covers_the_register(B, k2):
    params, tables, _ = _signs_case(B, k2, seed=11)
    assert 2**params.w_b <= -(-(B + 1) // 2**k2) * 2**k2
    assert np.array_equal(our_failure_signs(params, tables), np.ones(params.RC))
    assert np.allclose(dfthcbe_failure_signs(params, tables), our_failure_signs(params, tables),
                       rtol=0, atol=1e-12)


@pytest.mark.parametrize(("B", "k2", "reads"), [(4, 1, (1, 1, "B")), (6, 0, ("B",))])
def test_failure_signs_differ_as_documented(B, k2, reads):
    """(B+1, k2) = (5, 1): failed b = 5, 6, 7 read padding, entry 4 (= B) and padding in dfthcbe;
    (7, 0): failed b = 7 reads entry 6 (= B). Ours read padding everywhere: +1, and the model
    follows ours."""
    params, tables, factors = _signs_case(B, k2, seed=5)
    kept = np.array(tables.inner_keep)[:, B] / 2**params.b_k2
    sgn, d_sgn = np.array(tables.inner_sgn)[:, B], np.array(tables.inner_d_sgn)[:, B]
    entry_sign = (1 - 2 * sgn) * (kept + (1 - kept) * (1 - 2 * d_sgn))
    expected = np.mean([entry_sign if r == "B" else np.ones(params.RC) for r in reads], axis=0)
    port = dfthcbe_failure_signs(params, tables)
    assert np.allclose(port, expected, rtol=0, atol=1e-12)
    assert np.abs(port - our_failure_signs(params, tables)).max() > FAILURE_SIGN_MARGIN
    h_tilde, lambda_tilde = encoded_block(params, tables, factors)
    assert np.allclose(h_tilde / lambda_tilde - np.eye(16), _lcu_block(params, tables, factors),
                       rtol=0, atol=1e-10)


def test_encoded_block_needs_dense_sizes():
    f = random_factors(5, 2, 2, 1, n_plus=2, seed=0)
    p = dataclasses.replace(H2_STO3G_TEST, N=5, n_plus=2)
    with pytest.raises(ValueError, match="N <= 4"):
        encoded_block(p, WalkTables.from_factors(f, p), f)
