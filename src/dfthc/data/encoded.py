"""The Hamiltonian and normalization the circuit encodes: the encoded-Hamiltonian model (spec 8.5).

The circuit block-encodes ``<0|BE|0> = H~_SA / Lambda~ - I``. ``H~_SA`` is the sum-of-squares
Hamiltonian of the *encoded factors*: the prepared alias probabilities instead of the weights, the
sign columns, the directions ``u~`` of the rounded angles, the identity term ``b = B``, and the
failed uniform preparation of ``b``. A failed uniform preparation of ``x_o`` makes ``REF_T2`` act
as ``-I`` there, which renormalizes: ``Lambda~ = Lambda / (1 - P_fail(x_o))``. The construction is
dfthcbe's ``encoded_factors``, with our failure signs (all +1, spec 8.5) in place of its
``failure_signs``. Only the walk-frame offset enters ``H~_SA``. This model is the oracle of the
end-to-end check E-1 (spec 12.4).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import hadamard

from dfthc.data.alias import alias_probabilities
from dfthc.data.angles import vector_from_angles
from dfthc.data.factors import DFTHCFactors
from dfthc.data.fermion import MAX_ORBITALS
from dfthc.data.sos import e_sos_walk, h_walk_dense, normalization
from dfthc.data.tables import WalkTables
from dfthc.params import WalkParams, lg, odd_part


def uniform_failure_probability(L: int, s: int) -> float:
    """``P(succ = 0)`` of the flagged uniform preparation over ``L`` values with ``s``-bit
    rotations: a port of dfthcbe's ``UniformViaPhaseGradient.failure_probability``.

    With ``L = 2**eta l``, ``l`` odd and ``n = lg(l)``, one round of amplitude amplification on
    the top ``n`` qubits uses the angle ``arccos(1 - 2**floor(log2 l) / l)``, loaded as
    ``k = floor(angle / (2 pi) 2**s)``: Qualtran's ``QFxp(s, s).to_fixed_width_int`` truncates.
    The failure probability is the weight left on ``t >= l``; it is 0 if ``l = 1``.
    """
    odd = odd_part(L)
    n = lg(odd)
    if not n:
        return 0.0
    angle = np.arccos(1 - 2 ** np.floor(np.log2(odd)) / odd)
    k = int(np.floor(angle / (2 * np.pi) * 2**s)) % 2**s
    phase, size = np.exp(2j * np.pi * k / 2**s), 2**n
    good, h = np.arange(size) < odd, hadamard(size) / np.sqrt(size)
    psi = h @ (np.where(good, phase, 1) / np.sqrt(size))
    psi[0] *= phase
    return float(np.sum(np.abs(h @ psi)[~good] ** 2))


def our_failure_signs(params: WalkParams, tables: WalkTables) -> np.ndarray:
    """The sign every failed ``b`` of an SF row carries: +1, because our exact read loads zeros
    there, so the branch acts as ``+I`` (spec 8.5)."""
    return np.ones(params.RC)


def _leaf(v: int, n: int, length: int) -> int:
    """The leaf that ``v`` reaches in Qualtran's truncated unary iteration over ``[0, length)``
    on ``n`` bits, also for ``v >= length`` (dfthcbe's ``_leaf``)."""
    lo = 0
    for j in reversed(range(n)):
        if lo + 2**j < length and (v >> j) & 1:
            lo += 2**j
    return lo


def dfthcbe_failure_signs(params: WalkParams, tables: WalkTables) -> np.ndarray:
    """dfthcbe's ``InnerPrepare.failure_signs``, ported for comparison (spec 8.5).

    Per SF row, the mean over the failed ``b in [B+1, 2**w_b)`` of the keep-averaged sign of the
    entry that ``b`` reaches through the truncated iteration, or +1 past the table. It equals
    :func:`our_failure_signs` when ``2**w_b <= ceil((B+1) / 2**k2) 2**k2``.
    """
    p = params
    failed = range(p.B + 1, 2**p.w_b)
    if not failed:
        return np.ones(p.RC)
    kept = np.array(tables.inner_keep) / 2**p.b_k2
    sgn, d_sgn = np.array(tables.inner_sgn), np.array(tables.inner_d_sgn)
    signs = (1 - 2 * sgn) * (kept + (1 - kept) * (1 - 2 * d_sgn))
    blocks = -(-(p.B + 1) // 2**p.k2)
    loaded = [_leaf(b >> p.k2, p.w_b - p.k2, blocks) * 2**p.k2 + b % 2**p.k2 for b in failed]
    return np.mean([signs[:, e] if e <= p.B else np.ones(p.RC) for e in loaded], axis=0)


def encoded_factors(params: WalkParams, tables: WalkTables,
                    factors: DFTHCFactors) -> DFTHCFactors:
    """The factors the circuit encodes, in dfthcbe's convention (its ``encoded_factors``).

    With ``Lambda`` of the input factors, ``p`` and ``q`` the prepared outer and inner
    probabilities and ``P_b`` the failure probability of the uniform preparation of ``b``:
    ``a_b = (1 - P_b) sign(w_b) q_b``, ``a_B += P_b sigma_fail`` (``sigma_fail = +1``),
    ``w~_b = a_b sqrt(4 Lambda p_rc)``, ``w~_pm = Lambda p``, and the unit vectors are
    ``vector_from_angles`` of the loaded angles. ``h1``, ``h_sym``, ``beta_1``, ``e_upper`` and
    ``eta`` are carried over; ``h1_shifted``, ``lambda_h1`` and ``lambda_h2`` are recomputed.
    """
    p, t = params, tables
    lam = normalization(factors)
    outer = alias_probabilities(t.outer_alt, t.outer_keep, p.b_k1)
    inner = np.array([alias_probabilities(alt, keep, p.b_k2)
                      for alt, keep in zip(t.inner_alt, t.inner_keep, strict=True)])
    fail = uniform_failure_probability(p.B + 1, p.s_u)
    amps = (1 - fail) * (1 - 2 * np.array(t.inner_sgn)) * inner
    amps[:, p.B] += fail * our_failure_signs(p, t)
    w = (amps * np.sqrt(4 * lam * outer[:p.RC])[:, None]).reshape(p.R, p.C, p.B + 1)
    w_tail = lam * outer[p.RC:]
    u_tail = np.array([vector_from_angles(row, p.b_rot) for row in t.tail_angles])
    U = np.array([[vector_from_angles(row, p.b_rot) for row in plane] for plane in t.sf_angles])
    w_plus, w_minus = w_tail[:p.n_plus], w_tail[p.n_plus:]
    u_plus, u_minus = u_tail[:p.n_plus], u_tail[p.n_plus:]
    W, w_B = w[..., :p.B].transpose(0, 2, 1), w[..., p.B]
    return DFTHCFactors(
        U=U, W=W, w_B=w_B, w_plus=w_plus, u_plus=u_plus, w_minus=w_minus, u_minus=u_minus,
        h1=factors.h1, h_sym=factors.h_sym,
        h1_shifted=u_plus.T @ np.diag(w_plus) @ u_plus - u_minus.T @ np.diag(w_minus) @ u_minus,
        beta_1=factors.beta_1, lambda_h1=float(w_tail.sum()),
        lambda_h2=float((np.abs(w).sum(axis=-1) ** 2).sum() / 4), e_upper=factors.e_upper,
        eta=factors.eta,
    )


def encoded_block(params: WalkParams, tables: WalkTables,
                  factors: DFTHCFactors) -> tuple[np.ndarray, float]:
    """``(H~_SA, Lambda~)`` with ``<0|BE|0> = H~_SA / Lambda~ - I`` (spec 8.5), for ``N <= 4``.

    ``H~_SA = H_walk - E_SoS^walk`` of :func:`encoded_factors` (a dense ``4**N`` matrix), and
    ``Lambda~ = Lambda / (1 - P_fail(x_o))`` with ``Lambda`` of the input factors.
    """
    if params.N > MAX_ORBITALS:
        raise ValueError(f"the dense encoded model needs N <= {MAX_ORBITALS}; got N = {params.N}")
    enc = encoded_factors(params, tables, factors)
    h_tilde = h_walk_dense(enc) - e_sos_walk(enc) * np.eye(4**params.N)
    lambda_tilde = normalization(factors) / (1 - uniform_failure_probability(params.X_o,
                                                                              params.s_u))
    return h_tilde, lambda_tilde
