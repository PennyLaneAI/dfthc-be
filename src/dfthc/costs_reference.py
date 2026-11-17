"""Reference costs for the resource report (spec 13.1 and 13.3): the paper's Table III formulas,
dfthcbe's per-block Toffoli counts, and the decomposition of our deltas against both into reason
codes.

All three are closed forms in pure Python, so the report never imports Qualtran in the main
environment. :func:`paper_table3` evaluates the equations of Low et al. (PRX 15, 041016;
arXiv:2502.15882v1, App. B and Table III) as Table III does: the inner entry has
``b_2 = w_b + b_k2`` bits (Table III's 20 at FeMoco-54; dfthcbe's README notes that B36 also counts
a sign bit, which Table III's numbers leave out) and the erase one-hots are ``K_5 = 2**ke1`` and
``K_4 = 2**ke2 (B + 1)``. :func:`dfthcbe_blocks` ports the per-step formulas of dfthcbe's README
("Comparison with [1]") at commit 930255020cec30172e0a7d17c0071d9476a687d1, with Qualtran's
truncated unary iteration (``max(L + c - 2, 0)``, over the live leaves and the dead stops of the
README's zero rows together) wherever the README writes ``L - 2``. The README's formulas are
piecewise, so the port is guaranteed only where it is pinned: at the eight parameter sets of the
report (the six vendored systems at their default precisions, FeMoco-54 and ``H2_STO3G_TEST``),
where ``tests/report/test_report.py::test_dfthcbe_blocks_equal_the_call_graph`` checks it against
dfthcbe's own call graph, evaluated in ``.venv-ref`` by ``scripts/make_report_fixture.py``. A new
parameter set needs a fixture entry before its dfthcbe values are trusted. Values are Toffolis per
call (``Toffoli + CSwap + And + T/4`` in Qualtran's ``QECGatesCost``); dfthcbe's Qualtran
"measurement" count is our ``A``, not our ``M``, and is not ported.
"""

from __future__ import annotations

import dataclasses

from dfthc import costs
from dfthc.params import WalkParams, lg, odd_part

DEFAULT_PRECISIONS = {"b_rot": 15, "b_k1": 15, "b_k2": 15, "s_u": 7}
"""The precisions of the paper's Table III, which the report gives every vendored system
(spec 8.3: FeMoco-54's and ``MID_NAPHTHALENE``'s)."""

PAPER_QUBITS_FEMOCO = 1131
"""The paper's logical qubits at FeMoco-54 (spec 10.3, 13.2)."""

PAPER_QUBIT_GROUPS_FEMOCO = {"system": 108, "outer": 50, "inner": 53, "select": 795,
                             "phase_grad": 15, "work": 110}
"""The paper's FeMoco-54 qubits (arXiv:2502.15882v1 Table 3) under the names of our groups: the
system, the persistent ancillas of the outer prepare, the inner prepare, RPREP (the angles) and
SEL (the rotations' b_rot), and its largest temporary ancilla count, the outer prepare's, which
it adds; the inner prepare's 313 temporaries share RPREP's 795 qubits. They sum to
:data:`PAPER_QUBITS_FEMOCO`."""


def default_params(N: int, R: int, B: int, C: int, n_plus: int) -> WalkParams:
    """A system's report parameters: :data:`DEFAULT_PRECISIONS` and the block bits that minimize
    our closed-form walk Toffolis (:func:`dfthc.costs.cheapest_block_bits`, spec 8.3)."""
    p = WalkParams(N=N, R=R, B=B, C=C, n_plus=n_plus, **DEFAULT_PRECISIONS, k1=0, k2=0, ke1=0,
                   ke2=0)
    k1, k2, ke1, ke2 = costs.cheapest_block_bits(p)
    return dataclasses.replace(p, k1=k1, k2=k2, ke1=ke1, ke2=ke2)


def popcount(value: int) -> int:
    """``s(D)`` of dfthcbe's README: the number of ones of ``value`` in binary."""
    return bin(value).count("1")


def aligned_blocks(lo: int, hi: int) -> int:
    """``beta(lo, hi)`` of dfthcbe's README: the aligned blocks ``[j 2^h, (j + 1) 2^h)`` that
    ``[lo, hi)`` splits into, each as large as alignment allows."""
    count = 0
    while lo < hi:
        size = lo & -lo if lo else 1 << (hi.bit_length() - 1)
        while lo + size > hi:
            size //= 2
        lo += size
        count += 1
    return count


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def paper_table3(p: WalkParams) -> dict[str, int]:
    """Toffolis per call of the paper's blocks (Table III; App. B equations), keyed by our row
    names. Table III charges the tail-angle fix-up ``N`` (Eq. (123), B50) to the inner erase; the
    keys ending in ``_step18`` move it to the angle erase, where dfthcbe and we apply it (spec 13.1,
    footnote 1). ``spin_mux`` is Table III's 0; ``spin_mux_app_b3`` the 1 per application that
    App. B.3 and B.7.1 count (``4(N + 1)`` for the four swaps). The paper's QPE step controls only
    ``REF_{A u B}`` (App. B.7.2, Eq. (128)), so its controlled reflection ``REF_T2``, controlled
    block encoding and controlled walk are its uncontrolled ones (keys ``controlled_ref_t2``,
    ``controlled_block_encoding``, ``controlled_walk``). The paper does not split a read from its
    keep test and swap, nor give a standalone rotation, so those rows have no key."""
    b1, b2 = p.w_x + p.b_k1, p.w_b + p.b_k2
    K5, K4 = 2**p.ke1, 2**p.ke2 * (p.B + 1)
    unif_x, unif_b = 4 * lg(p.X_o), 4 * lg(p.B + 1)                       # B24, B54; B32, B52
    outer_erase = _ceil(p.X_o, K5) + K5                                    # B53, the fix-up
    inner_erase = _ceil(p.RC * (p.B + 1), K4) + K4                         # B50, the fix-up
    rows = {
        "uniform_x_o": unif_x, "uniform_x_o_adjoint": unif_x,
        "uniform_b": unif_b, "uniform_b_adjoint": unif_b,
        "outer_erase": outer_erase, "inner_erase": inner_erase,
        "outer_prepare": unif_x + p.M1 + p.K1 * b1,                        # B24, B29
        "outer_unprepare": b1 + outer_erase + unif_x,                      # B53, B54
        "inner_prepare": unif_b + p.M2 + p.K2 * b2,                        # B32, B35, B37
        "inner_unprepare": b2 + inner_erase + p.N + unif_b,                # B50 (with N), B52
        "inner_signs": 0,                                                  # B19
        "tail_angle_load": p.N,                                            # B41
        "sf_angle_load": p.R * p.B,                                        # Table III
        "spin_mux": 0, "spin_mux_app_b3": 1,
        "spin_swap": p.N,                                                  # Table III, 2N each
        "givens_rotation": 2 * p.b_rot, "givens_rotation_adjoint": 2 * p.b_rot,
        "givens_ladder": 2 * p.b_rot * (p.N - 1),                          # Table III
        "givens_ladder_dagger": 2 * p.b_rot * (p.N - 1),
        "majorana_gate": 7, "majorana_gate_adjoint": 7,                    # Fig. 4
        "angle_erase": p.R + p.B,                                          # B47
        "ref_t2": b2 + 1,                                                  # B11
        "controlled_ref_t2": b2 + 1,                                       # B11 uncontrolled
        "ref_ab": b1 + b2 + 2,                                             # B55, QPE control
        "controlled_ref_ab": b1 + b2 + 2,                                  # B55 is controlled
    }
    rows["inner_unprepare_step18"] = rows["inner_unprepare"] - p.N
    rows["angle_erase_step18"] = rows["angle_erase"] + p.N
    select = (rows["tail_angle_load"] + rows["sf_angle_load"] + 2 * rows["spin_mux"]
              + 2 * rows["spin_swap"] + 2 * rows["givens_ladder"] + rows["majorana_gate"]
              + rows["angle_erase"])
    rows["inner_select"] = rows["inner_select_dagger"] = select
    rows["inner_select_step18"] = rows["inner_select_dagger_step18"] = select + p.N
    rows["block_encoding"] = (rows["outer_prepare"] + rows["outer_unprepare"]
                              + 2 * (rows["inner_prepare"] + select + rows["inner_unprepare"])
                              + rows["ref_t2"])
    rows["controlled_block_encoding"] = rows["block_encoding"]
    rows["walk"] = rows["block_encoding"] + rows["ref_ab"]
    rows["controlled_walk"] = rows["walk"]
    return rows


def dfthcbe_blocks(p: WalkParams) -> dict[str, int]:
    """dfthcbe's Toffolis per call, keyed by our row names: the README's per-step formulas at the
    parameters of ``p`` (its block bits included). ``spin_mux`` is half of dfthcbe's
    ``_build_sigma`` pair (4 per select); ``givens_rotation`` one rung of its ladder;
    ``rotation`` one Qualtran ``ZPowConstViaPhaseGradient`` on ``s_u`` bits, ``s_u - 2``. The
    controlled block encoding controls only ``ref_t2`` (+1); the controlled walk's ``ref_ab``
    reuses the control as its phase target (``b_1 + b_2 + 1``, as uncontrolled). dfthcbe builds
    no phase estimation, so the L4 rows have no key."""
    b1, b2 = p.w_x + p.b_k1, p.w_b + p.b_k2
    Ke1, Ke2 = 2**p.ke1, 2**p.ke2
    b_hat = _ceil(p.B + 1, p.K2)
    n_x, n_b = lg(odd_part(p.X_o)), lg(odd_part(p.B + 1))

    def uniform(n: int) -> int:
        return 4 * n + 2 * p.s_u - 5 if n else 0

    def onehot(K: int) -> int:
        return max(K - 2, 0)

    rows = {
        "rotation": p.s_u - 2,
        "uniform_x_o": uniform(n_x), "uniform_x_o_adjoint": uniform(n_x),
        "uniform_b": uniform(n_b), "uniform_b_adjoint": uniform(n_b),
        "outer_read": costs.truncated_unary(p.M1, False) + (p.K1 - 1) * b1,
        "inner_read": (costs.truncated_unary(p.M2 + aligned_blocks(p.RC, p.X_o)
                                             * popcount(b_hat), False)
                       + (p.K2 - 1) * (b2 + 2)),
        "outer_erase": costs.truncated_unary(_ceil(p.X_o, Ke1), False) + onehot(Ke1),
        "inner_erase": (costs.truncated_unary(_ceil(p.RC, Ke2) + aligned_blocks(
            _ceil(p.RC, Ke2), _ceil(p.X_o, Ke2)), False) + onehot(Ke2 * (p.B + 1))),
        "inner_signs": 0,
        "tail_angle_load": costs.truncated_unary(p.N + popcount(p.RC), False),
        "sf_angle_load": costs.truncated_unary(p.R * p.B + popcount(p.R), True),
        "spin_mux": 2,
        "spin_swap": p.N,
        "givens_rotation": 2 * (p.b_rot - 1), "givens_rotation_adjoint": 2 * (p.b_rot - 1),
        "givens_ladder": 2 * (p.N - 1) * (p.b_rot - 1),
        "givens_ladder_dagger": 2 * (p.N - 1) * (p.b_rot - 1),
        "majorana_gate": 5, "majorana_gate_adjoint": 5,
        "angle_erase": (p.R + p.B - 3) + costs.truncated_unary(p.N + popcount(p.RC), False),
        "ref_t2": b2, "controlled_ref_t2": b2 + 1,
        "ref_ab": b1 + b2 + 1, "controlled_ref_ab": b1 + b2 + 1,
    }
    rows["outer_prepare"] = rows["uniform_x_o"] + rows["outer_read"] + b1
    rows["outer_unprepare"] = b1 + rows["outer_erase"] + rows["uniform_x_o_adjoint"]
    rows["inner_prepare"] = rows["uniform_b"] + rows["inner_read"] + b2
    rows["inner_unprepare"] = b2 + rows["inner_erase"] + rows["uniform_b_adjoint"]
    select = (rows["tail_angle_load"] + rows["sf_angle_load"] + 2 * rows["spin_mux"]
              + 2 * rows["spin_swap"] + 2 * rows["givens_ladder"] + rows["majorana_gate"]
              + rows["angle_erase"])
    rows["inner_select"] = rows["inner_select_dagger"] = select
    rows["block_encoding"] = (rows["outer_prepare"] + rows["outer_unprepare"]
                              + 2 * (rows["inner_prepare"] + select + rows["inner_unprepare"])
                              + rows["ref_t2"])
    rows["controlled_block_encoding"] = rows["block_encoding"] - rows["ref_t2"] + rows[
        "controlled_ref_t2"]
    rows["walk"] = rows["block_encoding"] + rows["ref_ab"]
    rows["controlled_walk"] = rows["controlled_block_encoding"] + rows["controlled_ref_ab"]
    return rows


def _nonzero(causes: dict[str, int]) -> dict[str, int]:
    return {code: value for code, value in causes.items() if value}


def _add(*parts: tuple[dict[str, int], int]) -> dict[str, int]:
    """The sum of ``(causes, calls)`` parts, code by code."""
    total: dict[str, int] = {}
    for causes, calls in parts:
        for code, value in causes.items():
            total[code] = total.get(code, 0) + calls * value
    return total


def iteration_counts(p: WalkParams) -> dict[str, tuple[int, int, int, int]]:
    """The unary iterations of the blocks as ``(ours, intermediate, dfthcbe, paper)`` Toffolis per
    call: the counts at which :func:`iteration_causes` splits each iteration difference by
    mechanism.

    ``ours`` is exact unary iteration over the live range (spec 9.24); for the reads it is the
    select's, PennyLane's unary ``QROM`` over the columns up to the last visited block
    (:func:`dfthc.costs.qrom`), also exact, which saves the root's AND. dfthcbe's walk truncates
    and prunes: Qualtran's truncated iteration (``L + c - 2``) over its live leaves and one stop on
    each aligned block of a dead interval that the register can hold (its README, "Zero rows").
    The paper counts one Toffoli per live leaf. The intermediate is an accounting convention, a
    cost model and not a circuit: on the walks from address 0 it is truncated iteration over the
    live leaves alone, with no stops (:func:`dfthc.costs.truncated_unary`, nested walks nested),
    and on the tail rows ``[RC, X_o)`` it is our exact count. It coincides with a valid
    construction only on the outer read, the outer erase's fix-up and the SF fix-up, where
    dfthcbe's walk has no stops and the intermediate is that walk (equal at the report's eight
    parameter sets). It does not elsewhere: truncated iteration routes a dead address that carries
    amplitude onto a live leaf, so dfthcbe stops on the zero D1/Q1 rows in the inner read, on the
    dead tail rows in the inner erase's fix-up and on the identity row ``b = B`` in the SF load,
    and the intermediate leaves those stops out. On the tail rows the dead SF prefix ``[0, RC)``
    carries amplitude and no truncated walk without its stops is valid (truncated iteration over
    ``[RC, X_o)`` assumes an in-range selection), so the convention takes our exact count there:
    the whole difference to dfthcbe is ``no-prune``, and the paper is reached through dfthcbe's
    walk (:func:`iteration_causes`).

    Keys: ``outer_read``, ``inner_read`` (nested over ``x_o`` and the inner block address),
    ``outer_erase`` and ``inner_erase`` (their fix-ups' high address), ``tail`` (the tail angle
    load and the angle erase's tail fix-up), ``sf_load`` (nested over ``r`` and ``b``, enabled on
    ``G0``) and ``sf_fixup`` (the angle erase's walk over ``r``, without its enable, which is
    ``sf-enable``)."""
    N, R, B, w_x = p.N, p.R, p.B, p.w_x
    trunc = costs.truncated_unary
    Ke1, Ke2, K4 = 2**p.ke1, 2**p.ke2, 2**p.ke2 * (B + 1)
    inner_blocks = _ceil(B + 1, p.K2)
    outer_fix, inner_fix = _ceil(p.M1 * p.K1, Ke1), _ceil(p.RC, Ke2)
    tail = costs.unary(p.RC, p.X_o, w_x, False)
    outer_columns, outer_controls = costs.outer_read_select(p)
    inner_columns, inner_controls = costs.inner_read_select(p)
    return {
        "outer_read": (costs.qrom(outer_controls, outer_columns), trunc(p.M1, False),
                       trunc(p.M1, False), p.M1),
        "inner_read": (
            costs.qrom(inner_controls, inner_columns),
            trunc(p.RC, False) + p.RC * trunc(inner_blocks, w_x > 0),
            trunc(p.M2 + aligned_blocks(p.RC, p.X_o) * popcount(inner_blocks), False), p.M2),
        "outer_erase": (costs.unary(0, outer_fix, w_x - p.ke1, False), trunc(outer_fix, False),
                        trunc(_ceil(p.X_o, Ke1), False), _ceil(p.X_o, Ke1)),
        "inner_erase": (
            costs.unary(0, inner_fix, w_x - p.ke2, False), trunc(inner_fix, False),
            trunc(inner_fix + aligned_blocks(inner_fix, _ceil(p.X_o, Ke2)), False),
            _ceil(p.RC * (B + 1), K4)),
        "tail": (tail, tail, trunc(N + popcount(p.RC), False), N),
        "sf_load": (costs.unary(0, R, p.w_r, True) + R * costs.unary(0, B, p.w_b, True),
                    trunc(R, True) + R * trunc(B, True), trunc(R * B + popcount(R), True), R * B),
        "sf_fixup": (costs.unary(0, R, p.w_r, False), trunc(R, False), R - 2, R),
    }


def iteration_causes(p: WalkParams) -> dict[str, tuple[dict[str, int], dict[str, int]]]:
    """For every unary iteration of :func:`iteration_counts`: the signed contribution of each
    reason code to ours minus dfthcbe's and to ours minus the paper's, per call, zeros included.

    Against dfthcbe, ``exact-iter`` is ours minus the intermediate (exact against truncated
    iteration) and ``no-prune`` the intermediate minus dfthcbe (no aligned-block pruning, signed:
    negative where dfthcbe's stops cost more than the intermediate). Against the paper, on the
    walks from address 0, ``exact-iter`` again and ``iter-const``, the intermediate minus the
    paper (truncated iteration's ``c - 2``). The tail walk goes to the paper through dfthcbe's:
    ours minus dfthcbe as above, plus dfthcbe minus the paper by dfthcbe's mechanisms, its
    ``s(RC)`` stops on the dead SF prefix ``[0, RC)`` (``prefix-stops``) and its truncated
    iteration's constant over the ``N + s(RC)`` leaves and stops (``iter-const``)."""
    causes = {}
    for key, (ours, mid, ref, paper) in iteration_counts(p).items():
        against_ref = {"exact-iter": ours - mid, "no-prune": mid - ref}
        if key == "tail":
            stops = popcount(p.RC)
            against_paper = against_ref | {"prefix-stops": stops,
                                           "iter-const": ref - stops - paper}
        else:
            against_paper = {"exact-iter": ours - mid, "iter-const": mid - paper}
        causes[key] = (against_ref, against_paper)
    return causes


def delta_causes(p: WalkParams) -> dict[str, tuple[dict[str, int], dict[str, int]]]:
    """For every row key: the signed contribution of each reason code to ours minus dfthcbe's and
    to ours minus the paper's (Table III's attribution), per call; codes with a zero contribution
    are left out, and two different codes may cancel (for example ``ref-unif`` -1 and
    ``ctrl-ref-t2`` +1 in the controlled ``REF_T2`` against the paper).

    Each block is split into parts that the three constructions share, and each part's difference
    goes to the code that names its mechanism: the uniform preparation's comparators (``cmp-free``)
    and rotations (``rot-add``) against dfthcbe, and dfthcbe's uniform-preparation formula against
    the paper's ``4 lg L`` (``unif-odd``); every unary iteration split by
    :func:`iteration_causes` into exact against truncated iteration (``exact-iter``), no
    aligned-block pruning (``no-prune``) and, against the paper, truncated iteration's constants
    (``iter-const``) and, on the tail walks, dfthcbe's stops on the dead SF prefix
    (``prefix-stops``); the inner entry's ``r`` (``flat-r``) and sign bits (``sign-bits``); the
    one-hots' padding (``onehot-pad``) and constants (``onehot-const``); the SF fix-up's enable
    (``sf-enable``); the tail fix-up the paper charges to the inner erase (``attrib``); the spin
    multiplexer (``mux``), the Givens rotations (``rz-addsub``) and the Majorana gate
    (``maj-and``); the reflections' target (``ref-target``), the paper's extra reflected qubit
    (``ref-unif``) and QPE control (``qpe-ctrl``), and our controlled ``REF_T2``
    (``ctrl-ref-t2``). The report checks at every system that each row's contributions sum to its
    deltas exactly."""
    ref, paper = dfthcbe_blocks(p), paper_table3(p)
    s, N, R, B = p.s_u, p.N, p.R, p.B
    Ke1, K4 = 2**p.ke1, 2**p.ke2 * (B + 1)

    def uniform(L: int, adjoint: bool) -> tuple[dict, dict]:
        n = lg(odd_part(L))
        ours_rest = (2 * n - 2 if adjoint else 3 * n - 3) if n else 0
        ours_rot, ref_rot, ref_rest = (2 * (s - 1), 2 * (s - 2), 4 * n - 1) if n else (0, 0, 0)
        against_ref = {"cmp-free": ours_rest - ref_rest, "rot-add": ours_rot - ref_rot}
        return against_ref, against_ref | {"unif-odd": ref_rot + ref_rest - 4 * lg(L)}

    x_prep, x_unprep = uniform(p.X_o, False), uniform(p.X_o, True)
    b_prep, b_unprep = uniform(B + 1, False), uniform(B + 1, True)
    iteration = iteration_causes(p)
    inner_onehot = costs.onehot(p.ke2 + p.w_b)
    causes: dict[str, tuple[dict, dict]] = {}
    causes["outer_read"] = (iteration["outer_read"][0], {})
    causes["inner_read"] = (_add((iteration["inner_read"][0], 1),
                                 ({"flat-r": (p.K2 - 1) * p.w_r}, 1)), {})
    causes["outer_erase"] = (iteration["outer_erase"][0],
                             _add((iteration["outer_erase"][1], 1),
                                  ({"onehot-const": costs.onehot(p.ke1) - Ke1}, 1)))
    pad = {"onehot-pad": inner_onehot - max(K4 - 2, 0)}
    causes["inner_erase"] = (
        _add((iteration["inner_erase"][0], 1), (pad, 1)),
        _add((iteration["inner_erase"][1], 1), (pad, 1),
             ({"onehot-const": max(K4 - 2, 0) - K4}, 1)))
    causes["uniform_x_o"], causes["uniform_x_o_adjoint"] = x_prep, x_unprep
    causes["uniform_b"], causes["uniform_b_adjoint"] = b_prep, b_unprep
    causes["rotation"] = ({"rot-add": 1}, {})
    causes["outer_prepare"] = (_add((x_prep[0], 1), (iteration["outer_read"][0], 1)),
                               _add((x_prep[1], 1), (iteration["outer_read"][1], 1)))
    causes["outer_unprepare"] = (
        _add((x_unprep[0], 1), ({"cmp-free": -p.b_k1}, 1), (causes["outer_erase"][0], 1)),
        _add((x_unprep[1], 1), ({"cmp-free": -p.b_k1}, 1), (causes["outer_erase"][1], 1)))
    causes["inner_prepare"] = (
        _add((b_prep[0], 1), (causes["inner_read"][0], 1)),
        _add((b_prep[1], 1), (iteration["inner_read"][1], 1),
             ({"flat-r": (p.K2 - 1) * p.w_r, "sign-bits": 2 * (p.K2 - 1)}, 1)))
    causes["inner_unprepare"] = (
        _add((b_unprep[0], 1), ({"cmp-free": -p.b_k2}, 1), (causes["inner_erase"][0], 1)),
        _add((b_unprep[1], 1), ({"cmp-free": -p.b_k2}, 1), (causes["inner_erase"][1], 1),
             ({"attrib": -N}, 1)))
    causes["inner_signs"] = ({}, {})
    causes["tail_angle_load"] = iteration["tail"]
    causes["sf_angle_load"] = iteration["sf_load"]
    causes["spin_mux"] = ({"mux": 1 - ref["spin_mux"]}, {"mux": 1 - paper["spin_mux"]})
    causes["spin_swap"] = ({}, {})
    for key in ("givens_rotation", "givens_rotation_adjoint"):
        causes[key] = ({}, {"rz-addsub": 2 * (p.b_rot - 1) - paper[key]})
    for key in ("givens_ladder", "givens_ladder_dagger"):
        causes[key] = ({}, {"rz-addsub": 2 * (p.b_rot - 1) * (N - 1) - paper[key]})
    for key in ("majorana_gate", "majorana_gate_adjoint"):
        causes[key] = ({}, {"maj-and": 5 - paper[key]})
    fixup = {"sf-enable": costs.unary(0, R, p.w_r, True) - costs.unary(0, R, p.w_r, False),
             "onehot-pad": costs.onehot(p.w_b) - (B - 1)}
    causes["angle_erase"] = (
        _add((fixup, 1), (iteration["sf_fixup"][0], 1), (iteration["tail"][0], 1)),
        _add((fixup, 1), ({"onehot-const": -1, "attrib": N}, 1), (iteration["sf_fixup"][1], 1),
             (iteration["tail"][1], 1)))
    select_parts = ("tail_angle_load", "sf_angle_load", "spin_mux", "spin_mux", "spin_swap",
                    "spin_swap", "givens_ladder", "givens_ladder_dagger", "majorana_gate",
                    "angle_erase")
    causes["inner_select"] = tuple(_add(*((causes[k][i], 1) for k in select_parts))
                                   for i in (0, 1))
    causes["inner_select_dagger"] = causes["inner_select"]
    causes["ref_t2"] = ({}, {"ref-unif": -1})
    causes["controlled_ref_t2"] = ({}, {"ref-unif": -1, "ctrl-ref-t2": 1})
    causes["ref_ab"] = ({"ref-target": -1}, {"ref-target": -1, "qpe-ctrl": -1})
    causes["controlled_ref_ab"] = ({}, {"ref-target": -1})
    inner_round = ("inner_prepare", "inner_select", "inner_unprepare", "inner_signs")
    for name, reflection in (("block_encoding", "ref_t2"),
                             ("controlled_block_encoding", "controlled_ref_t2")):
        causes[name] = tuple(_add((causes["outer_prepare"][i], 1),
                                  (causes["outer_unprepare"][i], 1),
                                  *((causes[k][i], 2) for k in inner_round),
                                  (causes[reflection][i], 1)) for i in (0, 1))
    causes["walk"] = tuple(_add((causes["block_encoding"][i], 1), (causes["ref_ab"][i], 1))
                           for i in (0, 1))
    causes["controlled_walk"] = tuple(
        _add((causes["controlled_block_encoding"][i], 1), (causes["controlled_ref_ab"][i], 1))
        for i in (0, 1))
    return {key: (_nonzero(against_ref), _nonzero(against_paper))
            for key, (against_ref, against_paper) in causes.items()}
