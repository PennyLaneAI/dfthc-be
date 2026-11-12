"""Spike K-1: compile a complete FeMoco-54-shaped stand-in walk and controlled walk.

Spec section 16, K-1. The stand-in has the walk's structure: WalkLayout-style registers, the
operator nesting of spec 9.21-9.22 with pinned adjoint and controlled rules, prototype QROAM
reads, measured erases, angle loads and angle erase with random tables of FeMoco shape, and
53-rung Givens ladders built exactly as spec 9.14. Placeholders: the flagged uniform preparation
(Hadamards and an X on the success flag) and the Majorana gate (five Toffoli-class gates).

Usage:
    python spikes/k1_whole_program.py --shape small --program walk   # one compile, JSON line
    python spikes/k1_whole_program.py --shape mid14 --all            # driver: one process per
                                                                     # program, under the watchdog
    python spikes/k1_whole_program.py --project                      # FeMoco projection
The driver compiles SHAPE_PROGRAMS[shape] within LIMITS[shape] (seconds, and the watchdog's memory
cap per program, which kills at effective_limit_mb(cap): 15/16 of it) and
writes spikes/results/k1_<shape>.json. Memory is MEMORY_METHOD. The controlled walk must cost
exactly 2 more Toffolis, 2 more measured uncomputes and the same explicit measurements as the walk.
Every warning of a compile is recorded and classified (spikes/prototypes.classify_warnings); the
result files keep each text up to its first "{" plus the suffix after its last "}".
Exit status 1 if a compile fails, is killed or over its limits, a warning names a prototype
operator, or the controlled-walk deltas differ. Catalyst also lowers the generic adjoint and
controlled candidates of operators with pinned rules and drops the measuring ones with a "Skipped
... it contains a mid-circuit measurement" warning, which for the measured erases
(EXPECTED_MEASURING) is recorded as "expected" and does not fail the verdict. A needed measuring
path that is missing fails the compile with Catalyst's CompileError when the modified target has
no other rule (both measuring prototypes have exactly one rule); a substitute rule would be caught
by the exact counts.

The FeMoco shape is a large-scale compile (master plan section 12.9). --project fits the small and
mid results (PROJECTION_METHOD) and writes k1_projection.json. The FeMoco driver refuses, with exit
status 2 and without compiling, unless that projection belongs to the current code_hash() (the
driver, the prototypes, dfthc.compile, dfthc.testing.resources, the watchdog, the toolchain
versions and the lock file) and keeps the walk and the controlled walk within 60 minutes and the
effective limit of the 8 GB cap (7,680 MB), by the sampled peak and the getrusage companion.
Each FeMoco program compiles at most once per code_hash(): k1_femoco.json is checkpointed after
every program, a completed program is reused, and a failed one is retried only with
--allow-retry, which needs the user's approval. A FeMoco --program run without the token of a
running driver is refused.

--remedy 1 (Task 0.9b) compiles the partial remedy-1 stand-in instead: the reads and angle loads
take their table bits from a dynamic int64 array (ProtoTableReadR1), the erases are unchanged. It
writes k1r1_<shape>.json, requires T, M and A equal to Task 0.9's (EXPECTED_TMA), kept bit loops
and no prototype warning per program, stops at the first program that fails, starts a mid shape
only after the smaller shapes passed for this code_hash(), and compiles no FeMoco shape.
--project --remedy 1 writes k1r1_projection.json (REMEDY_PROJECTION_METHOD).

--arm F|FP|FPG (Task 0.9c, on top of partial remedy 1) compiles an arm: F uses the spec's erase
fixup with one-hot bits (erase_geometry; ProtoTableEraseF, ProtoAngleEraseF), FP also runs the
conditional CZs of each fixup leaf in one static-bound qp.for_loop, and FPG also loops the Givens
ladder rungs (ProtoGivensLadderLoop, with a pinned explicit adjoint rule). --fixup-check runs arm
F's tape-mode correctness check first (fixup_check). --all --arm writes k1c_<arm>_<shape>.json
with the stop gate of arm_problems (T/M/A equal to the arm's resource functions, the independent
count and arm F; loops per class; conditional sites), refuses per arm_refusal, applies the mid14
controlled-walk rule of arm_controlled_skip and, with DFTHC_K1C_LEDGER set, the budget guard.
--project --arm writes that arm into k1c_projection.json (ARM_PROJECTION_METHOD). Task 0.9c
compiles no FeMoco shape.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import resource
import secrets
import subprocess
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.core.operator import abstractify
from pennylane.decomposition import DecompositionGraph
from pennylane.transforms import _tape_decompose
from pennylane.typing import AbstractArray, Wire

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prototypes  # noqa: E402
from prototypes import (  # noqa: E402
    WORD,
    ProtoTableErase,
    ProtoTableEraseF,
    ProtoTableEraseFP,
    ProtoTableRead,
    _Count,
    _Emit,
    _erase_counts,
    _measure_x,
    _split_high,
    classify_warnings,
    counts_to_resources,
    fixup_leaves,
    from_words,
    loaded_masks,
    measured_parities,
    nested_iterate,
    onehot_erase_body,
    onehot_erase_counts,
    phase_fixup,
    to_words,
    unary_toffolis,
)

import dfthc.compile  # noqa: E402
import dfthc.testing.resources  # noqa: E402
import dfthc.testing.watchdog  # noqa: E402
from dfthc.compile import (  # noqa: E402
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    mcm_parities,
    register_rule,
    resource_key,
)
from dfthc.testing.watchdog import (  # noqa: E402
    HEAVY_MEMORY_MB,
    MEMORY_CAP_MB,
    effective_limit_mb,
    run_watched,
    within_effective_limit,
)


def lg(value: int) -> int:
    return max(1, math.ceil(math.log2(value)))


@dataclass(frozen=True)
class Shape:
    N: int
    R: int
    B: int
    C: int
    n_plus: int
    b_rot: int
    b_k1: int
    b_k2: int
    k1: int
    k2: int

    @property
    def X_o(self):
        return self.N + self.R * self.C

    @property
    def RC(self):
        return self.R * self.C

    @property
    def w_x(self):
        return lg(self.X_o)

    @property
    def w_b(self):
        return lg(self.B + 1)

    @property
    def w_r(self):
        return lg(self.R)

    @property
    def d1(self):
        return self.w_x + self.b_k1

    @property
    def d2(self):
        return self.w_b + self.b_k2 + 2 + self.w_r


SHAPES = {
    "femoco": Shape(N=54, R=10, B=27, C=27, n_plus=27, b_rot=15, b_k1=15, b_k2=15, k1=2, k2=4),
    "mid14": Shape(N=14, R=6, B=13, C=8, n_plus=4, b_rot=15, b_k1=15, b_k2=15, k1=2, k2=2),
    "mid10": Shape(N=10, R=8, B=6, C=5, n_plus=5, b_rot=15, b_k1=15, b_k2=15, k1=2, k2=1),
    "small": Shape(N=4, R=2, B=2, C=2, n_plus=2, b_rot=4, b_k1=2, b_k2=2, k1=1, k2=1),
}


class Layout:
    """Stand-in for WalkLayout: consecutive named registers (spec 10.1)."""

    def __init__(self, s: Shape):
        self._next = 0
        reg = self._take
        self.psi_up, self.psi_dn = reg(s.N), reg(s.N)
        self.x_o, self.succ_x_o, self.sigma_1 = reg(s.w_x), reg(1), reg(1)
        self.m_x_o, self.lt_x_o, self.cmp_x_o = reg(s.b_k1), reg(1), reg(s.b_k1 - 1)
        self.alt_x_o, self.keep_x_o = reg(s.w_x), reg(s.b_k1)
        self.junk_x_o = reg((2**s.k1 - 1) * s.d1)
        self.b, self.succ_b, self.sigma_0 = reg(s.w_b), reg(1), reg(1)
        self.m_b, self.lt_b, self.cmp_b = reg(s.b_k2), reg(1), reg(s.b_k2 - 1)
        self.alt_b, self.keep_b, self.sgn, self.d_sgn = reg(s.w_b), reg(s.b_k2), reg(1), reg(1)
        self.r = reg(s.w_r)
        self.junk_b = reg((2**s.k2 - 1) * s.d2)
        self.angles = reg((s.N - 1) * s.b_rot)
        self.G0, self.G1, self.b_neq_B, self.sg = reg(1), reg(1), reg(1), reg(1)
        self.phase_grad = reg(s.b_rot)
        refab = s.w_x + s.b_k1 + s.w_b + s.b_k2 + 2
        self.work = reg(max(refab, s.w_x + s.w_b + 2, s.b_rot) + 2)
        self.num_wires = self._next

    def _take(self, size):
        wires = list(range(self._next, self._next + size))
        self._next += size
        return wires


def random_tables(s: Shape, seed: int = 54):
    """Random tables of the walk's shapes, as compilable tuples of 32-bit words."""
    rng = np.random.default_rng(seed)
    outer = tuple(
        (x, to_words(int(rng.integers(0, s.X_o)) | int(rng.integers(0, 2**s.b_k1)) << s.w_x, s.d1))
        for x in range(s.X_o)
    )
    inner = []
    for row in range(s.RC):
        for b in range(s.B + 1):
            value = int(rng.integers(0, 2**s.d2 // 2**s.w_r)) | (row // s.C) << (s.d2 - s.w_r)
            inner.append(((row << s.w_b) | b, to_words(value, s.d2)))
    angle_bits = (s.N - 1) * s.b_rot

    def angles():
        value = 0
        for k in range(s.N - 1):
            value |= int(rng.integers(0, 2 ** (s.b_rot - 1))) << (k * s.b_rot)
        return value

    tail = tuple(
        (s.RC + j, to_words(angles() | 1 << angle_bits | int(j >= s.n_plus) << (angle_bits + 1),
                            angle_bits + 2))
        for j in range(s.N)
    )
    sf = tuple(
        ((r << s.w_b) | b, to_words(angles() | 1 << angle_bits, angle_bits + 1))
        for r in range(s.R) for b in range(s.B)
    )
    return outer, tuple(inner), tail, sf


def _abstract(op_class, *args, **kwargs):
    return resource_key(lambda: op_class(*args, **kwargs))


# --------------------------------------------------------------------------- small blocks


class ProtoUniformPrep(Operator2):
    """Placeholder for FlaggedUniformPrep: Hadamards on target, X on succ."""

    wire_argnames = ("target", "succ")

    def __init__(self, target, succ):
        super().__init__(target, succ)


@qp.register_resources(lambda target, succ: {qp.H: len(target), qp.X: 1})
def _proto_uniform(target, succ):
    for wire in target:
        qp.H(wire)
    qp.X(succ[0])


register_rule(ProtoUniformPrep, _proto_uniform)


class ProtoSpinMux(Operator2):
    wire_argnames = ("G0", "sigma_0", "sigma_1", "sg")

    def __init__(self, G0, sigma_0, sigma_1, sg):
        super().__init__(G0, sigma_0, sigma_1, sg)


@qp.register_resources({qp.CNOT: 3, qp.Toffoli: 1})
def _proto_spin_mux(G0, sigma_0, sigma_1, sg):
    qp.CNOT([sigma_0[0], sg[0]])
    qp.CNOT([sigma_0[0], sigma_1[0]])
    qp.Toffoli([G0[0], sigma_1[0], sg[0]])
    qp.CNOT([sigma_0[0], sigma_1[0]])


def _self_adjoint_resources(base):
    return {_abstract(type(base), **base.arguments): 1}


def _self_adjoint(base):
    type(base)(**base.arguments)


register_rule(ProtoSpinMux, _proto_spin_mux)
register_rule("Adjoint(ProtoSpinMux)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))


class ProtoSpinSwap(Operator2):
    wire_argnames = ("sg", "psi_up", "psi_dn")

    def __init__(self, sg, psi_up, psi_dn):
        super().__init__(sg, psi_up, psi_dn)


@qp.register_resources(lambda sg, psi_up, psi_dn: {qp.CSWAP: len(psi_up)})
def _proto_spin_swap(sg, psi_up, psi_dn):
    for up, dn in zip(psi_up, psi_dn, strict=True):
        qp.CSWAP([sg[0], up, dn])


register_rule(ProtoSpinSwap, _proto_spin_swap)
register_rule("Adjoint(ProtoSpinSwap)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))


class ProtoInnerSigns(Operator2):
    wire_argnames = ("sgn", "d_sgn", "less_than_b")

    def __init__(self, sgn, d_sgn, less_than_b):
        super().__init__(sgn, d_sgn, less_than_b)


@qp.register_resources({qp.Z: 1, qp.CZ: 1})
def _proto_inner_signs(sgn, d_sgn, less_than_b):
    qp.Z(sgn[0])
    qp.CZ([less_than_b[0], d_sgn[0]])


register_rule(ProtoInnerSigns, _proto_inner_signs)
register_rule("Adjoint(ProtoInnerSigns)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))


class ProtoMajorana(Operator2):
    """Placeholder Majorana gate with five Toffoli-class gates (spec 9.16 cost)."""

    wire_argnames = ("succ_x_o", "succ_b", "G0", "G1", "b_neq_B", "flavour", "target", "work")

    def __init__(self, succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
        super().__init__(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work)


def _majorana_resources(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    return {resource_key(lambda: qp.TemporaryAND(Wire[3])): 2,
            resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))): 2,
            qp.Toffoli: 3, qp.H: 4, qp.CZ: 1}


@qp.register_resources(_majorana_resources)
def _proto_majorana(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    a, s = work[0], work[1]
    qp.TemporaryAND([succ_x_o[0], G0[0], a])
    qp.TemporaryAND([succ_x_o[0], succ_b[0], s])
    qp.H(target[0])
    qp.Toffoli([s, b_neq_B[0], target[0]])
    qp.H(target[0])
    qp.Toffoli([a, G1[0], target[0]])
    qp.H(flavour[0])
    qp.Toffoli([a, target[0], flavour[0]])
    qp.H(flavour[0])
    qp.CZ([a, target[0]])
    qp.adjoint(qp.TemporaryAND([succ_x_o[0], succ_b[0], s]))
    qp.adjoint(qp.TemporaryAND([succ_x_o[0], G0[0], a]))


register_rule(ProtoMajorana, _proto_majorana)


# --------------------------------------------------------------------------- Givens (spec 9.14)


class ProtoGivensRotation(Operator2):
    wire_argnames = ("q_i", "q_j", "angle", "phase_grad", "work")

    def __init__(self, q_i, q_j, angle, phase_grad, work):
        super().__init__(q_i, q_j, angle, phase_grad, work)


def _givens_resources(q_i, q_j, angle, phase_grad, work):
    b = len(angle)
    adder = resource_key(lambda: qp.SemiAdder(Wire[b], Wire[b], Wire[b - 1]))
    return {qp.H: 8 + 6, qp.S: 3, resource_key(lambda: qp.adjoint(qp.S(Wire[1]))): 3,
            qp.CNOT: 2 + 4 * b, qp.X: 4, adder: 2}


def _rz(q, angle, phase_grad, work):
    qp.X(q)
    for wire in phase_grad:
        qp.CNOT([q, wire])
    qp.SemiAdder(angle, phase_grad, work)
    for wire in phase_grad:
        qp.CNOT([q, wire])
    qp.X(q)


@qp.register_resources(_givens_resources)
def _proto_givens(q_i, q_j, angle, phase_grad, work):
    i, j = q_i[0], q_j[0]
    qp.H(i), qp.adjoint(qp.S(i)), qp.H(i)
    qp.S(j), qp.H(j), qp.adjoint(qp.S(j)), qp.H(j)
    qp.CNOT([i, j])
    qp.H(i)
    _rz(i, angle, phase_grad, work)
    _rz(j, angle, phase_grad, work)
    qp.H(i)
    qp.CNOT([i, j])
    qp.H(i), qp.S(i), qp.H(i)
    qp.H(j), qp.S(j), qp.H(j), qp.adjoint(qp.S(j))


register_rule(ProtoGivensRotation, _proto_givens)


class ProtoGivensLadder(Operator2):
    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        super().__init__(b_rot, system, angles, phase_grad, work)


def _ladder_resources(b_rot, system, angles, phase_grad, work):
    rung = _abstract(ProtoGivensRotation, Wire[1], Wire[1], Wire[b_rot], Wire[b_rot],
                     Wire[b_rot - 1])
    return {rung: len(system) - 1}


@qp.register_resources(_ladder_resources)
def _proto_ladder(b_rot, system, angles, phase_grad, work):
    for k in range(len(system) - 1):
        ProtoGivensRotation([system[k]], [system[k + 1]], angles[k * b_rot:(k + 1) * b_rot],
                            phase_grad, work[: b_rot - 1])


register_rule(ProtoGivensLadder, _proto_ladder)


# --------------------------------------------------------------------------- reflections


class ProtoZeroReflection(Operator2):
    """2 Pi - I on zero_wires (pattern 0) and one_wires (pattern 1), spec 9.6."""

    wire_argnames = ("zero_wires", "one_wires", "work")

    def __init__(self, zero_wires, one_wires, work):
        super().__init__(zero_wires, one_wires, work)


def _mcz(controls, values, target, work):
    """Multi-controlled Z with pattern ``values`` on controls, 1 on target.

    Inside a rule, wire registers are arrays: never test them for truth (``work or None``).
    """
    num_work = max(len(controls) - 2, 0)
    qp.H(target)
    qp.MultiControlledX(wires=list(controls) + [target], control_values=list(values),
                        work_wires=list(work[:num_work]) if num_work else None,
                        work_wire_type="zeroed")
    qp.H(target)


def _mcz_key(num_controls, values):
    return resource_key(lambda: qp.MultiControlledX(
        Wire[num_controls + 1], control_values=list(values),
        work_wires=Wire[max(num_controls - 2, 0)] if num_controls > 2 else None,
        work_wire_type="zeroed"))


def _reflection_parts(zero_wires, one_wires):
    pattern = [0] * len(zero_wires) + [1] * len(one_wires)
    wires = list(zero_wires) + list(one_wires)
    return wires[:-1], pattern[:-1], wires[-1], pattern[-1]


def _zero_reflection_resources(zero_wires, one_wires, work):
    controls, values, _, last = _reflection_parts(zero_wires, one_wires)
    return {_mcz_key(len(controls), values): 1, qp.H: 2, qp.X: 2 * (last == 0), qp.GlobalPhase: 1}


@qp.register_resources(_zero_reflection_resources)
def _proto_zero_reflection(zero_wires, one_wires, work):
    controls, values, target, last = _reflection_parts(zero_wires, one_wires)
    if last == 0:
        qp.X(target)
    _mcz(controls, values, target, work)
    if last == 0:
        qp.X(target)
    qp.GlobalPhase(np.pi)


def _c_zero_reflection_resources(base, control_wires, control_values, work_wires, work_wire_type):
    controls, values, _, last = _reflection_parts(base.zero_wires, base.one_wires)
    return {_mcz_key(len(controls) + 1, [1] + list(values)): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.Z: 1}


@qp.register_resources(_c_zero_reflection_resources)
def _proto_c_zero_reflection(base, control_wires, control_values, work_wires, work_wire_type):
    controls, values, target, last = _reflection_parts(base.zero_wires, base.one_wires)
    if last == 0:
        qp.X(target)
    _mcz([control_wires[0]] + list(controls), [1] + list(values), target, base.work)
    if last == 0:
        qp.X(target)
    qp.Z(control_wires[0])


register_rule(ProtoZeroReflection, _proto_zero_reflection)
register_rule("Adjoint(ProtoZeroReflection)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))
register_rule("C(ProtoZeroReflection)", _proto_c_zero_reflection)


# --------------------------------------------------------------------------- walk assembly


class ProtoWalkPart(Operator2):
    """Base for the stand-in blocks that take the whole layout through one compilable shape."""

    compilable_argnames = ("shape", "tables")
    wire_argnames = ("wires",)

    def __init__(self, shape, tables, wires):
        super().__init__(shape, tables, wires)


def _layout(shape_tuple):
    return Layout(Shape(*shape_tuple))


def _variant(remedy):
    """The stand-in variant of a ``remedy`` switch: None (baseline), "R1" (partial remedy 1, for
    ``True``) or a Task 0.9c arm name of ARMS."""
    if isinstance(remedy, str):
        return remedy
    return "R1" if remedy else None


def _w(wires, register):
    return [wires[i] for i in register]


def _read_args(s, lay, wires, which, tables):
    outer, inner, tail, sf = tables
    if which == "outer":
        return ((outer, ((s.w_x - s.k1, 0, -(-s.X_o // 2**s.k1)),), s.k1, s.d1),
                (_w(wires, lay.x_o[: s.w_x - s.k1]), _w(wires, lay.x_o[s.w_x - s.k1:]),
                 _w(wires, lay.alt_x_o + lay.keep_x_o), _w(wires, lay.junk_x_o),
                 _w(wires, lay.work)))
    return ((inner, ((s.w_x, 0, s.RC), (s.w_b - s.k2, 0, -(-(s.B + 1) // 2**s.k2))), s.k2, s.d2),
            (_w(wires, lay.x_o + lay.b[: s.w_b - s.k2]), _w(wires, lay.b[s.w_b - s.k2:]),
             _w(wires, lay.alt_b + lay.keep_b + lay.sgn + lay.d_sgn + lay.r),
             _w(wires, lay.junk_b), _w(wires, lay.work)))


def _prepare_body(s, lay, wires, which, tables, unprepare, remedy=False):
    static, wire_args = _read_args(s, lay, wires, which, tables)
    if which == "outer":
        index, succ, extra = lay.x_o, lay.succ_x_o, lay.sigma_1
        m, lt, cmp_work, alt, keep = lay.m_x_o, lay.lt_x_o, lay.cmp_x_o, lay.alt_x_o, lay.keep_x_o
    else:
        index, succ, extra = lay.b, lay.succ_b, lay.sigma_0
        m, lt, cmp_work, alt, keep = lay.m_b, lay.lt_b, lay.cmp_b, lay.alt_b, lay.keep_b
    def comparator():
        return qp.LeftQuantumComparator(_w(wires, keep), _w(wires, m), _w(wires, lt),
                                        _w(wires, cmp_work), "<=")

    swaps = [(lt[0], a, x) for a, x in zip(alt, index, strict=True)]
    if not unprepare:
        ProtoUniformPrep(_w(wires, index), _w(wires, succ))
        for wire in m + extra:
            qp.H(wires[wire])
        if remedy:
            _r1_read(static, wire_args)
        else:
            ProtoTableRead(*static, *wire_args)
        comparator()
        for c, a, x in swaps:
            qp.CSWAP([wires[c], wires[a], wires[x]])
    else:
        for c, a, x in swaps:
            qp.CSWAP([wires[c], wires[a], wires[x]])
        qp.adjoint(comparator())
        erase_cls = _table_erase_cls(remedy)
        if erase_cls is ProtoTableErase:
            ProtoTableErase(*static, *wire_args)
        else:
            erase_cls(*static, erase_geometry(s)[which], *wire_args)
        for wire in m + extra:
            qp.H(wires[wire])
        qp.adjoint(ProtoUniformPrep(_w(wires, index), _w(wires, succ)))


def _prepare_resources(s, lay, which, tables, unprepare, remedy=False):
    static, _ = _read_args(s, lay, list(range(lay.num_wires)), which, tables)
    nh, nl = sum(w for w, _, _ in static[1]), static[2]
    width, junk = static[3], (2**static[2] - 1) * static[3]
    n_index = s.w_x if which == "outer" else s.w_b
    b_k = s.b_k1 if which == "outer" else s.b_k2
    table_op = _table_erase_cls(remedy) if unprepare else ProtoTableRead
    def maybe_adjoint(op):
        return qp.adjoint(op) if unprepare else op

    cmp_key = resource_key(lambda: maybe_adjoint(qp.LeftQuantumComparator(
        Wire[b_k], Wire[b_k], Wire[1], Wire[b_k - 1], "<=")))
    uniform = resource_key(lambda: maybe_adjoint(ProtoUniformPrep(Wire[n_index], Wire[1])))
    wire_keys = (Wire[nh], Wire[nl], Wire[width], Wire[junk], Wire[len(lay.work)])
    if table_op in (ProtoTableEraseF, ProtoTableEraseFP):
        table = _abstract(table_op, *static, erase_geometry(s)[which], *wire_keys)
    else:
        table = _abstract(table_op, *static, *wire_keys)
    if remedy and not unprepare:
        table = _r1_read_key(static, (nh, nl, width, junk, len(lay.work)))
    return {uniform: 1, qp.H: b_k + 1, table: 1, cmp_key: 1, qp.CSWAP: n_index}


def _make_prepare(name, which, remedy=False):
    cls = type(name, (ProtoWalkPart,), {"__doc__": f"Stand-in {which} prepare (spec 9.7/9.8)."})

    def resources(shape, tables, wires):
        return _prepare_resources(Shape(*shape), _layout(shape), which, tables, False, remedy)

    def rule(shape, tables, wires):
        _prepare_body(Shape(*shape), _layout(shape), wires, which, tables, False, remedy)

    def adjoint_resources(base):
        shape, tables = base.shape, base.tables
        return _prepare_resources(Shape(*shape), _layout(shape), which, tables, True, remedy)

    def adjoint_rule(base):
        _prepare_body(Shape(*base.shape), _layout(base.shape), base.wires, which, base.tables, True,
                      remedy)

    rule.__name__ = f"_{name.lower()}_rule"
    adjoint_rule.__name__ = f"_{name.lower()}_unprepare"
    register_rule(cls, qp.register_resources(resources)(rule))
    register_rule(f"Adjoint({name})", qp.register_resources(adjoint_resources)(adjoint_rule))
    return cls


ProtoOuterPrepare = _make_prepare("ProtoOuterPrepare", "outer")
ProtoInnerPrepare = _make_prepare("ProtoInnerPrepare", "inner")


class ProtoAngleErase(ProtoWalkPart):
    """Stand-in AngleErase: measure angles and flags, SF fixup on G0 = 0, then tail fixup."""


def _angle_erase_body(emit, s, lay, wires, tables, measure, cond_z, parities=mcm_parities):
    _, _, tail, sf = tables
    angle_wires = _w(wires, lay.angles + lay.b_neq_B)
    outcomes = [measure(q) for q in angle_wires]
    sf_masks = [from_words(e) for _, e in sf]
    sf_parities = parities(outcomes, sf_masks) if outcomes else []
    registers = [(_w(wires, lay.r), 0, s.R), (_w(wires, lay.b), 0, s.B)]
    emit.x(wires[lay.G0[0]])
    index = iter(range(len(sf_masks)))
    nested_iterate(emit, registers, _w(wires, lay.work),
                   lambda flag, v: cond_z(sf_parities[next(index)], flag),
                   control=wires[lay.G0[0]])
    emit.x(wires[lay.G0[0]])
    flag_outcomes = [measure(wires[lay.G0[0]]), measure(wires[lay.G1[0]])]
    angle_bits = (s.N - 1) * s.b_rot
    tail_masks = []
    for _, entry in tail:
        value = from_words(entry)
        tail_masks.append((value & (2**angle_bits - 1)) | (value >> angle_bits) << (angle_bits + 1))
    tail_parities = parities(outcomes + flag_outcomes, tail_masks)
    index = iter(range(len(tail_masks)))
    nested_iterate(emit, [(_w(wires, lay.x_o), s.RC, s.X_o)], _w(wires, lay.work),
                   lambda flag, v: cond_z(tail_parities[next(index)], flag))


class _AngleEmit(_Emit):
    pass


class _AngleCount(_Count):
    pass


def _angle_erase_resources(shape, tables, wires):
    s, lay = Shape(*shape), _layout(shape)
    counter = _AngleCount()
    measured = len(lay.angles) + 3
    counter.counts["Hadamard"] += measured
    counter.counts["MidMeasure"] += measured

    def count_measure(q):
        return 1

    def count_cond(parity, flag):
        if parity is not False:
            counter.counts["PauliZ"] += 1

    # mcm_parities needs real measurement values in tape mode; the count needs only which masks
    # are nonzero (mcm_parities returns False exactly for those that are 0).
    _angle_erase_body(counter, s, lay, list(range(lay.num_wires)), tables, count_measure,
                      count_cond, parities=lambda outcomes, masks: [bool(m) for m in masks])
    return counts_to_resources(counter.counts)


@qp.register_resources(_angle_erase_resources)
def _proto_angle_erase(shape, tables, wires):
    _angle_erase_body(_AngleEmit(), Shape(*shape), _layout(shape), wires, tables, _measure_x,
                      lambda parity, flag: qp.cond(parity, qp.Z)(flag) if parity is not False
                      else None)


register_rule(ProtoAngleErase, _proto_angle_erase)


class ProtoTailAngleLoad(ProtoWalkPart):
    """Stand-in TailAngleLoad: a read of rows [RC, X_o) without swap network (spec 9.10)."""


class ProtoSFAngleLoad(ProtoWalkPart):
    """Stand-in SFAngleLoad: a read over (r, b), enabled on G0 = 0 (spec 9.11)."""


def _angle_read(s, lay, wires, tables, which):
    _, _, tail, sf = tables
    angle_bits = (s.N - 1) * s.b_rot
    if which == "tail":
        static = (tail, ((s.w_x, s.RC, s.X_o),), 0, angle_bits + 2)
        return static, (_w(wires, lay.x_o), [], _w(wires, lay.angles + lay.G0 + lay.G1), [],
                        _w(wires, lay.work))
    static = (sf, ((s.w_r, 0, s.R), (s.w_b, 0, s.B)), 0, angle_bits + 1)
    return static, (_w(wires, lay.r + lay.b), [], _w(wires, lay.angles + lay.b_neq_B), [],
                    _w(wires, lay.work))


def _make_angle_load(cls, which, remedy=False):
    def resources(shape, tables, wires):
        s, lay = Shape(*shape), _layout(shape)
        static, wire_args = _angle_read(s, lay, list(range(lay.num_wires)), tables, which)
        key = _abstract(ProtoTableRead, *static, *(Wire[len(w)] for w in wire_args))
        if remedy:
            key = _r1_read_key(static, [len(w) for w in wire_args])
        return {key: 1, qp.X: 2 * (which == "sf")}

    def rule(shape, tables, wires):
        s, lay = Shape(*shape), _layout(shape)
        static, wire_args = _angle_read(s, lay, wires, tables, which)
        if which == "sf":
            qp.X(wires[lay.G0[0]])
        # The SF load is enabled on G0 = 0; the stand-in read ignores the enable (placeholder).
        if remedy:
            _r1_read(static, wire_args)
        else:
            ProtoTableRead(*static, *wire_args)
        if which == "sf":
            qp.X(wires[lay.G0[0]])

    rule.__name__ = f"_proto_{which}_angle_load" + ("_r1" if remedy else "")
    register_rule(cls, qp.register_resources(resources)(rule))


_make_angle_load(ProtoTailAngleLoad, "tail")
_make_angle_load(ProtoSFAngleLoad, "sf")


class ProtoInnerSelect(ProtoWalkPart):
    """Stand-in InnerSelect (spec 9.18); the pinned adjoint uses the adjoint Majorana gate."""


def _angle_loads(remedy):
    if remedy:
        return ProtoTailAngleLoadR1, ProtoSFAngleLoadR1
    return ProtoTailAngleLoad, ProtoSFAngleLoad


def _select_ops(shape, tables, wires, adjoint_majorana, remedy=False):
    s, lay = Shape(*shape), _layout(shape)
    part = (shape, tables, wires)
    mux = (_w(wires, lay.G0), _w(wires, lay.sigma_0), _w(wires, lay.sigma_1), _w(wires, lay.sg))
    swap = (_w(wires, lay.sg), _w(wires, lay.psi_up), _w(wires, lay.psi_dn))
    ladder = (s.b_rot, _w(wires, lay.psi_dn), _w(wires, lay.angles), _w(wires, lay.phase_grad),
              _w(wires, lay.work))
    majorana = (_w(wires, lay.succ_x_o), _w(wires, lay.succ_b), _w(wires, lay.G0),
                _w(wires, lay.G1), _w(wires, lay.b_neq_B), _w(wires, lay.sigma_0),
                _w(wires, lay.psi_dn[:1]), _w(wires, lay.work[:2]))
    tail_load, sf_load = _angle_loads(remedy)
    ladder_cls = _ladder_cls(remedy)
    tail_load(*part)
    sf_load(*part)
    ProtoSpinMux(*mux)
    ProtoSpinSwap(*swap)
    ladder_cls(*ladder)
    if adjoint_majorana:
        qp.adjoint(ProtoMajorana(*majorana))
    else:
        ProtoMajorana(*majorana)
    qp.adjoint(ladder_cls(*ladder))
    ProtoSpinSwap(*swap)
    ProtoSpinMux(*mux)
    _angle_erase_cls(remedy)(*part)


def _select_resources(shape, tables, adjoint_majorana, remedy=False):
    s, lay = Shape(*shape), _layout(shape)
    part_keys = [_abstract(c, shape, tables, Wire[lay.num_wires])
                 for c in (*_angle_loads(remedy), _angle_erase_cls(remedy))]
    mux = _abstract(ProtoSpinMux, Wire[1], Wire[1], Wire[1], Wire[1])
    swap = _abstract(ProtoSpinSwap, Wire[1], Wire[s.N], Wire[s.N])
    ladder_args = (s.b_rot, Wire[s.N], Wire[len(lay.angles)], Wire[s.b_rot], Wire[len(lay.work)])
    ladder_cls = _ladder_cls(remedy)
    ladder = _abstract(ladder_cls, *ladder_args)
    ladder_dagger = resource_key(lambda: qp.adjoint(ladder_cls(*ladder_args)))
    maj_args = (Wire[1],) * 7 + (Wire[2],)
    majorana = resource_key(lambda: qp.adjoint(ProtoMajorana(*maj_args)) if adjoint_majorana
                            else ProtoMajorana(*maj_args))
    return {part_keys[0]: 1, part_keys[1]: 1, part_keys[2]: 1, mux: 2, swap: 2, ladder: 1,
            ladder_dagger: 1, majorana: 1}


register_rule(ProtoInnerSelect, qp.register_resources(
    lambda shape, tables, wires: _select_resources(shape, tables, False))(
    lambda shape, tables, wires: _select_ops(shape, tables, wires, False)))
register_rule("Adjoint(ProtoInnerSelect)", qp.register_resources(
    lambda base: _select_resources(base.shape, base.tables, True))(
    lambda base: _select_ops(base.shape, base.tables, base.wires, True)))


def _refs(shape, wires):
    lay = _layout(shape)
    t2 = (_w(wires, lay.b + lay.m_b + lay.sigma_0), _w(wires, lay.succ_x_o), _w(wires, lay.work))
    ab = (_w(wires, lay.x_o + lay.m_x_o + lay.sigma_1 + lay.b + lay.m_b + lay.sigma_0), [],
          _w(wires, lay.work))
    return t2, ab


class ProtoBlockEncoding(ProtoWalkPart):
    """Stand-in DFTHCBlockEncoding (spec 9.21)."""


class ProtoWalk(ProtoWalkPart):
    """Stand-in DFTHCWalk (spec 9.22)."""


def _parts(remedy):
    """(outer prepare, inner prepare, inner select, block encoding, walk) of a variant."""
    return {None: (ProtoOuterPrepare, ProtoInnerPrepare, ProtoInnerSelect, ProtoBlockEncoding,
                   ProtoWalk),
            "R1": (ProtoOuterPrepareR1, ProtoInnerPrepareR1, ProtoInnerSelectR1,
                   ProtoBlockEncodingR1, ProtoWalkR1),
            "F": (ProtoOuterPrepareF, ProtoInnerPrepareF, ProtoInnerSelectF, ProtoBlockEncodingF,
                  ProtoWalkF),
            "FP": (ProtoOuterPrepareFP, ProtoInnerPrepareFP, ProtoInnerSelectFP,
                   ProtoBlockEncodingFP, ProtoWalkFP),
            "FPG": (ProtoOuterPrepareFPG, ProtoInnerPrepareFPG, ProtoInnerSelectFPG,
                    ProtoBlockEncodingFPG, ProtoWalkFPG)}[_variant(remedy)]


def _table_erase_cls(remedy):
    return {None: ProtoTableErase, "R1": ProtoTableErase, "F": ProtoTableEraseF,
            "FP": ProtoTableEraseFP, "FPG": ProtoTableEraseFP}[_variant(remedy)]


def _angle_erase_cls(remedy):
    return {None: ProtoAngleErase, "R1": ProtoAngleErase, "F": ProtoAngleEraseF,
            "FP": ProtoAngleEraseFP, "FPG": ProtoAngleEraseFP}[_variant(remedy)]


def _ladder_cls(remedy):
    return ProtoGivensLadderLoop if _variant(remedy) == "FPG" else ProtoGivensLadder


def _be_parts(remedy):
    """The outer prepare, inner prepare and inner select of a variant (baseline, partial remedy 1
    or a Task 0.9c arm)."""
    return _parts(remedy)[:3]


def _be_ops(shape, tables, wires, control=None, remedy=False):
    part = (shape, tables, wires)
    t2, _ = _refs(shape, wires)
    outer_prepare, inner_prepare, inner_select = _be_parts(remedy)
    outer_prepare(*part)
    inner_prepare(*part)
    ProtoInnerSigns(*_signs(shape, wires))
    inner_select(*part)
    qp.adjoint(inner_prepare(*part))
    if control is None:
        ProtoZeroReflection(*t2)
    else:
        qp.ctrl(ProtoZeroReflection(*t2), control=control)
    inner_prepare(*part)
    qp.adjoint(inner_select(*part))
    ProtoInnerSigns(*_signs(shape, wires))
    qp.adjoint(inner_prepare(*part))
    qp.adjoint(outer_prepare(*part))


def _signs(shape, wires):
    lay = _layout(shape)
    return _w(wires, lay.sgn), _w(wires, lay.d_sgn), _w(wires, lay.lt_b)


def _be_resources(shape, tables, controlled, remedy=False):
    lay = _layout(shape)
    s = Shape(*shape)
    n = Wire[lay.num_wires]
    keys = {}
    parts = _be_parts(remedy)
    for cls in parts:
        keys[_abstract(cls, shape, tables, n)] = 2 if cls is parts[1] else 1
        keys[resource_key(lambda c=cls: qp.adjoint(c(shape, tables, n)))] = (
            2 if cls is parts[1] else 1)
    keys[_abstract(ProtoInnerSigns, Wire[1], Wire[1], Wire[1])] = 2
    t2_args = (Wire[s.w_b + s.b_k2 + 1], Wire[1], Wire[len(lay.work)])
    ref = (resource_key(lambda: qp.ctrl(ProtoZeroReflection(*t2_args), control=Wire[1]))
           if controlled else _abstract(ProtoZeroReflection, *t2_args))
    keys[ref] = 1
    return keys


def _ab_args(shape):
    s, lay = Shape(*shape), _layout(shape)
    return (Wire[s.w_x + s.b_k1 + s.w_b + s.b_k2 + 2], Wire[0], Wire[len(lay.work)])


register_rule(ProtoBlockEncoding, qp.register_resources(
    lambda shape, tables, wires: _be_resources(shape, tables, False))(
    lambda shape, tables, wires: _be_ops(shape, tables, wires)))


def _c_be_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return _be_resources(base.shape, base.tables, True)


def _c_be(base, control_wires, control_values, work_wires, work_wire_type):
    _be_ops(base.shape, base.tables, base.wires, control=control_wires[0])


register_rule("C(ProtoBlockEncoding)", qp.register_resources(_c_be_resources)(_c_be))


def _walk_keys(shape, tables, controlled, remedy):
    n = Wire[_layout(shape).num_wires]
    block_encoding = _parts(remedy)[3]
    if not controlled:
        return {_abstract(block_encoding, shape, tables, n): 1,
                _abstract(ProtoZeroReflection, *_ab_args(shape)): 1}
    return {resource_key(lambda: qp.ctrl(block_encoding(shape, tables, n), control=Wire[1])): 1,
            resource_key(lambda: qp.ctrl(ProtoZeroReflection(*_ab_args(shape)),
                                         control=Wire[1])): 1}


def _walk_ops(shape, tables, wires, control, remedy):
    block_encoding = _parts(remedy)[3]
    _, ab = _refs(shape, wires)
    if control is None:
        block_encoding(shape, tables, wires)
        ProtoZeroReflection(*ab)
    else:
        qp.ctrl(block_encoding(shape, tables, wires), control=control)
        qp.ctrl(ProtoZeroReflection(*ab), control=control)


def _walk_resources(shape, tables, wires):
    return _walk_keys(shape, tables, False, False)


def _walk(shape, tables, wires):
    _walk_ops(shape, tables, wires, None, False)


def _c_walk_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return _walk_keys(base.shape, base.tables, True, False)


def _c_walk(base, control_wires, control_values, work_wires, work_wire_type):
    _walk_ops(base.shape, base.tables, base.wires, control_wires[0], False)


register_rule(ProtoWalk, qp.register_resources(_walk_resources)(_walk))
register_rule("C(ProtoWalk)", qp.register_resources(_c_walk_resources)(_c_walk))


# --------------------------------------------------------------------------- partial remedy 1
# Remedy 1 of spec 16, K-1, applied to the reads and angle loads only (--remedy 1); the erases stay
# as above. The parallel classes end in R1; the baseline classes are unchanged.

WORD_DTYPE = np.int64
"""dtype of the remedy-1 ``words`` arrays: int64, because Catalyst builds no decomposition rule
for an unsigned dynamic argument, and 32 data bits per word keep every word non-negative.
ProtoTableReadR1 canonicalizes to it and _r1_read_key declares it: under capture the graph ID
spells the dtype (tensor<...xi64>), and a declared key of another dtype would leave the leaf
undecomposed (tape mode does not check it)."""
NOTES = (
    "Table words are int64 holding 32 data bits each: with uint32 words, Catalyst builds no rule "
    "for ProtoTableReadR1 ('Failed to synthesize the distributed decomposition rules ...: ui32') "
    "and the compile fails; reproducer: .scratch/task-0.9b/dtype_repro.py (not committed)",
)
"""Notes written into every k1r1_<shape>.json."""


def _word_array(words):
    """Canonicalize table words once for the dynamic ``words`` argument (L-23).

    Abstract arrays (resource keys) and tracers (operators rebuilt by Catalyst) pass unchanged.
    Raises ``ValueError`` for a word that is not an integer in [0, 2**32), checked on the Python
    integers before the WORD_DTYPE cast (which would overflow first for words beyond int64).
    """
    if isinstance(words, AbstractArray) or qp.math.is_tracer(words):
        return words
    values = np.asarray(words, dtype=object)
    bad = [w for w in values.flat
           if isinstance(w, bool) or not isinstance(w, (int, np.integer)) or not 0 <= w < 2**WORD]
    if bad:
        raise ValueError(f"table words must be integers in [0, 2**{WORD}), got {bad[0]!r}")
    array = values.astype(WORD_DTYPE)
    return jnp.asarray(array) if qp.capture.enabled() else array


def _wire_array(wires):
    """A wire register that a loop index can address: a jax array under capture."""
    return jnp.asarray(wires) if qp.capture.enabled() else wires


class ProtoTableReadR1(Operator2):
    """ProtoTableRead with its table split for remedy 1 (prototype only).

    ``addresses`` (compilable) lists the loaded addresses. ``words`` (dynamic, WORD_DTYPE, shape
    ``(len(addresses), ceil(width / 32))``) holds in row k the 32-bit words of the entry at
    ``addresses[k]``, least significant first. The unary iteration and the swap network stay
    unrolled in Python, so T, M and A are exact. At every loaded address of a leaf, a
    ``qp.for_loop`` over the ``width`` bits applies ``qp.cond(bit, qp.CNOT)`` from the leaf flag
    to a dynamically indexed wire; padding addresses load zero. The resource function sees only
    the shape of ``words`` (L-5): its CNOT count is an upper bound (every loaded bit set), and the
    rule is registered with ``exact=False``.
    """

    dynamic_argnames = ("words",)
    compilable_argnames = ("addresses", "high_ranges", "block_bits", "width")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, words, addresses, high_ranges, block_bits, width, high, low, target, junk,
                 work):
        super().__init__(_word_array(words), addresses, high_ranges, block_bits, width, high, low,
                         target, junk, work)


def _r1_read_body(emit, addresses, high_ranges, block_bits, width, high, low, target, junk, work,
                  load, cswap):
    """ProtoTableReadR1's gates, with ``load(flag, row, register)`` per loaded address of a leaf."""
    rows = {address: row for row, address in enumerate(addresses)}
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    widths = [w for w, _, _ in high_ranges]

    def leaf(flag, values):
        prefix = 0
        for value, w in zip(values, widths, strict=True):
            prefix = (prefix << w) | value
        for position in range(2**block_bits):
            row = rows.get((prefix << block_bits) | position)
            if row is not None:
                load(flag, row, registers[position])

    nested_iterate(emit, _split_high(high, high_ranges), work, leaf)
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _bit_loader(words, width):
    def load(flag, row, register):
        register = _wire_array(register)

        @qp.for_loop(0, width, 1)
        def bit_loop(i):
            bit = (words[row, i // WORD] >> (i % WORD)) & 1
            qp.cond(bit == 1, qp.CNOT)([flag, register[i]])

        bit_loop()

    return load


def _r1_read_counts(addresses, high_ranges, block_bits, width, count_loads=True):
    """Gate counts of ProtoTableReadR1; "CNOT" is an upper bound (every loaded bit set). Without
    ``count_loads`` the bit loops' bodies are left out (zero table words)."""
    counter = _Count()
    dummy = list(range(10_000))

    def load(flag, row, register):
        counter.counts["CNOT"] += width if count_loads else 0

    _r1_read_body(counter, addresses, high_ranges, block_bits, width,
                  dummy[: sum(w for w, _, _ in high_ranges)], dummy[:block_bits], dummy[:width],
                  dummy[: (2**block_bits - 1) * width], dummy[:64], load,
                  lambda c, a, b: counter.counts.update({"CSWAP": 1}))
    return counter.counts


def _r1_read_resources(words, addresses, high_ranges, block_bits, width, high, low, target, junk,
                       work):
    return counts_to_resources(_r1_read_counts(addresses, high_ranges, block_bits, width))


@qp.register_resources(_r1_read_resources, exact=False)
def _proto_table_read_r1(words, addresses, high_ranges, block_bits, width, high, low, target, junk,
                         work):
    _r1_read_body(_Emit(), addresses, high_ranges, block_bits, width, high, low, target, junk,
                  work, _bit_loader(words, width), lambda c, a, b: qp.CSWAP([c, a, b]))


register_rule(ProtoTableReadR1, _proto_table_read_r1)


def _r1_read(static, wire_args):
    """Emit ProtoTableReadR1 for the baseline read arguments ``static`` = (table, high ranges,
    block bits, width); the composite builds the leaf's ``words`` from its compilable table."""
    table, high_ranges, block_bits, width = static
    ProtoTableReadR1([entry for _, entry in table], tuple(a for a, _ in table), high_ranges,
                     block_bits, width, *wire_args)


def _r1_read_key(static, wire_sizes):
    """Resource key of :func:`_r1_read` (random_tables stores ceil(width / 32) words per entry)."""
    table, high_ranges, block_bits, width = static
    words = AbstractArray((len(table), -(-width // WORD)), WORD_DTYPE)
    return _abstract(ProtoTableReadR1, words, tuple(a for a, _ in table), high_ranges, block_bits,
                     width, *(Wire[n] for n in wire_sizes))


ProtoOuterPrepareR1 = _make_prepare("ProtoOuterPrepareR1", "outer", remedy=True)
ProtoInnerPrepareR1 = _make_prepare("ProtoInnerPrepareR1", "inner", remedy=True)


class ProtoTailAngleLoadR1(ProtoWalkPart):
    """ProtoTailAngleLoad through ProtoTableReadR1 (partial remedy 1)."""


class ProtoSFAngleLoadR1(ProtoWalkPart):
    """ProtoSFAngleLoad through ProtoTableReadR1 (partial remedy 1)."""


_make_angle_load(ProtoTailAngleLoadR1, "tail", remedy=True)
_make_angle_load(ProtoSFAngleLoadR1, "sf", remedy=True)


class ProtoInnerSelectR1(ProtoWalkPart):
    """ProtoInnerSelect with the remedy-1 angle loads; the angle erase is unchanged."""


register_rule(ProtoInnerSelectR1, qp.register_resources(
    lambda shape, tables, wires: _select_resources(shape, tables, False, remedy=True))(
    lambda shape, tables, wires: _select_ops(shape, tables, wires, False, remedy=True)))
register_rule("Adjoint(ProtoInnerSelectR1)", qp.register_resources(
    lambda base: _select_resources(base.shape, base.tables, True, remedy=True))(
    lambda base: _select_ops(base.shape, base.tables, base.wires, True, remedy=True)))


class ProtoBlockEncodingR1(ProtoWalkPart):
    """ProtoBlockEncoding with the remedy-1 prepares and select."""


class ProtoWalkR1(ProtoWalkPart):
    """ProtoWalk with ProtoBlockEncodingR1."""


register_rule(ProtoBlockEncodingR1, qp.register_resources(
    lambda shape, tables, wires: _be_resources(shape, tables, False, remedy=True))(
    lambda shape, tables, wires: _be_ops(shape, tables, wires, remedy=True)))


def _c_be_r1_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return _be_resources(base.shape, base.tables, True, remedy=True)


def _c_be_r1(base, control_wires, control_values, work_wires, work_wire_type):
    _be_ops(base.shape, base.tables, base.wires, control=control_wires[0], remedy=True)


register_rule("C(ProtoBlockEncodingR1)", qp.register_resources(_c_be_r1_resources)(_c_be_r1))


def _walk_r1_resources(shape, tables, wires):
    return _walk_keys(shape, tables, False, True)


def _walk_r1(shape, tables, wires):
    _walk_ops(shape, tables, wires, None, True)


def _c_walk_r1_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return _walk_keys(base.shape, base.tables, True, True)


def _c_walk_r1(base, control_wires, control_values, work_wires, work_wire_type):
    _walk_ops(base.shape, base.tables, base.wires, control_wires[0], True)


register_rule(ProtoWalkR1, qp.register_resources(_walk_r1_resources)(_walk_r1))
register_rule("C(ProtoWalkR1)", qp.register_resources(_c_walk_r1_resources)(_c_walk_r1))


# --------------------------------------------------------------------------- Task 0.9c arms
# On top of partial remedy 1 (--arm): F, the spec's erase fixup (one-hot bits, spec 9.4/9.5/9.17);
# FP, as F with the conditional CZs of each fixup leaf in one static-bound qp.for_loop; FPG, as FP
# with ProtoGivensLadderLoop (the rungs in a qp.for_loop, with an explicit pinned adjoint rule).

ARMS = ("F", "FP", "FPG")
ARM_PARENT = {"F": None, "FP": "F", "FPG": "FP"}
"""The arm each arm builds on (Task 0.9c requirement 3: a failed arm stops the arms on top)."""
ERASE_OBS = {"small": (1, 2), "mid10": (3, 4), "mid14": (3, 5), "femoco": (4, 7)}
"""(outer, inner) table-erase one-hot bits of the rule of erase_geometry (brief, requirement 1)."""
SF_OBS = {"small": 2, "mid10": 3, "mid14": 4, "femoco": 5}
"""One-hot bits of the angle erase's SF fixup, w_b (spec 9.17); its tail fixup has ob = 0."""


def onehot_cost(onehot_bits: int) -> int:
    """O(ob) of spec 9.24: TemporaryANDs of the one-hot tree."""
    return 2**onehot_bits - 2 if onehot_bits else 0


@functools.cache
def erase_geometry(s: Shape) -> dict[str, int]:
    """One-hot bits of the stand-in's fixups: one spike-local rule for every shape (Task 0.9c).

    Outer erase: covered set [0, L), L = ceil(X_o / K1) K1; ob = q minimizes
    U(0, ceil(L / 2**q), w_x - q, 0) + O(q) over 0 <= q <= w_x with 2**q <= K1 d1. Inner erase:
    covered x in [0, RC), b in [0, ceil((B+1) / K2) K2), flattened (x << w_b) | b; ob = w_b + q
    minimizes U(0, ceil(RC / 2**q), w_x - q, 0) + O(ob) over 0 <= q <= w_x with 2**ob <= K2 d2.
    Ties go to the smallest ob. SF fixup: ob = w_b; tail fixup: ob = 0. Raises ``ValueError`` if
    the padded b range is not the whole w_b-bit register (the inner rule assumes it).
    """
    k1, k2 = 2**s.k1, 2**s.k2
    covered_outer = -(-s.X_o // k1) * k1
    outer = min((unary_toffolis(0, -(-covered_outer // 2**q), s.w_x - q, False) + onehot_cost(q), q)
                for q in range(s.w_x + 1) if 2**q <= k1 * s.d1)[1]
    if -(-(s.B + 1) // k2) * k2 != 2**s.w_b:
        raise ValueError(f"padded b range {-(-(s.B + 1) // k2) * k2} != 2**w_b = {2**s.w_b}")
    inner = min((unary_toffolis(0, -(-s.RC // 2**q), s.w_x - q, False)
                 + onehot_cost(s.w_b + q), s.w_b + q)
                for q in range(s.w_x + 1) if 2 ** (s.w_b + q) <= k2 * s.d2)[1]
    return {"outer": outer, "inner": inner, "sf": s.w_b, "tail": 0}


for _name, _shape in SHAPES.items():
    _geometry = erase_geometry(_shape)
    if ((_geometry["outer"], _geometry["inner"]), _geometry["sf"]) != (ERASE_OBS[_name],
                                                                       SF_OBS[_name]):
        raise RuntimeError(f"erase_geometry({_name}) = {_geometry}, not {ERASE_OBS[_name]}, "
                           f"SF {SF_OBS[_name]}")


class ProtoAngleEraseF(ProtoWalkPart):
    """Stand-in AngleErase with spec 9.17's fixups (K-1 arm F): measure the angles and b_neq_B,
    the SF fixup over (r, b) in [0, R) x [0, B) with ob = w_b on the first 2**w_b reset angle
    wires, enabled on G0 = 0 while G0 is live, then measure G0 and G1, then the tail fixup over
    x_o in [RC, X_o) with ob = 0. Never adjointed or controlled."""


class ProtoAngleEraseFP(ProtoWalkPart):
    """ProtoAngleEraseF with the SF fixup's conditional CZs in one qp.for_loop per r leaf (FP)."""


def _angle_erase_onehot_body(emit, s, lay, wires, tables, measure, parities, looped):
    _, _, tail, sf = tables
    angle_wires = _w(wires, lay.angles + lay.b_neq_B)
    if 2**s.w_b > len(angle_wires):
        raise ValueError(f"the SF fixup needs {2**s.w_b} one-hot wires, has {len(angle_wires)}")
    outcomes = [measure(q) for q in angle_wires]
    sf_parities = parities(outcomes, [from_words(e) for _, e in sf])
    g0, g1 = wires[lay.G0[0]], wires[lay.G1[0]]
    emit.x(g0)
    phase_fixup(emit, _w(wires, lay.r + lay.b), fixup_leaves([a for a, _ in sf], s.w_b), s.w_b,
                angle_wires[: 2**s.w_b], _w(wires, lay.work), sf_parities, control=g0,
                looped=looped)
    emit.x(g0)
    flag_outcomes = [measure(g0), measure(g1)]
    angle_bits = (s.N - 1) * s.b_rot
    tail_masks = []
    for _, entry in tail:
        value = from_words(entry)
        tail_masks.append((value & (2**angle_bits - 1)) | (value >> angle_bits) << (angle_bits + 1))
    phase_fixup(emit, _w(wires, lay.x_o), fixup_leaves([a for a, _ in tail], 0), 0, [],
                _w(wires, lay.work), parities(outcomes + flag_outcomes, tail_masks))


def angle_erase_onehot_counts(shape, tables) -> Counter:
    """Gate counts of ProtoAngleEraseF and of ProtoAngleEraseFP (loop bodies as if unrolled)."""
    s, lay = Shape(*shape), _layout(shape)
    counter = _Count()

    def measure(qubit):
        counter.counts.update({"Hadamard": 1, "MidMeasure": 1})

    _angle_erase_onehot_body(counter, s, lay, list(range(lay.num_wires)), tables, measure,
                             lambda outcomes, masks: [None] * len(masks), looped=False)
    return counter.counts


def _angle_erase_onehot_rule(name, looped):
    def resources(shape, tables, wires):
        return counts_to_resources(angle_erase_onehot_counts(shape, tables))

    def rule(shape, tables, wires):
        _angle_erase_onehot_body(_Emit(), Shape(*shape), _layout(shape), wires, tables,
                                 _measure_x, measured_parities, looped)

    rule.__name__ = name
    return qp.register_resources(resources)(rule)


register_rule(ProtoAngleEraseF, _angle_erase_onehot_rule("_proto_angle_erase_f", looped=False))
register_rule(ProtoAngleEraseFP, _angle_erase_onehot_rule("_proto_angle_erase_fp", looped=True))


class ProtoGivensLadderLoop(Operator2):
    """ProtoGivensLadder with its rungs in one static-bound ``qp.for_loop`` (K-1 arm FPG).

    Iteration t applies rung t on (system[t], system[t + 1], angle row t), with ``system`` and the
    ``(N - 1, b_rot)`` angle rows as indexable arrays. The adjoint has its own pinned rule (rungs
    k = N - 2 - t, each ``qp.adjoint(ProtoGivensRotation(...))``): no auto-derived adjoint of a
    loop body is relied on (L-6). The rung is unchanged.
    """

    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        super().__init__(b_rot, system, angles, phase_grad, work)


def _rung_key(b_rot, adjoint):
    def rung():
        op = ProtoGivensRotation(Wire[1], Wire[1], Wire[b_rot], Wire[b_rot], Wire[b_rot - 1])
        return qp.adjoint(op) if adjoint else op

    return resource_key(rung)


def _ladder_arrays(system, angles, b_rot):
    """``system`` and the angle rows, shape (N - 1, b_rot), indexable by a traced loop index."""
    rows = len(system) - 1
    if qp.capture.enabled():
        return jnp.asarray(system), jnp.asarray(angles).reshape(rows, b_rot)
    return list(system), np.asarray(list(angles), dtype=object).reshape(rows, b_rot)


@qp.register_resources(lambda b_rot, system, angles, phase_grad, work: {
    _rung_key(b_rot, False): len(system) - 1})
def _proto_ladder_loop(b_rot, system, angles, phase_grad, work):
    system_array, rows = _ladder_arrays(system, angles, b_rot)

    @qp.for_loop(0, len(system) - 1, 1)
    def rung(t):
        ProtoGivensRotation([system_array[t]], [system_array[t + 1]], rows[t], phase_grad,
                            work[: b_rot - 1])

    rung()


@qp.register_resources(lambda base: {_rung_key(base.b_rot, True): len(base.system) - 1})
def _adjoint_proto_ladder_loop(base):
    b_rot, last = base.b_rot, len(base.system) - 2
    system_array, rows = _ladder_arrays(base.system, base.angles, b_rot)

    @qp.for_loop(0, last + 1, 1)
    def rung(t):
        k = last - t
        qp.adjoint(ProtoGivensRotation([system_array[k]], [system_array[k + 1]], rows[k],
                                       base.phase_grad, base.work[: b_rot - 1]))

    rung()


register_rule(ProtoGivensLadderLoop, _proto_ladder_loop)
register_rule("Adjoint(ProtoGivensLadderLoop)", _adjoint_proto_ladder_loop)


def _make_select(name, arm):
    cls = type(name, (ProtoWalkPart,), {"__doc__": f"ProtoInnerSelect of arm {arm} (Task 0.9c)."})

    def resources(shape, tables, wires):
        return _select_resources(shape, tables, False, remedy=arm)

    def rule(shape, tables, wires):
        _select_ops(shape, tables, wires, False, remedy=arm)

    def adjoint_resources(base):
        return _select_resources(base.shape, base.tables, True, remedy=arm)

    def adjoint_rule(base):
        _select_ops(base.shape, base.tables, base.wires, True, remedy=arm)

    rule.__name__, adjoint_rule.__name__ = f"_{name.lower()}_rule", f"_{name.lower()}_adjoint"
    register_rule(cls, qp.register_resources(resources)(rule))
    register_rule(f"Adjoint({name})", qp.register_resources(adjoint_resources)(adjoint_rule))
    return cls


def _make_block_encoding(name, arm):
    cls = type(name, (ProtoWalkPart,), {"__doc__": f"ProtoBlockEncoding of arm {arm}."})

    def resources(shape, tables, wires):
        return _be_resources(shape, tables, False, remedy=arm)

    def rule(shape, tables, wires):
        _be_ops(shape, tables, wires, remedy=arm)

    def c_resources(base, control_wires, control_values, work_wires, work_wire_type):
        return _be_resources(base.shape, base.tables, True, remedy=arm)

    def c_rule(base, control_wires, control_values, work_wires, work_wire_type):
        _be_ops(base.shape, base.tables, base.wires, control=control_wires[0], remedy=arm)

    rule.__name__, c_rule.__name__ = f"_{name.lower()}_rule", f"_c_{name.lower()}"
    register_rule(cls, qp.register_resources(resources)(rule))
    register_rule(f"C({name})", qp.register_resources(c_resources)(c_rule))
    return cls


def _make_walk(name, arm):
    cls = type(name, (ProtoWalkPart,), {"__doc__": f"ProtoWalk of arm {arm} (Task 0.9c)."})

    def resources(shape, tables, wires):
        return _walk_keys(shape, tables, False, arm)

    def rule(shape, tables, wires):
        _walk_ops(shape, tables, wires, None, arm)

    def c_resources(base, control_wires, control_values, work_wires, work_wire_type):
        return _walk_keys(base.shape, base.tables, True, arm)

    def c_rule(base, control_wires, control_values, work_wires, work_wire_type):
        _walk_ops(base.shape, base.tables, base.wires, control_wires[0], arm)

    rule.__name__, c_rule.__name__ = f"_{name.lower()}_rule", f"_c_{name.lower()}"
    register_rule(cls, qp.register_resources(resources)(rule))
    register_rule(f"C({name})", qp.register_resources(c_resources)(c_rule))
    return cls


ProtoOuterPrepareF = _make_prepare("ProtoOuterPrepareF", "outer", remedy="F")
ProtoInnerPrepareF = _make_prepare("ProtoInnerPrepareF", "inner", remedy="F")
ProtoInnerSelectF = _make_select("ProtoInnerSelectF", "F")
ProtoBlockEncodingF = _make_block_encoding("ProtoBlockEncodingF", "F")
ProtoWalkF = _make_walk("ProtoWalkF", "F")
ProtoOuterPrepareFP = _make_prepare("ProtoOuterPrepareFP", "outer", remedy="FP")
ProtoInnerPrepareFP = _make_prepare("ProtoInnerPrepareFP", "inner", remedy="FP")
ProtoInnerSelectFP = _make_select("ProtoInnerSelectFP", "FP")
ProtoBlockEncodingFP = _make_block_encoding("ProtoBlockEncodingFP", "FP")
ProtoWalkFP = _make_walk("ProtoWalkFP", "FP")
ProtoOuterPrepareFPG = _make_prepare("ProtoOuterPrepareFPG", "outer", remedy="FPG")
ProtoInnerPrepareFPG = _make_prepare("ProtoInnerPrepareFPG", "inner", remedy="FPG")
ProtoInnerSelectFPG = _make_select("ProtoInnerSelectFPG", "FPG")
ProtoBlockEncodingFPG = _make_block_encoding("ProtoBlockEncodingFPG", "FPG")
ProtoWalkFPG = _make_walk("ProtoWalkFPG", "FPG")


# --------------------------------------------------------------------------- arm F fixup check
# Task 0.9c requirement 2, tape mode only (no qjit): the reduced 11-wire read/erase geometry of
# K-3/K-10 with an ob = 1 one-hot fixup, and the phase-fixup helper alone on the small shape's
# outer, inner, SF and tail covered geometries against a classical phase oracle, plus 3 mutants.

FIXUP_WIDTH, FIXUP_BLOCK_BITS, FIXUP_ONEHOT_BITS = 2, 1, 1
FIXUP_HIGH_RANGES = ((2, 0, 3),)
"""Blocks 0..2 visited: addresses 0..4 hold entries, 5 is a visited dead position, 6 and 7 are
never loaded (spikes/k3_mode_agnostic_erase.py, spikes/k10_measurement_arrays.py)."""
FIXUP_WIRES = ([0, 1], [2], [3, 4], [5, 6], [7, 8, 9, 10])
"""(high, low, target, junk, work) of the 11-wire round trip."""
FIXUP_SEEDS = (0, 1, 2)
FIXUP_TOLERANCE = 1e-10


def _fixup_table(seed):
    rng = np.random.default_rng(seed)
    return tuple((a, to_words(int(v), FIXUP_WIDTH))
                 for a, v in enumerate(rng.integers(0, 2**FIXUP_WIDTH, size=5)))


def _fixup_round_trip(table, erase) -> dict:
    """|+> on the 3 address wires, the remedy-1 read, then ``erase()``; tape-decomposed with the
    pinned rules, measurements deferred, simulated on default.qubit. Compares the full reduced
    address density matrix with |+><+| elementwise and requires probability one on all read,
    one-hot and work ancillas being zero."""
    high, low, target, junk, work = FIXUP_WIRES
    qp.decomposition.enable_graph()
    with qp.queuing.AnnotatedQueue() as queue:
        for wire in high + low:
            qp.H(wire)
        _r1_read((table, FIXUP_HIGH_RANGES, FIXUP_BLOCK_BITS, FIXUP_WIDTH),
                 (high, low, target, junk, work))
        erase()
        qp.density_matrix(wires=high + low)
        qp.probs(wires=target + junk + work)
    [tape], _ = _tape_decompose(qp.tape.QuantumScript.from_queue(queue), gate_set=GATE_SET,
                                fixed_decomps=dict(FIXED_DECOMPS))
    [deferred], _ = qp.defer_measurements(tape)
    rho, probs = qp.devices.DefaultQubit().execute(deferred)
    error = float(np.max(np.abs(rho - np.full((8, 8), 1 / 8))))
    return {"density_max_error": error, "ancillas_zero_probability": float(probs[0]),
            "passed": bool(error < FIXUP_TOLERANCE and abs(probs[0] - 1) < FIXUP_TOLERANCE)}


def _fixup_erase_op(table, cls):
    return lambda: cls(table, FIXUP_HIGH_RANGES, FIXUP_BLOCK_BITS, FIXUP_WIDTH,
                       FIXUP_ONEHOT_BITS, *FIXUP_WIRES)


def _fixup_mutant_erase(table, **hooks):
    return lambda: onehot_erase_body(_Emit(), table, FIXUP_HIGH_RANGES, FIXUP_BLOCK_BITS,
                                     FIXUP_WIDTH, FIXUP_ONEHOT_BITS, *FIXUP_WIRES, _measure_x,
                                     measured_parities, False, **hooks)


def _flip_parity(index):
    def hook(values):
        values = values.copy()
        values[index] = (not values[index]) if isinstance(values[index], bool) else ~values[index]
        return values

    return hook


def _swap_first_positions(leaves):
    (h, positions, base), *rest = leaves
    return [(h, (positions[1], positions[0], *positions[2:]), base), *rest]


def fixup_oracle_geometries() -> dict:
    """The small shape's covered geometries: what the arm's code feeds the helper (``covered``,
    from loaded_masks and the angle tables) and, independently, the oracle's covered set from the
    shape formulas of requirement 1."""
    s, lay = SHAPES["small"], Layout(SHAPES["small"])
    tables, geometry = random_tables(s), erase_geometry(SHAPES["small"])
    outer_static, _ = _read_args(s, lay, list(range(lay.num_wires)), "outer", tables)
    inner_static, _ = _read_args(s, lay, list(range(lay.num_wires)), "inner", tables)
    padded_b = -(-(s.B + 1) // 2**s.k2) * 2**s.k2
    return {
        "outer": {"address_bits": s.w_x, "onehot_bits": geometry["outer"], "control": False,
                  "covered": [a for a, _ in loaded_masks(*outer_static)],
                  "oracle": list(range(-(-s.X_o // 2**s.k1) * 2**s.k1))},
        "inner": {"address_bits": s.w_x + s.w_b, "onehot_bits": geometry["inner"],
                  "control": False, "covered": [a for a, _ in loaded_masks(*inner_static)],
                  "oracle": [(x << s.w_b) | b for x in range(s.RC) for b in range(padded_b)]},
        "sf": {"address_bits": s.w_r + s.w_b, "onehot_bits": geometry["sf"], "control": True,
               "covered": [a for a, _ in tables[3]],
               "oracle": [(r << s.w_b) | b for r in range(s.R) for b in range(s.B)]},
        "tail": {"address_bits": s.w_x, "onehot_bits": geometry["tail"], "control": False,
                 "covered": [a for a, _ in tables[2]], "oracle": list(range(s.RC, s.X_o))},
    }


def _oracle_case(geometry, parities, parities_hook=None, leaves_hook=None,
                 covered_hook=None) -> float:
    """Max amplitude error of the helper against the phase oracle with ``parities``, over every
    computational address (and both G0 values for the SF geometry, enabled on G0 = 0), with the
    one-hot and work ancillas zero. The hooks mutate only what the helper receives."""
    n, ob, control = geometry["address_bits"], geometry["onehot_bits"], geometry["control"]
    covered = covered_hook(geometry["covered"]) if covered_hook else geometry["covered"]
    address = list(range(n))
    g0 = [n] if control else []
    first_onehot = n + len(g0)
    onehot = list(range(first_onehot, first_onehot + (2**ob if ob else 0)))
    work = list(range(first_onehot + len(onehot), first_onehot + len(onehot) + n))
    leaves = fixup_leaves(covered, ob)
    leaves = leaves_hook(leaves) if leaves_hook else leaves
    num_wires = n + len(g0) + len(onehot) + len(work)
    with qp.queuing.AnnotatedQueue() as queue:
        for wire in range(num_wires):  # fixes the state's wire order (address, G0, one-hot, work)
            qp.Identity(wire)
        for wire in address + g0:
            qp.H(wire)
        if control:
            qp.X(g0[0])
        phase_fixup(_Emit(), address, leaves, ob, onehot, work,
                    parities_hook(parities) if parities_hook else parities,
                    control=g0[0] if control else None)
        if control:
            qp.X(g0[0])
        qp.state()
    state = qp.devices.DefaultQubit(wires=num_wires).execute(
        qp.tape.QuantumScript.from_queue(queue))
    oracle = {a: i for i, a in enumerate(geometry["oracle"])}
    expected = np.zeros(2**num_wires, dtype=complex)
    norm = 1 / np.sqrt(2 ** (n + len(g0)))
    for basis in range(2 ** (n + len(g0))):
        a, enabled = (basis >> 1, not basis & 1) if control else (basis, True)
        sign = -1 if enabled and a in oracle and parities[oracle[a]] else 1
        expected[basis << (num_wires - n - len(g0))] = sign * norm
    return float(np.max(np.abs(np.asarray(state) - expected)))


def fixup_check() -> dict:
    """Task 0.9c requirement 2 (tape mode): see the section comment. ``passed`` needs every
    round trip and oracle case exact and every mutant rejected."""
    start = time.perf_counter()
    tables = {seed: _fixup_table(seed) for seed in FIXUP_SEEDS}
    record = {"geometry": {
        "round_trip": {"wires": dict(zip(("high", "low", "target", "junk", "work"), FIXUP_WIRES,
                                         strict=True)),
                       "width": FIXUP_WIDTH, "block_bits": FIXUP_BLOCK_BITS,
                       "high_ranges": FIXUP_HIGH_RANGES, "onehot_bits": FIXUP_ONEHOT_BITS,
                       "onehot_wires": (FIXUP_WIRES[2] + FIXUP_WIRES[3])[: 2**FIXUP_ONEHOT_BITS],
                       "covered": list(range(6)), "dead": [5], "unvisited": [6, 7]}},
              "seeds": list(FIXUP_SEEDS),
              "tables": {seed: [from_words(e) for _, e in t] for seed, t in tables.items()}}
    record["round_trip"] = {seed: _fixup_round_trip(t, _fixup_erase_op(t, ProtoTableEraseF))
                            for seed, t in tables.items()}
    record["round_trip_fp"] = {seed: _fixup_round_trip(t, _fixup_erase_op(t, ProtoTableEraseFP))
                               for seed, t in tables.items()}
    oracle = {}
    for name, geometry in fixup_oracle_geometries().items():
        size = len(geometry["covered"])
        vectors = [[False] * size] + [[i == j for j in range(size)] for i in range(size)]
        errors = [_oracle_case(geometry, v) for v in vectors]
        oracle[name] = {**{k: geometry[k] for k in ("address_bits", "onehot_bits", "control")},
                        "covered": len(geometry["covered"]),
                        "covered_matches_oracle": geometry["covered"] == geometry["oracle"],
                        "parity_vectors": len(vectors),
                        "addresses_checked": 2 ** (geometry["address_bits"] + geometry["control"]),
                        "max_error": max(errors),
                        "passed": max(errors) < FIXUP_TOLERANCE
                        and geometry["covered"] == geometry["oracle"]}
    record["oracle"] = oracle
    mutant_seed = next(seed for seed, t in tables.items()
                       if from_words(t[4][1]) and from_words(t[0][1]) != from_words(t[1][1]))
    table = tables[mutant_seed]
    inner = fixup_oracle_geometries()["inner"]
    dead_inner = SHAPES["small"].B + 1  # (x, b) = (0, B + 1), a dead position of a visited block
    unit = [[i == j for j in range(len(inner["covered"]))] for i in range(len(inner["covered"]))]
    mutants = {
        "flip_parity_0": {
            "round_trip": _fixup_round_trip(table, _fixup_mutant_erase(
                table, parities_hook=_flip_parity(0))),
            "oracle_inner": _oracle_case(inner, [False] * len(inner["covered"]),
                                         parities_hook=_flip_parity(0))},
        "drop_dead_address": {
            "round_trip": _fixup_round_trip(table, _fixup_mutant_erase(
                table, masks_hook=lambda masks: [(a, m) for a, m in masks if a != 5])),
            "oracle_inner": _oracle_case(
                inner, unit[inner["covered"].index(dead_inner)],
                covered_hook=lambda covered: [a for a in covered if a != dead_inner])},
        "swap_onehot_0_1": {
            "round_trip": _fixup_round_trip(table, _fixup_mutant_erase(
                table, leaves_hook=_swap_first_positions)),
            "oracle_inner": _oracle_case(inner, unit[0], leaves_hook=_swap_first_positions)},
    }
    for mutant in mutants.values():
        mutant["rejected"] = (not mutant["round_trip"]["passed"]
                              and mutant["oracle_inner"] > FIXUP_TOLERANCE)
    record["mutants"] = {"seed": mutant_seed, "dead_mask": from_words(table[4][1]) << FIXUP_WIDTH,
                         "inner_dead_address": dead_inner, **mutants}
    record["passed"] = (all(r["passed"] for r in record["round_trip"].values())
                        and all(r["passed"] for r in record["round_trip_fp"].values())
                        and all(o["passed"] for o in oracle.values())
                        and all(m["rejected"] for m in mutants.values()))
    record["seconds"] = round(time.perf_counter() - start, 1)
    return record


# --------------------------------------------------------------------------- driver

PROGRAMS = ("walk", "controlled_walk", "inner_prepare", "inner_select")
SHAPE_PROGRAMS = {
    "small": PROGRAMS,
    "mid10": ("walk", "controlled_walk"),
    "mid14": PROGRAMS,
    "femoco": ("walk", "controlled_walk"),
}
CALIBRATION_SHAPES = ("small", "mid10", "mid14")
LIMITS = {
    "small": (300, HEAVY_MEMORY_MB),
    "mid10": (900, HEAVY_MEMORY_MB),
    "mid14": (900, HEAVY_MEMORY_MB),
    "femoco": (3600, MEMORY_CAP_MB),
}
CONTROL_DELTAS = {"toffoli": 2, "measured_and_uncomputes": 2, "measurements": 0}
PROTOTYPE_NAMES = frozenset(
    obj.__name__ for obj in [*vars(prototypes).values(), *globals().values()]
    if isinstance(obj, type) and issubclass(obj, Operator2) and obj.__name__.startswith("Proto"))
"""Class names of the Proto* operators of this driver and spikes/prototypes.py; a recorded
warning that names one fails the verdict (spec 7.4, rule 12)."""
EXPECTED_MEASURING = frozenset(cls.__name__ for cls in (
    ProtoTableErase, ProtoAngleErase, ProtoTableEraseF, ProtoTableEraseFP, ProtoAngleEraseF,
    ProtoAngleEraseFP))
"""Measuring operators whose measured rule Catalyst may skip under Adjoint(...) or C(...)."""
MEMORY_METHOD = (
    "peak_mb: sampled peak of the summed physical footprint of the compile's process tree "
    "(dfthc.testing.watchdog), the gate metric; the watchdog kills the tree above the limit. "
    "rusage_sum_mb: the getrusage companion, ru_maxrss of the compile process plus ru_maxrss of "
    "its finished children. It is not an upper bound: under memory pressure the resident size "
    "drops compressed pages that the footprint keeps counting, so it can fall below peak_mb"
)
PROJECTION_METHOD = (
    "size measure: the stand-in program's emitted gates, from its resource functions "
    "(DecompositionGraph.resource_estimate, no compile). Time: power law through the two largest "
    "calibration shapes, exponent at least 1. Memory: least-squares line through all calibration "
    "shapes, slope at least 0, for peak_mb (the gate metric) and rusage_sum_mb (the getrusage "
    "companion). The FeMoco compile may run only if, for the walk and the controlled walk, the "
    "projected time, peak_mb and rusage_sum_mb are all within the time limit of LIMITS['femoco'] "
    "and the watchdog's effective limit of its cap (7,680 MB of 8 GB), where execution would be "
    "killed"
)
REMEDY_PROJECTION_METHOD = (
    "size measure: projection_size_gates(shape, program), from the baseline stand-in's resource "
    "functions (no compile): every gate of the baseline program with all table words zero (the "
    "skeleton that partial remedy 1 keeps unrolled), plus the conditional-Z leaves of the "
    "unchanged erases with random_tables(shape, seed=54). It must reproduce the walk sizes of "
    "PROBE_A_WALK_SIZES. A remedy compile's analyzed gates (analyzed_gates_upper_bound) count "
    "every loaded bit and are recorded, not fitted. Fit points: a shape's program record in a "
    "k1r1 file of this code_hash() with successful counts, within_budget true and a passed stop "
    "gate; at least two per program (fit_shapes). Time: power law through the two largest fit "
    "shapes, exponent at least 1. Memory: least-squares line through all fit shapes, slope at "
    "least 0, for peak_mb (the gate metric) and rusage_sum_mb (the getrusage companion). "
    "femoco_allowed as in Task 0.9: the walk and the controlled walk within the time limit of "
    "LIMITS['femoco'] and the watchdog's effective limit of its cap (7,680 MB of 8 GB) by both "
    "memory series. Task 0.9b compiles no FeMoco shape, whatever the projection says"
)
PROBE_A_WALK_SIZES = {"mid10": 20599, "mid14": 32822, "femoco": 193053}
"""projection_size_gates of the walk, from the Opus probe A of Task 0.9's review."""
RESULTS = Path(__file__).resolve().parent / "results"
FEMOCO_LOCK = RESULTS / "k1_femoco.lock"
TOKEN_VARIABLE = "DFTHC_K1_FEMOCO_TOKEN"
LOCK_FILE = Path(__file__).resolve().parents[1] / "requirements-nightly.lock"
_HASHED_FILES = (Path(__file__).resolve(), Path(__file__).resolve().parent / "prototypes.py",
                 Path(dfthc.compile.__file__).resolve(),
                 Path(dfthc.testing.resources.__file__).resolve(),
                 Path(dfthc.testing.watchdog.__file__).resolve())
_DISTRIBUTIONS = ("pennylane", "pennylane_catalyst", "pennylane_lightning", "jax", "jaxlib",
                  "numpy")


def environment() -> dict[str, str]:
    """Python, platform, the installed toolchain versions and the lock file's SHA-256."""
    record = {"python": platform.python_version(), "platform": platform.platform()}
    for name in _DISTRIBUTIONS:
        try:
            record[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record[name] = "not installed"
    record["lock_sha256"] = (hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest()
                             if LOCK_FILE.exists() else "missing")
    return record


def code_hash() -> str:
    """SHA-256 over what a compile result depends on: this driver, the prototypes,
    dfthc.compile, dfthc.testing.resources (the warning classifier), the watchdog and
    environment()."""
    digest = hashlib.sha256()
    for path in _HASHED_FILES:
        digest.update(path.read_bytes())
    digest.update(json.dumps(environment(), sort_keys=True).encode())
    return digest.hexdigest()


def build(shape_name: str, program: str, remedy: bool = False, tables=None):
    s = SHAPES[shape_name]
    shape = tuple(getattr(s, f) for f in Shape.__dataclass_fields__)
    lay = Layout(s)
    tables = random_tables(s) if tables is None else tables
    wires = list(range(lay.num_wires))
    control = lay.num_wires
    _, inner_prepare, inner_select, _, walk = _parts(remedy)

    def qfunc():
        qp.PhaseGradientStatePrep(lay.phase_grad)
        if program == "walk":
            walk(shape, tables, wires)
        elif program == "controlled_walk":
            qp.ctrl(walk(shape, tables, wires), control=control)
        elif program == "inner_prepare":
            inner_prepare(shape, tables, wires)
            qp.adjoint(inner_prepare(shape, tables, wires))
        else:
            inner_select(shape, tables, wires)

    return qfunc, lay.num_wires + 1


def stand_in_gates(shape_name: str, program: str) -> int:
    """Gates the walk or controlled walk emits, from the resource functions alone (no compile)."""
    s = SHAPES[shape_name]
    shape = tuple(getattr(s, f) for f in Shape.__dataclass_fields__)
    lay = Layout(s)
    with qp.queuing.AnnotatedQueue():
        op = ProtoWalk(shape, random_tables(s), list(range(lay.num_wires)))
        if program == "controlled_walk":
            op = qp.ctrl(op, control=lay.num_wires)
    graph = DecompositionGraph(operations=[op], gate_set=GATE_SET, fixed_decomps=FIXED_DECOMPS)
    return int(graph.solve().resource_estimate(op).num_gates)


def declared_counts(shape_name: str, program: str, remedy: bool = False, tables=None) -> Counter:
    """Gate counts of a program from the resource functions alone (no compile), by default with
    random_tables(). With the remedy, "CNOT" is an upper bound (ProtoTableReadR1 counts every
    loaded bit)."""
    qfunc, _ = build(shape_name, program, remedy, tables)
    with qp.queuing.AnnotatedQueue() as queue:
        qfunc()
    graph = DecompositionGraph(operations=queue.queue, gate_set=GATE_SET,
                               fixed_decomps=FIXED_DECOMPS)
    solution = graph.solve()
    counts = Counter()
    for op in queue.queue:
        if op.name in GATE_SET:
            counts[op.name] += 1
            continue
        for key, value in solution.resource_estimate(op).gate_counts.items():
            counts[getattr(key, "name", str(key))] += int(value)
    return counts


def zero_word_tables(tables):
    """``tables`` with every word zero (same addresses and word counts)."""
    return tuple(tuple((address, tuple(0 for _ in words)) for address, words in table)
                 for table in tables)


@functools.cache
def projection_size_parts(shape_name: str, program: str) -> dict:
    """The size measure of the remedy projection and its parts, from the baseline resource
    functions alone (no compile; REMEDY_PROJECTION_METHOD).

    ``skeleton_gates``: the baseline program with every table word zero, the part remedy 1 keeps
    unrolled (unary iteration, swaps, Givens ladders, comparators, ...). ``erase_conditional_z``:
    the conditional-Z leaves of the unchanged erases with random_tables(shape, seed=54).
    ``projection_size_gates`` is their sum; ``baseline_gates`` is the baseline program with
    those tables (stand_in_gates for the walk). PhaseGradientStatePrep is not counted.
    """
    tables = random_tables(SHAPES[shape_name], seed=54)
    full = declared_counts(shape_name, program, tables=tables)
    zero = declared_counts(shape_name, program, tables=zero_word_tables(tables))
    for counts in (full, zero):
        counts.pop("PhaseGradientStatePrep", None)
    skeleton, erase_z = sum(zero.values()), full["PauliZ"] - zero["PauliZ"]
    return {"projection_size_gates": skeleton + erase_z, "skeleton_gates": skeleton,
            "erase_conditional_z": erase_z, "baseline_gates": sum(full.values())}


def projection_size_gates(shape_name: str, program: str) -> int:
    return projection_size_parts(shape_name, program)["projection_size_gates"]


# --------------------------------------------------------------------------- Task 0.9c sizes


def _names(counts: Counter, drop=()) -> dict:
    return {name: n for name, n in counts.items() if n and name not in drop}


def _erase_f_structure(op, looped):
    a = op.arguments
    counts = onehot_erase_counts(a["table"], a["high_ranges"], a["block_bits"], a["width"],
                                 a["onehot_bits"])
    covered = [x for x, _ in loaded_masks(a["table"], a["high_ranges"], a["block_bits"],
                                          a["width"])]
    loops = Counter({"fixup_leaf": len(fixup_leaves(covered, a["onehot_bits"]))} if looped else {})
    return counts_to_resources(Counter(_names(counts, ("CZ",) if looped else ()))), loops, \
        counts["CZ"]


def _angle_erase_f_structure(op, looped):
    a, s = op.arguments, Shape(*op.arguments["shape"])
    counts = angle_erase_onehot_counts(a["shape"], a["tables"])
    loops = Counter({"fixup_leaf": len(fixup_leaves([x for x, _ in a["tables"][3]], s.w_b))}
                    if looped else {})
    return counts_to_resources(Counter(_names(counts, ("CZ",) if looped else ()))), loops, \
        counts["CZ"] + counts["PauliZ"]


def _read_r1_structure(op):
    a = op.arguments
    counts = _r1_read_counts(a["addresses"], a["high_ranges"], a["block_bits"], a["width"],
                             count_loads=False)
    return counts_to_resources(counts), Counter({"read_bit": len(a["addresses"])}), 0


def _plain_erase_structure(op):
    a = op.arguments
    counts = _erase_counts(a["table"], a["high_ranges"], a["block_bits"], a["width"])
    return counts_to_resources(counts), Counter(), counts["PauliZ"]


def _plain_angle_erase_structure(op):
    a = op.arguments
    resources = _angle_erase_resources(a["shape"], a["tables"], a["wires"])
    return resources, Counter(), resources.get(qp.Z, 0)


STRUCTURE = {
    "ProtoTableReadR1": _read_r1_structure,
    "ProtoTableErase": _plain_erase_structure,
    "ProtoAngleErase": _plain_angle_erase_structure,
    "ProtoTableEraseF": lambda op: _erase_f_structure(op, False),
    "ProtoTableEraseFP": lambda op: _erase_f_structure(op, True),
    "ProtoAngleEraseF": lambda op: _angle_erase_f_structure(op, False),
    "ProtoAngleEraseFP": lambda op: _angle_erase_f_structure(op, True),
    "ProtoGivensLadderLoop": lambda op: ({}, Counter({"givens_base": 1}), 0),
    "Adjoint(ProtoGivensLadderLoop)": lambda op: ({}, Counter({"givens_adjoint": 1}), 0),
}
"""Operators whose structure the static-size counter does not take from their resource function
alone: ``(non-loop resources, loop sites by class, logical conditional sites)``. Loop bodies
contribute no static gate: the bit loads of the remedy-1 reads (zero table words), the CZs of
the FP fixup leaves and the rungs of the looped ladders."""


def _key_name(key) -> str:
    return key.name if isinstance(getattr(key, "name", None), str) else str(key)


@functools.cache
def structural_size(shape_name: str, program: str, remedy=False) -> dict:
    """Static size of a program, independent of qp.analyze (Task 0.9c requirement 5).

    Recursively expands the program's resource graph (the pinned rules the solver uses) with
    program multiplicities and counts every emitted gate-set primitive outside a qp.for_loop:
    neither the loop operation nor its body counts (STRUCTURE). PennyLane operators count with
    their resource estimate; PhaseGradientStatePrep is not counted, as in Task 0.9b. Tables are
    random_tables(shape). Returns ``static_size``, ``static_gates`` by name,
    ``projection_loop_sites`` by loop class (with program multiplicity) and
    ``conditional_sites`` (one per covered address of every erase fixup, unhinted).
    """
    qfunc, _ = build(shape_name, program, remedy)
    with qp.queuing.AnnotatedQueue() as queue:
        qfunc()
    solution = DecompositionGraph(operations=queue.queue, gate_set=GATE_SET,
                                  fixed_decomps=FIXED_DECOMPS).solve()
    memo = {}

    def expand(op):
        name = _key_name(op)
        if name in GATE_SET:
            return Counter({name: 1}), Counter(), 0
        if "Proto" not in name:
            estimate = solution.resource_estimate(op).gate_counts
            return Counter({_key_name(k): int(v) for k, v in estimate.items()}), Counter(), 0
        if op in memo:
            return memo[op]
        if name in STRUCTURE:
            keys, loops, sites = STRUCTURE[name](op)
            keys = {abstractify(k): n for k, n in keys.items()}
        else:
            rule = solution.decomposition(op)
            keys, loops, sites = rule.compute_resources(**op.arguments).gate_counts, Counter(), 0
        gates, loops = Counter(), Counter(loops)
        for key, n in keys.items():
            key_gates, key_loops, key_sites = expand(key)
            for k, v in key_gates.items():
                gates[k] += n * v
            for k, v in key_loops.items():
                loops[k] += n * v
            sites += n * key_sites
        memo[op] = gates, loops, sites
        return memo[op]

    gates, loops, sites = Counter(), Counter(), 0
    for op in queue.queue:
        if op.name == "PhaseGradientStatePrep":
            continue
        op_gates, op_loops, op_sites = expand(abstractify(op))
        gates.update(op_gates)
        loops.update(op_loops)
        sites += op_sites
    return {"static_size": sum(gates.values()), "static_gates": dict(sorted(gates.items())),
            "projection_loop_sites": {kind: loops[kind] for kind in LOOP_CLASSES},
            "conditional_sites": sites}


def check_r1_counter() -> dict:
    """Apply structural_size to partial remedy 1: it must reproduce PROBE_A_WALK_SIZES."""
    sizes = {shape: structural_size(shape, "walk", True)["static_size"]
             for shape in PROBE_A_WALK_SIZES}
    if sizes != PROBE_A_WALK_SIZES:
        raise ValueError(f"structural_size of the remedy-1 walk {sizes} != {PROBE_A_WALK_SIZES}")
    return sizes


ERASE_MULTIPLICITY = {"walk": {"outer": 1, "inner": 2, "angle": 2},
                      "controlled_walk": {"outer": 1, "inner": 2, "angle": 2},
                      "inner_prepare": {"inner": 1}, "inner_select": {"angle": 1}}
"""Erase operators per program: the walk unprepares the outer state once and the inner state
twice, and runs two inner selects (each ends with an angle erase)."""
FEMOCO_ERASE_TMA = {"outer": (36, 96, 36), "inner": (196, 416, 196), "angle": (102, 798, 102)}
FEMOCO_ERASE_SITES = {"outer": 324, "inner": 8640, "sf": 270, "tail": 54}
FEMOCO_WALK_SITES = 18252
"""FeMoco targets of Task 0.9c requirement 3 (spec 9.5, 9.17, 9.24)."""


def independent_erase_counts(shape_name: str) -> dict:
    """Each erase's T, M, A and logical conditional sites, directly from U and O of spec 9.24 and
    the covered sets of requirement 1 (no resource function), and the plain-fixup T of partial
    remedy 1 (the nested exact iteration over the loaded set)."""
    s, ob = SHAPES[shape_name], erase_geometry(SHAPES[shape_name])
    u = unary_toffolis
    k1, k2 = 2**s.k1, 2**s.k2
    covered_outer, blocks_outer = -(-s.X_o // k1) * k1, -(-s.X_o // k1)
    padded_b, blocks_inner, q = -(-(s.B + 1) // k2) * k2, -(-(s.B + 1) // k2), ob["inner"] - s.w_b
    outer = u(0, -(-covered_outer // 2 ** ob["outer"]), s.w_x - ob["outer"], False) \
        + onehot_cost(ob["outer"])
    inner = u(0, -(-s.RC // 2**q), s.w_x - q, False) + onehot_cost(ob["inner"])
    sf, tail = u(0, s.R, s.w_r, True) + onehot_cost(s.w_b), u(s.RC, s.X_o, s.w_x, False)
    return {
        "outer": {"T": outer, "M": k1 * s.d1, "A": outer, "sites": covered_outer,
                  "onehot_bits": ob["outer"],
                  "plain_T": u(0, blocks_outer, s.w_x - s.k1, False)
                  + blocks_outer * u(0, k1, s.k1, True)},
        "inner": {"T": inner, "M": k2 * s.d2, "A": inner, "sites": s.RC * padded_b,
                  "onehot_bits": ob["inner"],
                  "plain_T": u(0, s.RC, s.w_x, False)
                  + s.RC * u(0, blocks_inner, s.w_b - s.k2, True)
                  + s.RC * blocks_inner * u(0, k2, s.k2, True)},
        "angle": {"T": sf + tail, "M": (s.N - 1) * s.b_rot + 3, "A": sf + tail,
                  "sites": s.R * s.B + s.N, "sf_sites": s.R * s.B, "tail_sites": s.N,
                  "sf_T": sf, "tail_T": tail, "onehot_bits": s.w_b,
                  "plain_T": u(0, s.R, s.w_r, True) + s.R * u(0, s.B, s.w_b, True) + tail},
    }


def _tma(counts) -> list[int]:
    return [counts.get("Toffoli", 0) + counts.get("TemporaryAND", 0), counts.get("MidMeasure", 0),
            counts.get("Adjoint(TemporaryAND)", 0)]


def resource_erase_counts(shape_name: str, remedy) -> dict:
    """Each erase's T, M, A and conditional sites from the arm's operator resource functions."""
    s, lay = SHAPES[shape_name], Layout(SHAPES[shape_name])
    shape, tables = tuple(getattr(s, f) for f in Shape.__dataclass_fields__), random_tables(s)
    wires, ob = list(range(lay.num_wires)), erase_geometry(s)
    out = {}
    for which in ("outer", "inner"):
        static, _ = _read_args(s, lay, wires, which, tables)
        counts = onehot_erase_counts(*static, ob[which])
        out[which] = {"tma": _tma(counts), "sites": counts["CZ"]}
    counts = angle_erase_onehot_counts(shape, tables)
    out["angle"] = {"tma": _tma(counts), "sites": counts["CZ"] + counts["PauliZ"]}
    return out


def independent_program(shape_name: str, program: str) -> dict:
    """An arm's program T/M/A and conditional sites from Task 0.9's compiled remedy-1 T/M/A
    (EXPECTED_TMA) plus the independent erase deltas; T/M/A is None at the FeMoco shape."""
    erases = independent_erase_counts(shape_name)
    multiplicity = ERASE_MULTIPLICITY[program]
    sites = sum(n * erases[e]["sites"] for e, n in multiplicity.items())
    base = EXPECTED_TMA.get(shape_name, {}).get(program)
    if base is None:
        return {"tma": None, "conditional_sites": sites}
    delta = sum(n * (erases[e]["T"] - erases[e]["plain_T"]) for e, n in multiplicity.items())
    return {"tma": [base[0] + delta, base[1], base[2] + delta], "conditional_sites": sites}


def independent_count_check() -> dict:
    """Task 0.9c requirement 3: per shape, the independent erase counts against the resource
    functions, and the FeMoco targets. ``ok`` needs every comparison equal."""
    out, ok = {}, True
    for shape_name in (*CALIBRATION_SHAPES, "femoco"):
        independent, declared = independent_erase_counts(shape_name), resource_erase_counts(
            shape_name, "F")
        rows = {}
        for e in ("outer", "inner", "angle"):
            tma = [independent[e][k] for k in ("T", "M", "A")]
            rows[e] = {"independent": tma, "resource_functions": declared[e]["tma"],
                       "sites": independent[e]["sites"],
                       "resource_function_sites": declared[e]["sites"],
                       "onehot_bits": independent[e]["onehot_bits"],
                       "plain_T": independent[e]["plain_T"]}
            ok &= tma == declared[e]["tma"] and independent[e]["sites"] == declared[e]["sites"]
        out[shape_name] = rows
    femoco = out["femoco"]
    targets = {e: list(FEMOCO_ERASE_TMA[e]) == femoco[e]["independent"] for e in FEMOCO_ERASE_TMA}
    sites = independent_erase_counts("femoco")
    walk_sites = independent_program("femoco", "walk")["conditional_sites"]
    targets["sites"] = ({"outer": sites["outer"]["sites"], "inner": sites["inner"]["sites"],
                         "sf": sites["angle"]["sf_sites"], "tail": sites["angle"]["tail_sites"]}
                        == FEMOCO_ERASE_SITES and walk_sites == FEMOCO_WALK_SITES)
    out["femoco_targets"] = targets
    out["ok"] = bool(ok and all(targets.values()))
    return out


@contextlib.contextmanager
def rule_capture_log():
    """Record every rule target Catalyst captures (target ID, seconds, equations, bit loops),
    by wrapping its capture functions as the Opus probe B of Task 0.9's review does."""
    import catalyst.decomposition.decomposition_rules as rules
    import catalyst.from_plxpr.qfunc_interpreter as interpreter

    log = []
    convert, materialize = (interpreter._convert_decomp_target_spec,
                            rules.materialize_decomp_target_spec)

    def entry(target_spec, seconds, traced):
        target = str(target_spec.target_id)
        kind = loop_class(target)
        loops = sum(_count_equations(t.closed_jaxpr, "for_loop") for t in traced) if kind else 0
        return {"id": target.split("{", 1)[0],
                "signature": hashlib.sha256(target.encode()).hexdigest()[:16],
                "seconds": seconds,
                "equations": sum(len(t.closed_jaxpr.jaxpr.eqns) for t in traced),
                "for_loops": loops if kind == "read_bit" else 0,
                "loop_class": kind, "class_for_loops": loops}

    def timed_convert(interp, target_spec):
        start = time.perf_counter()
        traced = convert(interp, target_spec)
        log.append(entry(target_spec, time.perf_counter() - start, traced))
        return traced

    def timed_materialize(target_spec):
        start = time.perf_counter()
        module = materialize(target_spec)
        log.append(entry(target_spec, time.perf_counter() - start, []))
        return module

    interpreter._convert_decomp_target_spec = timed_convert
    rules.materialize_decomp_target_spec = timed_materialize
    try:
        yield log
    finally:
        interpreter._convert_decomp_target_spec = convert
        rules.materialize_decomp_target_spec = materialize


LOOP_CLASSES = ("read_bit", "fixup_leaf", "givens_base", "givens_adjoint")
LOOP_TARGET_PREFIXES = {
    "read_bit": (f"{ProtoTableReadR1.__name__}{{",),
    "fixup_leaf": tuple(f"{c.__name__}{{" for c in (ProtoTableEraseF, ProtoTableEraseFP,
                                                     ProtoAngleEraseF, ProtoAngleEraseFP)),
    "givens_base": tuple(f"{c.__name__}{{" for c in (ProtoGivensLadder, ProtoGivensLadderLoop)),
    "givens_adjoint": tuple(f"Adjoint({c.__name__}){{"
                            for c in (ProtoGivensLadder, ProtoGivensLadderLoop)),
}
"""Captured rule targets whose qp.for_loop equations each loop class counts: the bare operators
that may hold the loop (and the ladders' adjoints); unused distributed Adjoint/C candidates of
the other operators are not counted."""


def loop_class(target: str) -> str | None:
    """The loop class of a captured rule target ID, or None."""
    for kind, prefixes in LOOP_TARGET_PREFIXES.items():
        if target.startswith(prefixes):
            return kind
    return None


def captured_loops(log: list[dict]) -> dict:
    """Per loop class, the sorted nonzero for_loop counts of its distinct captured targets (one
    per full target signature, the maximum over repeated captures, as in Task 0.9b)."""
    per_class = {kind: {} for kind in LOOP_CLASSES}
    for e in log:
        if e.get("loop_class"):
            seen = per_class[e["loop_class"]]
            seen[e["signature"]] = max(seen.get(e["signature"], 0), e["class_for_loops"])
    return {"captured": {kind: sorted(v for v in seen.values() if v)
                         for kind, seen in per_class.items()},
            "targets": {kind: len(seen) for kind, seen in per_class.items()}}


def _count_equations(jaxpr, primitive: str) -> int:
    """Equations named ``primitive`` in a jaxpr and its nested regions."""
    total = 0
    for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        total += eqn.primitive.name == primitive
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if hasattr(item, "eqns") or hasattr(item, "jaxpr"):
                    total += _count_equations(item, primitive)
    return total


def capture_diagnostics(log: list[dict], compile_seconds: float) -> dict:
    """Requirement 5: rule-capture time against compile time, and the modified (Adjoint/C) targets
    Catalyst lowered. IDs are cut at the first "{"; signatures count distinct full IDs."""
    modified = [e for e in log if e["id"].startswith(("Adjoint(", "C("))]
    return {"compile_seconds": round(compile_seconds, 1),
            "rule_capture_seconds": round(sum(e["seconds"] for e in log), 2),
            "rule_targets": len(log),
            "rule_equations": sum(e["equations"] for e in log),
            "modified_target_lowerings": len(modified),
            "modified_target_seconds": round(sum(e["seconds"] for e in modified), 2),
            "modified_target_equations": sum(e["equations"] for e in modified),
            "modified_target_signatures": len({e["signature"] for e in modified}),
            "modified_target_ids": sorted({e["id"] for e in modified}),
            "modified_target_lowerings_by_id": dict(sorted(Counter(e["id"] for e in modified
                                                                   ).items()))}


R1_READ_TABLES = {"walk": (0, 1, 2, 3), "controlled_walk": (0, 1, 2, 3), "inner_prepare": (1,),
                  "inner_select": (2, 3)}
"""Indices into random_tables() (outer, inner, tail, sf) of the remedy-1 reads of each program."""


def bit_loop_evidence(shape_name: str, program: str, log: list[dict], counts: dict,
                      remedy=True) -> dict:
    """Whether the compile kept the remedy's bit loops (requirement 1): the captured
    ProtoTableReadR1 rules hold one qp.for_loop per loaded address, and the analyzed CNOTs equal
    the declared upper bound (every loaded bit), not the baseline's set-bit count, which a
    constant-folded table would give. Catalyst may capture a target more than once per compile,
    so loops are counted once per distinct target. For a Task 0.9c arm the set-bit count is the
    arm's declared CNOTs with the reads' upper bound replaced by the baseline's set bits."""
    tables = random_tables(SHAPES[shape_name])
    expected_loops = sum(len(tables[i]) for i in R1_READ_TABLES[program])
    loops = {}
    for e in log:
        loops[e["signature"]] = max(loops.get(e["signature"], 0), e["for_loops"])
    captured = sum(loops.values())
    upper_bound = declared_counts(shape_name, program, remedy=remedy)["CNOT"]
    set_bits = declared_counts(shape_name, program)["CNOT"]
    if _variant(remedy) != "R1":
        set_bits += upper_bound - declared_counts(shape_name, program, remedy=True)["CNOT"]
    cnot = counts.get("CNOT", 0)
    return {"for_loops_captured": captured, "for_loops_expected": expected_loops,
            "cnot_analyzed": cnot, "cnot_declared_upper_bound": upper_bound,
            "cnot_set_bits": set_bits,
            "retained": captured == expected_loops and cnot == upper_bound != set_bits}


def compile_one(shape_name: str, program: str, remedy: bool = False) -> dict:
    qfunc, num_wires = build(shape_name, program, remedy)
    with rule_capture_log() as log, warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        start = time.perf_counter()
        resources = compiled_resources(qfunc, num_wires=num_wires)
        wall = time.perf_counter() - start
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024
    record = {"program": program, "shape": shape_name, "remedy": int(bool(remedy)),
              "num_wires": num_wires, "seconds": round(wall, 1),
              "cpu_seconds": round(own.ru_utime + own.ru_stime + children.ru_utime
                                   + children.ru_stime, 1),
              "rusage_sum_mb": round((own.ru_maxrss + children.ru_maxrss) * scale / 2**20, 1),
              # With the remedy, qp.analyze counts every loaded bit: not the emitted gates.
              "analyzed_gates_upper_bound" if remedy else "compiled_gates":
                  sum(resources.counts.values()),
              "toffoli": resources.toffoli, "measurements": resources.measurements,
              "measured_and_uncomputes": resources.measured_and_uncomputes,
              "counts": resources.counts,
              "diagnostics": capture_diagnostics(log, wall),
              "warnings": short_warnings(classify_warnings(records, PROTOTYPE_NAMES,
                                                           expected_measuring=EXPECTED_MEASURING))}
    if remedy:
        record["table_word_dtype"] = np.dtype(WORD_DTYPE).name
        record["bit_loops"] = bit_loop_evidence(shape_name, program, log, resources.counts,
                                                remedy)
        if _variant(remedy) in ARMS:
            record["arm"] = _variant(remedy)
            record["loops"] = captured_loops(log)
            record["projection_size"] = structural_size(shape_name, program, remedy)
        else:
            record["projection_size"] = projection_size_parts(shape_name, program)
    return record


def short_warning(message: str) -> str:
    """The text up to its first "{" plus the suffix after its last "}" (no attribute dumps)."""
    if "{" not in message:
        return message
    return message[: message.index("{")] + message[message.rindex("}") + 1:]


def short_warnings(classified: dict) -> dict:
    return {kind: [short_warning(m) for m in texts] if isinstance(texts, list) else texts
            for kind, texts in classified.items()}


def controlled_walk_check(results: dict) -> dict:
    """Compare the controlled walk's counts with the walk's against CONTROL_DELTAS."""
    walk, controlled = results.get("walk", {}), results.get("controlled_walk", {})
    if not all(key in walk and key in controlled for key in CONTROL_DELTAS):
        return {"ok": False, "reason": "the walk or the controlled walk has no counts"}
    deltas = {key: controlled[key] - walk[key] for key in CONTROL_DELTAS}
    return {"ok": deltas == CONTROL_DELTAS, "deltas": deltas, "expected": CONTROL_DELTAS}


def _fit_time(points, x):
    (x1, t1), (x2, t2) = sorted(points)[-2:]
    exponent = max(1.0, math.log(t2 / t1) / math.log(x2 / x1))
    return t2 * (x / x2) ** exponent, exponent


def _fit_memory(points, x):
    xs, ys = np.array([p[0] for p in points], float), np.array([p[1] for p in points], float)
    slope = max(0.0, float(np.polyfit(xs, ys, 1)[0]))
    intercept = float(ys.mean() - slope * xs.mean())
    return intercept + slope * x, slope


def project(calibration: dict[str, dict]) -> dict:
    """Project the FeMoco walk and controlled walk from the calibration results (PROJECTION_METHOD).

    ``calibration`` maps each of CALIBRATION_SHAPES to its k1_<shape>.json content. Raises
    ValueError if a result is missing, failed, or belongs to another code_hash().
    """
    current = code_hash()
    for shape in CALIBRATION_SHAPES:
        record = calibration.get(shape)
        if record is None or record.get("verdict") != "pass":
            raise ValueError(f"no passing k1_{shape}.json: run --shape {shape} --all first")
        if record.get("code_hash") != current:
            raise ValueError(f"k1_{shape}.json belongs to other code: rerun --shape {shape} --all")
    seconds_limit, cap_mb = LIMITS["femoco"]
    memory_limit = effective_limit_mb(cap_mb)
    programs, reasons = {}, []
    for program in SHAPE_PROGRAMS["femoco"]:
        shapes = [s for s in CALIBRATION_SHAPES if program in calibration[s]]
        gates = {s: stand_in_gates(s, program) for s in [*shapes, "femoco"]}
        seconds, exponent = _fit_time(
            [(gates[s], calibration[s][program]["seconds"]) for s in shapes], gates["femoco"])
        peak, peak_slope = _fit_memory(
            [(gates[s], calibration[s][program]["peak_mb"]) for s in shapes], gates["femoco"])
        companion, _ = _fit_memory(
            [(gates[s], calibration[s][program]["rusage_sum_mb"]) for s in shapes], gates["femoco"])
        programs[program] = {
            "gates": gates, "seconds": round(seconds), "time_exponent": round(exponent, 2),
            "peak_mb": round(peak), "mb_per_1000_gates": round(1000 * peak_slope, 2),
            "rusage_sum_mb_companion": round(companion)}
        if seconds > seconds_limit:
            reasons.append(f"{program}: {seconds / 60:.0f} min > {seconds_limit / 60:.0f} min")
        for label, value in (("sampled peak", peak), ("getrusage companion", companion)):
            if not within_effective_limit(value, cap_mb):
                reasons.append(f"{program}: {label} {value / 1024:.1f} GB > "
                               f"{memory_limit / 1024:.1f} GB")
    return {"method": PROJECTION_METHOD,
            "limits": {"seconds": seconds_limit, "mb": memory_limit, "cap_mb": cap_mb},
            "programs": programs, "femoco_allowed": not reasons,
            "reason": "; ".join(reasons) or "within the limits", "code_hash": current,
            "environment": environment()}


def fit_point(saved: dict, program: str, current: str) -> tuple[dict | None, str]:
    """A program's fit point from one k1r1_<shape>.json, or None and the reason it is missing."""
    record = saved.get(program)
    if saved.get("code_hash") != current:
        return None, "no result file for this code_hash()"
    if record is None:
        return None, saved.get("skipped", {}).get(program, {}).get("reason", "not compiled")
    if "error" in record or record.get("toffoli") is None:
        return None, f"no counts: {record.get('error', '')[-200:]}"
    if not record.get("within_budget"):
        return None, "not within_budget"
    if not record.get("gate", {"ok": True})["ok"]:
        return None, f"stop gate: {record['gate']['problems']}"
    return record, ""


def project_remedy(calibration: dict[str, dict]) -> dict:
    """Project the remedy-1 FeMoco walk and controlled walk (REMEDY_PROJECTION_METHOD).

    ``calibration`` maps calibration shapes to their k1r1_<shape>.json content. Raises
    ValueError if projection_size_gates misses a PROBE_A_WALK_SIZES value or a program has fewer
    than two fit points.
    """
    for shape, expected in PROBE_A_WALK_SIZES.items():
        got = projection_size_gates(shape, "walk")
        if got != expected:
            raise ValueError(f"projection_size_gates({shape!r}, 'walk') = {got}, not the "
                             f"probe A value {expected} (difference {got - expected})")
    current = code_hash()
    seconds_limit, cap_mb = LIMITS["femoco"]
    memory_limit = effective_limit_mb(cap_mb)
    programs, reasons = {}, []
    for program in SHAPE_PROGRAMS["femoco"]:
        points, missing = {}, {}
        for shape in CALIBRATION_SHAPES:
            record, why = fit_point(calibration.get(shape, {}), program, current)
            if record is None:
                missing[shape] = why
            else:
                points[shape] = record
        if len(points) < 2:
            raise ValueError(f"{program}: fewer than two fit points ({missing})")
        sizes = {s: projection_size_parts(s, program) for s in [*points, "femoco"]}
        x = {s: part["projection_size_gates"] for s, part in sizes.items()}
        seconds, exponent = _fit_time([(x[s], r["seconds"]) for s, r in points.items()],
                                      x["femoco"])
        peak, peak_slope = _fit_memory([(x[s], r["peak_mb"]) for s, r in points.items()],
                                       x["femoco"])
        companion, _ = _fit_memory([(x[s], r["rusage_sum_mb"]) for s, r in points.items()],
                                   x["femoco"])
        programs[program] = {
            "fit_shapes": list(points), "missing_points": missing, "sizes": sizes,
            "analyzed_gates_upper_bound": {s: r.get("analyzed_gates_upper_bound")
                                           for s, r in points.items()},
            "seconds": round(seconds), "time_exponent": round(exponent, 2),
            "peak_mb": round(peak), "mb_per_1000_gates": round(1000 * peak_slope, 2),
            "rusage_sum_mb_companion": round(companion)}
        if seconds > seconds_limit:
            reasons.append(f"{program}: {seconds / 60:.0f} min > {seconds_limit / 60:.0f} min")
        for label, value in (("sampled peak", peak), ("getrusage companion", companion)):
            if not within_effective_limit(value, cap_mb):
                reasons.append(f"{program}: {label} {value / 1024:.1f} GB > "
                               f"{memory_limit / 1024:.1f} GB")
    return {"method": REMEDY_PROJECTION_METHOD,
            "limits": {"seconds": seconds_limit, "mb": memory_limit, "cap_mb": cap_mb},
            "programs": programs, "femoco_allowed": not reasons,
            "reason": "; ".join(reasons) or "within the limits", "code_hash": current,
            "environment": environment()}


ARM_PROJECTION_METHOD = (
    "x: structural_size(shape, program, arm)['static_size'], the static size (every emitted "
    "gate-set primitive outside a qp.for_loop, by recursive expansion of the arm's resource graph "
    "with program multiplicities; neither loop operations nor loop bodies count; "
    "PhaseGradientStatePrep not counted); the same counter reproduces PROBE_A_WALK_SIZES on "
    "partial remedy 1. Estimate (a), Task 0.9b's formulas on the arm's fit points (k1c files of "
    "this code_hash() with counts, within_budget and a passed stop gate): time a power law "
    "through the two largest fit shapes, exponent at least 1; memory a least-squares line, slope "
    "at least 0, for peak_mb and rusage_sum_mb. femoco_allowed uses estimate (a) only, as in Task "
    "0.9b. Estimate (b), sensitivity evidence: with P, S, W the static sizes of inner_prepare, "
    "inner_select and walk and R = W - 2P - 2S, per memory metric y: H = 300 MB, "
    "cP = max(yP14 - H, 0)/P14, cS = max(yS14 - H, 0)/S14, cR = estimate (a)'s walk slope, "
    "alpha = (yW14 - H)/(2 cP P14 + 2 cS S14 + cR R14), prediction H + alpha (2 cP P + 2 cS S + "
    "cR R) at FeMoco (size-scaled, the best value); the address-scaled variant replaces P_femoco "
    "by P14 times the inner erase's covered-address growth. Time: H = 0 and cR = yW14/W14. The "
    "walk's range is the hull of estimate (a) and both variants. The controlled walk multiplies "
    "each component endpoint by the observed mid10 controlled/walk ratio of the same metric (the "
    "best value uses the size-scaled endpoint), widened by the small-shape ratio and the "
    "controlled walk's estimate (a); it is not an independently calibrated component model. "
    "Task 0.9c compiles no FeMoco shape"
)
COMPONENT_OVERHEAD_MB = 300


def _hull(values: dict) -> list[float]:
    return [round(min(values.values())), round(max(values.values()))]


def component_model(calibration: dict, sizes: dict, estimate_a: dict, current: str) -> dict:
    """Estimate (b) of Task 0.9c requirement 5 (ARM_PROJECTION_METHOD) for one arm."""
    points = {}
    for shape, programs in (("mid14", ("walk", "inner_prepare", "inner_select")),
                            ("mid10", ("walk", "controlled_walk")),
                            ("small", ("walk", "controlled_walk"))):
        for program in programs:
            record, why = fit_point(calibration.get(shape, {}), program, current)
            if record is None:
                raise ValueError(f"component model: no {shape} {program} point ({why})")
            points[shape, program] = record
    size = {shape: {p: sizes[shape][p]["static_size"]
                    for p in ("walk", "inner_prepare", "inner_select")}
            for shape in ("mid14", "femoco")}
    rest = {shape: size[shape]["walk"] - 2 * size[shape]["inner_prepare"]
            - 2 * size[shape]["inner_select"] for shape in size}
    if min(rest.values()) < 0:
        raise ValueError(f"component model: R = W - 2P - 2S < 0 ({rest})")
    growth = (independent_erase_counts("femoco")["inner"]["sites"]
              / independent_erase_counts("mid14")["inner"]["sites"])
    p14, s14, w14, r14 = (size["mid14"]["inner_prepare"], size["mid14"]["inner_select"],
                          size["mid14"]["walk"], rest["mid14"])
    pf, sf, rf = size["femoco"]["inner_prepare"], size["femoco"]["inner_select"], rest["femoco"]
    out = {"static_sizes": size, "rest": rest, "inner_address_growth": growth,
           "walk": {}, "controlled_walk": {}}
    slopes = {"peak_mb": estimate_a["walk"]["mb_per_gate"],
              "rusage_sum_mb": estimate_a["walk"]["companion_mb_per_gate"],
              "seconds": points["mid14", "walk"]["seconds"] / w14}
    a_values = {"peak_mb": "peak_mb", "rusage_sum_mb": "rusage_sum_mb_companion",
                "seconds": "seconds"}
    for metric, slope in slopes.items():
        overhead = 0 if metric == "seconds" else COMPONENT_OVERHEAD_MB
        y_p, y_s, y_w = (points["mid14", p][metric]
                         for p in ("inner_prepare", "inner_select", "walk"))
        c_p, c_s = max(y_p - overhead, 0) / p14, max(y_s - overhead, 0) / s14
        alpha = (y_w - overhead) / (2 * c_p * p14 + 2 * c_s * s14 + slope * r14)
        size_scaled = overhead + alpha * (2 * c_p * pf + 2 * c_s * sf + slope * rf)
        address_scaled = overhead + alpha * (2 * c_p * p14 * growth + 2 * c_s * sf + slope * rf)
        walk_ends = {"estimate_a": estimate_a["walk"][a_values[metric]],
                     "size_scaled": size_scaled, "address_scaled": address_scaled}
        out["walk"][metric] = {
            "H": overhead, "cP": c_p, "cS": c_s, "cR": slope, "alpha": alpha,
            "best": round(size_scaled), "endpoints": {k: round(v) for k, v in walk_ends.items()},
            "range": _hull(walk_ends)}
        ratio_mid10 = points["mid10", "controlled_walk"][metric] / points["mid10", "walk"][metric]
        ratio_small = points["small", "controlled_walk"][metric] / points["small", "walk"][metric]
        controlled_ends = {"estimate_a": estimate_a["controlled_walk"][a_values[metric]],
                           "size_scaled_x_mid10": size_scaled * ratio_mid10,
                           "address_scaled_x_mid10": address_scaled * ratio_mid10,
                           "size_scaled_x_small": size_scaled * ratio_small,
                           "address_scaled_x_small": address_scaled * ratio_small}
        out["controlled_walk"][metric] = {
            "ratio_mid10": round(ratio_mid10, 3), "ratio_small": round(ratio_small, 3),
            "best": round(size_scaled * ratio_mid10),
            "endpoints": {k: round(v) for k, v in controlled_ends.items()},
            "range": _hull(controlled_ends)}
    return out


def femoco_counts() -> dict:
    """The stand-in walk's FeMoco T/M/A from the resource functions (no compile): partial remedy
    1 (Task 0.9b's stand-in) and arm F (FP and FPG have F's counts)."""
    return {variant: _tma(declared_counts("femoco", program, remedy=remedy))
            for variant, remedy in (("R1", True), ("F", "F"))
            for program in ("walk",)}


def project_arm(arm: str, calibration: dict[str, dict]) -> dict:
    """Project one arm's FeMoco walk and controlled walk (ARM_PROJECTION_METHOD).

    ``calibration`` maps calibration shapes to their k1c_<arm>_<shape>.json content. Raises
    ValueError if the structural counter misses PROBE_A_WALK_SIZES on partial remedy 1, a program
    has fewer than two fit points, or the component model lacks a point.
    """
    r1_sizes = check_r1_counter()
    current = code_hash()
    seconds_limit, cap_mb = LIMITS["femoco"]
    memory_limit = effective_limit_mb(cap_mb)
    sizes = {shape: {p: structural_size(shape, p, arm) for p in PROGRAMS}
             for shape in (*CALIBRATION_SHAPES, "femoco")}
    estimate_a, reasons = {}, []
    for program in SHAPE_PROGRAMS["femoco"]:
        points, missing = {}, {}
        for shape in CALIBRATION_SHAPES:
            record, why = fit_point(calibration.get(shape, {}), program, current)
            if record is None:
                missing[shape] = why
            else:
                points[shape] = record
        if len(points) < 2:
            raise ValueError(f"{arm} {program}: fewer than two fit points ({missing})")
        x = {s: sizes[s][program]["static_size"] for s in [*points, "femoco"]}
        seconds, exponent = _fit_time([(x[s], r["seconds"]) for s, r in points.items()],
                                      x["femoco"])
        peak, peak_slope = _fit_memory([(x[s], r["peak_mb"]) for s, r in points.items()],
                                       x["femoco"])
        companion, companion_slope = _fit_memory(
            [(x[s], r["rusage_sum_mb"]) for s, r in points.items()], x["femoco"])
        estimate_a[program] = {
            "fit_shapes": list(points), "missing_points": missing, "static_size": x,
            "points": {s: {"seconds": r["seconds"], "peak_mb": r["peak_mb"],
                           "rusage_sum_mb": r["rusage_sum_mb"]} for s, r in points.items()},
            "seconds": round(seconds), "time_exponent": round(exponent, 2),
            "peak_mb": round(peak), "mb_per_gate": peak_slope,
            "rusage_sum_mb_companion": round(companion), "companion_mb_per_gate": companion_slope}
        if seconds > seconds_limit:
            reasons.append(f"{program}: {seconds / 60:.0f} min > {seconds_limit / 60:.0f} min")
        for label, value in (("sampled peak", peak), ("getrusage companion", companion)):
            if not within_effective_limit(value, cap_mb):
                reasons.append(f"{program}: {label} {value / 1024:.1f} GB > "
                               f"{memory_limit / 1024:.1f} GB")
    return {"estimate_a": estimate_a,
            "estimate_b": component_model(calibration, sizes, estimate_a, current),
            "sizes": {shape: {p: {k: v for k, v in r.items() if k != "static_gates"}
                              for p, r in by_program.items()}
                      for shape, by_program in sizes.items()},
            "femoco_allowed": not reasons, "reason": "; ".join(reasons) or "within the limits",
            "r1_counter_check": r1_sizes}


def femoco_refusal() -> str | None:
    """Why the projection does not allow the FeMoco compile now, or None if it does."""
    projection_file = RESULTS / "k1_projection.json"
    if not projection_file.exists():
        return "no projection: run the small and mid shapes, then --project"
    projection = json.loads(projection_file.read_text())
    if projection.get("code_hash") != code_hash():
        return "the projection belongs to other code: rerun the mid shapes and --project"
    if not projection.get("femoco_allowed"):
        return f"the projection exceeds the limits ({projection.get('reason')}); escalate at G0"
    return None


def femoco_checkpoint() -> dict:
    """The FeMoco programs already compiled (or attempted) for the current code_hash()."""
    path = RESULTS / "k1_femoco.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("code_hash") != code_hash():
        return {}
    return {program: saved[program] for program in SHAPE_PROGRAMS["femoco"] if program in saved}


def _driver_token_valid() -> bool:
    token = os.environ.get(TOKEN_VARIABLE)
    return bool(token) and FEMOCO_LOCK.exists() and FEMOCO_LOCK.read_text() == token


def results_path(name: str, remedy: bool = False) -> Path:
    """spikes/results/k1_<name>.json, k1r1_<name>.json for partial remedy 1, k1c_<arm>_<name>.json
    for a Task 0.9c arm, and k1c_projection.json for every arm's projection."""
    variant = _variant(remedy)
    if variant in ARMS:
        return RESULTS / ("k1c_projection.json" if name == "projection"
                          else f"k1c_{variant}_{name}.json")
    return RESULTS / f"{'k1r1' if remedy else 'k1'}_{name}.json"


EXPECTED_TMA = {
    "small": {"walk": (236, 68, 175), "controlled_walk": (238, 68, 177),
              "inner_prepare": (35, 14, 24), "inner_select": (66, 15, 53)},
    "mid10": {"walk": (2592, 452, 2407), "controlled_walk": (2594, 452, 2409)},
    "mid14": {"walk": (4275, 672, 3972), "controlled_walk": (4277, 672, 3974),
              "inner_prepare": (1053, 96, 973), "inner_select": (972, 198, 939)},
}
"""Task 0.9's compiled (T, M, A); the remedy must reproduce them exactly."""
CONTROLLED_SKIP = (1.8, 3500)
"""The remedy's mid14 controlled walk runs only if 1.8 times the mid14 walk's peak is below
3,500 MB (Task 0.9b, requirement 2)."""


def remedy_problems(shape_name: str, program: str, record: dict) -> list[str]:
    """Why a remedy-1 program record trips the stop gate (Task 0.9b, requirement 2)."""
    if "error" in record:
        return [f"error: {record['error'][-300:]}"]
    problems = []
    if not record.get("within_budget"):
        problems.append("over the time or memory limit")
    if record["warnings"]["prototype"]:
        problems.append("a warning names a prototype")
    tma = (record["toffoli"], record["measurements"], record["measured_and_uncomputes"])
    if tma != EXPECTED_TMA[shape_name][program]:
        problems.append(f"T/M/A {tma} != {EXPECTED_TMA[shape_name][program]}")
    if not record["bit_loops"]["retained"]:
        problems.append("the bit loop is lost")
    return problems


def remedy_refusal(shape_name: str) -> str | None:
    """Why a remedy-1 shape may not compile now: the FeMoco shape never does in Task 0.9b, and a
    mid shape needs passing results of every smaller calibration shape for this code_hash()."""
    if shape_name == "femoco":
        return "Task 0.9b compiles no FeMoco shape; the orchestrator decides after --project"
    current = code_hash()
    for shape in CALIBRATION_SHAPES[: CALIBRATION_SHAPES.index(shape_name)]:
        path = results_path(shape, remedy=True)
        saved = json.loads(path.read_text()) if path.exists() else {}
        if saved.get("verdict") != "pass" or saved.get("code_hash") != current:
            return f"stop gate: no passing {path.name} for this code (run --shape {shape} first)"
    return None


def expected_loops(shape_name: str, program: str, remedy) -> dict:
    """Per loop class, the sorted nonzero for_loop counts per distinct captured target that the
    arm's static geometry implies (requirement 3): one bit loop per loaded address of each read,
    one fixup-leaf loop per high leaf of each one-hot erase in FP/FPG, one loop in each looped
    ladder rule in FPG."""
    variant, s = _variant(remedy), SHAPES[shape_name]
    out = {kind: [] for kind in LOOP_CLASSES}
    if variant is None:
        return out
    tables, ob = random_tables(s), erase_geometry(s)
    out["read_bit"] = sorted(len(tables[i]) for i in R1_READ_TABLES[program])
    if variant in ("FP", "FPG"):
        covered_outer = -(-s.X_o // 2**s.k1) * 2**s.k1
        leaves = {"outer": -(-covered_outer // 2 ** ob["outer"]),
                  "inner": -(-s.RC // 2 ** (ob["inner"] - s.w_b)), "angle": s.R}
        out["fixup_leaf"] = sorted(leaves[e] for e in ERASE_MULTIPLICITY[program])
    if variant == "FPG" and "angle" in ERASE_MULTIPLICITY[program]:
        out["givens_base"], out["givens_adjoint"] = [1], [1]
    return out


def arm_expectations(shape_name: str, program: str, arm: str) -> dict:
    """What a Task 0.9c arm's program must give, derived before its compile (no compile):
    T/M/A from the arm's resource functions, from the independent count (Task 0.9's T/M/A plus
    the erase deltas of independent_erase_counts) and, for FP and FPG, from arm F's resource
    functions; the loops per class; the logical conditional sites."""
    independent = independent_program(shape_name, program)
    expected = {"tma_resource_functions": _tma(declared_counts(shape_name, program, remedy=arm)),
                "tma_independent": independent["tma"],
                "conditional_sites": independent["conditional_sites"],
                "loops": expected_loops(shape_name, program, arm)}
    if arm != "F":
        expected["tma_F"] = _tma(declared_counts(shape_name, program, remedy="F"))
    return expected


def arm_problems(shape_name: str, program: str, record: dict, expected: dict) -> list[str]:
    """Why a Task 0.9c arm's program record trips the stop gate (requirement 3, as Task 0.9b
    requirement 2): an error (kill, survivors, nonzero status, malformed output, symbolic counts,
    CompileError), over the limits, a prototype warning, T/M/A unequal to the resource functions,
    the independent count or arm F, folded table loads, a lost or extra loop in a class, or wrong
    logical conditional sites."""
    if "error" in record:
        return [f"error: {record['error'][-300:]}"]
    problems = []
    if not record.get("within_budget"):
        problems.append("over the time or memory limit")
    if record["warnings"]["prototype"]:
        problems.append("a warning names a prototype")
    tma = [record["toffoli"], record["measurements"], record["measured_and_uncomputes"]]
    for label in ("tma_resource_functions", "tma_independent", "tma_F"):
        if expected.get(label) is not None and tma != list(expected[label]):
            problems.append(f"T/M/A {tma} != {label} {expected[label]}")
    if not record["bit_loops"]["retained"]:
        problems.append("the read's bit loads are folded or lost")
    for kind in LOOP_CLASSES:
        got = record["loops"]["captured"].get(kind)
        if got != expected["loops"][kind]:
            problems.append(f"{kind} loops {got} != {expected['loops'][kind]}")
    if record["projection_size"]["conditional_sites"] != expected["conditional_sites"]:
        problems.append(f"conditional sites {record['projection_size']['conditional_sites']} "
                        f"!= {expected['conditional_sites']}")
    return problems


def _passing(path: Path, current: str) -> bool:
    saved = json.loads(path.read_text()) if path.exists() else {}
    return saved.get("verdict") == "pass" and saved.get("code_hash") == current


def arm_refusal(shape_name: str, arm: str) -> str | None:
    """Why a Task 0.9c arm's shape may not compile now: never the FeMoco shape; arm F needs a
    passing fixup check (requirement 2) for this code_hash(); a mid shape needs the arm's passing
    smaller shapes; FP and FPG need their parent arm's passing file at the same shape."""
    if shape_name == "femoco":
        return "Task 0.9c compiles no FeMoco shape; the projection is evidence for G0"
    current = code_hash()
    if arm == "F":
        saved = (json.loads(results_path("small", "F").read_text())
                 if results_path("small", "F").exists() else {})
        if not (saved.get("fixup_check", {}).get("passed")
                and saved.get("code_hash") == current):
            return "stop gate: no passing fixup check for this code (run --fixup-check first)"
    for shape in CALIBRATION_SHAPES[: CALIBRATION_SHAPES.index(shape_name)]:
        if not _passing(results_path(shape, arm), current):
            return f"stop gate: no passing {results_path(shape, arm).name} for this code"
    parent = ARM_PARENT[arm]
    if parent and not _passing(results_path(shape_name, parent), current):
        return (f"stop gate: arm {arm} builds on {parent}, which has no passing "
                f"{results_path(shape_name, parent).name} for this code")
    return None


def arm_controlled_skip(arm: str, walk14: dict) -> tuple[dict | None, dict]:
    """The mid14 controlled-walk rule of requirement 4: control_ratio = max(1.8, mid10 controlled
    walk peak / mid10 walk peak) from the arm's passing mid10 pair; run only if control_ratio
    times the mid14 walk's peak is below 3,500 MB. Returns (skip record or None, the rule's
    values); a missing or failed mid10 pair or mid14 walk means skip, not an assumed 1.8."""
    floor, limit_mb = CONTROLLED_SKIP
    current = code_hash()
    path = results_path("mid10", arm)
    mid10 = json.loads(path.read_text()) if path.exists() else {}
    pair = [fit_point(mid10, p, current)[0] for p in ("walk", "controlled_walk")]
    walk_ok = walk14.get("gate", {}).get("ok") and "peak_mb" in walk14
    if None in pair or not walk_ok:
        reason = ("no passing mid10 walk and controlled walk for this code" if None in pair
                  else "the mid14 walk did not pass")
        return {"reason": f"{reason}: skipped, not an assumed {floor}"}, {}
    ratio = max(floor, pair[1]["peak_mb"] / pair[0]["peak_mb"])
    rule = {"control_ratio": round(ratio, 3), "mid10_walk_peak_mb": pair[0]["peak_mb"],
            "mid10_controlled_walk_peak_mb": pair[1]["peak_mb"],
            "mid14_walk_peak_mb": walk14["peak_mb"],
            "projected_peak_mb": round(ratio * walk14["peak_mb"]), "limit_mb": limit_mb}
    if ratio * walk14["peak_mb"] < limit_mb:
        return None, rule
    return {"reason": f"control_ratio {ratio:.3f} x the mid14 walk's peak is "
                      f"{ratio * walk14['peak_mb']:.0f} MB, not below {limit_mb} MB "
                      "(Task 0.9c, requirement 4)", **rule}, rule


BUDGET_SECONDS = 2700
"""Task 0.9c's watched compile budget (requirement 4), with inclusive accounting."""
LEDGER_VARIABLE = "DFTHC_K1C_LEDGER"
"""Path of the JSON-lines budget ledger; when set, drive() appends every watched child and
refuses a child whose timeout would cross BUDGET_SECONDS unless its expected duration fits."""


def ledger_total() -> float:
    path = os.environ.get(LEDGER_VARIABLE)
    if not path or not Path(path).exists():
        return 0.0
    return sum(json.loads(line)["compile_seconds"]
               for line in Path(path).read_text().splitlines() if line.strip())


def ledger_append(label: str, seconds: float, compiles: bool = True) -> None:
    """Persist one watched child's seconds and the running total (no-op without the ledger)."""
    path = os.environ.get(LEDGER_VARIABLE)
    if not path:
        return
    total = ledger_total() + (seconds if compiles else 0.0)
    with open(path, "a") as ledger:
        ledger.write(json.dumps({"label": label, "seconds": round(seconds, 1),
                                 "compile_seconds": round(seconds, 1) if compiles else 0.0,
                                 "total_compile_seconds": round(total, 1),
                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def expected_seconds(shape_name: str, program: str, remedy) -> float | None:
    """A child's expected duration from the immediately preceding same-program shape of the same
    variant, scaled by Task 0.9b's remedy-1 growth between the two shapes for that program (the
    walk's if remedy 1 lacks the program); None without a preceding record."""
    previous = [s for s in CALIBRATION_SHAPES[: CALIBRATION_SHAPES.index(shape_name)]
                if program in SHAPE_PROGRAMS[s]] if shape_name in CALIBRATION_SHAPES else []
    if not previous:
        return None
    path = results_path(previous[-1], remedy)
    record = (json.loads(path.read_text()) if path.exists() else {}).get(program, {})
    if "watched_seconds" not in record:
        return None
    r1 = {s: json.loads(results_path(s, True).read_text())
          for s in (previous[-1], shape_name) if results_path(s, True).exists()}
    for name in (program, "walk"):
        try:
            growth = (r1[shape_name][name]["watched_seconds"]
                      / r1[previous[-1]][name]["watched_seconds"])
            return record["watched_seconds"] * growth
        except (KeyError, ZeroDivisionError):
            continue
    return None


def budget_refusal(timeout_s: float, expected_s: float | None) -> str | None:
    """Why a watched child may not start under the ledger's budget, or None."""
    if not os.environ.get(LEDGER_VARIABLE):
        return None
    remaining = BUDGET_SECONDS - ledger_total()
    if timeout_s <= remaining or (expected_s is not None and expected_s <= remaining):
        return None
    return (f"budget: {remaining:.0f} of {BUDGET_SECONDS} watched compile seconds remain, the "
            f"timeout is {timeout_s:.0f} s and the expected duration "
            f"{'unknown' if expected_s is None else f'{expected_s:.0f} s'}")


ARM_NOTES = (
    "Arm F follows one spike-local ob rule at every shape (erase_geometry). At mid10 the spec's "
    "MID_NAPHTHALENE parameters (k_e1 = 2, k_e2 = 1) would give the outer erase ob 2 (T 15) "
    "instead of this rule's 3 (T 12); the inner ob 4 agrees",
    "All arms remain stand-ins: erase masks and tables are compilable constants (no dynamic-mask "
    "erase path), and the SF angle load keeps its enable placeholder",
    "Erase conditionals carry qp.hint({'branch-prob': 0.5}): analyzed CZ and Z counts are "
    "expected values; T, M and A are exact",
)
"""Notes written into every k1c_<arm>_<shape>.json, after NOTES."""


def write_results(shape_name: str, results: dict, remedy: bool = False,
                  skipped: dict | None = None, stopped: str | None = None) -> dict:
    """Write k1_<shape>.json (k1r1_<shape>.json with the remedy, k1c_<arm>_<shape>.json for an
    arm) from the program records so far; the verdict covers all of them, and a skipped program
    counts as complete. ``stopped`` (the budget guard's reason) leaves the shape incomplete. Arm
    F's small file keeps its fixup check, and fails without a passing one for this code."""
    skipped = skipped or {}
    path, variant = results_path(shape_name, remedy), _variant(remedy)
    record = {p: results[p] for p in SHAPE_PROGRAMS[shape_name] if p in results}
    complete = len(record) + len(skipped) == len(SHAPE_PROGRAMS[shape_name])
    failed = any(not program.get("within_budget") or program.get("warnings", {}).get("prototype")
                 or (remedy and not program.get("gate", {}).get("ok"))
                 for program in record.values())
    if complete and "controlled_walk" in record:
        record["controlled_walk_check"] = controlled_walk_check(record)
        failed |= not record["controlled_walk_check"]["ok"]
    if skipped:
        record["skipped"] = skipped
    if stopped:
        record["stopped"] = stopped
    if variant == "F" and shape_name == "small":
        saved = json.loads(path.read_text()) if path.exists() else {}
        fixup = saved.get("fixup_check") if saved.get("code_hash") == code_hash() else None
        record["fixup_check"] = fixup
        failed |= not (fixup or {}).get("passed")
    if remedy:
        record["notes"] = list(NOTES) + (list(ARM_NOTES) if variant in ARMS else [])
    seconds_limit, cap_mb = LIMITS[shape_name]
    record.update(limits={"seconds": seconds_limit, "mb": effective_limit_mb(cap_mb),
                          "cap_mb": cap_mb},
                  memory_method=MEMORY_METHOD, code_hash=code_hash(), environment=environment(),
                  verdict="fail" if failed else "pass" if complete and not stopped
                  else "incomplete")
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def compressor_mb() -> float | None:
    """Memory stored in the macOS compressor: vm_stat's "Pages stored in compressor" times its
    page size, in MB (None if vm_stat fails)."""
    try:
        text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        page = int(re.search(r"page size of (\d+) bytes", text)[1])
        pages = int(re.search(r"Pages stored in compressor:\s+(\d+)", text)[1])
    except (OSError, subprocess.CalledProcessError, TypeError):
        return None
    return round(pages * page / 2**20, 1)


_NUMBER = (int, float)
CHILD_FIELDS = {"program": str, "shape": str, "seconds": _NUMBER, "cpu_seconds": _NUMBER,
                "rusage_sum_mb": _NUMBER, "toffoli": int, "measurements": int,
                "measured_and_uncomputes": int, "counts": dict, "diagnostics": dict,
                "warnings": dict}
"""Fields (and types) every child record needs before the driver uses it."""
REMEDY_CHILD_FIELDS = {"table_word_dtype": str, "bit_loops": dict, "projection_size": dict}
"""Further fields of a remedy-1 child record."""
ARM_CHILD_FIELDS = {"arm": str, "loops": dict}
"""Further fields of a Task 0.9c arm's child record."""


def child_record(line: str, shape_name: str, program: str, remedy) -> tuple[dict | None, str]:
    """Parse and validate a child's JSON line: the record, or None and why it is malformed
    (Task 0.9b review I1). A malformed record becomes an ``error`` record in drive()."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, f"not JSON ({exc})"
    if not isinstance(record, dict):
        return None, f"a JSON {type(record).__name__}, not an object"
    arm = _variant(remedy) in ARMS
    fields = {**CHILD_FIELDS, **(REMEDY_CHILD_FIELDS if remedy else {}),
              **(ARM_CHILD_FIELDS if arm else {})}
    for name, kind in fields.items():
        if name not in record:
            return None, f"missing field {name!r}"
        if isinstance(record[name], bool) or not isinstance(record[name], kind):
            return None, f"field {name!r} has type {type(record[name]).__name__}"
    if (record["shape"], record["program"]) != (shape_name, program):
        return None, f"record of {record['shape']}/{record['program']}, not {shape_name}/{program}"
    if not isinstance(record["warnings"].get("prototype"), list):
        return None, "warnings['prototype'] is not a list"
    if remedy and not isinstance(record["bit_loops"].get("retained"), bool):
        return None, "bit_loops['retained'] is not a bool"
    if arm and (not isinstance(record["loops"].get("captured"), dict)
                or not isinstance(record["projection_size"].get("conditional_sites"), int)):
        return None, "loops['captured'] or projection_size['conditional_sites'] is malformed"
    return record, ""


def drive(shape_name: str, allow_retry: bool = False, remedy: bool = False) -> int:
    seconds_limit, cap_mb = LIMITS[shape_name]
    results, programs, env = {}, list(SHAPE_PROGRAMS[shape_name]), dict(os.environ)
    skipped, stopped = {}, None
    arm = _variant(remedy) if _variant(remedy) in ARMS else None
    if remedy:
        refusal = arm_refusal(shape_name, arm) if arm else remedy_refusal(shape_name)
        if refusal:
            print("refused:", refusal, flush=True)
            return 2
    elif shape_name == "femoco":
        refusal = femoco_refusal()
        if refusal is None:
            results = femoco_checkpoint()
            done = [p for p in programs if results.get(p, {}).get("within_budget")]
            failed_before = [p for p in programs if p in results and p not in done]
            programs = [p for p in programs if p not in done]
            if failed_before and not allow_retry:
                refusal = (f"{', '.join(failed_before)} already failed for this code; a retry "
                           "needs the user's approval (--allow-retry)")
            elif not programs:
                refusal = "the FeMoco compile already completed for this code; it runs once"
        if refusal:
            print("refused:", refusal, flush=True)
            return 2
        for program in done:
            print(program, "reused from k1_femoco.json", flush=True)
        RESULTS.mkdir(exist_ok=True)
        try:
            with open(FEMOCO_LOCK, "x") as lock:
                lock.write(env.setdefault(TOKEN_VARIABLE, secrets.token_hex(16)))
        except FileExistsError:
            print(f"refused: {FEMOCO_LOCK} exists: another FeMoco driver runs, or one crashed "
                  "(check, then delete the lock)", flush=True)
            return 2
    try:
        for program in programs:
            rule = None
            if arm and shape_name == "mid14" and program == "controlled_walk":
                skip, rule = arm_controlled_skip(arm, results.get("walk", {}))
                if skip:
                    skipped[program] = skip
                    print(program, "skipped", json.dumps(skip), flush=True)
                    continue
            elif remedy and shape_name == "mid14" and program == "controlled_walk":
                factor, limit_mb = CONTROLLED_SKIP
                projected = factor * results["walk"]["peak_mb"]
                if projected >= limit_mb:
                    skipped[program] = {
                        "reason": f"{factor} x the mid14 walk's peak is {projected:.0f} MB, not "
                                  f"below {limit_mb} MB (Task 0.9b, requirement 2)",
                        "projected_peak_mb": round(projected)}
                    print(program, "skipped", json.dumps(skipped[program]), flush=True)
                    continue
            expected = arm_expectations(shape_name, program, arm) if arm else None
            refusal = budget_refusal(seconds_limit + 60,
                                     expected_seconds(shape_name, program, remedy))
            if refusal:
                stopped = f"{program}: {refusal}"
                print("stopped:", stopped, flush=True)
                break
            command = [sys.executable, __file__, "--shape", shape_name, "--program", program,
                       *(["--arm", arm] if arm else ["--remedy", "1"] if remedy else [])]
            before = compressor_mb()
            run = run_watched(command, cap_mb=cap_mb, timeout_s=seconds_limit + 60, env=env)
            after = compressor_mb()
            ledger_append(f"{_variant(remedy) or 'baseline'}/{shape_name}/{program}", run.seconds)
            load = {"watched_seconds": round(run.seconds, 1), "compressor_mb_before": before,
                    "compressor_mb_after": after,
                    "compressor_mb_delta": (None if None in (before, after)
                                            else round(after - before, 1))}
            lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
            record, malformed = (child_record(lines[-1], shape_name, program, remedy) if lines
                                 else (None, "no line starting with '{'"))
            if run.killed or run.survivors or run.returncode != 0 or malformed:
                if run.killed:
                    reason = f"killed: {run.killed}"
                elif run.survivors or run.returncode != 0:
                    reason = run.stderr[-3000:]
                else:
                    reason = f"malformed output: {malformed}; stdout tail: {run.stdout[-300:]!r}"
                results[program] = {"error": reason, **run.summary(), **load}
            else:
                record["peak_mb"] = round(run.peak_mb, 1)
                record["within_budget"] = (record["seconds"] <= seconds_limit
                                           and within_effective_limit(run.peak_mb, cap_mb))
                results[program] = {**record, **load}
            if arm:
                results[program]["expected"] = expected
                if rule:
                    results[program]["control_rule"] = rule
                problems = arm_problems(shape_name, program, results[program], expected)
            elif remedy:
                problems = remedy_problems(shape_name, program, results[program])
            if remedy:
                results[program]["gate"] = {"ok": not problems, "problems": problems}
            console = {k: v for k, v in results[program].items()
                       if k not in ("warnings", "counts", "diagnostics", "expected",
                                    "projection_size")}
            print(program, json.dumps(console), flush=True)
            if shape_name == "femoco":
                write_results(shape_name, results)
            if remedy and not results[program]["gate"]["ok"]:
                print("stop gate tripped:", "; ".join(problems), flush=True)
                break
    finally:
        if shape_name == "femoco":
            FEMOCO_LOCK.unlink(missing_ok=True)
    record = write_results(shape_name, results, remedy, skipped, stopped)
    if "controlled_walk_check" in record:
        print("controlled_walk_check", json.dumps(record["controlled_walk_check"]), flush=True)
    print("verdict:", record["verdict"])
    return 0 if record["verdict"] == "pass" else 1


def write_projection(remedy: bool = False) -> int:
    calibration = {}
    for shape in CALIBRATION_SHAPES:
        path = results_path(shape, remedy)
        if path.exists():
            calibration[shape] = json.loads(path.read_text())
    if _variant(remedy) in ARMS:
        return write_arm_projection(_variant(remedy), calibration)
    projection = project_remedy(calibration) if remedy else project(calibration)
    results_path("projection", remedy).write_text(json.dumps(projection, indent=2) + "\n")
    for program, values in projection["programs"].items():
        print(program, json.dumps(values), flush=True)
    print("femoco_allowed:", projection["femoco_allowed"], "-", projection["reason"])
    return 0


def write_arm_projection(arm: str, calibration: dict) -> int:
    """Add (or replace) one arm in k1c_projection.json; arms of another code_hash() are dropped."""
    path = results_path("projection", arm)
    saved = json.loads(path.read_text()) if path.exists() else {}
    arms = saved.get("arms", {}) if saved.get("code_hash") == code_hash() else {}
    arms[arm] = project_arm(arm, calibration)
    seconds_limit, cap_mb = LIMITS["femoco"]
    projection = {
        "method": ARM_PROJECTION_METHOD,
        "limits": {"seconds": seconds_limit, "mb": effective_limit_mb(cap_mb), "cap_mb": cap_mb},
        "arms": dict(sorted(arms.items())),
        "erase_geometry": {shape: erase_geometry(s) for shape, s in SHAPES.items()},
        "independent_count_check": independent_count_check(),
        "femoco_counts": femoco_counts(),
        "notes": list(ARM_NOTES), "code_hash": code_hash(), "environment": environment()}
    path.write_text(json.dumps(projection, indent=2) + "\n")
    result = arms[arm]
    for program, values in result["estimate_a"].items():
        print(arm, program, "estimate (a)", json.dumps({k: values[k] for k in (
            "fit_shapes", "static_size", "seconds", "time_exponent", "peak_mb",
            "rusage_sum_mb_companion")}), flush=True)
    for program, metrics in result["estimate_b"].items():
        if program in ("walk", "controlled_walk"):
            print(arm, program, "estimate (b)", json.dumps(
                {m: {"best": v["best"], "range": v["range"]} for m, v in metrics.items()}),
                flush=True)
    print(arm, "femoco_allowed:", result["femoco_allowed"], "-", result["reason"])
    return 0


def run_fixup_check() -> int:
    """--fixup-check: fixup_check() in a child under the watchdog (tape mode, no qjit, so no
    compile seconds); the record goes into k1c_F_small.json as ``fixup_check``."""
    run = run_watched([sys.executable, __file__, "--fixup-check-child"], cap_mb=2048,
                      timeout_s=600, env=dict(os.environ))
    ledger_append("F/small/fixup_check (tape only)", run.seconds, compiles=False)
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    try:
        record = json.loads(lines[-1])
        malformed = "" if isinstance(record, dict) and isinstance(record.get("passed"), bool) \
            else "not a fixup-check record"
    except (IndexError, json.JSONDecodeError) as exc:
        record, malformed = {}, f"no fixup-check record ({exc})"
    if run.killed or run.survivors or run.returncode != 0 or malformed:
        record = {"passed": False, "error": f"killed: {run.killed}" if run.killed
                  else malformed or run.stderr[-3000:], **run.summary()}
    record.update(watched_seconds=round(run.seconds, 1), peak_mb=round(run.peak_mb, 1),
                  method="tape mode: _tape_decompose with the pinned rules, deferred "
                         "measurements, default.qubit; no qjit")
    path = results_path("small", "F")
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("code_hash") != code_hash():
        saved = {"verdict": "incomplete"}
    saved.update(fixup_check=record, code_hash=code_hash(), environment=environment())
    if not record["passed"]:
        saved["verdict"] = "fail"
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(saved, indent=2) + "\n")
    print("fixup_check", json.dumps({k: record.get(k) for k in (
        "passed", "seconds", "watched_seconds", "peak_mb", "error")}), flush=True)
    return 0 if record["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", choices=sorted(SHAPES))
    parser.add_argument("--program", choices=PROGRAMS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--allow-retry", action="store_true")
    parser.add_argument("--remedy", type=int, choices=(1,),
                        help="partial remedy 1 (reads and angle loads); default: the baseline")
    parser.add_argument("--arm", choices=ARMS,
                        help="Task 0.9c arm on top of partial remedy 1 (implies --remedy 1)")
    parser.add_argument("--fixup-check", action="store_true",
                        help="arm F's tape-mode fixup check into k1c_F_small.json")
    parser.add_argument("--fixup-check-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    remedy = args.arm or args.remedy == 1
    if args.fixup_check_child:
        print(json.dumps(fixup_check(), default=str), flush=True)
        return
    if args.fixup_check:
        sys.exit(run_fixup_check())
    if args.project:
        sys.exit(write_projection(remedy))
    if args.shape is None:
        parser.error("--shape is required unless --project is given")
    if args.all:
        sys.exit(drive(args.shape, allow_retry=args.allow_retry, remedy=remedy))
    if args.shape == "femoco" and not _driver_token_valid():
        sys.exit("refused: the FeMoco shape compiles only through --shape femoco --all, which runs "
                 "each program once under the watchdog")
    print(json.dumps(compile_one(args.shape, args.program, remedy)), flush=True)


if __name__ == "__main__":
    main()
