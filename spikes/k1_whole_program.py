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
Every warning of a compile is recorded and classified (spikes/prototypes.classify_warnings).
Exit status 1 if a compile fails, is killed or over its limits, a warning names a prototype
operator, or the controlled-walk deltas differ. Catalyst also lowers the generic adjoint and
controlled candidates of operators with pinned rules and drops the measuring ones with a "Skipped
... it contains a mid-circuit measurement" warning, which for the measured erases
(EXPECTED_MEASURING) is recorded as "expected" and does not fail the verdict. A needed measuring
path that is missing still fails loudly through the hard pins, the gate-set closure in
compiled_resources and the exact counts.

The FeMoco shape is a large-scale compile (master plan section 12.9). --project fits the small and
mid results (PROJECTION_METHOD) and writes k1_projection.json. The FeMoco driver refuses, with exit
status 2 and without compiling, unless that projection belongs to the current code_hash() (the
driver, the prototypes, dfthc.compile, the watchdog, the toolchain versions and the lock file) and
keeps the walk and the controlled walk within 60 minutes and the effective limit of the 8 GB cap
(7,680 MB), by sampled peak and upper bound.
Each FeMoco program compiles at most once per code_hash(): k1_femoco.json is checkpointed after
every program, a completed program is reused, and a failed one is retried only with
--allow-retry, which needs the user's approval. A FeMoco --program run without the token of a
running driver is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import secrets
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.decomposition import DecompositionGraph
from pennylane.typing import Wire

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prototypes  # noqa: E402
from prototypes import (  # noqa: E402
    ProtoTableErase,
    ProtoTableRead,
    _Count,
    _Emit,
    _measure_x,
    classify_warnings,
    counts_to_resources,
    from_words,
    nested_iterate,
    to_words,
)

import dfthc.compile  # noqa: E402
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


def _prepare_body(s, lay, wires, which, tables, unprepare):
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
        ProtoTableRead(*static, *wire_args)
        comparator()
        for c, a, x in swaps:
            qp.CSWAP([wires[c], wires[a], wires[x]])
    else:
        for c, a, x in swaps:
            qp.CSWAP([wires[c], wires[a], wires[x]])
        qp.adjoint(comparator())
        ProtoTableErase(*static, *wire_args)
        for wire in m + extra:
            qp.H(wires[wire])
        qp.adjoint(ProtoUniformPrep(_w(wires, index), _w(wires, succ)))


def _prepare_resources(s, lay, which, tables, unprepare):
    static, _ = _read_args(s, lay, list(range(lay.num_wires)), which, tables)
    nh, nl = sum(w for w, _, _ in static[1]), static[2]
    width, junk = static[3], (2**static[2] - 1) * static[3]
    n_index = s.w_x if which == "outer" else s.w_b
    b_k = s.b_k1 if which == "outer" else s.b_k2
    table_op = ProtoTableErase if unprepare else ProtoTableRead
    def maybe_adjoint(op):
        return qp.adjoint(op) if unprepare else op

    cmp_key = resource_key(lambda: maybe_adjoint(qp.LeftQuantumComparator(
        Wire[b_k], Wire[b_k], Wire[1], Wire[b_k - 1], "<=")))
    uniform = resource_key(lambda: maybe_adjoint(ProtoUniformPrep(Wire[n_index], Wire[1])))
    table = _abstract(table_op, *static, Wire[nh], Wire[nl], Wire[width], Wire[junk],
                      Wire[len(lay.work)])
    return {uniform: 1, qp.H: b_k + 1, table: 1, cmp_key: 1, qp.CSWAP: n_index}


def _make_prepare(name, which):
    cls = type(name, (ProtoWalkPart,), {"__doc__": f"Stand-in {which} prepare (spec 9.7/9.8)."})

    def resources(shape, tables, wires):
        return _prepare_resources(Shape(*shape), _layout(shape), which, tables, False)

    def rule(shape, tables, wires):
        _prepare_body(Shape(*shape), _layout(shape), wires, which, tables, False)

    def adjoint_resources(base):
        shape, tables = base.shape, base.tables
        return _prepare_resources(Shape(*shape), _layout(shape), which, tables, True)

    def adjoint_rule(base):
        _prepare_body(Shape(*base.shape), _layout(base.shape), base.wires, which, base.tables, True)

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
    counter.counts["Hadamard"] -= 0
    counter.counts["MidMeasure"] -= 0
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


def _make_angle_load(cls, which):
    def resources(shape, tables, wires):
        s, lay = Shape(*shape), _layout(shape)
        static, wire_args = _angle_read(s, lay, list(range(lay.num_wires)), tables, which)
        key = _abstract(ProtoTableRead, *static, *(Wire[len(w)] for w in wire_args))
        return {key: 1, qp.X: 2 * (which == "sf")}

    def rule(shape, tables, wires):
        s, lay = Shape(*shape), _layout(shape)
        static, wire_args = _angle_read(s, lay, wires, tables, which)
        if which == "sf":
            qp.X(wires[lay.G0[0]])
        # The SF load is enabled on G0 = 0; the stand-in read ignores the enable (placeholder).
        ProtoTableRead(*static, *wire_args)
        if which == "sf":
            qp.X(wires[lay.G0[0]])

    rule.__name__ = f"_proto_{which}_angle_load"
    register_rule(cls, qp.register_resources(resources)(rule))


_make_angle_load(ProtoTailAngleLoad, "tail")
_make_angle_load(ProtoSFAngleLoad, "sf")


class ProtoInnerSelect(ProtoWalkPart):
    """Stand-in InnerSelect (spec 9.18); the pinned adjoint uses the adjoint Majorana gate."""


def _select_ops(shape, tables, wires, adjoint_majorana):
    s, lay = Shape(*shape), _layout(shape)
    part = (shape, tables, wires)
    mux = (_w(wires, lay.G0), _w(wires, lay.sigma_0), _w(wires, lay.sigma_1), _w(wires, lay.sg))
    swap = (_w(wires, lay.sg), _w(wires, lay.psi_up), _w(wires, lay.psi_dn))
    ladder = (s.b_rot, _w(wires, lay.psi_dn), _w(wires, lay.angles), _w(wires, lay.phase_grad),
              _w(wires, lay.work))
    majorana = (_w(wires, lay.succ_x_o), _w(wires, lay.succ_b), _w(wires, lay.G0),
                _w(wires, lay.G1), _w(wires, lay.b_neq_B), _w(wires, lay.sigma_0),
                _w(wires, lay.psi_dn[:1]), _w(wires, lay.work[:2]))
    ProtoTailAngleLoad(*part)
    ProtoSFAngleLoad(*part)
    ProtoSpinMux(*mux)
    ProtoSpinSwap(*swap)
    ProtoGivensLadder(*ladder)
    if adjoint_majorana:
        qp.adjoint(ProtoMajorana(*majorana))
    else:
        ProtoMajorana(*majorana)
    qp.adjoint(ProtoGivensLadder(*ladder))
    ProtoSpinSwap(*swap)
    ProtoSpinMux(*mux)
    ProtoAngleErase(*part)


def _select_resources(shape, tables, adjoint_majorana):
    s, lay = Shape(*shape), _layout(shape)
    part_keys = [_abstract(c, shape, tables, Wire[lay.num_wires])
                 for c in (ProtoTailAngleLoad, ProtoSFAngleLoad, ProtoAngleErase)]
    mux = _abstract(ProtoSpinMux, Wire[1], Wire[1], Wire[1], Wire[1])
    swap = _abstract(ProtoSpinSwap, Wire[1], Wire[s.N], Wire[s.N])
    ladder_args = (s.b_rot, Wire[s.N], Wire[len(lay.angles)], Wire[s.b_rot], Wire[len(lay.work)])
    ladder = _abstract(ProtoGivensLadder, *ladder_args)
    ladder_dagger = resource_key(lambda: qp.adjoint(ProtoGivensLadder(*ladder_args)))
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


def _be_ops(shape, tables, wires, control=None):
    part = (shape, tables, wires)
    t2, _ = _refs(shape, wires)
    ProtoOuterPrepare(*part)
    ProtoInnerPrepare(*part)
    ProtoInnerSigns(*_signs(shape, wires))
    ProtoInnerSelect(*part)
    qp.adjoint(ProtoInnerPrepare(*part))
    if control is None:
        ProtoZeroReflection(*t2)
    else:
        qp.ctrl(ProtoZeroReflection(*t2), control=control)
    ProtoInnerPrepare(*part)
    qp.adjoint(ProtoInnerSelect(*part))
    ProtoInnerSigns(*_signs(shape, wires))
    qp.adjoint(ProtoInnerPrepare(*part))
    qp.adjoint(ProtoOuterPrepare(*part))


def _signs(shape, wires):
    lay = _layout(shape)
    return _w(wires, lay.sgn), _w(wires, lay.d_sgn), _w(wires, lay.lt_b)


def _be_resources(shape, tables, controlled):
    lay = _layout(shape)
    s = Shape(*shape)
    n = Wire[lay.num_wires]
    keys = {}
    for cls in (ProtoOuterPrepare, ProtoInnerPrepare, ProtoInnerSelect):
        keys[_abstract(cls, shape, tables, n)] = 2 if cls is ProtoInnerPrepare else 1
        keys[resource_key(lambda c=cls: qp.adjoint(c(shape, tables, n)))] = (
            2 if cls is ProtoInnerPrepare else 1)
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


def _walk_resources(shape, tables, wires):
    n = Wire[_layout(shape).num_wires]
    return {_abstract(ProtoBlockEncoding, shape, tables, n): 1,
            _abstract(ProtoZeroReflection, *_ab_args(shape)): 1}


def _walk(shape, tables, wires):
    ProtoBlockEncoding(shape, tables, wires)
    _, ab = _refs(shape, wires)
    ProtoZeroReflection(*ab)


def _c_walk_resources(base, control_wires, control_values, work_wires, work_wire_type):
    n = Wire[_layout(base.shape).num_wires]
    return {resource_key(lambda: qp.ctrl(ProtoBlockEncoding(base.shape, base.tables, n),
                                         control=Wire[1])): 1,
            resource_key(lambda: qp.ctrl(ProtoZeroReflection(*_ab_args(base.shape)),
                                         control=Wire[1])): 1}


def _c_walk(base, control_wires, control_values, work_wires, work_wire_type):
    qp.ctrl(ProtoBlockEncoding(base.shape, base.tables, base.wires), control=control_wires[0])
    _, ab = _refs(base.shape, base.wires)
    qp.ctrl(ProtoZeroReflection(*ab), control=control_wires[0])


register_rule(ProtoWalk, qp.register_resources(_walk_resources)(_walk))
register_rule("C(ProtoWalk)", qp.register_resources(_c_walk_resources)(_c_walk))


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
EXPECTED_MEASURING = frozenset({ProtoTableErase.__name__, ProtoAngleErase.__name__})
"""Measuring operators whose measured rule Catalyst may skip under Adjoint(...) or C(...)."""
MEMORY_METHOD = (
    "peak_mb: sampled peak of the summed physical footprint of the compile's process tree "
    "(dfthc.testing.watchdog), the gate metric; the watchdog kills the tree above the limit. "
    "rusage_sum_mb: ru_maxrss of the compile process plus ru_maxrss of its finished children "
    "(getrusage), an upper bound, because Catalyst runs its compiler and linker as blocking "
    "subprocesses while Python holds the traced program"
)
PROJECTION_METHOD = (
    "size measure: the stand-in program's emitted gates, from its resource functions "
    "(DecompositionGraph.resource_estimate, no compile). Time: power law through the two largest "
    "calibration shapes, exponent at least 1. Memory: least-squares line through all calibration "
    "shapes, slope at least 0, for peak_mb (the gate metric) and rusage_sum_mb (the upper bound). "
    "The FeMoco compile may run only if, for the walk and the controlled walk, the projected time, "
    "peak_mb and rusage_sum_mb are all within the time limit of LIMITS['femoco'] and the "
    "watchdog's effective limit of its cap (7,680 MB of 8 GB), where execution would be killed"
)
RESULTS = Path(__file__).resolve().parent / "results"
FEMOCO_LOCK = RESULTS / "k1_femoco.lock"
TOKEN_VARIABLE = "DFTHC_K1_FEMOCO_TOKEN"
LOCK_FILE = Path(__file__).resolve().parents[1] / "requirements-nightly.lock"
_HASHED_FILES = (Path(__file__).resolve(), Path(__file__).resolve().parent / "prototypes.py",
                 Path(dfthc.compile.__file__).resolve(),
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
    dfthc.compile, the watchdog and environment()."""
    digest = hashlib.sha256()
    for path in _HASHED_FILES:
        digest.update(path.read_bytes())
    digest.update(json.dumps(environment(), sort_keys=True).encode())
    return digest.hexdigest()


def build(shape_name: str, program: str):
    s = SHAPES[shape_name]
    shape = tuple(getattr(s, f) for f in Shape.__dataclass_fields__)
    lay = Layout(s)
    tables = random_tables(s)
    wires = list(range(lay.num_wires))
    control = lay.num_wires

    def qfunc():
        qp.PhaseGradientStatePrep(lay.phase_grad)
        if program == "walk":
            ProtoWalk(shape, tables, wires)
        elif program == "controlled_walk":
            qp.ctrl(ProtoWalk(shape, tables, wires), control=control)
        elif program == "inner_prepare":
            ProtoInnerPrepare(shape, tables, wires)
            qp.adjoint(ProtoInnerPrepare(shape, tables, wires))
        else:
            ProtoInnerSelect(shape, tables, wires)

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


def compile_one(shape_name: str, program: str) -> dict:
    qfunc, num_wires = build(shape_name, program)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        start = time.perf_counter()
        resources = compiled_resources(qfunc, num_wires=num_wires)
        wall = time.perf_counter() - start
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024
    return {"program": program, "shape": shape_name, "num_wires": num_wires,
            "seconds": round(wall, 1),
            "cpu_seconds": round(own.ru_utime + own.ru_stime + children.ru_utime
                                 + children.ru_stime, 1),
            "rusage_sum_mb": round((own.ru_maxrss + children.ru_maxrss) * scale / 2**20, 1),
            "compiled_gates": sum(resources.counts.values()),
            "toffoli": resources.toffoli, "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "warnings": classify_warnings(records, PROTOTYPE_NAMES,
                                          expected_measuring=EXPECTED_MEASURING)}


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
        upper, _ = _fit_memory(
            [(gates[s], calibration[s][program]["rusage_sum_mb"]) for s in shapes], gates["femoco"])
        programs[program] = {
            "gates": gates, "seconds": round(seconds), "time_exponent": round(exponent, 2),
            "peak_mb": round(peak), "mb_per_1000_gates": round(1000 * peak_slope, 2),
            "rusage_sum_mb_upper_bound": round(upper)}
        if seconds > seconds_limit:
            reasons.append(f"{program}: {seconds / 60:.0f} min > {seconds_limit / 60:.0f} min")
        for label, value in (("sampled peak", peak), ("upper bound", upper)):
            if not within_effective_limit(value, cap_mb):
                reasons.append(f"{program}: {label} {value / 1024:.1f} GB > "
                               f"{memory_limit / 1024:.1f} GB")
    return {"method": PROJECTION_METHOD,
            "limits": {"seconds": seconds_limit, "mb": memory_limit, "cap_mb": cap_mb},
            "programs": programs, "femoco_allowed": not reasons,
            "reason": "; ".join(reasons) or "within the limits", "code_hash": current,
            "environment": environment()}


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


def write_results(shape_name: str, results: dict) -> dict:
    """Write k1_<shape>.json from the program records so far; the verdict covers all of them."""
    record = {p: results[p] for p in SHAPE_PROGRAMS[shape_name] if p in results}
    complete = len(record) == len(SHAPE_PROGRAMS[shape_name])
    failed = any(not program.get("within_budget") or program.get("warnings", {}).get("prototype")
                 for program in record.values())
    if complete and "controlled_walk" in record:
        record["controlled_walk_check"] = controlled_walk_check(record)
        failed |= not record["controlled_walk_check"]["ok"]
    seconds_limit, cap_mb = LIMITS[shape_name]
    record.update(limits={"seconds": seconds_limit, "mb": effective_limit_mb(cap_mb),
                          "cap_mb": cap_mb},
                  memory_method=MEMORY_METHOD, code_hash=code_hash(), environment=environment(),
                  verdict="fail" if failed else "pass" if complete else "incomplete")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"k1_{shape_name}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def drive(shape_name: str, allow_retry: bool = False) -> int:
    seconds_limit, cap_mb = LIMITS[shape_name]
    results, programs, env = {}, list(SHAPE_PROGRAMS[shape_name]), dict(os.environ)
    if shape_name == "femoco":
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
            command = [sys.executable, __file__, "--shape", shape_name, "--program", program]
            run = run_watched(command, cap_mb=cap_mb, timeout_s=seconds_limit + 60, env=env)
            lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
            if run.killed or run.survivors or run.returncode != 0 or not lines:
                reason = f"killed: {run.killed}" if run.killed else run.stderr[-3000:]
                results[program] = {"error": reason, **run.summary()}
            else:
                record = json.loads(lines[-1])
                record["peak_mb"] = round(run.peak_mb, 1)
                record["within_budget"] = (record["seconds"] <= seconds_limit
                                           and within_effective_limit(run.peak_mb, cap_mb))
                results[program] = record
            print(program, json.dumps(results[program])[:600], flush=True)
            if shape_name == "femoco":
                write_results(shape_name, results)
    finally:
        if shape_name == "femoco":
            FEMOCO_LOCK.unlink(missing_ok=True)
    record = write_results(shape_name, results)
    if "controlled_walk_check" in record:
        print("controlled_walk_check", json.dumps(record["controlled_walk_check"]), flush=True)
    print("verdict:", record["verdict"])
    return 0 if record["verdict"] == "pass" else 1


def write_projection() -> int:
    calibration = {}
    for shape in CALIBRATION_SHAPES:
        path = RESULTS / f"k1_{shape}.json"
        if path.exists():
            calibration[shape] = json.loads(path.read_text())
    projection = project(calibration)
    (RESULTS / "k1_projection.json").write_text(json.dumps(projection, indent=2) + "\n")
    for program, values in projection["programs"].items():
        print(program, json.dumps(values), flush=True)
    print("femoco_allowed:", projection["femoco_allowed"], "-", projection["reason"])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", choices=sorted(SHAPES))
    parser.add_argument("--program", choices=PROGRAMS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--allow-retry", action="store_true")
    args = parser.parse_args()
    if args.project:
        sys.exit(write_projection())
    if args.shape is None:
        parser.error("--shape is required unless --project is given")
    if args.all:
        sys.exit(drive(args.shape, allow_retry=args.allow_retry))
    if args.shape == "femoco" and not _driver_token_valid():
        sys.exit("refused: the FeMoco shape compiles only through --shape femoco --all, which runs "
                 "each program once under the watchdog")
    print(json.dumps(compile_one(args.shape, args.program)), flush=True)


if __name__ == "__main__":
    main()
