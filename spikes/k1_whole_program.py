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

--arm B|P|X|PX (Task 0.9d, on top of arm F, one compile per count): B is arm F as merged; P reads
every table through one load loop per unary leaf (ProtoTableReadP); X calls explicit classes
instead of the adjoint and controlled forms of the composites (CONVERSIONS); PX does both.
--read-check runs P's compiled-path read check (four lightning.qubit compiles). --all --arm
writes k1d_<arm>_<shape>.json for small, mid10, mid10k4 and mid14 with the capture inventory of
capture_instrument and the stop gate of d_problems; --project --arm B writes k1d_projection.json
for every arm (D_PROJECTION_METHOD). Task 0.9d compiles no FeMoco shape (d_refusal).

--shape femoco --all --arm B|P|X|PX [--program controlled_walk|walk] [--dry-run] (Task 0.9e) is
the one FeMoco confirmation path of the Task 0.9d arms (drive_femoco_d). It compiles exactly one
program in one watched child under Task 0.9's FeMoco cap (8 GB, effective 7,680 MB) and time limit
(LIMITS['femoco']), never retried, and writes k1d_<arm>_femoco_<program>.json with the prediction
of k1d_projection.json and its comparison with the measurement. It refuses, before compiling, an
arm whose femoco_allowed is not exactly true, a missing token in DFTHC_K1_FEMOCO_TOKEN, an existing
lock (Task 0.9's, created with mode x, holding the token, removed at the end) and an existing
result file. --dry-run runs every refusal check and prints the child command, the cap, the
timeout and the prediction, without compiling or creating the lock.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import resource
import secrets
import shlex
import subprocess
import sys
import threading
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
    WatchdogError,
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
    "mid10k4": Shape(N=10, R=8, B=13, C=5, n_plus=5, b_rot=15, b_k1=15, b_k2=15, k1=2, k2=4),
}
"""mid10k4 (Task 0.9d) keeps mid10's values except B = 13 and k2 = 4: one K2 = 16 block per inner
row (FeMoco has two), and B + 1 = 14 padded to 16 = 2**w_b, as erase_geometry requires."""


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


def _c_reflection_keys(zero_wires, one_wires):
    """Resources of the controlled reflection (the C(ProtoZeroReflection) rule and, in arms X and
    PX, ProtoControlledRefT2X and ProtoControlledRefABX)."""
    controls, values, _, last = _reflection_parts(zero_wires, one_wires)
    return {_mcz_key(len(controls) + 1, [1] + list(values)): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.Z: 1}


def _c_reflection_ops(zero_wires, one_wires, work, control):
    """The gates of the controlled reflection on one ``control`` wire."""
    controls, values, target, last = _reflection_parts(zero_wires, one_wires)
    if last == 0:
        qp.X(target)
    _mcz([control] + list(controls), [1] + list(values), target, work)
    if last == 0:
        qp.X(target)
    qp.Z(control)


def _c_zero_reflection_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return _c_reflection_keys(base.zero_wires, base.one_wires)


@qp.register_resources(_c_zero_reflection_resources)
def _proto_c_zero_reflection(base, control_wires, control_values, work_wires, work_wire_type):
    _c_reflection_ops(base.zero_wires, base.one_wires, base.work, control_wires[0])


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
            _remedy_read(remedy, static, wire_args)
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
        table = _remedy_read_key(remedy, static, (nh, nl, width, junk, len(lay.work)))
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
            key = _remedy_read_key(remedy, static, [len(w) for w in wire_args])
        return {key: 1, qp.X: 2 * (which == "sf")}

    def rule(shape, tables, wires):
        s, lay = Shape(*shape), _layout(shape)
        static, wire_args = _angle_read(s, lay, wires, tables, which)
        if which == "sf":
            qp.X(wires[lay.G0[0]])
        # The SF load is enabled on G0 = 0; the stand-in read ignores the enable (placeholder).
        if remedy:
            _remedy_read(remedy, static, wire_args)
        else:
            ProtoTableRead(*static, *wire_args)
        if which == "sf":
            qp.X(wires[lay.G0[0]])

    suffix = {False: "", True: "_r1"}.get(remedy, f"_{str(remedy).lower()}")
    rule.__name__ = f"_proto_{which}_angle_load{suffix}"
    register_rule(cls, qp.register_resources(resources)(rule))


_make_angle_load(ProtoTailAngleLoad, "tail")
_make_angle_load(ProtoSFAngleLoad, "sf")


class ProtoInnerSelect(ProtoWalkPart):
    """Stand-in InnerSelect (spec 9.18); the pinned adjoint uses the adjoint Majorana gate."""


def _angle_loads(remedy):
    if _variant(remedy) in PER_LEAF_ARMS:
        return ProtoTailAngleLoadP, ProtoSFAngleLoadP
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
    _adjoint(remedy, ladder_cls, *ladder)
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
    ladder_dagger = _adjoint_key(remedy, ladder_cls, *ladder_args)
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
                    ProtoBlockEncodingFPG, ProtoWalkFPG),
            "B": (ProtoOuterPrepareF, ProtoInnerPrepareF, ProtoInnerSelectF, ProtoBlockEncodingF,
                  ProtoWalkF),
            "P": (ProtoOuterPrepareP, ProtoInnerPrepareP, ProtoInnerSelectP, ProtoBlockEncodingP,
                  ProtoWalkP),
            "X": (ProtoOuterPrepareF, ProtoInnerPrepareF, ProtoInnerSelectX, ProtoBlockEncodingX,
                  ProtoWalkX),
            "PX": (ProtoOuterPrepareP, ProtoInnerPrepareP, ProtoInnerSelectPX,
                   ProtoBlockEncodingPX, ProtoWalkPX)}[_variant(remedy)]


def _table_erase_cls(remedy):
    return {None: ProtoTableErase, "R1": ProtoTableErase, "F": ProtoTableEraseF,
            "FP": ProtoTableEraseFP, "FPG": ProtoTableEraseFP, "B": ProtoTableEraseF,
            "P": ProtoTableEraseF, "X": ProtoTableEraseF, "PX": ProtoTableEraseF}[_variant(remedy)]


def _angle_erase_cls(remedy):
    return {None: ProtoAngleErase, "R1": ProtoAngleErase, "F": ProtoAngleEraseF,
            "FP": ProtoAngleEraseFP, "FPG": ProtoAngleEraseFP, "B": ProtoAngleEraseF,
            "P": ProtoAngleEraseF, "X": ProtoAngleEraseF, "PX": ProtoAngleEraseF}[_variant(remedy)]


def _adjoint(remedy, cls, *args):
    """Emit ``qp.adjoint(cls(*args))``; arms X and PX call the explicit class of
    EXPLICIT_ADJOINT instead (KeyError for an unconverted class, so no old key slips through)."""
    if _variant(remedy) in EXPLICIT_ARMS:
        return EXPLICIT_ADJOINT[cls](*args)
    return qp.adjoint(cls(*args))


def _adjoint_key(remedy, cls, *args):
    """Resource key of :func:`_adjoint`."""
    if _variant(remedy) in EXPLICIT_ARMS:
        return _abstract(EXPLICIT_ADJOINT[cls], *args)
    return resource_key(lambda: qp.adjoint(cls(*args)))


def _controlled_reflection(remedy, which, args, control):
    """Emit ``qp.ctrl(ProtoZeroReflection(*args), control)``; arms X and PX call the explicit class
    of the reflection ``which`` ("t2" or "ab") instead."""
    if _variant(remedy) in EXPLICIT_ARMS:
        return EXPLICIT_CLASSES[_variant(remedy)][f"ref_{which}"](*args, [control])
    return qp.ctrl(ProtoZeroReflection(*args), control=control)


def _controlled_reflection_key(remedy, which, args):
    """Resource key of :func:`_controlled_reflection` (``args`` are abstract wire registers)."""
    if _variant(remedy) in EXPLICIT_ARMS:
        return _abstract(EXPLICIT_CLASSES[_variant(remedy)][f"ref_{which}"], *args, Wire[1])
    return resource_key(lambda: qp.ctrl(ProtoZeroReflection(*args), control=Wire[1]))


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
    _adjoint(remedy, inner_prepare, *part)
    if control is None:
        ProtoZeroReflection(*t2)
    else:
        _controlled_reflection(remedy, "t2", t2, control)
    inner_prepare(*part)
    _adjoint(remedy, inner_select, *part)
    ProtoInnerSigns(*_signs(shape, wires))
    _adjoint(remedy, inner_prepare, *part)
    _adjoint(remedy, outer_prepare, *part)


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
        keys[_adjoint_key(remedy, cls, shape, tables, n)] = 2 if cls is parts[1] else 1
    keys[_abstract(ProtoInnerSigns, Wire[1], Wire[1], Wire[1])] = 2
    t2_args = (Wire[s.w_b + s.b_k2 + 1], Wire[1], Wire[len(lay.work)])
    ref = (_controlled_reflection_key(remedy, "t2", t2_args)
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
    if _variant(remedy) in EXPLICIT_ARMS:
        return {_abstract(EXPLICIT_CLASSES[_variant(remedy)]["controlled_block_encoding"], shape,
                          tables, n, Wire[1]): 1,
                _controlled_reflection_key(remedy, "ab", _ab_args(shape)): 1}
    return {resource_key(lambda: qp.ctrl(block_encoding(shape, tables, n), control=Wire[1])): 1,
            resource_key(lambda: qp.ctrl(ProtoZeroReflection(*_ab_args(shape)),
                                         control=Wire[1])): 1}


def _walk_ops(shape, tables, wires, control, remedy):
    block_encoding = _parts(remedy)[3]
    _, ab = _refs(shape, wires)
    if control is None:
        block_encoding(shape, tables, wires)
        ProtoZeroReflection(*ab)
    elif _variant(remedy) in EXPLICIT_ARMS:
        EXPLICIT_CLASSES[_variant(remedy)]["controlled_block_encoding"](shape, tables, wires,
                                                                         [control])
        _controlled_reflection(remedy, "ab", ab, control)
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


def _remedy_read(remedy, static, wire_args):
    """The read of a remedy variant: ProtoTableReadP in arms P and PX, else ProtoTableReadR1."""
    if _variant(remedy) in PER_LEAF_ARMS:
        return _p_read(static, wire_args)
    return _r1_read(static, wire_args)


def _remedy_read_key(remedy, static, wire_sizes):
    if _variant(remedy) in PER_LEAF_ARMS:
        return _p_read_key(static, wire_sizes)
    return _r1_read_key(static, wire_sizes)


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
ERASE_OBS = {"small": (1, 2), "mid10": (3, 4), "mid14": (3, 5), "femoco": (4, 7),
             "mid10k4": (3, 5)}
"""(outer, inner) table-erase one-hot bits of the rule of erase_geometry (brief, requirement 1;
mid10k4: Task 0.9d requirement 5)."""
SF_OBS = {"small": 2, "mid10": 3, "mid14": 4, "femoco": 5, "mid10k4": 4}
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


# --------------------------------------------------------------------------- Task 0.9d arms
# All on top of arm F (the spec's one-hot erase fixup, unrolled Givens ladders, partial remedy 1):
# B is arm F as merged; P reads every table through one load loop per unary leaf
# (ProtoTableReadP); X calls explicit classes instead of the Adjoint/C forms of the composites
# (EXPLICIT_ADJOINT, EXPLICIT_CLASSES), so Catalyst never synthesizes their distributed
# candidates; PX does both.

D_ARMS = ("B", "P", "X", "PX")
PER_LEAF_ARMS = ("P", "PX")
EXPLICIT_ARMS = ("X", "PX")


def p_num_leaves(high_ranges) -> int:
    return math.prod(hi - lo for _, lo, hi in high_ranges)


def p_leaf_prefixes(high_ranges) -> list[int]:
    """The high prefix of every unary leaf, in the visitation order of nested_iterate (which is
    the lexicographic order of the register values); raises RuntimeError if they differ."""
    widths = [w for w, _, _ in high_ranges]
    dummy = list(range(10_000))
    prefixes = []

    def leaf(flag, values):
        prefix = 0
        for value, w in zip(values, widths, strict=True):
            prefix = (prefix << w) | value
        prefixes.append(prefix)

    nested_iterate(_Count(), _split_high(dummy[: sum(widths)], high_ranges), dummy[:64], leaf)
    lexicographic = []
    for values in itertools.product(*(range(lo, hi) for _, lo, hi in high_ranges)):
        prefix = 0
        for value, w in zip(values, widths, strict=True):
            prefix = (prefix << w) | value
        lexicographic.append(prefix)
    if prefixes != lexicographic:
        raise RuntimeError("nested_iterate does not visit the leaves in lexicographic order")
    return prefixes


def p_leaf_words(table, high_ranges, block_bits, width) -> np.ndarray:
    """The dynamic ``leaf_words`` of ProtoTableReadP for a compilable ``table``.

    Shape ``(num_leaves, K, ceil(width / 32))``, WORD_DTYPE, 32 data bits per word (L-26). Row
    ``ell`` is the ``ell``-th unary leaf of nested_iterate; position ``p`` holds the words of the
    entry at ``(prefix << block_bits) | p``; padding and dead positions are all-zero. Raises
    ``ValueError`` for a malformed entry, an entry the unary iteration never reaches, a wrong
    shape or a nonzero padding position.
    """
    entries, block = dict(table), 2**block_bits
    num_words = -(-width // WORD)
    prefixes = p_leaf_prefixes(high_ranges)
    array = np.zeros((len(prefixes), block, num_words), dtype=WORD_DTYPE)
    live = np.zeros((len(prefixes), block), dtype=bool)
    for leaf, prefix in enumerate(prefixes):
        for position in range(block):
            words = entries.pop((prefix << block_bits) | position, None)
            if words is None:
                continue
            if len(words) != num_words or any(isinstance(w, bool) or not 0 <= w < 2**WORD
                                              for w in words):
                raise ValueError(f"entry words {words!r} are not {num_words} words of {WORD} bits")
            array[leaf, position] = words
            live[leaf, position] = True
    if entries:
        raise ValueError(f"table addresses outside the unary iteration: {sorted(entries)[:8]}")
    if array.shape != (p_num_leaves(high_ranges), block, num_words):
        raise ValueError(f"leaf_words shape {array.shape} is not "
                         f"{(p_num_leaves(high_ranges), block, num_words)}")
    if np.any(array[~live]):
        raise ValueError("a padding or dead position of leaf_words holds a nonzero word")
    return array


class ProtoTableReadP(Operator2):
    """ProtoTableReadR1 with one load loop per unary leaf (K-1 arm P, Task 0.9d; prototype).

    ``leaf_words`` (dynamic, WORD_DTYPE) is :func:`p_leaf_words` of the table. The unary iteration
    over ``high_ranges`` and the swap network stay unrolled in Python, so T, M, A and the CSWAPs
    equal ProtoTableReadR1's. At unary leaf ``ell`` one ``qp.for_loop(0, K * width, 1)`` runs:
    iteration ``t`` sets ``p = t // width`` and ``i = t % width``, reads bit
    ``(leaf_words[ell, p, i // 32] >> (i % 32)) & 1`` and applies ``qp.cond(bit, qp.CNOT)`` from
    the leaf flag to ``(target ‖ junk)[p * width + i]``: position-major, bit-minor, as
    ProtoTableReadR1. The loop index addresses only JAX arrays. The resource function declares
    ``num_leaves * K * width`` CNOTs, an upper bound (rule registered with ``exact=False``).
    """

    dynamic_argnames = ("leaf_words",)
    compilable_argnames = ("high_ranges", "block_bits", "width")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, leaf_words, high_ranges, block_bits, width, high, low, target, junk, work):
        expected = (p_num_leaves(high_ranges), 2**block_bits, -(-width // WORD))
        shape = tuple(getattr(leaf_words, "shape", np.shape(leaf_words)))
        if shape != expected:
            raise ValueError(f"leaf_words shape {shape} is not {expected}")
        super().__init__(_word_array(leaf_words), high_ranges, block_bits, width, high, low,
                         target, junk, work)


def _register_array(target, junk):
    """``target ‖ junk`` as one array a traced loop index can address."""
    if qp.capture.enabled():
        if len(junk) == 0:
            return jnp.asarray(target)
        return jnp.concatenate([jnp.asarray(target), jnp.asarray(junk)])
    return list(target) + list(junk)


def _p_read_body(emit, high_ranges, block_bits, width, high, low, target, junk, work, load,
                 cswap):
    """ProtoTableReadP's gates, with ``load(flag, leaf)`` once per unary leaf."""
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    leaves = itertools.count()
    nested_iterate(emit, _split_high(high, high_ranges), work,
                   lambda flag, values: load(flag, next(leaves)))
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _p_leaf_loader(leaf_words, block_bits, width, target, junk):
    register_array = _register_array(target, junk)

    def load(flag, leaf):
        @qp.for_loop(0, 2**block_bits * width, 1)
        def bit_loop(t):
            p, i = t // width, t % width
            bit = (leaf_words[leaf, p, i // WORD] >> (i % WORD)) & 1
            qp.cond(bit == 1, qp.CNOT)([flag, register_array[p * width + i]])

        bit_loop()

    return load


def _p_read_counts(high_ranges, block_bits, width, count_loads=True):
    """Gate counts of ProtoTableReadP; "CNOT" is an upper bound (every position and bit set).
    Without ``count_loads`` the leaf loops' bodies are left out."""
    counter = _Count()
    dummy = list(range(10_000))

    def load(flag, leaf):
        counter.counts["CNOT"] += 2**block_bits * width if count_loads else 0

    _p_read_body(counter, high_ranges, block_bits, width,
                 dummy[: sum(w for w, _, _ in high_ranges)], dummy[:block_bits], dummy[:width],
                 dummy[: (2**block_bits - 1) * width], dummy[:64], load,
                 lambda c, a, b: counter.counts.update({"CSWAP": 1}))
    return counter.counts


def _p_read_resources(leaf_words, high_ranges, block_bits, width, high, low, target, junk, work):
    return counts_to_resources(_p_read_counts(high_ranges, block_bits, width))


@qp.register_resources(_p_read_resources, exact=False)
def _proto_table_read_p(leaf_words, high_ranges, block_bits, width, high, low, target, junk,
                        work):
    _p_read_body(_Emit(), high_ranges, block_bits, width, high, low, target, junk, work,
                 _p_leaf_loader(leaf_words, block_bits, width, target, junk),
                 lambda c, a, b: qp.CSWAP([c, a, b]))


register_rule(ProtoTableReadP, _proto_table_read_p)


def _p_read(static, wire_args):
    """Emit ProtoTableReadP for the read arguments ``static`` = (table, high ranges, block bits,
    width); the composite builds ``leaf_words`` from its compilable table (and checks it)."""
    table, high_ranges, block_bits, width = static
    ProtoTableReadP(p_leaf_words(table, high_ranges, block_bits, width), high_ranges, block_bits,
                    width, *wire_args)


def _p_read_key(static, wire_sizes):
    _, high_ranges, block_bits, width = static
    leaf_words = AbstractArray((p_num_leaves(high_ranges), 2**block_bits, -(-width // WORD)),
                               WORD_DTYPE)
    return _abstract(ProtoTableReadP, leaf_words, high_ranges, block_bits, width,
                     *(Wire[n] for n in wire_sizes))


READ_IMPLEMENTATION = (p_num_leaves, p_leaf_prefixes, p_leaf_words, ProtoTableReadP,
                       _register_array, _p_read_body, _p_leaf_loader, _p_read_counts,
                       _p_read_resources, _proto_table_read_p, _p_read, _p_read_key)
"""The per-leaf read layout of arms P and PX; read_implementation_hash() hashes their source."""


def read_implementation_hash() -> str:
    import inspect

    digest = hashlib.sha256()
    for obj in READ_IMPLEMENTATION:
        digest.update(inspect.getsource(getattr(obj, "__wrapped__", obj)).encode())
    return digest.hexdigest()


class ProtoTailAngleLoadP(ProtoWalkPart):
    """ProtoTailAngleLoad through ProtoTableReadP (arms P and PX)."""


class ProtoSFAngleLoadP(ProtoWalkPart):
    """ProtoSFAngleLoad through ProtoTableReadP (arms P and PX)."""


_make_angle_load(ProtoTailAngleLoadP, "tail", remedy="P")
_make_angle_load(ProtoSFAngleLoadP, "sf", remedy="P")
ProtoOuterPrepareP = _make_prepare("ProtoOuterPrepareP", "outer", remedy="P")
ProtoInnerPrepareP = _make_prepare("ProtoInnerPrepareP", "inner", remedy="P")
ProtoInnerSelectP = _make_select("ProtoInnerSelectP", "P")
ProtoBlockEncodingP = _make_block_encoding("ProtoBlockEncodingP", "P")
ProtoWalkP = _make_walk("ProtoWalkP", "P")


class ProtoControlledPart(Operator2):
    """Base for the explicit controlled composites of arms X and PX: the layout and one control."""

    compilable_argnames = ("shape", "tables")
    wire_argnames = ("wires", "control")

    def __init__(self, shape, tables, wires, control):
        super().__init__(shape, tables, wires, control)


class ProtoControlledReflectionPart(Operator2):
    """Base for the explicit controlled reflections of arms X and PX."""

    wire_argnames = ("zero_wires", "one_wires", "work", "control")

    def __init__(self, zero_wires, one_wires, work, control):
        super().__init__(zero_wires, one_wires, work, control)


def _make_unprepare(name, which):
    """The explicit class of Adjoint(Proto<which>Prepare{F,P}): the registered unprepare rule's
    body (no read, so arms X and PX share it)."""
    cls = type(name, (ProtoWalkPart,), {"__doc__": f"Explicit {which} unprepare (K-1 arms X, PX): "
                                                   f"the body of Adjoint(Proto{which.title()}"
                                                   "PrepareF)'s registered rule."})

    def resources(shape, tables, wires):
        return _prepare_resources(Shape(*shape), _layout(shape), which, tables, True, "X")

    def rule(shape, tables, wires):
        _prepare_body(Shape(*shape), _layout(shape), wires, which, tables, True, "X")

    rule.__name__ = f"_{name.lower()}_rule"
    register_rule(cls, qp.register_resources(resources)(rule))
    return cls


def _make_explicit_select(name, dagger_name, arm):
    """The forward select of arm X or PX (it calls ProtoGivensLadderDaggerX) and its explicit
    dagger, the body of the registered Adjoint(ProtoInnerSelect...) rule."""
    forward = type(name, (ProtoWalkPart,), {"__doc__": f"ProtoInnerSelect of arm {arm}."})
    dagger = type(dagger_name, (ProtoWalkPart,),
                  {"__doc__": f"Explicit inner-select dagger of arm {arm}."})

    def resources(shape, tables, wires):
        return _select_resources(shape, tables, False, remedy=arm)

    def rule(shape, tables, wires):
        _select_ops(shape, tables, wires, False, remedy=arm)

    def dagger_resources(shape, tables, wires):
        return _select_resources(shape, tables, True, remedy=arm)

    def dagger_rule(shape, tables, wires):
        _select_ops(shape, tables, wires, True, remedy=arm)

    rule.__name__, dagger_rule.__name__ = f"_{name.lower()}_rule", f"_{dagger_name.lower()}_rule"
    register_rule(forward, qp.register_resources(resources)(rule))
    register_rule(dagger, qp.register_resources(dagger_resources)(dagger_rule))
    return forward, dagger


class ProtoGivensLadderDaggerX(Operator2):
    """Explicit adjoint of ProtoGivensLadder (K-1 arms X and PX): rungs k = N - 2 - t for
    t = 0..N-2, each ``qp.adjoint(ProtoGivensRotation(...k...))``, unrolled. Arm F registers no
    ladder-adjoint rule, so this is specified directly; it keeps the distributed
    Adjoint(ProtoGivensRotation) leaf target."""

    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        super().__init__(b_rot, system, angles, phase_grad, work)


@qp.register_resources(lambda b_rot, system, angles, phase_grad, work: {
    _rung_key(b_rot, True): len(system) - 1})
def _proto_ladder_dagger_x(b_rot, system, angles, phase_grad, work):
    last = len(system) - 2
    for t in range(last + 1):
        k = last - t
        qp.adjoint(ProtoGivensRotation([system[k]], [system[k + 1]],
                                       angles[k * b_rot:(k + 1) * b_rot], phase_grad,
                                       work[: b_rot - 1]))


register_rule(ProtoGivensLadderDaggerX, _proto_ladder_dagger_x)


def _make_controlled_reflection(name, which):
    cls = type(name, (ProtoControlledReflectionPart,),
               {"__doc__": f"Explicit C(ProtoZeroReflection) at {which.upper()} (K-1 arms X, PX): "
                           "the body of the registered C(ProtoZeroReflection) rule."})

    def resources(zero_wires, one_wires, work, control):
        return _c_reflection_keys(zero_wires, one_wires)

    def rule(zero_wires, one_wires, work, control):
        _c_reflection_ops(zero_wires, one_wires, work, control[0])

    rule.__name__ = f"_{name.lower()}_rule"
    register_rule(cls, qp.register_resources(resources)(rule))
    return cls


def _make_explicit_composite(name, arm, kind, controlled):
    """Arm X's or PX's block encoding or walk, forward (a ProtoWalkPart) or controlled (the
    explicit class of C(...), a ProtoControlledPart with the registered C rule's body)."""
    base = ProtoControlledPart if controlled else ProtoWalkPart
    cls = type(name, (base,), {"__doc__": f"{'Explicit controlled ' if controlled else ''}"
                                          f"{kind} of arm {arm} (Task 0.9d)."})

    def keys(shape, tables, with_control):
        if kind == "block_encoding":
            return _be_resources(shape, tables, with_control, remedy=arm)
        return _walk_keys(shape, tables, with_control, arm)

    def ops(shape, tables, wires, control):
        if kind == "block_encoding":
            _be_ops(shape, tables, wires, control=control, remedy=arm)
        else:
            _walk_ops(shape, tables, wires, control, arm)

    if controlled:
        def resources(shape, tables, wires, control):
            return keys(shape, tables, True)

        def rule(shape, tables, wires, control):
            ops(shape, tables, wires, control[0])
    else:
        def resources(shape, tables, wires):
            return keys(shape, tables, False)

        def rule(shape, tables, wires):
            ops(shape, tables, wires, None)

    rule.__name__ = f"_{name.lower()}_rule"
    register_rule(cls, qp.register_resources(resources)(rule))
    return cls


ProtoOuterUnprepareX = _make_unprepare("ProtoOuterUnprepareX", "outer")
ProtoInnerUnprepareX = _make_unprepare("ProtoInnerUnprepareX", "inner")
ProtoInnerSelectX, ProtoInnerSelectDaggerX = _make_explicit_select(
    "ProtoInnerSelectX", "ProtoInnerSelectDaggerX", "X")
ProtoControlledRefT2X = _make_controlled_reflection("ProtoControlledRefT2X", "t2")
ProtoControlledRefABX = _make_controlled_reflection("ProtoControlledRefABX", "ab")
ProtoBlockEncodingX = _make_explicit_composite("ProtoBlockEncodingX", "X", "block_encoding", False)
ProtoControlledBlockEncodingX = _make_explicit_composite(
    "ProtoControlledBlockEncodingX", "X", "block_encoding", True)
ProtoWalkX = _make_explicit_composite("ProtoWalkX", "X", "walk", False)
ProtoControlledWalkX = _make_explicit_composite("ProtoControlledWalkX", "X", "walk", True)
ProtoInnerSelectPX, ProtoInnerSelectDaggerPX = _make_explicit_select(
    "ProtoInnerSelectPX", "ProtoInnerSelectDaggerPX", "PX")
ProtoBlockEncodingPX = _make_explicit_composite("ProtoBlockEncodingPX", "PX", "block_encoding",
                                                False)
ProtoControlledBlockEncodingPX = _make_explicit_composite(
    "ProtoControlledBlockEncodingPX", "PX", "block_encoding", True)
ProtoWalkPX = _make_explicit_composite("ProtoWalkPX", "PX", "walk", False)
ProtoControlledWalkPX = _make_explicit_composite("ProtoControlledWalkPX", "PX", "walk", True)

EXPLICIT_ADJOINT = {
    ProtoOuterPrepareF: ProtoOuterUnprepareX, ProtoOuterPrepareP: ProtoOuterUnprepareX,
    ProtoInnerPrepareF: ProtoInnerUnprepareX, ProtoInnerPrepareP: ProtoInnerUnprepareX,
    ProtoInnerSelectX: ProtoInnerSelectDaggerX, ProtoInnerSelectPX: ProtoInnerSelectDaggerPX,
    ProtoGivensLadder: ProtoGivensLadderDaggerX,
}
"""Arms X and PX: the explicit class called wherever arm F calls qp.adjoint(<key>(...))."""
EXPLICIT_CLASSES = {
    "X": {"ref_t2": ProtoControlledRefT2X, "ref_ab": ProtoControlledRefABX,
          "controlled_block_encoding": ProtoControlledBlockEncodingX,
          "controlled_walk": ProtoControlledWalkX},
    "PX": {"ref_t2": ProtoControlledRefT2X, "ref_ab": ProtoControlledRefABX,
           "controlled_block_encoding": ProtoControlledBlockEncodingPX,
           "controlled_walk": ProtoControlledWalkPX},
}
"""Arms X and PX: the explicit class called wherever arm F calls qp.ctrl(<key>(...))."""
CONVERSIONS = {
    "X": (("Adjoint(ProtoOuterPrepareF)", "ProtoOuterUnprepareX"),
          ("Adjoint(ProtoInnerPrepareF)", "ProtoInnerUnprepareX"),
          ("Adjoint(ProtoInnerSelectF)", "ProtoInnerSelectDaggerX"),
          ("Adjoint(ProtoGivensLadder)", "ProtoGivensLadderDaggerX"),
          ("C(ProtoZeroReflection) at T2", "ProtoControlledRefT2X"),
          ("C(ProtoZeroReflection) at AB", "ProtoControlledRefABX"),
          ("C(ProtoBlockEncodingF)", "ProtoControlledBlockEncodingX"),
          ("C(ProtoWalkF)", "ProtoControlledWalkX")),
    "PX": (("Adjoint(ProtoOuterPrepareP)", "ProtoOuterUnprepareX"),
           ("Adjoint(ProtoInnerPrepareP)", "ProtoInnerUnprepareX"),
           ("Adjoint(ProtoInnerSelectP)", "ProtoInnerSelectDaggerPX"),
           ("Adjoint(ProtoGivensLadder)", "ProtoGivensLadderDaggerX"),
           ("C(ProtoZeroReflection) at T2", "ProtoControlledRefT2X"),
           ("C(ProtoZeroReflection) at AB", "ProtoControlledRefABX"),
           ("C(ProtoBlockEncodingP)", "ProtoControlledBlockEncodingPX"),
           ("C(ProtoWalkP)", "ProtoControlledWalkPX")),
}
"""The closed conversion table of Task 0.9d requirement 1: (old modified key, explicit class)."""
FORWARD_ROLE_CLASSES = {
    "X": {"ProtoInnerSelectF": "ProtoInnerSelectX", "ProtoBlockEncodingF": "ProtoBlockEncodingX",
          "ProtoWalkF": "ProtoWalkX"},
    "PX": {"ProtoInnerSelectP": "ProtoInnerSelectPX", "ProtoBlockEncodingP": "ProtoBlockEncodingPX",
           "ProtoWalkP": "ProtoWalkPX"},
}
"""Forward classes whose bodies call an explicit class, so arms X and PX need a class of the same
role (the forward prepares are arm F's, or arm P's, unchanged)."""
RETAINED_PROTO_KEYS = ("Adjoint(ProtoMajorana)", "Adjoint(ProtoUniformPrep)",
                       "Adjoint(ProtoGivensRotation)")
"""Modified prototype keys arms X and PX keep: small or leaf modified operations legitimately used
by the explicit rules (Adjoint(ProtoGivensRotation) by ProtoGivensLadderDaggerX)."""
EXPECTED_RETAINED = {"walk": set(RETAINED_PROTO_KEYS), "controlled_walk": set(RETAINED_PROTO_KEYS),
                     "inner_prepare": {"Adjoint(ProtoUniformPrep)"},
                     "inner_select": {"Adjoint(ProtoGivensRotation)"}}
"""The retained modified prototype keys each program must show in arms X and PX."""


def explicit_class_names(arm: str) -> frozenset[str]:
    return frozenset(new for _, new in CONVERSIONS.get(arm, ()))


def expected_explicit(arm: str, program: str) -> set[str]:
    """The explicit classes a program of arm X or PX must capture."""
    new = [cls for _, cls in CONVERSIONS[arm]]
    walk = set(new[:4])  # the two unprepares, the select dagger and the ladder dagger
    return {"walk": walk, "controlled_walk": set(new), "inner_prepare": {new[1]},
            "inner_select": {new[3]}}[program]


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
    "mid10k4": ("walk", "controlled_walk"),
}
CALIBRATION_SHAPES = ("small", "mid10", "mid14")
LIMITS = {
    "small": (300, HEAVY_MEMORY_MB),
    "mid10": (900, HEAVY_MEMORY_MB),
    "mid10k4": (900, HEAVY_MEMORY_MB),
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
        elif program == "controlled_walk" and _variant(remedy) in EXPLICIT_ARMS:
            EXPLICIT_CLASSES[_variant(remedy)]["controlled_walk"](shape, tables, wires, [control])
        elif program == "controlled_walk":
            qp.ctrl(walk(shape, tables, wires), control=control)
        elif program == "inner_prepare":
            inner_prepare(shape, tables, wires)
            _adjoint(remedy, inner_prepare, shape, tables, wires)
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


def _read_p_structure(op):
    a = op.arguments
    counts = _p_read_counts(a["high_ranges"], a["block_bits"], a["width"], count_loads=False)
    return counts_to_resources(counts), Counter({"read_leaf": p_num_leaves(a["high_ranges"])}), 0


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
    "ProtoTableReadP": _read_p_structure,
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
    classes = (*LOOP_CLASSES, "read_leaf") if _variant(remedy) in D_ARMS else LOOP_CLASSES
    return {"static_size": sum(gates.values()), "static_gates": dict(sorted(gates.items())),
            "projection_loop_sites": {kind: loops[kind] for kind in classes},
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
    "read_leaf": (f"{ProtoTableReadP.__name__}{{",),
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


def captured_loops(log: list[dict], classes=LOOP_CLASSES) -> dict:
    """Per loop class of ``classes``, the sorted nonzero for_loop counts of its distinct captured
    targets (one per full target signature, the maximum over repeated captures, as in Task
    0.9b)."""
    per_class = {kind: {} for kind in classes}
    for e in log:
        if e.get("loop_class") in per_class:
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
    if _variant(remedy) in D_ARMS:
        return compile_one_d(shape_name, program, _variant(remedy))
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
    if variant in D_ARMS:
        return RESULTS / ("k1d_projection.json" if name == "projection"
                          else f"k1d_{variant}_{name}.json")
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


# --------------------------------------------------------------------------- Task 0.9d driver
# Arms B, P, X, PX on the one-compile pipeline of Task 0.2c (compiled_resources compiles once,
# without the AOT binary). Every compile records the capture inventory (capture_instrument): one
# record per DecompTargetSpec of the single analyze capture, registered and distributed specs
# kept apart, with nested JAXPR equations by target class and modifier bucket.

D_SHAPES = ("small", "mid10", "mid10k4", "mid14")
D_CALIBRATION_SHAPES = ("small", "mid10", "mid10k4")
D_SHAPE_PROGRAMS = {"small": PROGRAMS, "mid10": ("walk", "controlled_walk"),
                    "mid10k4": ("walk", "controlled_walk"),
                    "mid14": ("walk", "inner_prepare", "inner_select", "controlled_walk")}
D_LOOP_CLASSES = (*LOOP_CLASSES, "read_leaf")
D_CAP_MB = 4096
D_BUDGET_SECONDS = 2400
D_PLANNED_COMPILES = 48 + 4
D_LEDGER_VARIABLE = "DFTHC_K1D_LEDGER"
"""JSON-lines ledger of every watched child of Task 0.9d (seconds, compiles, running totals)."""
READ_ROLES = ("outer", "inner", "tail", "sf")
PROGRAM_READ_ROLES = {"walk": READ_ROLES, "controlled_walk": READ_ROLES,
                      "inner_prepare": ("inner",), "inner_select": ("tail", "sf")}
READ_CLASS_NAMES = frozenset({"ProtoTableRead", "ProtoTableReadR1", "ProtoTableReadP"})
SAMPLE_SECONDS = 0.1
D_ARM_PARENT = {"B": None, "P": "B", "X": "B", "PX": "B"}
"""P, X and PX compile a shape only after B passed it (their deltas are against B)."""


def _target_class_of(name: str) -> str:
    if name in READ_CLASS_NAMES:
        return "read"
    if re.fullmatch(r"Proto(Outer|Inner)(Prepare|Unprepare)\w*|ProtoInnerSelect\w*"
                    r"|Proto(Tail|SF)AngleLoad\w*|Proto(Controlled)?(BlockEncoding|Walk)(?!Part)\w*",
                    name):
        return "composite"
    if re.fullmatch(r"Proto(Table|Angle)Erase\w*", name):
        return "erase"
    if re.fullmatch(r"ProtoGivens(Ladder\w*|Rotation)", name):
        return "ladder"
    return "other"


TARGET_CLASSES = dict(sorted((name, _target_class_of(name)) for name in PROTOTYPE_NAMES))
"""The frozen name table of requirement 4: every prototype class name to read, composite, erase,
ladder or other (ProtoGivensRotation counts as ladder); any non-prototype name is pennylane."""
EXPLICIT_CLASS_NAMES = explicit_class_names("X") | explicit_class_names("PX")
MODIFIER_BUCKETS = ("plain", "registered_modified", "distributed_adjoint", "distributed_control",
                    "distributed_control_of_adjoint", "explicit_class")
EQUATION_CLASSES = ("read_load", "read_skeleton", "composite", "erase", "ladder", "pennylane",
                    "other")
_WRAPPER = re.compile(r"^((\d*C|Adjoint)\()+")


def core_name(name: str) -> str:
    """``C(Adjoint(ProtoInnerPrepareF))`` gives ``ProtoInnerPrepareF``."""
    return _WRAPPER.sub("", name).rstrip(")")


def is_modified_name(name: str) -> bool:
    return bool(_WRAPPER.match(name))


def target_class(op_name: str) -> str:
    return TARGET_CLASSES.get(op_name, "other" if op_name.startswith("Proto") else "pennylane")


def _bucket(origin: str, adjoint: bool, controls: int, op_name: str) -> str:
    if origin == "registered":
        return "registered_modified"
    if not adjoint and not controls:
        return "explicit_class" if op_name in EXPLICIT_CLASS_NAMES else "plain"
    if adjoint and controls:
        return "distributed_control_of_adjoint"
    return "distributed_adjoint" if adjoint else "distributed_control"


def _normalized(value):
    if isinstance(value, (tuple, list)):
        return tuple(_normalized(v) for v in value)
    return int(value) if isinstance(value, (int, np.integer)) else value


def read_static(shape_name: str, role: str) -> tuple:
    """(table, high_ranges, block_bits, width) of the read ``role`` at a shape."""
    s, lay = SHAPES[shape_name], Layout(SHAPES[shape_name])
    tables, wires = random_tables(s), list(range(lay.num_wires))
    if role in ("outer", "inner"):
        return _read_args(s, lay, wires, role, tables)[0]
    return _angle_read(s, lay, wires, tables, role)[0]


@functools.cache
def read_geometry(shape_name: str) -> dict:
    """Per read role: loaded entries, unary leaves, K, width and CSWAPs (no compile)."""
    out = {}
    for role in READ_ROLES:
        table, high_ranges, block_bits, width = read_static(shape_name, role)
        out[role] = {"entries": len(table), "leaves": p_num_leaves(high_ranges),
                     "K": 2**block_bits, "width": width, "cswaps": (2**block_bits - 1) * width,
                     "high_ranges": _normalized(high_ranges), "block_bits": block_bits}
    return out


def _read_spec_info(op_name, static, roles):
    if op_name not in READ_CLASS_NAMES or static is None:
        return None
    key = (_normalized(static["high_ranges"]), int(static["block_bits"]), int(static["width"]))
    leaves = p_num_leaves(key[0])
    loads = len(static["addresses"]) if "addresses" in static else leaves
    return {"role": roles.get(key), "leaves": leaves, "cswaps": (2**key[1] - 1) * key[2],
            "loads_expected": loads}


def equation_counts(jaxpr) -> tuple[int, int, int, int, int]:
    """(equations, for_loops, conds, load equations, top-level loops) of a jaxpr, each equation
    counted once, recursing into loop bodies, cond branches and modifier regions. Load equations
    are every for_loop equation plus its nested equations."""
    j = getattr(jaxpr, "jaxpr", jaxpr)
    eqns = loops = conds = loads = top_loops = 0
    for e in j.eqns:
        sub, seen = [0, 0, 0, 0], set()
        for value in e.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if (hasattr(item, "eqns") or hasattr(item, "jaxpr")) and id(item) not in seen:
                    seen.add(id(item))
                    for k, v in enumerate(equation_counts(item)[:4]):
                        sub[k] += v
        is_loop = e.primitive.name == "for_loop"
        eqns += 1 + sub[0]
        loops += is_loop + sub[1]
        conds += (e.primitive.name == "cond") + sub[2]
        loads += (1 + sub[0]) if is_loop else sub[3]
        top_loops += is_loop
    return eqns, loops, conds, loads, top_loops


def _spec_record(target_spec, traced, seconds) -> dict:
    target = str(target_spec.target_id)
    name = target.split("{", 1)[0]
    op_name = getattr(target_spec, "k1d_op_name", None) or core_name(name)
    modifiers = getattr(target_spec, "k1d_modifiers", None)
    adjoint = bool(modifiers.adjoint) if modifiers is not None else name.startswith("Adjoint(")
    controls = (int(modifiers.control_count) if modifiers is not None
                else int(bool(re.match(r"\d*C\(", name))))
    origin = getattr(target_spec, "k1d_origin", "unknown")
    cls = target_class(op_name)
    record = {"name": name, "op": op_name, "signature": hashlib.sha256(target.encode()).hexdigest()
              [:16], "class": cls, "bucket": _bucket(origin, adjoint, controls, op_name),
              "origin": origin, "adjoint": adjoint, "controls": controls,
              "rules": len(target_spec.rules), "traced": len(traced), "seconds": round(seconds, 3),
              "children": sorted({str(rid).split("{", 1)[0] for rule in target_spec.rules
                                  for rid in rule.resource_counts_by_id})}
    totals, per_rule_loops = [0, 0, 0, 0, 0], []
    top = 0
    for rule in traced:
        counts = equation_counts(rule.closed_jaxpr)
        totals = [a + b for a, b in zip(totals, counts, strict=True)]
        per_rule_loops.append(counts[1])
        top += len(rule.closed_jaxpr.jaxpr.eqns)
    record.update(top=top, nested=totals[0], loops=totals[1], conds=totals[2])
    if cls == "read":
        info = getattr(target_spec, "k1d_read", None) or {}
        record.update(read=info, read_load=totals[3], read_skeleton=totals[0] - totals[3],
                      read_load_top=totals[4], read_skeleton_top=top - totals[4],
                      read_loops_ok=all(n == info.get("loads_expected") for n in per_rule_loops))
    return record


class CaptureRecord:
    """What capture_instrument collected during one compile: spec records, marks and samples."""

    def __init__(self):
        self.specs, self.marks, self.samples, self.errors = [], [], [], []
        self.scopes, self.instrument_seconds = 0, 0.0

    def inventory(self) -> dict:
        """Requirement 4: cells per class x modifier bucket (targets, rules, top and nested
        equations, loops, read roles, ladder and rung targets), the total and its sum check, every
        target name with its parents, the other-class names, the read targets and their
        load/skeleton partition, and the number of rule captures."""
        cells, keys, parents = {}, {}, {}
        total = Counter()
        for r in self.specs:
            total.update({k: r[k] for k in ("rules", "traced", "top", "nested", "loops")})
            total["targets"] += 1
            for cls in ("read_load", "read_skeleton") if r["class"] == "read" else (r["class"],):
                cell = cells.setdefault(f"{cls}|{r['bucket']}", {
                    "targets": 0, "traced_targets": 0, "rules": 0, "top": 0, "nested": 0,
                    "loops": 0, "read_roles": Counter(), "ladder_targets": 0, "rung_targets": 0})
                cell["targets"] += 1
                cell["traced_targets"] += r["traced"] > 0
                cell["rules"] += r["rules"]
                if r["class"] == "read":
                    part = "load" if cls == "read_load" else "skeleton"
                    cell["nested"] += r[f"read_{part}"]
                    cell["top"] += r[f"read_{part}_top"]
                    cell["loops"] += r["loops"] if cls == "read_load" else 0
                    if r["traced"]:
                        cell["read_roles"][str(r["read"].get("role"))] += 1
                else:
                    cell["nested"] += r["nested"]
                    cell["top"] += r["top"]
                    cell["loops"] += r["loops"]
                if r["class"] == "ladder" and r["traced"]:
                    cell["rung_targets" if r["op"] == "ProtoGivensRotation"
                         else "ladder_targets"] += 1
            key = keys.setdefault(r["name"], {"targets": 0, "nested": 0, "class": r["class"],
                                              "buckets": set()})
            key["targets"] += 1
            key["nested"] += r["nested"]
            key["buckets"].add(r["bucket"])
            for child in r["children"]:
                parents.setdefault(child, set()).add(r["name"])
        for name, key in keys.items():
            key["buckets"] = sorted(key["buckets"])
            key["parents"] = sorted(parents.get(name, ()))
        for cell in cells.values():
            cell["read_roles"] = dict(sorted(cell["read_roles"].items()))
        reads = [r for r in self.specs if r["class"] == "read"]
        mismatches = [{"name": r["name"], "bucket": r["bucket"], "loops": r["loops"],
                       "expected": r["read"].get("loads_expected")}
                      for r in reads if r["traced"] and not r["read_loops_ok"]]
        cell_sum = sum(c["nested"] for c in cells.values())
        return {
            "cells": dict(sorted(cells.items())),
            "total": dict(total), "cell_sum": cell_sum, "sum_check": cell_sum == total["nested"],
            "by_class": {cls: sum(c["nested"] for k, c in cells.items()
                                  if k.split("|")[0] == cls) for cls in EQUATION_CLASSES},
            "by_bucket": {b: sum(c["nested"] for k, c in cells.items() if k.split("|")[1] == b)
                          for b in MODIFIER_BUCKETS},
            "loops_by_bucket": {b: sum(c["loops"] for k, c in cells.items()
                                       if k.split("|")[1] == b) for b in MODIFIER_BUCKETS},
            "keys": dict(sorted(keys.items())),
            "other_names": sorted({r["op"] for r in self.specs if r["class"] == "other"}),
            "read_targets": [{k: r[k] for k in ("name", "bucket", "traced", "loops", "read_load",
                                                "read_skeleton")} | {"read": r["read"]}
                             for r in reads],
            "read_partition": {"ok": not mismatches, "mismatches": mismatches,
                               "read_targets": len(reads)},
            "scopes": self.scopes, "instrument_seconds": round(self.instrument_seconds, 2),
            "instrument_errors": self.errors,
            "rule_capture_seconds": round(sum(r["seconds"] for r in self.specs), 2)}

    def timeline(self, pre_capture_tree_mb: float) -> dict:
        """The Opus timeline: phase windows of every QJIT capture/generate_ir/compile, the rule
        capture and the compiler child, with the pre-capture and peak footprints."""
        def window(a, b):
            sel = [s for s in self.samples if a <= s[0] <= b]
            if not sel:
                return {}
            return {"own_max_mb": round(max(s[1] for s in sel), 1),
                    "tree_max_mb": round(max(s[2] for s in sel), 1),
                    "child_max_mb": round(max(s[2] - s[1] for s in sel), 1)}

        phases, opened = [], {}
        for t, label in self.marks:
            name, _, edge = label.rpartition("_")
            if edge == "start":
                opened.setdefault(name, []).append(t)
            elif edge == "end" and opened.get(name):
                t0 = opened[name].pop()
                phases.append({"phase": name, "start": t0, "end": t,
                               "seconds": round(t - t0, 2), **window(t0, t)})
        phases.sort(key=lambda p: p["start"])
        samples = self.samples or [(0.0, 0.0, 0.0)]
        return {"pre_capture_tree_mb": round(pre_capture_tree_mb, 1),
                "peak_own_mb": round(max(s[1] for s in samples), 1),
                "peak_child_mb": round(max(s[2] - s[1] for s in samples), 1),
                "peak_tree_mb": round(max(s[2] for s in samples), 1),
                "phases": phases, "marks": self.marks,
                "every_second": [s for i, s in enumerate(self.samples)
                                 if i % round(1 / SAMPLE_SECONDS) == 0]}


@contextlib.contextmanager
def capture_instrument(shape_name: str):
    """Instrument one compile (requirement 4): tag every DecompTargetSpec built for the capture
    with its origin (registered or distributed), operator and modifiers; count each converted
    spec's equations; mark every QJIT capture/generate_ir/compile, the rule capture and the
    Catalyst CLI child; sample this process and its tree every SAMPLE_SECONDS."""
    import inspect

    import catalyst.compiler as compiler_module
    import catalyst.decomposition.decomposition_rules as rules
    import catalyst.from_plxpr.qfunc_interpreter as interpreter
    from catalyst.jit import QJIT

    from dfthc.testing.watchdog import memory_mb, tree_memory_mb

    record, start, pid = CaptureRecord(), time.perf_counter(), os.getpid()
    roles = {(read_geometry(shape_name)[r]["high_ranges"], read_geometry(shape_name)[r]
              ["block_bits"], read_geometry(shape_name)[r]["width"]): r for r in READ_ROLES}

    def now():
        return round(time.perf_counter() - start, 2)

    def mark(label):
        record.marks.append((now(), label))

    halt = threading.Event()

    def sample():
        while not halt.is_set():
            with contextlib.suppress(Exception):
                record.samples.append((now(), memory_mb(pid), tree_memory_mb(pid)))
            halt.wait(SAMPLE_SECONDS)

    patches = []

    def patch(owner, name, replacement):
        patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def marked(original, label):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            mark(f"{label}_start")
            try:
                return original(*args, **kwargs)
            finally:
                mark(f"{label}_end")
        return wrapper

    for method in ("capture", "generate_ir", "compile"):
        patch(QJIT, method, marked(getattr(QJIT, method), method))
    scope_rules = marked(interpreter._capture_scope_rules, "rules")

    def capture_scope_rules(*args, **kwargs):
        record.scopes += 1
        return scope_rules(*args, **kwargs)

    patch(interpreter, "_capture_scope_rules", capture_scope_rules)

    class _Subprocess:
        def run(self, *args, **kwargs):
            return marked(subprocess.run, "cli")(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(subprocess, item)

    patch(compiler_module, "subprocess", _Subprocess())

    def tagging(builder, origin):
        signature = inspect.signature(builder)

        def build(*args, **kwargs):
            target_spec = builder(*args, **kwargs)
            if target_spec is not None:
                arguments = signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                a = arguments.arguments
                target_spec.k1d_origin = origin
                target_spec.k1d_op_name = a["op_name"]
                target_spec.k1d_modifiers = a["modifiers"]
                target_spec.k1d_read = _read_spec_info(a["op_name"], a["static_data"], roles)
            return target_spec
        return build

    patch(rules, "build_registered_decomp_target_spec",
          tagging(rules.build_registered_decomp_target_spec, "registered"))
    patch(rules, "build_decomp_target_spec", tagging(rules.build_decomp_target_spec,
                                                     "distributed"))
    convert = interpreter._convert_decomp_target_spec

    def counting_convert(interp, target_spec):
        t0 = time.perf_counter()
        traced = convert(interp, target_spec)
        t1 = time.perf_counter()
        try:
            record.specs.append(_spec_record(target_spec, traced, t1 - t0))
        except Exception as exc:  # noqa: BLE001 - an instrument bug must not end the compile
            record.errors.append(f"{target_spec.target_id}: {exc!r}"[:300])
        record.instrument_seconds += time.perf_counter() - t1
        return traced

    patch(interpreter, "_convert_decomp_target_spec", counting_convert)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    mark("instrument_start")
    try:
        yield record
    finally:
        mark("instrument_end")
        halt.set()
        sampler.join()
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def compile_one_d(shape_name: str, program: str, arm: str) -> dict:
    """One compile of a Task 0.9d arm's program, with the capture inventory and the timeline."""
    from dfthc.testing.watchdog import tree_memory_mb

    qfunc, num_wires = build(shape_name, program, arm)
    pre_capture = tree_memory_mb(os.getpid())
    with capture_instrument(shape_name) as instrument, rule_capture_log() as log, \
            warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        start = time.perf_counter()
        resources = compiled_resources(qfunc, num_wires=num_wires)
        wall = time.perf_counter() - start
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024
    record = {"program": program, "shape": shape_name, "arm": arm, "num_wires": num_wires,
              "seconds": round(wall, 1),
              "cpu_seconds": round(own.ru_utime + own.ru_stime + children.ru_utime
                                   + children.ru_stime, 1),
              "rusage_sum_mb": round((own.ru_maxrss + children.ru_maxrss) * scale / 2**20, 1),
              "rusage_self_mb": round(own.ru_maxrss * scale / 2**20, 1),
              "rusage_children_mb": round(children.ru_maxrss * scale / 2**20, 1),
              "analyzed_gates_upper_bound": sum(resources.counts.values()),
              "toffoli": resources.toffoli, "measurements": resources.measurements,
              "measured_and_uncomputes": resources.measured_and_uncomputes,
              "counts": resources.counts, "cnot_analyzed": resources.counts.get("CNOT", 0),
              "table_word_dtype": np.dtype(WORD_DTYPE).name,
              "diagnostics": capture_diagnostics(log, wall),
              "warnings": short_warnings(classify_warnings(records, PROTOTYPE_NAMES,
                                                           expected_measuring=EXPECTED_MEASURING)),
              "loops": captured_loops(log, D_LOOP_CLASSES),
              "inventory": instrument.inventory(),
              "timeline": instrument.timeline(pre_capture)}
    if arm in PER_LEAF_ARMS:
        record["read_implementation_hash"] = read_implementation_hash()
    if arm in EXPLICIT_ARMS:
        record["class_check"] = class_check(arm, program, record["inventory"])
    return record


def class_check(arm: str, program: str, inventory: dict) -> dict:
    """Requirement 2 for X and PX over the complete captured target inventory (candidates the
    graph decomposition does not select included): every expected explicit class present, no
    converted old key (nor any other modified form of a composite, read or explicit class), and
    the modified prototype keys exactly the program's retained list. Non-prototype modified keys
    are retained as gate-set or PennyLane-template modifiers."""
    names = set(inventory["keys"])
    modified = {n: core_name(n) for n in names if is_modified_name(n)}
    old_keys = {old.split(" at ")[0] for a in ("X", arm) for old, _ in CONVERSIONS[a]}
    proto_modified = {n for n, core in modified.items() if core.startswith("Proto")}
    forward_role = {new for roles in FORWARD_ROLE_CLASSES.values() for new in roles.values()}
    found = {
        "explicit_missing": sorted(expected_explicit(arm, program) - names),
        "converted_old_keys_found": sorted(names & old_keys),
        "unexpected_modified_prototype_keys": sorted(proto_modified - EXPECTED_RETAINED[program]),
        "retained_missing": sorted(EXPECTED_RETAINED[program] - proto_modified),
        "modified_read_keys": sorted(n for n, core in modified.items()
                                     if core in READ_CLASS_NAMES),
        "modified_explicit_or_forward_role_keys": sorted(
            n for n, core in modified.items() if core in EXPLICIT_CLASS_NAMES | forward_role),
    }
    return {
        "ok": not any(found.values()), **found,
        "explicit_present": sorted(expected_explicit(arm, program) & names),
        "retained_prototype_keys": sorted(proto_modified & EXPECTED_RETAINED[program]),
        "retained_non_prototype_keys": {
            n: "gate-set modifier" if core in GATE_SET else "PennyLane-template modifier"
            for n, core in sorted(modified.items()) if not core.startswith("Proto")},
        "retained_reason": "small or leaf modified operation legitimately used by the explicit "
                           "rule",
        "class_check_hash": class_check_hash()}


def class_check_hash() -> str:
    """SHA-256 of the class check and its fixed tables (PX must use X's)."""
    import inspect

    digest = hashlib.sha256(inspect.getsource(class_check).encode())
    digest.update(json.dumps([CONVERSIONS, RETAINED_PROTO_KEYS,
                              {k: sorted(v) for k, v in EXPECTED_RETAINED.items()}]).encode())
    return digest.hexdigest()


def b_small_key_table(x_programs: dict) -> dict:
    """Requirement 1: every target name of B's small programs, marked against the fixed table:
    a converted key, retained (with the reason), plain (kept, or replaced by the class of the same
    role), or removed because every parent that demanded it is converted or removed; anything else
    is unexplained (NEEDS_CONTEXT). ``x_programs`` maps program to X's small record."""
    path = results_path("small", "B")
    b = json.loads(path.read_text()) if path.exists() else {}
    converted = {}
    for old, new in CONVERSIONS["X"]:
        converted.setdefault(old.split(" at ")[0], []).append(new)
    table = {}
    for program in PROGRAMS:
        if program not in b or "inventory" not in b[program]:
            continue
        keys = b[program]["inventory"]["keys"]
        x_keys = set(x_programs.get(program, {}).get("inventory", {}).get("keys", {}))
        status, removed = {}, set()
        changed = True
        while changed:
            changed = False
            for name, key in keys.items():
                if (name not in removed and name not in converted and is_modified_name(name)
                        and core_name(name).startswith("Proto") and name not in RETAINED_PROTO_KEYS
                        and key["parents"]
                        and all(p in converted or p in removed for p in key["parents"])):
                    removed.add(name)
                    changed = True
        for name, key in keys.items():
            core = core_name(name)
            if name in converted:
                kind = f"converted to {' / '.join(converted[name])}"
            elif not is_modified_name(name):
                kind = (f"plain, replaced by {FORWARD_ROLE_CLASSES['X'][name]} (same role)"
                        if name in FORWARD_ROLE_CLASSES["X"] else "plain")
            elif name in RETAINED_PROTO_KEYS:
                kind = "retained: small or leaf modified operation legitimately used by the " \
                       "explicit rule"
            elif not core.startswith("Proto"):
                kind = ("retained: gate-set modifier" if core in GATE_SET
                        else "retained: PennyLane-template modifier")
            elif name in removed:
                kind = "removed: demanded only by converted or removed keys"
            else:
                kind = "unexplained"
            status[name] = {"status": kind, "in_X": name in x_keys, "nested": key["nested"],
                            "targets": key["targets"], "buckets": key["buckets"],
                            "parents": key["parents"]}
        table[program] = status
    unexplained = sorted({n for rows in table.values() for n, r in rows.items()
                          if r["status"] == "unexplained"})
    removed_present = sorted({n for rows in table.values() for n, r in rows.items()
                              if r["status"].startswith(("removed", "converted")) and r["in_X"]})
    return {"programs": table, "unexplained": unexplained,
            "removed_or_converted_but_present_in_X": removed_present,
            "ok": not unexplained and not removed_present}


# ----------------------------------------------------------------- P read check (requirement 2)

READ_CHECK_TAG = 0x9E3779B1
"""Address-tagged rows: the entry at address a is ((a + 1) * READ_CHECK_TAG) mod 2**width, odd
multiplier, so distinct addresses (a + 1 <= 2**width) give distinct nonzero entries."""


def read_check_static(role: str) -> tuple:
    """The small shape's read ``role`` with address-tagged rows."""
    table, high_ranges, block_bits, width = read_static("small", role)
    tagged = tuple((a, to_words(((a + 1) * READ_CHECK_TAG) % 2**width, width)) for a, _ in table)
    return tagged, high_ranges, block_bits, width


def read_check_child(role: str) -> dict:
    """Requirement 2, P compiled path, one read signature: one lightning.qubit qjit whose qnode
    takes a signed int64 address and the signed int64 leaf_words as dynamic arguments, prepares
    the address with conditional X gates, applies ProtoTableReadP and returns the probabilities of
    target ‖ junk. Invoked for every computational address without recompiling, and compared
    with the B read's loaded_masks; then one bit of one live row is flipped and the same callable
    must fail the comparison at the affected address. The work register is the minimal one the
    iteration needs (the walk's is wider; it does not affect the loaded layout)."""
    from catalyst.jit import QJIT

    start = time.perf_counter()
    qp.decomposition.enable_graph()
    table, high_ranges, block_bits, width = read_check_static(role)
    leaf_words = p_leaf_words(table, high_ranges, block_bits, width)
    nh = sum(w for w, _, _ in high_ranges)
    nxt = itertools.count()
    high = [next(nxt) for _ in range(nh)]
    low = [next(nxt) for _ in range(block_bits)]
    target = [next(nxt) for _ in range(width)]
    junk = [next(nxt) for _ in range((2**block_bits - 1) * width)]
    work = [next(nxt) for _ in range(max(nh, 1))]
    address_wires, out = high + low, target + junk
    n = len(address_wires)
    compiles = []
    original = QJIT.compile

    def counted(self, *args, **kwargs):
        compiles.append(time.perf_counter())
        return original(self, *args, **kwargs)

    QJIT.compile = counted

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=len(address_wires + out + work)))
    def circuit(address, words):
        for k, wire in enumerate(address_wires):
            qp.cond(((address >> (n - 1 - k)) & 1) == 1, qp.X)(wire)
        ProtoTableReadP(words, high_ranges, block_bits, width, high, low, target, junk, work)
        return qp.probs(wires=out)

    def index(mask):
        return sum(((mask >> j) & 1) << (len(out) - 1 - j) for j in range(len(out)))

    from dfthc.testing.watchdog import tree_memory_mb

    expected = dict(loaded_masks(table, high_ranges, block_bits, width))
    expected_index = [index(expected.get(a, 0)) for a in range(2**n)]
    invocations, outcome = 0, []
    pre_capture = tree_memory_mb(os.getpid())
    with capture_instrument("small") as instrument:
        first = np.asarray(circuit(0, leaf_words))
    for a in range(2**n):
        probs = first if a == 0 else np.asarray(circuit(a, leaf_words))
        invocations += 1
        outcome.append((int(np.argmax(probs)), float(probs.max())))
    QJIT.compile = original
    inventory = instrument.inventory()
    nonzero = [i for i in expected_index if i]
    prefixes = p_leaf_prefixes(high_ranges)
    live = [(leaf, p) for leaf in range(len(prefixes)) for p in range(2**block_bits)
            if ((prefixes[leaf] << block_bits) | p) in dict(table)]
    leaf, position = max(live)
    bit = min(width, WORD) - 1
    mutant = leaf_words.copy()
    mutant[leaf, position, 0] ^= 1 << bit
    affected = (prefixes[leaf] << block_bits) | position
    block = [(prefixes[leaf] << block_bits) | q for q in range(2**block_bits)]
    QJIT.compile = counted
    failing = []
    for a in block:
        invocations += 1
        if int(np.argmax(np.asarray(circuit(a, mutant)))) != expected_index[a]:
            failing.append(a)
    QJIT.compile = original
    mismatches = [a for a in range(2**n) if outcome[a][0] != expected_index[a]]
    nondeterministic = max(1 - p for _, p in outcome)
    record = {
        "role": role, "geometry": {"high_ranges": high_ranges, "block_bits": block_bits,
                                   "width": width, "leaf_words_shape": list(leaf_words.shape),
                                   "wires": len(address_wires + out + work),
                                   "work_wires": len(work)},
        "domain": 2**n, "compiles": len(compiles), "invocations": invocations,
        "mismatches": mismatches, "max_nondeterministic_probability": nondeterministic,
        "zero_expected_addresses": expected_index.count(0),
        "nonzero_expected_distinct": len(set(nonzero)) == len(nonzero),
        "mutant": {"leaf": leaf, "position": position, "word": 0, "bit": bit,
                   "address": affected, "called_addresses": block, "failing_addresses": failing,
                   "detected": affected in failing},
        "capture_inventory": {k: inventory[k] for k in (
            "total", "sum_check", "by_class", "by_bucket", "read_targets", "read_partition",
            "scopes", "instrument_errors", "other_names")},
        "timeline": {k: v for k, v in instrument.timeline(pre_capture).items()
                     if k not in ("marks", "every_second")},
        "seconds": round(time.perf_counter() - start, 1)}
    record["passed"] = (not mismatches and len(compiles) == 1 and nondeterministic < 1e-9
                        and record["nonzero_expected_distinct"] and record["mutant"]["detected"])
    record["instrument_ok"] = (inventory["scopes"] == 1 and inventory["sum_check"]
                               and inventory["read_partition"]["ok"]
                               and not inventory["instrument_errors"])
    return record


def run_read_check() -> int:
    """--read-check: one watched child per read signature (four compiles, one each), written to
    k1d_P_small.json as ``read_check``."""
    roles, ok = {}, True
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for role in READ_ROLES:
        refusal = d_budget_refusal(None)
        if refusal:
            roles[role] = {"passed": False, "error": refusal}
            ok = False
            break
        before = compressor_mb()
        run = run_watched([sys.executable, __file__, "--read-check-child", role], cap_mb=D_CAP_MB,
                          timeout_s=900, env=env)
        after = compressor_mb()
        d_ledger_append(f"P/small/read_check_{role}", run.seconds, compiles=1)
        lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
        try:
            record = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            record = {"passed": False, "error": f"no read-check record ({exc})"}
        if run.killed or run.survivors or run.returncode != 0:
            record = {"passed": False, "error": f"killed: {run.killed}" if run.killed
                      else run.stderr[-3000:], **run.summary()}
        record.update(watched_seconds=round(run.seconds, 1), peak_mb=round(run.peak_mb, 1),
                      compressor_mb_before=before, compressor_mb_after=after)
        roles[role] = record
        ok &= bool(record.get("passed"))
        print("read_check", role, json.dumps({k: record.get(k) for k in (
            "passed", "domain", "compiles", "invocations", "mismatches",
            "max_nondeterministic_probability", "watched_seconds", "peak_mb", "error")}),
              flush=True)
        if not record.get("passed"):
            break
    check = {"passed": ok and len(roles) == len(READ_ROLES), "roles": roles,
             "compiles": sum(r.get("compiles", 0) for r in roles.values()),
             "invocations": sum(r.get("invocations", 0) for r in roles.values()),
             "read_implementation_hash": read_implementation_hash(), "code_hash": code_hash(),
             "method": "one lightning.qubit qjit per read signature of the small walk "
                       "(capture, graph decomposition with the pinned rules); address and "
                       "leaf_words dynamic int64 arguments; compared with loaded_masks"}
    path = results_path("small", "P")
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get("code_hash") != code_hash():
        saved = {"verdict": "incomplete"}
    saved.update(read_check=check, code_hash=code_hash(), environment=environment())
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(saved, indent=2, default=str) + "\n")
    print("read_check passed:", check["passed"], flush=True)
    return 0 if check["passed"] else 1


def read_check_valid() -> bool:
    path = results_path("small", "P")
    saved = json.loads(path.read_text()) if path.exists() else {}
    check = saved.get("read_check") or {}
    return bool(check.get("passed")) and check.get(
        "read_implementation_hash") == read_implementation_hash()


# ----------------------------------------------------------------- expectations and stop gate


def independent_program_d(shape_name: str, program: str) -> dict:
    """An arm's program T/M/A and conditional sites, generated from the Shape for every shape: the
    baseline stand-in's T/M/A from its own resource functions (plain reads and plain-fixup erases,
    no arm-F code) plus the independent erase deltas of independent_erase_counts (U and O of spec
    9.24). Where Task 0.9 compiled the baseline (EXPECTED_TMA), the base must equal it."""
    erases = independent_erase_counts(shape_name)
    multiplicity = ERASE_MULTIPLICITY[program]
    sites = sum(n * erases[e]["sites"] for e, n in multiplicity.items())
    if shape_name == "femoco":
        return {"tma": None, "conditional_sites": sites}
    base = _tma(declared_counts(shape_name, program))
    delta = sum(n * (erases[e]["T"] - erases[e]["plain_T"]) for e, n in multiplicity.items())
    compiled = EXPECTED_TMA.get(shape_name, {}).get(program)
    return {"tma": [base[0] + delta, base[1], base[2] + delta], "conditional_sites": sites,
            "baseline_tma": base,
            "baseline_equals_task_0_9_compiled": None if compiled is None
            else list(compiled) == base}


def independent_count_check_d() -> dict:
    """independent_count_check over the Task 0.9d shapes (mid10k4 included) and FeMoco."""
    out, ok = {}, True
    for shape_name in (*D_SHAPES, "femoco"):
        independent = independent_erase_counts(shape_name)
        declared = resource_erase_counts(shape_name, "F")
        rows = {}
        for e in ("outer", "inner", "angle"):
            tma = [independent[e][k] for k in ("T", "M", "A")]
            rows[e] = {"independent": tma, "resource_functions": declared[e]["tma"],
                       "sites": independent[e]["sites"],
                       "resource_function_sites": declared[e]["sites"],
                       "onehot_bits": independent[e]["onehot_bits"]}
            ok &= tma == declared[e]["tma"] and independent[e]["sites"] == declared[e]["sites"]
        out[shape_name] = rows
    base = {}
    for shape_name in D_SHAPES:
        for program in D_SHAPE_PROGRAMS[shape_name]:
            check = independent_program_d(shape_name, program)
            base[f"{shape_name}/{program}"] = check["baseline_equals_task_0_9_compiled"]
            ok &= check["baseline_equals_task_0_9_compiled"] is not False
    out["baseline_equals_task_0_9_compiled"] = base
    out["femoco_targets"] = independent_count_check()["femoco_targets"]
    out["ok"] = bool(ok and all(out["femoco_targets"].values()))
    return out


def expected_loops_d(shape_name: str, program: str, arm: str) -> dict:
    """Per loop class, the sorted per-target loop counts the geometry implies: one load loop per
    loaded entry of each distinct read (read_bit, B and X) or per unary leaf (read_leaf, P, PX)."""
    out = {kind: [] for kind in D_LOOP_CLASSES}
    geometry = read_geometry(shape_name)
    per_leaf = arm in PER_LEAF_ARMS
    out["read_leaf" if per_leaf else "read_bit"] = sorted(
        geometry[r]["leaves" if per_leaf else "entries"] for r in PROGRAM_READ_ROLES[program])
    return out


def _load_d(shape_name: str, arm: str) -> dict:
    path = results_path(shape_name, arm)
    saved = json.loads(path.read_text()) if path.exists() else {}
    return saved if saved.get("code_hash") == code_hash() else {}


def expected_d(shape_name: str, program: str, arm: str) -> dict:
    """What a Task 0.9d program must give, derived before its compile (no compile)."""
    independent = independent_program_d(shape_name, program)
    declared = declared_counts(shape_name, program, remedy=arm)
    set_bits = (declared_counts(shape_name, program)["CNOT"]
                + declared_counts(shape_name, program, remedy="F")["CNOT"]
                - declared_counts(shape_name, program, remedy=True)["CNOT"])
    static = structural_size(shape_name, program, arm)
    expected = {"tma_resource_functions": _tma(declared), "tma_independent": independent["tma"],
                "conditional_sites": independent["conditional_sites"],
                "loops": expected_loops_d(shape_name, program, arm),
                "cnot_upper_bound": declared["CNOT"], "cnot_set_bits": set_bits,
                "static": {k: v for k, v in static.items() if k != "static_gates"}}
    if arm != "B":
        expected["tma_B"] = _tma(declared_counts(shape_name, program, remedy="B"))
        b_record = _load_d(shape_name, "B").get(program, {})
        if "toffoli" in b_record:
            expected["tma_B_measured"] = [b_record["toffoli"], b_record["measurements"],
                                          b_record["measured_and_uncomputes"]]
    return expected


def d_problems(shape_name: str, program: str, record: dict, expected: dict, arm: str) -> list:
    """Why a Task 0.9d program record trips the stop gate (requirement 3)."""
    if "error" in record:
        return [f"error: {record['error'][-300:]}"]
    problems = []
    if not record.get("within_budget"):
        problems.append("over the time or memory limit")
    if record["warnings"]["prototype"]:
        problems.append("a warning names a prototype")
    tma = [record["toffoli"], record["measurements"], record["measured_and_uncomputes"]]
    for label in ("tma_resource_functions", "tma_independent", "tma_B", "tma_B_measured"):
        if expected.get(label) is not None and tma != list(expected[label]):
            problems.append(f"T/M/A {tma} != {label} {expected[label]}")
    if not record["cnot_analyzed"] == expected["cnot_upper_bound"] != expected["cnot_set_bits"]:
        problems.append(f"read loads folded or lost: CNOT {record['cnot_analyzed']}, upper bound "
                        f"{expected['cnot_upper_bound']}, set bits {expected['cnot_set_bits']}")
    for kind in D_LOOP_CLASSES:
        got = record["loops"]["captured"].get(kind)
        if got != expected["loops"][kind]:
            problems.append(f"{kind} loops {got} != {expected['loops'][kind]}")
    if expected["static"]["conditional_sites"] != expected["conditional_sites"]:
        problems.append(f"conditional sites {expected['static']['conditional_sites']} != "
                        f"{expected['conditional_sites']}")
    inventory = record["inventory"]
    if inventory.get("instrument_errors"):
        problems.append(f"instrument errors: {inventory['instrument_errors'][:2]}")
    if inventory["scopes"] != 1:
        problems.append(f"{inventory['scopes']} rule captures, not one")
    if not inventory["sum_check"]:
        problems.append("the class x bucket cells do not sum to the capture total")
    if not inventory["read_partition"]["ok"]:
        problems.append(f"read targets with loops other than the load loops: "
                        f"{inventory['read_partition']['mismatches'][:3]}")
    if arm in EXPLICIT_ARMS and not record.get("class_check", {}).get("ok"):
        problems.append(f"class check failed: {record.get('class_check')}")
    return problems


def d_refusal(shape_name: str, arm: str) -> str | None:
    """Why a Task 0.9d arm's shape may not compile now: only the four Task 0.9d shapes (never
    FeMoco); P and PX need a passing read check of the current read implementation; a shape needs
    the arm's passing smaller shapes; P, X and PX need B's passing file at the same shape."""
    if shape_name not in D_SHAPES:
        return "Task 0.9d compiles only small, mid10, mid10k4 and mid14 (no FeMoco shape)"
    current = code_hash()
    if arm in PER_LEAF_ARMS and not read_check_valid():
        return "no passing read check for the current read implementation (run --read-check)"
    for shape in D_SHAPES[: D_SHAPES.index(shape_name)]:
        if not _passing(results_path(shape, arm), current):
            return f"stop gate: no passing {results_path(shape, arm).name} for this code"
    parent = D_ARM_PARENT[arm]
    if parent and not _passing(results_path(shape_name, parent), current):
        return f"stop gate: no passing {results_path(shape_name, parent).name} for this code"
    return None


def d_controlled_skip(arm: str, walk14: dict) -> tuple[dict | None, dict]:
    """Task 0.9c's mid14 controlled-walk rule on the arm's own mid10 pair: control_ratio =
    max(1.8, mid10 controlled walk peak / mid10 walk peak), run only if control_ratio times the
    mid14 walk's peak is below 3,500 MB; a missing pair or failed mid14 walk means skip."""
    floor, limit_mb = CONTROLLED_SKIP
    mid10 = _load_d("mid10", arm)
    pair = [mid10.get(p) if mid10.get(p, {}).get("gate", {}).get("ok") else None
            for p in ("walk", "controlled_walk")]
    if None in pair or not (walk14.get("gate", {}).get("ok") and "peak_mb" in walk14):
        reason = f"no passing mid10 pair or mid14 walk: skipped, not an assumed {floor}"
        return {"reason": reason}, {}
    ratio = max(floor, pair[1]["peak_mb"] / pair[0]["peak_mb"])
    rule = {"control_ratio": round(ratio, 3), "mid10_walk_peak_mb": pair[0]["peak_mb"],
            "mid10_controlled_walk_peak_mb": pair[1]["peak_mb"],
            "mid14_walk_peak_mb": walk14["peak_mb"],
            "projected_peak_mb": round(ratio * walk14["peak_mb"]), "limit_mb": limit_mb}
    if ratio * walk14["peak_mb"] < limit_mb:
        return None, rule
    return {"reason": f"control_ratio {ratio:.3f} x the mid14 walk's peak is "
                      f"{ratio * walk14['peak_mb']:.0f} MB, not below {limit_mb} MB", **rule}, rule


def d_ledger() -> list[dict]:
    path = os.environ.get(D_LEDGER_VARIABLE)
    if not path or not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def d_ledger_append(label: str, seconds: float, compiles: int) -> None:
    """Persist one watched child (inclusive accounting: every compile counts) and the totals."""
    path = os.environ.get(D_LEDGER_VARIABLE)
    if not path:
        return
    ledger = d_ledger()
    total = sum(e["compile_seconds"] for e in ledger) + (seconds if compiles else 0.0)
    with open(path, "a") as out:
        out.write(json.dumps({"label": label, "seconds": round(seconds, 1),
                              "compile_seconds": round(seconds, 1) if compiles else 0.0,
                              "compiles": compiles,
                              "total_compile_seconds": round(total, 1),
                              "total_compiles": sum(e["compiles"] for e in ledger) + compiles,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def d_expected_seconds(shape_name: str, program: str) -> float | None:
    """A child's expected watched seconds: 1.3 x the largest watched seconds of the same shape
    and program in any arm so far, else 2.5 x the previous Task 0.9d shape's same program."""
    ledger = d_ledger()
    same = [e["seconds"] for e in ledger if e["label"].endswith(f"/{shape_name}/{program}")]
    if same:
        return 1.3 * max(same)
    previous = [s for s in D_SHAPES[: D_SHAPES.index(shape_name)]
                if program in D_SHAPE_PROGRAMS[s]][-1:]
    earlier = [e["seconds"] for e in ledger for s in previous
               if e["label"].endswith(f"/{s}/{program}")]
    return 2.5 * max(earlier) if earlier else None


def d_budget_refusal(expected_s: float | None) -> str | None:
    """Stop starting compiles at D_BUDGET_SECONDS accumulated watched compile seconds, when the
    expected duration would cross it, or at D_PLANNED_COMPILES compiles (needs the ledger)."""
    if not os.environ.get(D_LEDGER_VARIABLE):
        return "no budget ledger: set DFTHC_K1D_LEDGER"
    ledger = d_ledger()
    total = sum(e["compile_seconds"] for e in ledger)
    compiles = sum(e["compiles"] for e in ledger)
    if compiles >= D_PLANNED_COMPILES:
        return f"budget: {compiles} compiles, the planned maximum is {D_PLANNED_COMPILES}"
    if total + (expected_s or 0.0) > D_BUDGET_SECONDS:
        return (f"budget: {total:.0f} of {D_BUDGET_SECONDS} watched compile seconds used, expected "
                f"{'unknown' if expected_s is None else f'{expected_s:.0f} s'}")
    return None


D_CHILD_FIELDS = {"arm": str, "loops": dict, "inventory": dict, "timeline": dict,
                  "cnot_analyzed": int}


def d_child_record(line: str, shape_name: str, program: str, arm: str) -> tuple[dict | None, str]:
    record, malformed = child_record(line, shape_name, program, False)
    if record is None:
        return None, malformed
    for name, kind in D_CHILD_FIELDS.items():
        if not isinstance(record.get(name), kind) or isinstance(record.get(name), bool):
            return None, f"field {name!r} missing or malformed"
    if record["arm"] != arm:
        return None, f"record of arm {record['arm']}, not {arm}"
    for name in ("cells", "keys", "read_partition", "scopes", "sum_check", "total"):
        if name not in record["inventory"]:
            return None, f"inventory without {name!r}"
    return record, ""


D_NOTES = (
    "Task 0.9d: one compile per resource count (Task 0.2c, compiled_resources with "
    "enable_aot_compilation=False); earlier k1_*, k1r1_* and k1c_* records held two compiles",
    "nested equations: every DecompTargetSpec of the single analyze capture, registered and "
    "distributed specs counted separately, each equation once (loop bodies, cond branches and "
    "modifier regions included)",
    "mid10k4 has one K2 = 16 block per inner row; FeMoco has two",
)


def write_d_results(shape_name: str, arm: str, results: dict, skipped: dict,
                    stopped: str | None) -> dict:
    """Write k1d_<arm>_<shape>.json (checkpointed after every program). P's small file keeps its
    read check; X's and PX's small files hold the class checks (X also B's key table)."""
    path = results_path(shape_name, arm)
    saved = _load_d(shape_name, arm)
    programs = D_SHAPE_PROGRAMS[shape_name]
    record = {p: results[p] for p in programs if p in results}
    complete = len(record) + len(skipped) == len(programs)
    failed = any(not r.get("gate", {}).get("ok") for r in record.values())
    if complete and "controlled_walk" in record and "walk" in record:
        record["controlled_walk_check"] = controlled_walk_check(record)
        failed |= not record["controlled_walk_check"]["ok"]
    if skipped:
        record["skipped"] = skipped
    if stopped:
        record["stopped"] = stopped
    if arm in PER_LEAF_ARMS:
        record["read_implementation_hash"] = read_implementation_hash()
    if arm == "P" and shape_name == "small":
        record["read_check"] = saved.get("read_check")
        failed |= not read_check_valid()
    if arm == "PX":
        p_small = _load_d("small", "P").get("read_check") or {}
        record["read_check_reference"] = {
            "read_implementation_hash_of_P_check": p_small.get("read_implementation_hash"),
            "same_as_PX": p_small.get("read_implementation_hash") == read_implementation_hash(),
            "passed": p_small.get("passed")}
    if arm in EXPLICIT_ARMS and shape_name == "small":
        record["class_check"] = {p: r.get("class_check") for p, r in record.items()
                                 if p in programs}
        record["class_check_hash"] = class_check_hash()
        failed |= not all((c or {}).get("ok") for c in record["class_check"].values())
        if arm == "X" and complete:
            record["b_small_keys"] = b_small_key_table(record)
            failed |= not record["b_small_keys"]["ok"]
    seconds_limit, _ = LIMITS[shape_name]
    record.update(limits={"seconds": seconds_limit, "mb": effective_limit_mb(D_CAP_MB),
                          "cap_mb": D_CAP_MB},
                  memory_method=MEMORY_METHOD, notes=list(NOTES) + list(ARM_NOTES) + list(D_NOTES),
                  code_hash=code_hash(), environment=environment(),
                  verdict="fail" if failed else "pass" if complete and not stopped
                  else "incomplete")
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


def drive_d(shape_name: str, arm: str) -> int:
    """Compile a Task 0.9d arm's programs at a shape, one watched child each (4 GB cap), with the
    stop gate, the mid14 controlled-walk rule, the budget guard and checkpointing: a program
    already recorded for this code is never compiled again, and a failed one stops the arm."""
    refusal = d_refusal(shape_name, arm)
    if refusal:
        print("refused:", refusal, flush=True)
        return 2
    saved = _load_d(shape_name, arm)
    programs = D_SHAPE_PROGRAMS[shape_name]
    results = {p: saved[p] for p in programs if isinstance(saved.get(p), dict)}
    skipped, stopped = dict(saved.get("skipped", {})), None
    if any(not r.get("gate", {}).get("ok") for r in results.values()):
        print("refused: a recorded program failed its gate; no retry", flush=True)
        return 1
    if shape_name == "mid14":
        store_prediction(arm)
    seconds_limit, _ = LIMITS[shape_name]
    env = dict(os.environ)
    for program in programs:
        if program in results or program in skipped:
            print(program, "already recorded", flush=True)
            continue
        rule = None
        if shape_name == "mid14" and program == "controlled_walk":
            skip, rule = d_controlled_skip(arm, results.get("walk", {}))
            if skip:
                skipped[program] = skip
                print(program, "skipped", json.dumps(skip), flush=True)
                write_d_results(shape_name, arm, results, skipped, None)
                continue
        expected = expected_d(shape_name, program, arm)
        refusal = d_budget_refusal(d_expected_seconds(shape_name, program))
        if refusal:
            stopped = f"{program}: {refusal}"
            print("stopped:", stopped, flush=True)
            break
        command = [sys.executable, __file__, "--shape", shape_name, "--program", program,
                   "--arm", arm]
        before = compressor_mb()
        run = run_watched(command, cap_mb=D_CAP_MB, timeout_s=seconds_limit + 60, env=env)
        after = compressor_mb()
        d_ledger_append(f"{arm}/{shape_name}/{program}", run.seconds, compiles=1)
        load = {"watched_seconds": round(run.seconds, 1), "compressor_mb_before": before,
                "compressor_mb_after": after,
                "compressor_mb_delta": (None if None in (before, after)
                                        else round(after - before, 1))}
        lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
        record, malformed = (d_child_record(lines[-1], shape_name, program, arm) if lines
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
                                       and within_effective_limit(run.peak_mb, D_CAP_MB))
            results[program] = {**record, **load}
        results[program]["expected"] = expected
        if rule:
            results[program]["control_rule"] = rule
        problems = d_problems(shape_name, program, results[program], expected, arm)
        results[program]["gate"] = {"ok": not problems, "problems": problems}
        r = results[program]
        print(program, json.dumps({k: r.get(k) for k in (
            "toffoli", "measurements", "measured_and_uncomputes", "seconds", "cpu_seconds",
            "watched_seconds", "peak_mb", "rusage_sum_mb", "compressor_mb_delta", "error")}
            | {"nested": r.get("inventory", {}).get("total", {}).get("nested"),
               "gate": r["gate"]}), flush=True)
        write_d_results(shape_name, arm, results, skipped, None)
        if problems:
            print("stop gate tripped:", "; ".join(problems), flush=True)
            break
    record = write_d_results(shape_name, arm, results, skipped, stopped)
    if "controlled_walk_check" in record:
        print("controlled_walk_check", json.dumps(record["controlled_walk_check"]), flush=True)
    print("verdict:", record["verdict"], flush=True)
    return 0 if record["verdict"] == "pass" else 1


# -------------------------------------------- FeMoco confirmation of the Task 0.9d arms (0.9e)

FEMOCO_D_DEFAULT_PROGRAM = "controlled_walk"
FEMOCO_D_NOTES = (
    "Task 0.9e: the one FeMoco confirmation compile of a Task 0.9d arm, in one watched child "
    "under Task 0.9's FeMoco cap and time limit, started only through drive_femoco_d (token and "
    "lock of Task 0.9), never retried; no P read check is run for it",
    "prediction: the arm's entries of k1d_projection.json as stored before the compile "
    "(nested equations, best and range of peak_mb and rusage_sum_mb, indicative time); "
    "comparison: the measurement against it. A killed run's peak is a lower bound",
)


def femoco_d_result_path(arm: str, program: str) -> Path:
    return results_path(f"femoco_{program}", arm)


def femoco_d_prediction(arm: str, program: str) -> tuple[str | None, dict | None]:
    """The arm's FeMoco prediction for one program from k1d_projection.json, or why there is
    none: the arm's femoco_allowed must be exactly true (unknown, indicative and false refuse).
    The projection's code_hash is recorded, not required (a Task 0.9e edit makes it stale)."""
    path = results_path("projection", "B")
    if not path.exists():
        return f"no {path.name}: run --project --arm B first", None
    projection = json.loads(path.read_text())
    out = projection.get("arms", {}).get(arm)
    if out is None:
        return f"{path.name} has no projection for arm {arm}", None
    if out.get("femoco_allowed") is not True:
        return (f"arm {arm}: femoco_allowed is {out.get('femoco_allowed')!r}, not true "
                f"({out.get('reason')})"), None
    equations, memory = out["equations"].get(program, {}), out["femoco"].get(program)
    if "E_femoco" not in equations or not memory:
        return f"{path.name} has no FeMoco prediction of arm {arm} for {program}", None
    return None, {
        "source": path.name, "projection_code_hash": projection.get("code_hash"),
        "current_code_hash": code_hash(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": arm, "program": program, "femoco_allowed": out["femoco_allowed"],
        "reason": out["reason"], "limits": projection["limits"],
        "equations": {"nested": equations["E_femoco"], "method": equations["E_femoco_method"],
                      "validated": equations["validated"],
                      "mid14_holdout_error": equations.get("mid14", {}).get("error"),
                      "drivers": equations.get("femoco_drivers")},
        "peak_mb": memory["peak_mb"], "rusage_sum_mb": memory["rusage_sum_mb"],
        "model_comparison": memory.get("model_comparison"),
        "time_s": {k: v for k, v in out["time"].get(program, {}).items() if k != "points"}}


def femoco_d_refusal(arm: str, program: str) -> tuple[str | None, dict | None]:
    """Every refusal of the FeMoco confirmation path, before anything compiles (nothing is
    created): the program, the arm's femoco_allowed, an existing result file, a missing token,
    an existing lock. Returns the refusal (or None) and the stored prediction."""
    if program not in SHAPE_PROGRAMS["femoco"]:
        return f"the FeMoco shape compiles only {', '.join(SHAPE_PROGRAMS['femoco'])}", None
    refusal, prediction = femoco_d_prediction(arm, program)
    if refusal:
        return refusal, None
    result = femoco_d_result_path(arm, program)
    if result.exists():
        return (f"{result.name} exists: this arm's FeMoco {program} was already attempted, and "
                "no second run happens without the user's decision"), prediction
    if not os.environ.get(TOKEN_VARIABLE):
        return (f"{TOKEN_VARIABLE} is not set: the orchestrator sets the token of the run "
                "(the driver writes it into the lock and the child checks it)"), prediction
    if FEMOCO_LOCK.exists():
        held = FEMOCO_LOCK.read_text() == os.environ[TOKEN_VARIABLE]
        return (f"{FEMOCO_LOCK} exists (it {'holds' if held else 'does not hold'} this token): "
                "another FeMoco driver runs, or one crashed (check, then delete the lock)"
                ), prediction
    return None, prediction


def expected_femoco_d(program: str, arm: str) -> dict:
    """expected_d at FeMoco (no compile) plus the independent T/M/A that independent_program_d
    leaves out there: the baseline's resource functions plus the independent erase deltas."""
    expected = expected_d("femoco", program, arm)
    erases, multiplicity = independent_erase_counts("femoco"), ERASE_MULTIPLICITY[program]
    base = _tma(declared_counts("femoco", program))
    delta = sum(n * (erases[e]["T"] - erases[e]["plain_T"]) for e, n in multiplicity.items())
    expected["tma_independent"] = [base[0] + delta, base[1], base[2] + delta]
    return expected


def _range_comparison(measured, best: float, bounds: list) -> dict:
    if measured is None:
        return {"measured_mb": None, "best_mb": best, "range_mb": bounds, "inside_range": None}
    low, high = bounds
    outside = measured - high if measured > high else measured - low if measured < low else 0.0
    return {"measured_mb": measured, "best_mb": best, "range_mb": bounds,
            "error_vs_best": round((measured - best) / best, 4),
            "inside_range": low <= measured <= high, "outside_by_mb": round(outside, 1),
            "outside_by_pct_of_bound": round(
                100 * outside / (high if outside > 0 else low), 1) if outside else 0.0}


def femoco_d_comparison(prediction: dict, record: dict, killed: str | None) -> dict:
    """The measurement against the stored prediction: nested equations (signed relative error
    against the equation tolerance), both memory series against best and range (inside or not,
    by how much past the bound), and the wall seconds against the indicative time."""
    measured = (record.get("inventory") or {}).get("total", {}).get("nested")
    predicted = prediction["equations"]["nested"]
    equations = {"measured": measured, "predicted": predicted,
                 "tolerance": D_EQUATION_TOLERANCE}
    if measured is not None:
        error = (measured - predicted) / predicted
        equations.update(error=round(error, 4), abs_error=round(abs(error), 4),
                         within_tolerance=abs(error) <= D_EQUATION_TOLERANCE)
    out = {"equations": equations, "peak_lower_bound_only": bool(killed)}
    for metric in ("peak_mb", "rusage_sum_mb"):
        row = prediction[metric]
        out[metric] = _range_comparison(record.get(metric), row["best_mb"], row["range_mb"])
    seconds, time_row = record.get("seconds"), prediction["time_s"]
    out["seconds"] = {"measured": seconds, "indicative_best": time_row.get("indicative_best_s"),
                      "indicative_upper": time_row.get("indicative_upper_s"),
                      "within_indicative_upper": None if seconds is None
                      or "indicative_upper_s" not in time_row
                      else seconds <= time_row["indicative_upper_s"]}
    return out


def write_femoco_d_results(arm: str, program: str, record: dict | None, prediction: dict,
                           comparison: dict | None, watchdog: dict | None, verdict: str,
                           first: bool = False) -> dict:
    """k1d_<arm>_femoco_<program>.json: the D record structure (the program's record, the class
    check of X and PX, limits, notes, code_hash, environment, verdict) plus the prediction, the
    comparison and the watchdog line. ``first`` creates the file with mode x (the pre-compile
    stub, verdict "running"): a result file of any outcome blocks a second run."""
    seconds_limit, cap_mb = LIMITS["femoco"]
    out = {"shape": "femoco", "arm": arm, "program": program}
    if record is not None:
        out[program] = record
        if arm in EXPLICIT_ARMS:
            out["class_check"] = {program: record.get("class_check")}
            out["class_check_hash"] = class_check_hash()
    if arm in PER_LEAF_ARMS:
        out["read_implementation_hash"] = read_implementation_hash()
        out["read_check"] = "skipped: no compile at FeMoco for checks (Task 0.9e)"
    out.update(prediction=prediction, comparison=comparison, watchdog=watchdog,
               limits={"seconds": seconds_limit, "mb": effective_limit_mb(cap_mb),
                       "cap_mb": cap_mb},
               memory_method=MEMORY_METHOD,
               notes=list(NOTES) + list(ARM_NOTES) + list(D_NOTES) + list(FEMOCO_D_NOTES),
               code_hash=code_hash(), environment=environment(), verdict=verdict)
    RESULTS.mkdir(exist_ok=True)
    with open(femoco_d_result_path(arm, program), "x" if first else "w") as handle:
        handle.write(json.dumps(out, indent=2, default=str) + "\n")
    return out


def drive_femoco_d(arm: str, program: str, dry_run: bool = False) -> int:
    """The FeMoco confirmation compile of a Task 0.9d arm: one program, one watched child under
    Task 0.9's cap and time limit, with Task 0.9's token and lock, never retried (a result file
    of any outcome blocks a second run). Exit 0 for a pass, 2 for a refusal, 1 otherwise."""
    refusal, prediction = femoco_d_refusal(arm, program)
    if refusal:
        print("refused:", refusal, flush=True)
        return 2
    seconds_limit, cap_mb = LIMITS["femoco"]
    timeout_s = seconds_limit + 60
    command = [sys.executable, __file__, "--shape", "femoco", "--program", program, "--arm", arm]
    expected = expected_femoco_d(program, arm)
    if dry_run:
        print("dry run: nothing compiles, no lock is created", flush=True)
        print("child command:", shlex.join(command),
              f"(run under the watchdog with {TOKEN_VARIABLE} in its environment)", flush=True)
        print(f"cap: {cap_mb} MB, effective limit {effective_limit_mb(cap_mb):.0f} MB; timeout: "
              f"{timeout_s} s (time limit {seconds_limit} s + 60 s)", flush=True)
        print("result file:", femoco_d_result_path(arm, program), flush=True)
        print("expected:", json.dumps({k: expected[k] for k in (
            "tma_resource_functions", "tma_independent", "tma_B", "conditional_sites", "loops",
            "cnot_upper_bound", "cnot_set_bits") if k in expected}), flush=True)
        print("prediction:", json.dumps(prediction), flush=True)
        return 0
    env = dict(os.environ)
    RESULTS.mkdir(exist_ok=True)
    try:
        with open(FEMOCO_LOCK, "x") as lock:
            lock.write(env[TOKEN_VARIABLE])
    except FileExistsError:
        print(f"refused: {FEMOCO_LOCK} exists: another FeMoco driver runs, or one crashed "
              "(check, then delete the lock)", flush=True)
        return 2
    try:
        if not _driver_token_valid():
            print("refused: the lock does not hold this token", flush=True)
            return 2
        try:
            write_femoco_d_results(arm, program, None, prediction, None, None, "running",
                                   first=True)
        except FileExistsError:
            print(f"refused: {femoco_d_result_path(arm, program).name} exists", flush=True)
            return 2
        print(f"{arm} {program}: prediction stored in {femoco_d_result_path(arm, program).name}; "
              f"compiling the FeMoco shape once, cap {cap_mb} MB, timeout {timeout_s} s",
              flush=True)
        before = compressor_mb()
        try:
            run = run_watched(command, cap_mb=cap_mb, timeout_s=timeout_s, env=env)
        except WatchdogError as exc:
            print("verdict: fail (the watchdog could not start the child; nothing compiled)",
                  flush=True)
            write_femoco_d_results(arm, program, {"error": f"watchdog did not start: {exc}"},
                                   prediction, None, None, "fail")
            return 1
        after = compressor_mb()
        load = {"watched_seconds": round(run.seconds, 1), "compressor_mb_before": before,
                "compressor_mb_after": after,
                "compressor_mb_delta": (None if None in (before, after)
                                        else round(after - before, 1))}
        lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
        record, malformed = (d_child_record(lines[-1], "femoco", program, arm) if lines
                             else (None, "no line starting with '{'"))
        if run.killed or run.survivors or run.returncode != 0 or malformed:
            if run.killed:
                reason = f"killed: {run.killed}"
            elif run.survivors or run.returncode != 0:
                reason = run.stderr[-3000:]
            else:
                reason = f"malformed output: {malformed}; stdout tail: {run.stdout[-300:]!r}"
            result = {"error": reason, **run.summary(), **load}
        else:
            record["peak_mb"] = round(run.peak_mb, 1)
            record["within_budget"] = (record["seconds"] <= seconds_limit
                                       and within_effective_limit(run.peak_mb, cap_mb))
            result = {**record, **load}
        result["expected"] = expected
        try:
            problems = d_problems("femoco", program, result, expected, arm)
            comparison = femoco_d_comparison(prediction, result, run.killed)
        except Exception as exc:  # the compile's result is never lost to an evaluation error
            result["gate"] = {"ok": False, "problems": [f"evaluation error: {exc!r}"]}
            write_femoco_d_results(arm, program, result, prediction, None, run.summary(),
                                   "killed" if run.killed else "fail")
            raise
        result["gate"] = {"ok": not problems, "problems": problems}
        verdict = ("killed" if run.killed else "pass" if not problems else "fail")
        print(program, json.dumps({k: result.get(k) for k in (
            "toffoli", "measurements", "measured_and_uncomputes", "seconds", "cpu_seconds",
            "watched_seconds", "peak_mb", "rusage_sum_mb", "compressor_mb_delta", "error")}
            | {"nested": result.get("inventory", {}).get("total", {}).get("nested"),
               "gate": result["gate"]}), flush=True)
        print("comparison", json.dumps(comparison), flush=True)
        if run.killed:
            print("WATCHDOG", json.dumps(run.summary()), flush=True)
        write_femoco_d_results(arm, program, result, prediction, comparison, run.summary(),
                               verdict)
        print("verdict:", verdict, flush=True)
        return 0 if verdict == "pass" else 1
    finally:
        FEMOCO_LOCK.unlink(missing_ok=True)


# ----------------------------------------------------------------- projection (requirement 6)

D_DRIVERS = {
    "read_load": "captured load-loop bodies (loaded entries for B/X, unary leaves for P/PX): the "
                 "cell's captured for_loop count at a measured shape; for a prediction, the sum "
                 "over the reference inventory's read targets of the read's loads at the shape",
    "read_skeleton": "sum over the cell's read targets of 23 x unary leaves + 6 x CSWAPs",
    "composite": "layout wires x composite targets of the cell",
    "erase": "erase_score(shape): 5 sites + 9 measured + leaf_cost + 25 nodes + 16 tree ANDs "
             "(Opus model.py, without its factor 1.1 and constant 80)",
    "ladder": "rungs of the cell's targets: N - 1 per ladder target, 1 per rung target",
    "pennylane": "constant 1",
    "other": "layout wires, or constant 1 if the three calibration values are equal",
}
"""The driver table of requirement 6, frozen before any mid14 result exists."""
D_PROJECTION_METHOD = (
    "Equations: per (arm, program, class, modifier bucket) cell, E = a + b D, least squares with "
    "b >= 0 (b = 0 and a = mean when D is constant or the slope is negative) on small, mid10 and "
    "mid10k4 (D_DRIVERS); mid14 predicted from the calibration fit with D from the latest "
    "calibration inventory's multiplicities, before mid14 is used; a program is validated if its "
    "total mid14 error is at most 15%; validated programs are refit on all measured shapes and "
    "projected to FeMoco with D from Shape and Layout and the mid14 inventory's multiplicities. "
    "Memory: per arm and metric, peak = H + c E with H >= 0, c >= 0 on every calibration-shape "
    "program, held out on every measured mid14 program (validated if the largest error is at most "
    "20%), refit on all points after validation. delta = max(0.25, e_E, e_m, e_c); range "
    "[max(0, H + (1 - delta) c E_F), H + (1 + delta) c E_F]. Time: power law through the two "
    "largest measured equation counts per (arm, program), exponent at least 1, for wall and CPU; "
    "upper = max(wall, CPU) x max(2, largest measured wall/CPU ratio); indicative only"
)
MODEL_FPG_GB = {"B": ("no AOT, as built", 6.3, 11.5),
                "P": ("no AOT, per-leaf loads", 2.9, 4.7),
                "X": ("no AOT, no distributed copies", 3.9, 3.6),
                "PX": ("no AOT, per-leaf loads and no distributed copies", 2.1, 1.9)}
"""Opus model.py values (FPG-based estimates, not like-for-like arm-F measurements), walk and
controlled walk in GB."""
D_EQUATION_TOLERANCE, D_MEMORY_TOLERANCE = 0.15, 0.20


def erase_score(shape_name: str) -> int:
    """The fixed erase geometry score of the Opus model.py."""
    s, ob = SHAPES[shape_name], erase_geometry(SHAPES[shape_name])
    k1_, k2_ = 2**s.k1, 2**s.k2
    covered_outer = -(-s.X_o // k1_) * k1_
    sites = covered_outer + s.RC * 2**s.w_b + s.R * s.B + s.N
    measured = k1_ * s.d1 + k2_ * s.d2 + (s.N - 1) * s.b_rot + 3
    outer_leaves = -(-covered_outer // 2 ** ob["outer"])
    inner_leaves = -(-s.RC // 2 ** (ob["inner"] - s.w_b))
    leaf_cost = (outer_leaves * (2 ** ob["outer"] + 12) + inner_leaves * (2 ** ob["inner"] + 12)
                 + s.R * (2**s.w_b + 12) + s.N * 6)
    nodes = outer_leaves + inner_leaves + s.R + s.N
    tree_ands = sum(2**o - 2 for o in (ob["outer"], ob["inner"], s.w_b))
    return 5 * sites + 9 * measured + leaf_cost + 25 * nodes + 16 * tree_ands


def cell_driver(cell_key: str, cell: dict, shape_name: str, arm: str) -> float:
    """D of a cell at a shape from Shape, Layout and the cell's multiplicities (no compile)."""
    cls = cell_key.split("|")[0]
    s, lay, geometry = SHAPES[shape_name], Layout(SHAPES[shape_name]), read_geometry(shape_name)
    roles = {r: n for r, n in cell.get("read_roles", {}).items() if r in geometry}
    if cls == "read_load":
        return sum(n * geometry[r]["leaves" if arm in PER_LEAF_ARMS else "entries"]
                   for r, n in roles.items())
    if cls == "read_skeleton":
        return sum(n * (23 * geometry[r]["leaves"] + 6 * geometry[r]["cswaps"])
                   for r, n in roles.items())
    if cls == "composite":
        return cell["targets"] * lay.num_wires
    if cls == "erase":
        return erase_score(shape_name)
    if cls == "ladder":
        return cell["ladder_targets"] * (s.N - 1) + cell["rung_targets"]
    if cls == "pennylane":
        return 1
    return lay.num_wires


def _measured_driver(cell_key, cell, shape_name, arm):
    """D at a measured shape: the captured load loops for read_load, else cell_driver."""
    if cell_key.startswith("read_load|"):
        return cell["loops"]
    return cell_driver(cell_key, cell, shape_name, arm)


def _fit_affine(xs, ys) -> tuple[float, float]:
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 2 or np.ptp(xs) == 0:
        return float(ys.mean()), 0.0
    b = float(np.polyfit(xs, ys, 1)[0])
    if b < 0:
        return float(ys.mean()), 0.0
    return float(ys.mean() - b * xs.mean()), b


def _fit_line_nonnegative(xs, ys) -> tuple[float, float]:
    """y = H + c x, least squares with H >= 0 and c >= 0."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    c, h = (float(v) for v in np.polyfit(xs, ys, 1)) if np.ptp(xs) else (0.0, float(ys.mean()))
    if c < 0:
        return float(ys.mean()), 0.0
    if h < 0:
        return 0.0, float((xs * ys).sum() / (xs * xs).sum())
    return h, c


def _d_records(arm: str, any_code: bool = False) -> dict:
    """{shape: {program: record}} of an arm's passing program records for this code (with
    ``any_code``, of whatever code_hash the files carry; project_d reports those hashes)."""
    out = {}
    for shape in D_SHAPES:
        path = results_path(shape, arm)
        saved = (json.loads(path.read_text()) if path.exists() else {}) if any_code \
            else _load_d(shape, arm)
        out[shape] = {p: saved[p] for p in D_SHAPE_PROGRAMS[shape]
                      if saved.get(p, {}).get("gate", {}).get("ok")}
        out.setdefault("_code_hashes", set()).add(saved.get("code_hash"))
    return out


def equation_fit(arm: str, program: str, records: dict, shapes, reference: str) -> dict:
    """Fit every cell of a program on ``shapes`` and predict the other measured shapes and FeMoco
    with multiplicities from the ``reference`` shape's inventory."""
    keys = sorted({k for s in shapes for k in records[s][program]["inventory"]["cells"]})
    cells = {}
    for key in keys:
        ref = next((records[s][program]["inventory"]["cells"][key]
                    for s in (reference, *reversed(shapes))
                    if key in records.get(s, {}).get(program, {}).get("inventory", {})
                    .get("cells", {})), None)
        points = {}
        for s in shapes:
            cell = records[s][program]["inventory"]["cells"].get(key)
            points[s] = ((_measured_driver(key, cell, s, arm), cell["nested"]) if cell
                         else (cell_driver(key, ref, s, arm), 0))
        constant = key.split("|")[0] == "other" and len({e for _, e in points.values()}) == 1
        xs = [1 if constant else d for d, _ in points.values()]
        a, b = _fit_affine(xs, [e for _, e in points.values()])
        entry = {"D": {s: d for s, (d, _) in points.items()},
                 "E": {s: e for s, (_, e) in points.items()},
                 "a": round(a, 3), "b": round(b, 6), "constant_other": constant,
                 "residuals": {s: round(e - (a + b * (1 if constant else d)), 1)
                               for s, (d, e) in points.items()}}
        for target in [s for s in D_SHAPES if s not in shapes] + ["femoco"]:
            d = 1 if constant else cell_driver(key, ref, target, arm)
            entry.setdefault("predicted", {})[target] = {"D": d, "E": round(a + b * d, 1)}
        cells[key] = entry
    totals = {t: round(sum(c["predicted"][t]["E"] for c in cells.values()), 1)
              for t in next(iter(cells.values()))["predicted"]}
    return {"shapes": list(shapes), "reference_inventory": reference, "cells": cells,
            "predicted_totals": totals}


def _holdout(fit: dict, measured: dict, target: str) -> dict:
    """Per bucket and class, and in total, predicted against measured nested equations."""
    cells = measured["inventory"]["cells"]
    rows = {}
    for group, index in (("by_bucket", 1), ("by_class", 0)):
        sums = {}
        for key in set(fit["cells"]) | set(cells):
            label = key.split("|")[index]
            pred = fit["cells"].get(key, {}).get("predicted", {}).get(target, {}).get("E", 0)
            meas = cells.get(key, {}).get("nested", 0)
            p, m = sums.get(label, (0, 0))
            sums[label] = (p + pred, m + meas)
        rows[group] = {k: {"predicted": round(p), "measured": m,
                           "error": round((p - m) / m, 4) if m else None}
                       for k, (p, m) in sorted(sums.items())}
    predicted = fit["predicted_totals"][target]
    total = measured["inventory"]["total"]["nested"]
    return {"predicted": round(predicted), "measured": total,
            "error": round(abs(predicted - total) / total, 4), **rows}


def predict_mid14(arm: str) -> dict:
    """The mid14 equation prediction of an arm from its calibration shapes only."""
    records = _d_records(arm)
    records.pop("_code_hashes")
    out = {}
    for program in ("walk", "controlled_walk"):
        if all(program in records[s] for s in D_CALIBRATION_SHAPES):
            fit = equation_fit(arm, program, records, D_CALIBRATION_SHAPES, "mid10k4")
            out[program] = {"mid14_predicted_nested": fit["predicted_totals"].get("mid14"),
                            "cells": {k: c["predicted"].get("mid14")
                                      for k, c in fit["cells"].items()}}
    for program in ("inner_prepare", "inner_select"):
        out[program] = {"note": "no calibration at mid10/mid10k4: not projected"}
    return out


def store_prediction(arm: str) -> None:
    """Write the arm's mid14 prediction into k1d_projection.json before any mid14 compile."""
    path = results_path("projection", "B")
    saved = json.loads(path.read_text()) if path.exists() else {}
    predictions = saved.get("predictions", {})
    if predictions.get(arm, {}).get("code_hash") == code_hash() or _load_d("mid14", arm):
        return  # already stored, or a mid14 record exists: the prediction must precede it
    predictions[arm] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_hash": code_hash(),
                        "driver_table": D_DRIVERS, "programs": predict_mid14(arm)}
    saved["predictions"] = predictions
    RESULTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(saved, indent=2, default=str) + "\n")


def _power_fit(points, x):
    (x1, t1), (x2, t2) = sorted(points)[-2:]
    exponent = max(1.0, math.log(t2 / t1) / math.log(x2 / x1)) if x2 > x1 and t1 > 0 else 1.0
    return t2 * (x / x2) ** exponent, exponent


def project_d() -> dict:
    """Requirement 6 for every arm with data (see D_PROJECTION_METHOD)."""
    limit_mb, seconds_limit = effective_limit_mb(MEMORY_CAP_MB), LIMITS["femoco"][0]
    arms, memory_c = {}, {"peak_mb": {}, "rusage_sum_mb": {}}
    for arm in D_ARMS:
        records = _d_records(arm, any_code=True)
        hashes = sorted(h for h in records.pop("_code_hashes") if h)
        if not all(records[s].get("walk") for s in D_CALIBRATION_SHAPES):
            continue
        out = {"equations": {}, "memory": {}, "time": {}, "record_code_hashes": hashes,
               "records_of_current_code": hashes == [code_hash()]}
        for program in ("walk", "controlled_walk"):
            if not all(program in records[s] for s in D_CALIBRATION_SHAPES):
                out["equations"][program] = {"missing": "calibration points"}
                continue
            calibration = equation_fit(arm, program, records, D_CALIBRATION_SHAPES, "mid10k4")
            entry = {"calibration_fit": calibration}
            if program in records["mid14"]:
                entry["mid14"] = _holdout(calibration, records["mid14"][program], "mid14")
                entry["validated"] = entry["mid14"]["error"] <= D_EQUATION_TOLERANCE
            else:
                entry["validated"] = False
                entry["mid14"] = {"missing": "no passing mid14 record (skipped or not run)"}
            if entry["validated"]:
                final = equation_fit(arm, program, records, D_SHAPES, "mid14")
                entry["refit_all_shapes"] = final
                entry["E_femoco"] = final["predicted_totals"]["femoco"]
                entry["E_femoco_method"] = "refit on all measured shapes (validated)"
            else:
                entry["E_femoco"] = calibration["predicted_totals"]["femoco"]
                entry["E_femoco_method"] = "calibration fit (unvalidated)"
            entry["femoco_drivers"] = {k: c["predicted"]["femoco"]["D"]
                                       for k, c in (entry.get("refit_all_shapes")
                                                    or calibration)["cells"].items()}
            out["equations"][program] = entry
        for metric in ("peak_mb", "rusage_sum_mb"):
            cal = [(r["inventory"]["total"]["nested"], r[metric], s, p)
                   for s in D_CALIBRATION_SHAPES for p, r in records[s].items()]
            h, c = _fit_line_nonnegative([e for e, *_ in cal], [y for _, y, *_ in cal])
            holdout = {p: {"E": r["inventory"]["total"]["nested"], "measured": r[metric],
                           "predicted": round(h + c * r["inventory"]["total"]["nested"], 1),
                           "error": round(abs(h + c * r["inventory"]["total"]["nested"]
                                              - r[metric]) / r[metric], 4)}
                       for p, r in records["mid14"].items()}
            e_m = max((v["error"] for v in holdout.values()), default=None)
            validated = e_m is not None and e_m <= D_MEMORY_TOLERANCE
            entry = {"calibration": {"H": round(h, 1), "c_mb_per_equation": c,
                                     "points": [{"shape": s, "program": p, "E": e, metric: y}
                                                for e, y, s, p in cal]},
                     "holdout_mid14": holdout, "e_m": e_m, "validated": validated}
            if validated:
                pts = cal + [(r["inventory"]["total"]["nested"], r[metric], "mid14", p)
                             for p, r in records["mid14"].items()]
                h, c = _fit_line_nonnegative([e for e, *_ in pts], [y for _, y, *_ in pts])
                entry["refit_all"] = {"H": round(h, 1), "c_mb_per_equation": c,
                                      "points": len(pts)}
            entry["H"], entry["c"] = h, c
            entry["pre_capture_tree_mb"] = sorted(
                r["timeline"]["pre_capture_tree_mb"] for s in D_SHAPES for r in records[s].values())
            out["memory"][metric] = entry
            memory_c[metric][arm] = c
        for program in ("walk", "controlled_walk"):
            pts = [(r["inventory"]["total"]["nested"], r["seconds"], r["cpu_seconds"], s)
                   for s in D_SHAPES for p, r in records[s].items() if p == program]
            eq = out["equations"].get(program, {})
            if len(pts) < 2 or "E_femoco" not in eq:
                continue
            wall, k_wall = _power_fit([(e, w) for e, w, _, _ in pts], eq["E_femoco"])
            cpu, k_cpu = _power_fit([(e, c) for e, _, c, _ in pts], eq["E_femoco"])
            ratio = max(w / c for _, w, c, _ in pts if c)
            out["time"][program] = {
                "points": [{"shape": s, "E": e, "wall_s": w, "cpu_s": c,
                            "wall_cpu_ratio": round(w / c, 3) if c else None}
                           for e, w, c, s in pts],
                "wall_projection_s": round(wall), "wall_exponent": round(k_wall, 3),
                "cpu_projection_s": round(cpu), "cpu_exponent": round(k_cpu, 3),
                "max_wall_cpu_ratio": round(ratio, 3),
                "indicative_best_s": round(wall),
                "indicative_upper_s": round(max(wall, cpu) * max(2.0, ratio)),
                "label": "indicative (compressor and CPU contention)"}
        arms[arm] = out
    e_c = {}
    for metric, by_arm in memory_c.items():
        if by_arm:
            median = float(np.median(list(by_arm.values())))
            e_c[metric] = max(abs(c / median - 1) for c in by_arm.values()) if median else 0.0
    for arm, out in arms.items():
        femoco, reasons = {}, []
        validated_all = True
        for program in ("walk", "controlled_walk"):
            eq = out["equations"].get(program, {})
            if "E_femoco" not in eq:
                validated_all = False
                continue
            validated_all &= bool(eq.get("validated"))
            e_e = eq.get("mid14", {}).get("error")
            rows = {}
            for metric in ("peak_mb", "rusage_sum_mb"):
                mem = out["memory"][metric]
                validated_all &= bool(mem["validated"])
                delta = max(0.25, e_e or 0.0, mem["e_m"] or 0.0, e_c.get(metric, 0.0))
                h, c, e_f = mem["H"], mem["c"], eq["E_femoco"]
                best, upper = h + c * e_f, h + (1 + delta) * c * e_f
                rows[metric] = {
                    "best_mb": round(best), "range_mb": [round(max(0.0, h + (1 - delta) * c * e_f)),
                                                         round(upper)],
                    "delta": round(delta, 4), "delta_inputs": {
                        "e_E": e_e, "e_m": mem["e_m"], "e_c": round(e_c.get(metric, 0.0), 4)},
                    "best_value_within_limit": best <= limit_mb,
                    "upper_range_within_limit": upper <= limit_mb,
                    "headroom_best_mb": round(limit_mb - best),
                    "headroom_best_pct": round(100 * (limit_mb - best) / limit_mb, 1),
                    "headroom_upper_mb": round(limit_mb - upper),
                    "headroom_upper_pct": round(100 * (limit_mb - upper) / limit_mb, 1)}
                if upper > limit_mb:
                    reasons.append(f"{program} {metric} upper {upper:.0f} MB > {limit_mb:.0f} MB")
            t = out["time"].get(program, {})
            if t.get("indicative_upper_s", 0) > seconds_limit:
                reasons.append(f"{program} indicative time upper {t['indicative_upper_s']} s > "
                               f"{seconds_limit} s")
            label, walk_gb, cwalk_gb = MODEL_FPG_GB[arm]
            model = walk_gb if program == "walk" else cwalk_gb
            rows["model_comparison"] = {
                "opus_model": label, "opus_model_mb": round(model * 1024),
                "ratio_best_to_model": round(rows["peak_mb"]["best_mb"] / (model * 1024), 3),
                "differs_by_more_than_25_percent":
                    abs(rows["peak_mb"]["best_mb"] / (model * 1024) - 1) > 0.25,
                "note": "the model values are FPG-based estimates, not like-for-like arm-F "
                        "measurements"}
            femoco[program] = rows
        if not validated_all:
            allowed = "indicative"
            reasons.insert(0, "a projection is not validated")
        elif reasons:
            allowed = "unknown"
        else:
            allowed = True
        out["femoco"] = femoco
        out["femoco_allowed"] = allowed
        out["reason"] = "; ".join(reasons) or "validated, within the limits"
    return {"method": D_PROJECTION_METHOD, "driver_table": D_DRIVERS,
            "limits": {"mb": limit_mb, "seconds": seconds_limit,
                       "equation_tolerance": D_EQUATION_TOLERANCE,
                       "memory_tolerance": D_MEMORY_TOLERANCE},
            "e_c": e_c, "arms": arms,
            "femoco_geometry": {"layout_wires": Layout(SHAPES["femoco"]).num_wires,
                                "erase_score": erase_score("femoco"),
                                "reads": read_geometry("femoco")},
            "erase_geometry": {shape: erase_geometry(SHAPES[shape]) for shape in
                               (*D_SHAPES, "femoco")},
            "independent_count_check": independent_count_check_d(),
            "notes": list(ARM_NOTES) + list(D_NOTES)}


def write_d_projection() -> int:
    path = results_path("projection", "B")
    saved = json.loads(path.read_text()) if path.exists() else {}
    projection = project_d()
    projection["predictions"] = saved.get("predictions", {})
    for arm, out in projection["arms"].items():
        stored = projection["predictions"].get(arm, {}).get("programs", {})
        for program, eq in out["equations"].items():
            if "calibration_fit" in eq and program in stored:
                eq["matches_stored_prediction"] = (
                    stored[program]["mid14_predicted_nested"]
                    == eq["calibration_fit"]["predicted_totals"].get("mid14"))
    projection.update(code_hash=code_hash(), environment=environment())
    path.write_text(json.dumps(projection, indent=2, default=str) + "\n")
    for arm, out in projection["arms"].items():
        for program, rows in out["femoco"].items():
            print(arm, program, json.dumps({m: {k: rows[m][k] for k in ("best_mb", "range_mb")}
                                            for m in ("peak_mb", "rusage_sum_mb")}), flush=True)
        print(arm, "femoco_allowed:", out["femoco_allowed"], "-", out["reason"], flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", choices=sorted(SHAPES))
    parser.add_argument("--program", choices=PROGRAMS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--allow-retry", action="store_true")
    parser.add_argument("--remedy", type=int, choices=(1,),
                        help="partial remedy 1 (reads and angle loads); default: the baseline")
    parser.add_argument("--arm", choices=ARMS + D_ARMS,
                        help="Task 0.9c arm F|FP|FPG or Task 0.9d arm B|P|X|PX (on top of arm F); "
                             "implies --remedy 1")
    parser.add_argument("--fixup-check", action="store_true",
                        help="arm F's tape-mode fixup check into k1c_F_small.json")
    parser.add_argument("--fixup-check-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--read-check", action="store_true",
                        help="arm P's compiled-path read check into k1d_P_small.json")
    parser.add_argument("--read-check-child", choices=READ_ROLES, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true",
                        help="with --shape femoco --all --arm B|P|X|PX: every refusal check, then "
                             "print the child command, cap, timeout and prediction; compile "
                             "nothing, create no lock")
    args = parser.parse_args()
    remedy = args.arm or args.remedy == 1
    femoco_d = args.shape == "femoco" and args.all and args.arm in D_ARMS
    if args.dry_run and not femoco_d:
        parser.error("--dry-run applies only to --shape femoco --all --arm B|P|X|PX")
    if femoco_d and args.allow_retry:
        parser.error("the FeMoco confirmation of a Task 0.9d arm is never retried")
    if femoco_d:
        sys.exit(drive_femoco_d(args.arm, args.program or FEMOCO_D_DEFAULT_PROGRAM,
                                dry_run=args.dry_run))
    if args.fixup_check_child:
        print(json.dumps(fixup_check(), default=str), flush=True)
        return
    if args.read_check_child:
        print(json.dumps(read_check_child(args.read_check_child), default=str), flush=True)
        return
    if args.fixup_check:
        sys.exit(run_fixup_check())
    if args.read_check:
        sys.exit(run_read_check())
    if args.project:
        sys.exit(write_d_projection() if args.arm in D_ARMS else write_projection(remedy))
    if args.shape is None:
        parser.error("--shape is required unless --project is given")
    if args.all and args.arm in D_ARMS:
        sys.exit(drive_d(args.shape, args.arm))
    if args.all:
        sys.exit(drive(args.shape, allow_retry=args.allow_retry, remedy=remedy))
    if args.shape == "femoco" and not _driver_token_valid():
        sys.exit("refused: the FeMoco shape compiles only through --shape femoco --all, which runs "
                 "each program once under the watchdog")
    print(json.dumps(compile_one(args.shape, args.program, remedy)), flush=True)


if __name__ == "__main__":
    main()
