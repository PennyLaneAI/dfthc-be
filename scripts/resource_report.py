"""The resource report: ``docs/resources.md`` and ``docs/resources.json`` (spec 13.3; master plan
Tasks 4.3 and 4.5).

Usage, from the repository root (``PYTHONPATH=src``)::

    python scripts/resource_report.py [--systems NAME ...] [--large SYSTEM] [--rows [ID ...]]
                                      [--out DIR]

``--rows`` compiles only the rows named (none if it names none; the tests use it); the other rows
show their closed form, marked pending.

Every row is one program, the operator exactly as the walk calls it: the rows are the distinct
dfthc operator calls in the walk and the controlled walk (found by running each composite's pinned
rule in tape mode, nothing compiled), two programs the walk does not call (``PhaseFixup`` with each
erase's fix-up arguments, and ``C(ZeroReflection)`` on each reflection's pattern) and the L4
programs (``ControlledWalkPower`` with p = 1, 2; the QPE program with t = 1, 2, 3 at the H2
systems).
Each program is compiled once, in its own subprocess under ``run_watched``, with
``compiled_resources`` on ``null.qubit``; its counts go into the ``"compiled"`` section of the json
with :func:`~dfthc.testing.provenance.code_hash`, the environment, the compile time, the watched
peak memory and the gate count from the resource functions. An entry of the current code hash is
reused; an entry of another hash stays as history and is never shown as a current measurement.

The H2 systems and ``naphthalene`` compile every program that has no current entry. ``fe2s2``,
``fe4s4`` and FeMoco-54 compile only with ``--large SYSTEM``, run by the orchestrator alone
(master plan 12.9): each program is first projected with
:func:`~dfthc.testing.provenance.project_compile` from its current entries at smaller systems and
compiled only if the projection is within 7.5 GB and 60 min, under the 8 GB cap; a refused or
killed program is recorded with its projection and is not a measurement. At FeMoco-54 only the walk
and the controlled walk compile (Task 4.5); every other FeMoco-54 row is its closed form,
validated by the compiles at the other systems and by the cross-check.

The script exits with status 1 after writing both files if a check fails: a measured T, M or A
that differs from its closed form, a nonzero delta against dfthcbe or the paper without a reason
code, or, wherever the walks are measured, a cross-check (the sum over the partition rows of calls
per walk times the closed form, plus in M the block encoding's own junk measurements, against the
measured walk and controlled walk). The walk's unprepares take the junk outcomes the block
encoding measured (spec 16, Q-1 variant (b)): their programs pass constant outcomes, and the
``PhaseFixup`` rows take their erase arguments; the walk calls no ``QROAMErase``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pennylane as qp  # noqa: E402
from pennylane.decomposition import DecompositionGraph  # noqa: E402
from pennylane.typing import AbstractArray  # noqa: E402

from dfthc import costs, costs_reference  # noqa: E402
from dfthc.compile import (  # noqa: E402
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    surviving_dfthc_operators,
)
from dfthc.data.factors import load_factors, random_factors  # noqa: E402
from dfthc.data.tables import WalkTables  # noqa: E402
from dfthc.ops.prepare import InnerPrepare, OuterPrepare  # noqa: E402
from dfthc.ops.qroam import PhaseFixup, QROAMErase, QROAMRead  # noqa: E402
from dfthc.ops.reflections import ZeroReflection  # noqa: E402
from dfthc.ops.walk import ControlledDFTHCWalk, DFTHCWalk  # noqa: E402
from dfthc.params import FEMOCO54, H2_STO3G_TEST, WalkParams, lg, odd_part  # noqa: E402
from dfthc.qpe import ControlledWalkPower, qpe_program  # noqa: E402
from dfthc.registers import WalkLayout  # noqa: E402
from dfthc.testing.provenance import code_hash, environment, project_compile  # noqa: E402
from dfthc.testing.watchdog import MEMORY_CAP_MB, run_watched  # noqa: E402

SMALL = ("h2_sto3g_R2_B2_C1", "h2_631g_R2_B2_C2", "h2_631g_R2_B3_C3", "naphthalene")
LARGE = ("fe2s2", "fe4s4", "femoco54")
SYSTEMS = SMALL + LARGE
"""The report's systems (spec 13.3 step 1), smallest first."""
TEST_SYSTEMS = ("H2_STO3G_TEST",)
"""Accepted by ``--systems`` for the tests; never part of the default report."""
QPE_SYSTEMS = ("h2_sto3g_R2_B2_C1", "h2_631g_R2_B2_C2", "h2_631g_R2_B3_C3", "H2_STO3G_TEST")
"""Where the QPE program is compiled (spec 11.3, 13.3 step 2)."""
FEMOCO_PROGRAMS = ("DFTHCWalk", "ControlledDFTHCWalk")
"""The only programs compiled at FeMoco-54 (spec 13.3 step 2; Task 4.5)."""
FEMOCO_SEED = 0
"""``random_factors(54, 10, 27, 27, n_plus=27, seed=0)``, as the work-pool test uses."""
QPE_BITS = (1, 2, 3)
POWERS = (1, 2)
SMALL_CAP_MB, SMALL_TIMEOUT_S = 4096, 900
LARGE_CAP_MB, LARGE_TIMEOUT_S = MEMORY_CAP_MB, 3600
JSON_NAME, MD_NAME = "resources.json", "resources.md"
PREFIX = "RESULT "
CLIFFORDS = ("PauliX", "PauliY", "PauliZ", "Hadamard", "S", "Adjoint(S)", "CNOT", "CZ", "SWAP",
             "GlobalPhase")

REASON_CODES = {
    "flat-r": "r is carried through the inner swap; heterogeneous targets are deferred",
    "no-prune": "no aligned-block pruning of unary-iteration boundaries: dfthcbe stops once on "
                "each aligned block of a dead interval, and a walk over a sub-range [lo, hi) can "
                "be truncated only that way (costs_reference.iteration_counts)",
    "exact-iter": "exact unary iteration: out-of-range addresses raise no flag, which costs at "
                  "most one Toffoli per address bit at each range end; dfthcbe's truncated "
                  "iteration is cheaper (costs_reference.iteration_counts)",
    "iter-const": "truncated iteration over L leaves costs L + c - 2 Toffolis with c controls; the "
                  "paper counts L, one per leaf",
    "prefix-stops": "dfthcbe's walks over the tail rows [RC, X_o) (the tail load and the angle "
                    "erase's tail fix-up) stop once on each aligned block of the dead SF prefix "
                    "[0, RC), s(RC) stops; the paper's (B41, and B50's fix-up N) count their N "
                    "leaves alone (costs_reference.iteration_causes)",
    "onehot-pad": "the fixup one-hot is padded to a power of two",
    "adder-const": "SemiAdder constants differ from Qualtran's",
    "mux": "spin-multiplexer construction",
    "attrib": "the paper attributes the cost to a different step",
    "rz-addsub": "each Givens R_z is Qualtran's ControlledAddOrSubtract, b_rot - 1 Toffolis, as in "
                 "dfthcbe; the paper counts b_rot",
    "cmp-free": "PennyLane's comparators uncompute by measurement, so their adjoints cost no "
                "Toffoli",
    "maj-and": "the Majorana gate folds the success flags into two ANDs with G0 (2 + 2 + 1); the "
               "paper's Fig. 4 controls its four gates on one AND of both flags (1 + 2 + 4)",
    "sf-enable": "the SF fix-up of AngleErase is enabled on G0, one Toffoli more than dfthcbe's",
    "rot-add": "PhaseGradientRotation adds with one m-bit SemiAdder, m - 1 Toffolis; Qualtran's "
               "ZPowConstViaPhaseGradient (dfthcbe's rotation) costs m - 2",
    "unif-odd": "dfthcbe's uniform preparation, 4 n + 2 s_u - 5 on the n = lg l qubits of the odd "
                "part l of L (as ours), against the paper's 4 lg L over all lg L qubits",
    "sign-bits": "the inner entry carries both sign bits, sgn and d_sgn; Table III evaluates its "
                 "inner swaps with b_2 = w_b + b_k2 bits",
    "onehot-const": "a one-hot over K values costs K - 2 Toffolis; the paper's B50 and B53 count K "
                    "and its B47 counts B for the B + 1 values of b",
    "ref-target": "the reflection's multi-controlled Z targets one of its n_w pattern qubits "
                  "(n_w - 2 Toffolis); dfthcbe's uncontrolled REF_{A u B} and the paper's B55 add "
                  "a phase target (n_w - 1)",
    "ref-unif": "the paper's REF_T2 (B11) also reflects the rotated qubit of its Unif(b), which "
                "the flagged uniform preparation does not have",
    "qpe-ctrl": "the paper's B55 books one Toffoli for the unary iteration on the QPE control in "
                "the walk (App. B.7.2, Eq. (128)); our walk has no QPE control",
    "ctrl-ref-t2": "the controlled walk also controls REF_T2 (+1 per step), so C(W)^p is exact for "
                   "every p; the paper's QPE step controls only REF_{A u B} (App. B.7.2, "
                   "Eq. (128))",
}
"""Spec 13.3's reason codes, with the twelve this task adds (master plan Task 4.3). A row lists
the codes whose contribution to its delta is nonzero (``costs_reference.delta_causes``)."""

LEAVES = frozenset({"QROAMRead", "QROAMErase", "PhaseFixup", "AngleErase", "ZeroReflection",
                    "SpinMux", "SpinSwap", "MajoranaGate", "InnerSigns", "PhaseGradientRotation",
                    "GivensRotation"})
"""Operators whose rules call no dfthc operator; their rules are not run to find rows."""
COMPOSITES = frozenset({"DFTHCWalk", "ControlledDFTHCWalk", "DFTHCBlockEncoding",
                        "ControlledDFTHCBlockEncoding", "InnerSelect", "InnerSelectDagger"})
"""The partition rows are the children of these that are not in :data:`INNER_COMPOSITES`."""
INNER_COMPOSITES = frozenset({"DFTHCBlockEncoding", "ControlledDFTHCBlockEncoding",
                              "InnerSelect", "InnerSelectDagger"})
LEVELS = {"L1": {"PhaseGradientRotation", "FlaggedUniformPrep", "Adjoint(FlaggedUniformPrep)",
                 "QROAMRead", "PhaseFixup", "QROAMErase", "ZeroReflection", "C(ZeroReflection)"},
          "L3": {"RefT2", "RefAB", "ControlledRefT2", "ControlledRefAB", "DFTHCBlockEncoding",
                 "ControlledDFTHCBlockEncoding", "DFTHCWalk", "ControlledDFTHCWalk"},
          "L4": {"ControlledWalkPower", "QPE"}}
FROM_OUTCOMES = frozenset({"OuterUnprepareFromOutcomes", "InnerUnprepareFromOutcomes"})
"""The unprepares the walk calls (spec 16, Q-1 variant (b)): they take the junk outcomes the
block encoding measured, so their programs pass constant outcomes, and they emit the erase's
fix-up themselves, so the ``PhaseFixup`` rows take their erase arguments."""
NAME_KEYS = {
    "OuterPrepare": "outer_prepare", "OuterUnprepareFromOutcomes": "outer_unprepare",
    "InnerPrepare": "inner_prepare", "InnerUnprepareFromOutcomes": "inner_unprepare",
    "InnerSigns": "inner_signs", "TailAngleLoad": "tail_angle_load",
    "SFAngleLoad": "sf_angle_load", "SpinMux": "spin_mux", "SpinSwap": "spin_swap",
    "GivensRotation": "givens_rotation", "Adjoint(GivensRotation)": "givens_rotation_adjoint",
    "GivensLadder": "givens_ladder", "GivensLadderDagger": "givens_ladder_dagger",
    "MajoranaGate": "majorana_gate", "Adjoint(MajoranaGate)": "majorana_gate_adjoint",
    "AngleErase": "angle_erase", "InnerSelect": "inner_select",
    "InnerSelectDagger": "inner_select_dagger", "RefT2": "ref_t2",
    "ControlledRefT2": "controlled_ref_t2", "RefAB": "ref_ab",
    "ControlledRefAB": "controlled_ref_ab",
    "DFTHCBlockEncoding": "block_encoding",
    "ControlledDFTHCBlockEncoding": "controlled_block_encoding", "DFTHCWalk": "walk",
    "ControlledDFTHCWalk": "controlled_walk",
}
"""The ``costs_reference`` key of every row whose class appears with one signature."""
PARENT_KEYS = {
    ("FlaggedUniformPrep", "OuterPrepare"): "uniform_x_o",
    ("FlaggedUniformPrep", "InnerPrepare"): "uniform_b",
    ("Adjoint(FlaggedUniformPrep)", "OuterUnprepareFromOutcomes"): "uniform_x_o_adjoint",
    ("Adjoint(FlaggedUniformPrep)", "InnerUnprepareFromOutcomes"): "uniform_b_adjoint",
    ("PhaseGradientRotation", "FlaggedUniformPrep"): "rotation",
    ("QROAMRead", "OuterPrepare"): "outer_read", ("QROAMRead", "InnerPrepare"): "inner_read",
    ("QROAMRead", "TailAngleLoad"): "tail_angle_load",
    ("QROAMRead", "SFAngleLoad"): "sf_angle_load",
    ("PhaseFixup", "OuterUnprepareFromOutcomes"): "outer_erase",
    ("PhaseFixup", "InnerUnprepareFromOutcomes"): "inner_erase",
    ("ZeroReflection", "RefT2"): "ref_t2", ("ZeroReflection", "RefAB"): "ref_ab",
    ("C(ZeroReflection)", "RefT2"): "controlled_ref_t2",
    ("C(ZeroReflection)", "RefAB"): "controlled_ref_ab",
}
"""The ``costs_reference`` key of the L1 rows, by class and parent class."""
STEPS = {"OuterPrepare": "1-4", "InnerPrepare": "5-8", "InnerSigns": "9", "InnerSelect": "10-18",
         "TailAngleLoad": "10", "SFAngleLoad": "11", "SpinMux": "12, 17", "SpinSwap": "13, 17",
         "GivensLadder": "14", "GivensRotation": "14", "MajoranaGate": "15",
         "Adjoint(MajoranaGate)": "15", "GivensLadderDagger": "16",
         "Adjoint(GivensRotation)": "16", "AngleErase": "18", "InnerSelectDagger": "10-18",
         "InnerUnprepareFromOutcomes": "19-22", "RefT2": "23", "ControlledRefT2": "23",
         "OuterUnprepareFromOutcomes": "24-27", "RefAB": "28", "ControlledRefAB": "28",
         "DFTHCBlockEncoding": "1-27", "ControlledDFTHCBlockEncoding": "1-27",
         "DFTHCWalk": "1-28", "ControlledDFTHCWalk": "1-28", "ControlledWalkPower": "QPE",
         "QPE": "QPE"}
PARENT_STEPS = {("FlaggedUniformPrep", "OuterPrepare"): "1",
                ("FlaggedUniformPrep", "InnerPrepare"): "5",
                ("Adjoint(FlaggedUniformPrep)", "OuterUnprepareFromOutcomes"): "27",
                ("Adjoint(FlaggedUniformPrep)", "InnerUnprepareFromOutcomes"): "22",
                ("PhaseGradientRotation", "FlaggedUniformPrep"): "1, 5",
                ("QROAMRead", "OuterPrepare"): "2", ("QROAMRead", "InnerPrepare"): "6",
                ("QROAMRead", "TailAngleLoad"): "10", ("QROAMRead", "SFAngleLoad"): "11",
                ("PhaseFixup", "OuterUnprepareFromOutcomes"): "26",
                ("PhaseFixup", "InnerUnprepareFromOutcomes"): "21",
                ("ZeroReflection", "RefT2"): "23", ("ZeroReflection", "RefAB"): "28",
                ("C(ZeroReflection)", "RefT2"): "23", ("C(ZeroReflection)", "RefAB"): "28"}


@dataclasses.dataclass
class System:
    """One report system: its parameters, tables and layout."""

    name: str
    params: WalkParams
    tables: WalkTables
    layout: WalkLayout


@dataclasses.dataclass
class Row:
    """One program of the report at one system."""

    id: str
    name: str
    parent: str | None
    level: str
    steps: str
    calls: int
    controlled_calls: int
    partition: bool
    closed: costs.Cost
    uncomputes: int
    """The closed form of A, the measured AND uncomputes (:func:`closed_form`)."""
    wires: int
    program: Callable[[], tuple[Callable[[], None], int]]
    key: str | None
    descendants: frozenset[str]
    op: object = None
    size_op: Callable[[], object] | None = None
    """Builds, in tape mode, the operator whose resource functions give the row's gates."""


def build_system(name: str) -> System:
    """FeMoco-54 from ``random_factors``; the test preset; a vendored system at the report's
    default precisions (``costs_reference.default_params``)."""
    if name == "femoco54":
        p, factors = FEMOCO54, random_factors(54, 10, 27, 27, n_plus=27, seed=FEMOCO_SEED)
    elif name == "H2_STO3G_TEST":
        p, factors = H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")
    elif name in SYSTEMS:
        factors = load_factors(name)
        p = costs_reference.default_params(factors.N, factors.R, factors.B, factors.C,
                                           factors.n_plus)
    else:
        raise ValueError(f"unknown system {name!r}; expected one of {SYSTEMS + TEST_SYSTEMS}")
    return System(name, p, WalkTables.from_factors(factors, p), WalkLayout.from_params(p))


def _groups(s: System) -> dict:
    return {name: getattr(s.layout, name) for name in DFTHCWalk.wire_argnames}


def _walk_op(s: System, controlled: bool):
    p, t = s.params.as_compilable(), s.tables.as_compilable()
    if controlled:
        return ControlledDFTHCWalk(p, t, [s.layout.num_wires], **_groups(s))
    return DFTHCWalk(p, t, **_groups(s))


def walk_program(s: System, controlled: bool):
    """The walk or controlled walk with its phase gradient: the programs of Task 3.4."""

    def qfunc():
        qp.PhaseGradientStatePrep(s.layout.phase_grad)
        _walk_op(s, controlled)

    return qfunc, s.layout.num_wires + int(controlled)


def _power_op(s: System, power: int):
    groups = {name: getattr(s.layout, name) for name in ControlledWalkPower.wire_argnames[1:]}
    return ControlledWalkPower(power, s.params.as_compilable(), s.tables.as_compilable(),
                               [s.layout.num_wires], **groups)


def power_program(s: System, power: int):
    """``ControlledWalkPower`` on ``WalkLayout``'s groups and one control wire after them (the
    phase4-qpe plan's interface)."""
    groups = {name: getattr(s.layout, name) for name in ControlledWalkPower.wire_argnames[1:]}

    def qfunc():
        qp.PhaseGradientStatePrep(s.layout.phase_grad)
        ControlledWalkPower(power, s.params.as_compilable(), s.tables.as_compilable(),
                            [s.layout.num_wires], **groups)

    return qfunc, s.layout.num_wires + 1


def _children(op) -> list:
    """The dfthc operators the pinned rule of ``op`` calls (tape mode, nothing compiled)."""
    if op.name in LEAVES or op.name not in FIXED_DECOMPS:
        return []
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        FIXED_DECOMPS[op.name](**op.arguments)
    return [child for child in queue.queue if surviving_dfthc_operators([child.name])]


def _target(op):
    return op.base if op.name.startswith("Adjoint(") else op


def _signature(op) -> tuple:
    target = _target(op)
    return (op.name, tuple(sorted(target.compilable_args.items())),
            tuple((name, len(getattr(target, name))) for name in target.wire_argnames))


def _rebuilt(op, index: dict):
    """``op`` constructed again on the wires ``index`` maps its wires to (under capture this
    queues it); an unprepare from outcomes gets constant junk outcomes, a bool array, since its
    program has no junk measurement (its counts do not depend on the outcomes)."""
    if op.name.startswith("Adjoint("):
        return qp.adjoint(_rebuilt(op.base, index))
    arguments = dict(op.arguments)
    for name in op.wire_argnames:
        arguments[name] = [index[wire] for wire in getattr(op, name)]
    if op.name in FROM_OUTCOMES:
        arguments["junk_outcomes"] = np.zeros(len(arguments["junk_outcomes"]), dtype=bool)
    return type(op)(**arguments)


def _call_program(op):
    def program():
        index = {wire: i for i, wire in enumerate(op.wires)}
        return (lambda: _rebuilt(op, index)), len(op.wires)

    return program


def _erase_arguments(op) -> dict:
    """The erase an unprepare from outcomes performs: its read's ``address_widths``,
    ``live_ranges``, ``width`` and ``block_bits`` (no enable) and the fix-up's ``onehot_bits``
    (spec 9.7, 9.8)."""
    a = op.arguments
    if op.name == "OuterUnprepareFromOutcomes":
        shape, onehot_bits = OuterPrepare.read_shape(a["X_o"], a["b_k1"], a["k1"]), a["ke1"]
    else:
        shape = InnerPrepare.read_shape(a["N"], a["R"], a["B"], a["C"], a["b_k2"], a["k2"])
        onehot_bits = a["ke2"] + lg(a["B"] + 1)
    return shape | {"control_value": None, "onehot_bits": onehot_bits}


def _fixup_program(a: dict):
    """``PhaseFixup`` with an erase's fix-up arguments ``a`` (:func:`_erase_arguments`): its
    covered set, one-hot bits, enable and work, on a one-hot of ``2**onehot_bits`` wires; the
    program's parities are all ``False``, and the size operator (tape mode) takes an abstract
    parity array. Returns ``(program, size_op, wires)``."""
    covered = QROAMRead.loaded_ranges(a["address_widths"], a["live_ranges"], a["block_bits"])
    sizes = PhaseFixup.required_wires(a["address_widths"], covered, a["onehot_bits"],
                                      a["control_value"])
    size = len(PhaseFixup.address_order(a["address_widths"], covered))
    wires, start = {}, 0
    for name in ("address", "onehot", "work", "control"):
        wires[name] = list(range(start, start + sizes[name]))
        start += sizes[name]

    def build(parities):
        return PhaseFixup(parities, a["address_widths"], covered, a["onehot_bits"],
                          a["control_value"], **wires)

    def program():
        return (lambda: build(np.zeros(size, dtype=bool))), start

    return program, lambda: build(AbstractArray((size,), bool)), start


def _controlled_reflection_program(reflection):
    """``qp.ctrl`` of the reflection's ``ZeroReflection`` with one more work wire. Returns
    ``(program, size_op, wires)``."""
    zero, one = len(reflection.zero_wires), len(reflection.one_wires)
    work = ZeroReflection.required_wires(zero, one, True)["work"]
    n = zero + one + work + 1
    wires = list(range(n))

    def build():
        return qp.ctrl(ZeroReflection(wires[:zero], wires[zero:zero + one],
                                      wires[zero + one:n - 1]), control=wires[n - 1])

    return (lambda: (build, n)), build, n


def _dirty(L: int) -> int:
    """ANDs a flagged uniform preparation of ``L`` leaves for its adjoint to uncompute: its
    classical comparator's ``n - 1`` (``n = lg l``; none if ``l = 1``, spec 9.24)."""
    n = lg(odd_part(L))
    return n - 1 if n else 0


def closed_form(name: str, op, p: WalkParams) -> tuple[costs.Cost, int]:
    """The closed form (``dfthc.costs``) of one call: ``(Cost(T, M), A)``; for ``PhaseFixup``,
    ``op`` is the erase's arguments (:func:`_erase_arguments`). A counts the measured AND
    uncomputes inside the call: a prepare leaves its comparators' ANDs (``b_k`` and the uniform
    preparation's :func:`_dirty`) for its unprepare to measure (spec 9.24, 10.2), so per call A
    is not ``T`` minus the plain ``Toffoli`` gates; over the walk it is
    (``measured_and_uncomputes``)."""
    a = op if isinstance(op, dict) else _target(op).arguments if op is not None else {}
    if name == "PhaseGradientRotation":
        cost = costs.phase_gradient_rotation(a["k"], a["m"])
        return cost, cost.toffoli
    if name == "FlaggedUniformPrep":
        cost = costs.flagged_uniform_prep(a["L"], a["s"])
        return cost, cost.toffoli - _dirty(a["L"])
    if name == "Adjoint(FlaggedUniformPrep)":
        cost = costs.flagged_uniform_unprep(a["L"], a["s"])
        return cost, cost.toffoli + _dirty(a["L"])
    if name == "QROAMRead":
        ranges = QROAMRead.visited_ranges(a["address_widths"], a["live_ranges"], a["block_bits"])
        cost = costs.qroam_read(ranges, a["block_bits"], a["width"],
                                a["control_value"] is not None)
        return cost, cost.toffoli - (2**a["block_bits"] - 1) * a["width"]
    if name == "PhaseFixup":
        ranges = QROAMErase.fixup_high_ranges(a["address_widths"], a["live_ranges"],
                                              a["block_bits"], a["onehot_bits"])
        cost = costs.phase_fixup(ranges, a["onehot_bits"], a["control_value"] is not None)
        return cost, cost.toffoli
    if name in ("ZeroReflection", "C(ZeroReflection)"):
        n_w = len(op.zero_wires) + len(op.one_wires)
        controlled = name.startswith("C(")
        cost = costs.zero_reflection(n_w, controlled)
        return cost, cost.toffoli - int(n_w >= (2 if controlled else 3))
    walk_a = costs.measured_and_uncomputes(p, False)
    table = {
        "OuterPrepare": (costs.outer_prepare(p),
                         -_dirty(p.X_o) - (p.K1 - 1) * p.d1 - p.b_k1 - p.w_x),
        "OuterUnprepareFromOutcomes": (costs.outer_unprepare_from_outcomes(p),
                                       -p.w_x + p.b_k1 + _dirty(p.X_o)),
        "InnerPrepare": (costs.inner_prepare(p),
                         -_dirty(p.B + 1) - (p.K2 - 1) * p.d2 - p.b_k2 - p.w_b),
        "InnerUnprepareFromOutcomes": (costs.inner_unprepare_from_outcomes(p),
                                       -p.w_b + p.b_k2 + _dirty(p.B + 1)),
        "InnerSigns": (costs.inner_signs(), 0),
        "TailAngleLoad": (costs.tail_angle_load(p), 0),
        "SFAngleLoad": (costs.sf_angle_load(p), 0),
        "SpinMux": (costs.spin_mux(), -1), "SpinSwap": (costs.spin_swap(p), -p.N),
        "GivensRotation": (costs.givens_rotation(p), 0),
        "Adjoint(GivensRotation)": (costs.givens_rotation(p), 0),
        "GivensLadder": (costs.givens_ladder(p), 0),
        "GivensLadderDagger": (costs.givens_ladder(p), 0),
        "MajoranaGate": (costs.majorana_gate(), -3),
        "Adjoint(MajoranaGate)": (costs.majorana_gate(), -3),
        "AngleErase": (costs.angle_erase(p), 0),
        "InnerSelect": (costs.inner_select(p), -(2 * p.N + 5)),
        "InnerSelectDagger": (costs.inner_select(p), -(2 * p.N + 5)),
        "RefT2": (costs.ref_t2(p, False), -1), "ControlledRefT2": (costs.ref_t2(p, True), -1),
        "RefAB": (costs.ref_ab(p, False), -1), "ControlledRefAB": (costs.ref_ab(p, True), -1),
        "DFTHCBlockEncoding": (costs.block_encoding(p, False), 1 - costs.toffoli_only(p)),
        "ControlledDFTHCBlockEncoding": (costs.block_encoding(p, True),
                                         1 - costs.toffoli_only(p)),
        "DFTHCWalk": (costs.walk(p, False), walk_a - costs.walk(p, False).toffoli),
        "ControlledDFTHCWalk": (costs.walk(p, True), costs.measured_and_uncomputes(p, True)
                                - costs.walk(p, True).toffoli),
    }
    cost, offset = table[name]
    return cost, cost.toffoli + offset


def _level(name: str) -> str:
    return next((level for level, names in LEVELS.items() if name in names), "L2")


def build_rows(s: System) -> list[Row]:
    """Every row of ``s``: the distinct calls in the walk, then the controlled walk's own, in the
    order of first appearance; then the two programs the walk does not call, and L4. A row's id
    is its class name if the class appears with one signature, else ``name@parent id``, with
    ``#2``, ``#3`` for further calls of the class under the same parent."""
    p = s.params
    with qp.capture.pause(), qp.queuing.AnnotatedQueue():
        roots = [_walk_op(s, False), _walk_op(s, True)]
    order, calls, first, below = [], [{}, {}], {}, {}
    for tree, root in enumerate(roots):
        def visit(op, parent, tree=tree):
            sig = _signature(op)
            if sig not in first:
                first[sig] = (op, parent)
                order.append(sig)
            calls[tree][sig] = calls[tree].get(sig, 0) + 1
            names = {op.name}
            for child in _children(op):
                names |= visit(child, op)
            below[sig] = frozenset(names)
            return names

        visit(root, None)
    signatures_of = {}
    for sig in order:
        signatures_of[sig[0]] = signatures_of.get(sig[0], 0) + 1
    ids, used, rows = {}, {}, []
    for sig in order:
        op, parent = first[sig]
        base = sig[0] if signatures_of[sig[0]] == 1 else f"{sig[0]}@{ids[_signature(parent)]}"
        used[base] = used.get(base, 0) + 1
        ids[sig] = base if used[base] == 1 else f"{base}#{used[base]}"
        parent_name = parent.name if parent is not None else None
        closed, uncomputes = closed_form(op.name, op, p)
        rows.append(Row(
            id=ids[sig], name=op.name, parent=parent_name, level=_level(op.name),
            steps=PARENT_STEPS.get((op.name, parent_name), STEPS.get(op.name, "")),
            calls=calls[0].get(sig, 0), controlled_calls=calls[1].get(sig, 0),
            partition=parent_name in COMPOSITES and op.name not in INNER_COMPOSITES,
            closed=closed, uncomputes=uncomputes, wires=len(op.wires),
            program=(_walk_rows_program(s, op.name) if op.name in FEMOCO_PROGRAMS
                     else _call_program(op)),
            key=PARENT_KEYS.get((op.name, parent_name), NAME_KEYS.get(op.name)),
            descendants=below[sig], op=op, size_op=lambda op=op: op))
    rows += _extra_rows(s, rows)
    return rows


def _walk_rows_program(s: System, name: str):
    return lambda: walk_program(s, name == "ControlledDFTHCWalk")


def _extra_rows(s: System, rows: list[Row]) -> list[Row]:
    p, extra = s.params, []
    for row in rows:
        if row.name in FROM_OUTCOMES:
            erase = _erase_arguments(row.op)
            program, size_op, wires = _fixup_program(erase)
            closed, uncomputes = closed_form("PhaseFixup", erase, p)
            extra.append(dataclasses.replace(
                row, id=f"PhaseFixup@{row.id}", name="PhaseFixup", parent=row.name, level="L1",
                steps=PARENT_STEPS[("PhaseFixup", row.name)], calls=0, controlled_calls=0,
                partition=False, closed=closed, uncomputes=uncomputes, wires=wires, program=program,
                key=PARENT_KEYS[("PhaseFixup", row.name)], descendants=frozenset({
                    "PhaseFixup"}), size_op=size_op))
        if row.name == "ZeroReflection":
            program, size_op, wires = _controlled_reflection_program(row.op)
            closed, uncomputes = closed_form("C(ZeroReflection)", row.op, p)
            extra.append(dataclasses.replace(
                row, id=f"C(ZeroReflection)@{row.parent}", name="C(ZeroReflection)",
                steps=PARENT_STEPS[("C(ZeroReflection)", row.parent)], calls=0,
                controlled_calls=0, partition=False, closed=closed, uncomputes=uncomputes,
                wires=wires, program=program, key=PARENT_KEYS[("C(ZeroReflection)", row.parent)],
                descendants=frozenset({"C(ZeroReflection)"}), size_op=size_op))
    walk = next(row for row in rows if row.name == "ControlledDFTHCWalk")
    for power in POWERS:
        extra.append(Row(
            id=f"ControlledWalkPower(p={power})", name="ControlledWalkPower", parent=None,
            level="L4", steps=STEPS["ControlledWalkPower"], calls=0, controlled_calls=0,
            partition=False, closed=costs.controlled_walk_power(p, power),
            uncomputes=power * costs.measured_and_uncomputes(p, True),
            wires=s.layout.num_wires + 1,
            program=(lambda power=power: power_program(s, power)), key="controlled_walk_power",
            descendants=walk.descendants | {"ControlledWalkPower"},
            size_op=lambda power=power: _power_op(s, power)))
    if s.name in QPE_SYSTEMS or s.name == "femoco54":
        for bits in QPE_BITS:
            extra.append(Row(
                id=f"QPE(t={bits})", name="QPE", parent=None, level="L4", steps=STEPS["QPE"],
                calls=0, controlled_calls=0, partition=False,
                closed=costs.iterative_qpe(p, bits),
                uncomputes=(2**bits - 1) * costs.measured_and_uncomputes(p, True)
                + (bits**3 - bits) // 6, wires=qpe_wires(s, bits),
                program=(lambda bits=bits: qpe_program(s.params, s.tables, bits)), key=None,
                descendants=walk.descendants | {"ControlledWalkPower", "QPERound"}))
    return extra


def qpe_wires(s: System, bits: int) -> int:
    """The QPE program's wires without building it (the phase4-qpe plan's interface)."""
    from dfthc.qpe import spare_wires_needed

    sizes = tuple(len(getattr(s.layout, name)) for name in ControlledWalkPower.wire_argnames[1:])
    return (WalkLayout.from_params(s.params, qpe_bits=bits).num_wires
            + spare_wires_needed("ControlledWalkPower", sizes, bits))


def clifford_kind(row: Row) -> str:
    """How the row's Clifford total is to be read (spec 7.6): reads and resets are unhinted
    conditionals, counted at their maximum branch; erase fix-ups are hinted, counted at their
    expectation."""
    names = row.descendants
    unhinted = names & ({"QROAMRead", "QROAMErase", "AngleErase", "QPERound"} | FROM_OUTCOMES)
    hinted = names & ({"QROAMErase", "AngleErase", "PhaseFixup"} | FROM_OUTCOMES)
    parts = (["upper bound (CNOT, PauliX)"] if unhinted else []) + (
        ["expected, hinted (CZ, PauliZ)"] if hinted else [])
    return "; ".join(parts) or "exact"


def gates(rows: list[Row]) -> dict[str, int]:
    """Each row's emitted gates from the resource functions of its size operator (no compile):
    the size measure of the projection. One decomposition graph serves all rows; the QPE rows,
    which the large systems never compile, have none."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue():
        ops = {row.id: row.size_op() for row in rows if row.size_op is not None}
    solution = DecompositionGraph(operations=list(ops.values()), gate_set=GATE_SET,
                                  fixed_decomps=FIXED_DECOMPS).solve()
    return {row_id: int(solution.resource_estimate(op).num_gates) for row_id, op in ops.items()}


def compile_counts(s: System, row: Row) -> dict:
    """Compile one row's program in this process; its counts and compile time."""
    qfunc, num_wires = row.program()
    start = time.perf_counter()
    resources = compiled_resources(qfunc, num_wires)
    return {"toffoli": resources.toffoli, "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "counts": resources.counts, "num_wires": num_wires,
            "seconds": round(time.perf_counter() - start, 2)}


def _find_row(s: System, row_id: str) -> Row:
    if row_id in FEMOCO_PROGRAMS:
        controlled = row_id == "ControlledDFTHCWalk"
        closed, uncomputes = closed_form(row_id, None, s.params)
        return Row(id=row_id, name=row_id, parent=None, level="L3", steps=STEPS[row_id],
                   calls=1 - controlled, controlled_calls=int(controlled), partition=False,
                   closed=closed, uncomputes=uncomputes, wires=s.layout.num_wires + controlled,
                   program=_walk_rows_program(s, row_id), key=NAME_KEYS[row_id],
                   descendants=frozenset())
    return next(row for row in build_rows(s) if row.id == row_id)


def child_command(s: System, row: Row) -> list[str]:
    """The subprocess that compiles one row (``--compile-one``)."""
    return [sys.executable, str(Path(__file__).resolve()), "--compile-one", s.name, row.id]


def measure(s: System, row: Row, hash_: str, cap_mb: float, timeout_s: float) -> dict:
    """Compile one row in a subprocess under ``run_watched``; the cache entry. A killed or
    failed compile is recorded with its status and is never a measurement."""
    command = child_command(s, row)
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_watched(command, cap_mb=cap_mb, timeout_s=timeout_s, cwd=REPO, env=env)
    lines = [line for line in run.stdout.splitlines() if line.startswith(PREFIX)]
    summary = run.summary()
    summary["command"] = summary["command"][-2:]
    entry = {"code_hash": hash_, "environment": environment(), "watchdog": summary}
    if run.killed or run.survivors:
        return entry | {"status": "killed", "reason": run.killed}
    if run.returncode != 0 or len(lines) != 1:
        return entry | {"status": "failed", "reason": run.stderr[-2000:]}
    return entry | {"status": "compiled"} | json.loads(lines[0][len(PREFIX):])


def current(cache: dict, key: str, hash_: str) -> dict | None:
    """The newest compiled entry of ``key`` with code hash ``hash_``."""
    entries = [e for e in cache.get(key, []) if e["code_hash"] == hash_
               and e["status"] == "compiled"]
    return entries[-1] if entries else None


def family(row: Row) -> str:
    """The calibration family of a row: its class and comparison key, which name the same block
    at every system although a row's id can differ between systems (``PhaseGradientRotation`` has
    one call shape at the H2 systems and two elsewhere); an L4 row's id."""
    return row.id if row.level == "L4" else f"{row.name}:{row.key or row.id}"


def calibration(cache: dict, family_: str, hash_: str) -> list[tuple]:
    """``(gates, seconds, peak_mb)`` of every current entry of the family, at every system."""
    points = []
    for key, entries in cache.items():
        entry = current({key: entries}, key, hash_)
        if entry and entry.get("family") == family_ and entry.get("gates"):
            points.append((entry["gates"], entry["watchdog"]["seconds"],
                           entry["watchdog"]["peak_mb"]))
    return points


def projection(cache: dict, row: Row, hash_: str, size: int | None) -> dict:
    """The row's projection to ``size`` gates from the current entries of its family at every
    system, earlier large systems included (:func:`~dfthc.testing.provenance.project_compile`
    under the 8 GB cap: the family's largest measurement if some entry has at least ``size``
    gates, else extrapolated)."""
    if size is None:
        return {"allowed": False, "reason": "no size measure", "gates": None, "points": []}
    return project_compile(calibration(cache, family(row), hash_), size, cap_mb=LARGE_CAP_MB)


def compiled_here(s: System, row: Row) -> bool:
    """Whether the report compiles this row at this system at all (spec 13.3 step 2)."""
    if s.name == "femoco54":
        return row.id in FEMOCO_PROGRAMS
    return not (row.name == "QPE" and s.name not in QPE_SYSTEMS)


def run_system(s: System, rows: list[Row], cache: dict, hash_: str, *, projected: bool,
               only: set[str] | None) -> list[str]:
    """Compile the rows of ``s`` that this run compiles and record them in ``cache``; the log.

    Rows with a current entry are reused. ``projected`` (``--large``) first projects each row
    from its current entries (:func:`projection`) and compiles it under the 8 GB cap only if the
    projection allows; a refused row is recorded with its projection. Under ``projected`` the
    first ``killed`` or ``failed`` row stops the system's run, and so does the first ``refused``
    row at FeMoco-54: every remaining candidate is recorded as ``skipped``, never a measurement,
    and the log names the escalation (master plan 12.6, 12.9; Task 4.5). The large systems compile
    nothing without ``projected``; the others compile under a 4 GB cap."""
    log, stopped = [], None
    if s.name in LARGE and not projected:
        return log
    candidates = [row for row in rows if compiled_here(s, row)
                  and (only is None or row.id in only)
                  and not current(cache, f"{s.name}/{row.id}", hash_)]
    sizes = gates(candidates) if candidates else {}
    for row in candidates:
        key = f"{s.name}/{row.id}"
        if stopped:
            cache.setdefault(key, []).append(
                {"code_hash": hash_, "environment": environment(), "status": "skipped",
                 "reason": f"not compiled after {stopped}"})
            log.append(f"{key}: skipped (after {stopped})")
            continue
        if projected:
            estimate = projection(cache, row, hash_, sizes.get(row.id))
            if not estimate["allowed"]:
                cache.setdefault(key, []).append(
                    {"code_hash": hash_, "environment": environment(), "status": "refused",
                     "gates": sizes.get(row.id), "projection": estimate})
                log.append(f"{key}: refused ({estimate['reason']})")
                if s.name == "femoco54":
                    stopped = f"{key} refused"
                    log.append(f"{s.name}: stopped; escalate to the user with the projection "
                               "(master plan 12.6, \"Where compiles above the cap run\")")
                continue
            entry = measure(s, row, hash_, LARGE_CAP_MB, LARGE_TIMEOUT_S)
            entry["projection"] = estimate
        else:
            entry = measure(s, row, hash_, SMALL_CAP_MB, SMALL_TIMEOUT_S)
        entry["gates"], entry["family"] = sizes.get(row.id), family(row)
        cache.setdefault(key, []).append(entry)
        log.append(f"{key}: {entry['status']} ({entry['watchdog']['seconds']} s, "
                   f"{entry['watchdog']['peak_mb']} MB)")
        if projected and entry["status"] in ("killed", "failed"):
            stopped = f"{key} {entry['status']}"
            log.append(f"{s.name}: stopped; the orchestrator reports the {entry['status']} "
                       "compile and escalates (master plan 12.6, 12.9)")
    return log


def report_system(s: System, rows: list[Row], cache: dict, hash_: str) -> tuple[dict, list]:
    """The json record of ``s`` and the failed checks."""
    p, errors, records = s.params, [], []
    dfthcbe, paper = costs_reference.dfthcbe_blocks(p), costs_reference.paper_table3(p)
    causes = costs_reference.delta_causes(p)
    for row in rows:
        entry = current(cache, f"{s.name}/{row.id}", hash_)
        history = cache.get(f"{s.name}/{row.id}", [])
        closed = {"toffoli": row.closed.toffoli, "measurements": row.closed.measurements,
                  "measured_and_uncomputes": row.uncomputes}
        if entry:
            status = "measured"
            measured = {name: entry[name] for name in closed}
            if measured != closed:
                errors.append(f"{s.name}/{row.id}: measured {measured} != closed form {closed}")
        else:
            measured = None
            last = history[-1] if history else None
            status = ("closed form" if not compiled_here(s, row)
                      else last["status"] if last and last["code_hash"] == hash_
                      else "pending")
        ref_d = dfthcbe.get(row.key) if row.key else None
        ref_p = paper.get(row.key) if row.key else None
        codes_d, codes_p = causes.get(row.key, ({}, {}))
        if row.key == "controlled_walk_power":
            power = int(row.id.split("=")[1].rstrip(")"))
            ref_d, ref_p = None, power * paper["controlled_walk"]
            codes_d = {}
            codes_p = {code: power * value for code, value in causes["controlled_walk"][1].items()}
        delta_d = None if ref_d is None else row.closed.toffoli - ref_d
        delta_p = None if ref_p is None else row.closed.toffoli - ref_p
        for delta, codes, who in ((delta_d, codes_d, "dfthcbe"), (delta_p, codes_p, "paper")):
            if delta is None:
                continue
            unknown = sorted(set(codes) - set(REASON_CODES))
            if unknown:
                errors.append(f"{s.name}/{row.id}: unknown reason codes {unknown}")
            if sum(codes.values()) != delta:
                errors.append(f"{s.name}/{row.id}: delta {delta} against {who} != its reason "
                              f"codes {codes} (sum {sum(codes.values())})")
        counts = entry["counts"] if entry else {}
        records.append({
            "id": row.id, "level": row.level, "steps": row.steps, "calls": row.calls,
            "controlled_calls": row.controlled_calls, "partition": row.partition,
            "status": status, "measured": measured, "closed": closed,
            "dfthcbe": ref_d, "paper": ref_p,
            "paper_step18": paper.get(f"{row.key}_step18") if row.key else None,
            "delta_dfthcbe": delta_d, "delta_paper": delta_p,
            "codes_dfthcbe": dict(codes_d) if delta_d is not None else {},
            "codes_paper": dict(codes_p) if delta_p is not None else {},
            "cliffords": sum(counts.get(name, 0) for name in CLIFFORDS) if entry else None,
            "clifford_kind": clifford_kind(row), "wires": row.wires,
            "seconds": entry["seconds"] if entry else None,
            "peak_mb": entry["watchdog"]["peak_mb"] if entry else None,
            "gates": entry.get("gates") if entry else None,
            "projection": history[-1].get("projection") if history else None,
        })
    errors += cross_check(s, records)
    record = {"params": dataclasses.asdict(p), "wires": s.layout.num_wires,
              "qubit_groups": {g: len(getattr(s.layout, g)) for g in
                               ("system", "outer", "inner", "select", "phase_grad", "work")},
              "junk_wires": len(s.layout.junk_x_o) + len(s.layout.junk_b),
              "junk_measurements": costs.junk_measurements(p).measurements, "rows": records}
    if s.name == "femoco54":
        cwalk = next(r for r in records if r["id"] == "ControlledDFTHCWalk")
        walk = next(r for r in records if r["id"] == "DFTHCWalk")
        basis = cwalk["measured"] or cwalk["closed"]
        m_walk = (walk["measured"] or walk["closed"])["measurements"]
        formulas = ("T = (2^t - 1) T_C(W) + (t^3 - t)/6, M = (2^t - 1) M_W + t and "
                    "A = (2^t - 1) A_C(W) + (t^3 - t)/6, every correction at its maximum branch")
        record["qpe_basis"] = (
            "T and A from the compiled controlled walk, M from the walk (`--large femoco54`, once "
            f"each per code hash): {formulas}."
            if cwalk["measured"] else
            f"T and A from the closed form of the controlled walk, M from the walk: {formulas}; "
            "a compile with `--large femoco54` replaces them by the compiled T_C(W), M_W and "
            "A_C(W).")
        record["qpe"] = {str(bits): {
            "toffoli": (2**bits - 1) * basis["toffoli"] + (bits**3 - bits) // 6,
            "measurements": (2**bits - 1) * m_walk + bits,
            "measured_and_uncomputes": (2**bits - 1) * basis["measured_and_uncomputes"]
            + (bits**3 - bits) // 6,
            "wires": qpe_wires(s, bits)} for bits in (1, 2, 3, 10, 20, 23)}
    return record, errors


def cross_check(s: System, records: list[dict]) -> list[str]:
    """Spec 13.3 step 4: the partition rows times their calls, plus in M the block encoding's own
    junk measurements (:func:`dfthc.costs.junk_measurements`; spec 16, Q-1 variant (b)), give the
    walk and the controlled walk, in closed form always and against the measured walks where they
    are measured; and the partition sums of dfthcbe's and the paper's values give their walks."""
    errors, by_id = [], {r["id"]: r for r in records}
    junk = costs.junk_measurements(s.params).measurements
    for walk, calls in (("DFTHCWalk", "calls"), ("ControlledDFTHCWalk", "controlled_calls")):
        parts = [r for r in records if r["partition"] and r[calls]]
        total = {name: sum(r[calls] * r["closed"][name] for r in parts)
                 for name in ("toffoli", "measurements", "measured_and_uncomputes")}
        total["measurements"] += junk
        if total != by_id[walk]["closed"]:
            errors.append(f"{s.name}: partition {total} != closed {walk} {by_id[walk]['closed']}")
        if by_id[walk]["measured"] is not None and total != by_id[walk]["measured"]:
            errors.append(f"{s.name}: partition {total} != measured {walk}")
        for ref in ("dfthcbe", "paper"):
            if by_id[walk][ref] is not None and all(r[ref] is not None for r in parts):
                if sum(r[calls] * r[ref] for r in parts) != by_id[walk][ref]:
                    errors.append(f"{s.name}: {ref} partition != {ref} {walk}")
    return errors


def _cell(value) -> str:
    return "—" if value is None else f"{value:,}" if isinstance(value, int) else str(value)


def _tma(counts: dict | None) -> str:
    if counts is None:
        return "—"
    return " / ".join(f"{counts[n]:,}" for n in ("toffoli", "measurements",
                                                 "measured_and_uncomputes"))


def render(report: dict) -> str:
    """``docs/resources.md`` from the json record."""
    lines = ["# Resource report", "",
             "Generated by `scripts/resource_report.py` (spec 13.3). Every row is one program, "
             "the operator as the walk calls it, compiled on its own with `compiled_resources` "
             "on `null.qubit` where the status says measured; T, M and A are the Toffoli, "
             "explicit-measurement and measured-AND-uncompute counts (spec 7.6). A measured "
             "value always equals the closed form of `dfthc.costs`, or the script fails. "
             "dfthcbe's and the paper's values are Toffolis per call "
             "(`dfthc.costs_reference`); dfthcbe's Qualtran \"measurement\" count is our A, not "
             "our M, and the paper itemizes neither, so they are not shown. Calls are per walk "
             "and per controlled walk; the partition rows (P) sum to the walk exactly.", "",
             f"Code hash `{report['code_hash']}`; environment: "
             + ", ".join(f"{k} {v}" for k, v in report["environment"].items()) + ".", ""]
    for name, system in report["systems"].items():
        p = system["params"]
        lines += [f"## {name}", "",
                  "Parameters: " + ", ".join(f"{k} = {v}" for k, v in p.items())
                  + f". Walk qubits (`WalkLayout`): {system['wires']:,} ("
                  + ", ".join(f"{g} {n}" for g, n in system["qubit_groups"].items()) + ").", "",
                  "| Row | Level | Steps | Calls (W, C(W)) | T / M / A measured | "
                  "T / M / A closed form | Status | dfthcbe | Paper | Δ dfthcbe | Δ paper | "
                  "Reason codes | Cliffords | Qubits | Compile s | Peak MB |",
                  "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- "
                  "| --- | ---: | ---: | ---: |"]
        for r in system["rows"]:
            calls = f"{r['calls']}, {r['controlled_calls']}" + (" (P)" if r["partition"] else "")
            paper = _cell(r["paper"]) + (f" ({r['paper_step18']:,} at step 18)"
                                         if r["paper_step18"] is not None else "")
            codes = "; ".join(", ".join(f"`{c}` {v:+,}" for c, v in r[side].items()) or "—"
                              for side in ("codes_dfthcbe", "codes_paper"))
            cliffords = ("—" if r["cliffords"] is None
                         else f"{r['cliffords']:,} ({r['clifford_kind']})")
            lines.append(
                f"| `{r['id']}` | {r['level']} | {r['steps']} | {calls} | {_tma(r['measured'])} "
                f"| {_tma(r['closed'])} | {r['status']} | {_cell(r['dfthcbe'])} | {paper} | "
                f"{_cell(r['delta_dfthcbe'])} | {_cell(r['delta_paper'])} | {codes} | "
                f"{cliffords} | {r['wires']:,} | {_cell(r['seconds'])} | "
                f"{_cell(r['peak_mb'])} |")
        lines.append("")
    if "femoco54" in report["systems"]:
        lines += femoco_summary(report["systems"]["femoco54"])
    lines += large_compiles(report)
    lines += ["## Reason codes", "", "The codes of a row are the causes with a nonzero "
              "contribution to its delta, with that contribution (against dfthcbe; against the "
              "paper); the contributions sum to the delta exactly, and two different causes may "
              "cancel (`dfthc.costs_reference.delta_causes`).", "",
              "| Code | Meaning |", "| --- | --- |"]
    lines += [f"| `{code}` | {meaning} |" for code, meaning in REASON_CODES.items()]
    return "\n".join(lines) + "\n"


def femoco_summary(system: dict) -> list[str]:
    """Spec 13.1 at FeMoco-54 from the partition rows: per walk, ours, dfthcbe's and the paper's
    (Table III's attribution), the qubits, the QPE and the deferred refinements."""
    rows = {r["id"]: r for r in system["rows"]}
    parts = [r for r in system["rows"] if r["partition"] and r["calls"]]
    walk, cwalk = rows["DFTHCWalk"], rows["ControlledDFTHCWalk"]
    compiled = ("are compiled with `--large femoco54`, once each per code hash; their measured "
                "rows above equal their closed forms."
                if walk["measured"] and cwalk["measured"] else
                "are compiled only with `--large femoco54`; without a compile under the current "
                "code hash their rows above show their closed form.")
    lines = ["## FeMoco-54 summary (spec 13.1)", "",
             "Toffolis per walk: each partition row's calls per walk times its value. The walk "
             f"and the controlled walk {compiled}", "",
             "| Row | Calls | Ours | dfthcbe | Paper |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in parts:
        lines.append(f"| `{r['id']}` | {r['calls']} | {r['calls'] * r['closed']['toffoli']:,} "
                     f"| {_cell(None if r['dfthcbe'] is None else r['calls'] * r['dfthcbe'])} "
                     f"| {_cell(None if r['paper'] is None else r['calls'] * r['paper'])} |")
    lines.append(f"| **walk** | 1 | **{walk['closed']['toffoli']:,}** | **{walk['dfthcbe']:,}** | "
                 f"**{walk['paper']:,}** |")
    m, a = walk["closed"]["measurements"], walk["closed"]["measured_and_uncomputes"]
    ca = cwalk["closed"]["measured_and_uncomputes"]
    junk_m = system["junk_measurements"]
    junk = WalkLayout.junk_sizes(WalkParams(**system["params"]))
    ours, paper = system["qubit_groups"], costs_reference.PAPER_QUBIT_GROUPS_FEMOCO
    qubit_rows = [f"| `{group}` | {ours[group]:,} | {paper[group]:,} | "
                  f"{ours[group] - paper[group]:+,} |" for group in ours]
    lines += ["", f"- Ours minus dfthcbe: walk {walk['delta_dfthcbe']:+,}, controlled walk "
              f"{cwalk['delta_dfthcbe']:+,}. Ours minus the paper: walk {walk['delta_paper']:+,}. "
              f"The paper's {walk['paper']:,} is its QPE step, with the control on "
              f"REF_{{A u B}} only (App. B.7.2, Eq. (128)); against our controlled walk, "
              f"{cwalk['closed']['toffoli']:,}, the difference is {cwalk['delta_paper']:+,}: the "
              "block encoding's, with `ctrl-ref-t2` +1 (our controlled walk also controls "
              "REF_T2) and `ref-target` -1 (the controlled REF_{A u B} against B55), and no "
              "`qpe-ctrl`, since both carry the QPE control.",
              f"- Measurements per walk: M = {m:,}, A = {a:,} (controlled {ca:,}), so "
              f"M + A = {m + a:,} (controlled {m + ca:,}) physical measurements. {junk_m:,} of "
              "the M are the block encoding's own junk measurements, right after each prepare "
              "(spec 16, Q-1 variant (b)); the partition rows hold the rest.",
              "- The paper's Table III charges the tail-angle fix-up (N) to the inner erase; "
              "the rows show its value and, in parentheses, the value with the fix-up at step 18, "
              "where dfthcbe and we apply it. The spin multiplexer is 0 in Table III and 1 per "
              "application in App. B.3 and B.7.1 (4(N + 1) for the four swaps).",
              "", "### Qubits (spec 10.3, 13.2)", "",
              f"Ours: {system['wires']:,} wires from `WalkLayout`; the paper: "
              f"{costs_reference.PAPER_QUBITS_FEMOCO:,} (its Table 3, under the names of our "
              f"groups: the persistent ancillas of each subroutine and, as `work`, its largest "
              f"temporary count, the outer prepare's). The junk registers ({junk['junk_x_o']:,} "
              f"and {junk['junk_b']:,} wires, aliased onto the first {max(junk.values()):,} angle "
              "wires) are X-measured right after each prepare (spec 16, "
              "Q-1 variant (b)); the paper shares only its inner junk (313 temporaries) with its "
              "angles and adds the outer prepare's 110 temporaries.", "",
              "| Group | Ours | Paper | Ours - paper |", "| --- | ---: | ---: | ---: |",
              *qubit_rows,
              f"| **total** | **{system['wires']:,}** | **{sum(paper.values()):,}** | "
              f"**{system['wires'] - sum(paper.values()):+,}** |", "",
              "By group: `outer` holds the dirty comparator and uniform-preparation work "
              "registers (20 wires, which make the comparator adjoint free, `cmp-free`) and the "
              "keep test's `less_than_x_o`; `inner` holds the dirty work registers (16 wires), "
              "and its `less_than_b`, `sgn`, `d_sgn` and `r` take 7 wires against the paper's "
              "11 for G, r and c; `select` adds the flags `G0`, `G1`, `b_neq_B` and `sg` to the "
              "angles; the shared `work` pool (the controlled `RefAB`'s 44 wires, the largest "
              "need) takes the place of the paper's 110 outer temporaries, because our outer "
              "junk shares the angle register like the inner junk.", "",
              "### QPE at FeMoco-54 (spec 11.3)", "",
              "| t | T | M | A | Qubits |", "| ---: | ---: | ---: | ---: | ---: |"]
    for bits in (1, 2, 3, 10, 20, 23):
        qpe = system["qpe"][str(bits)]
        lines.append(f"| {bits} | {qpe['toffoli']:,} | {qpe['measurements']:,} | "
                     f"{qpe['measured_and_uncomputes']:,} | {qpe['wires']:,} |")
    lines += ["", system["qpe_basis"], "", "### Deferred refinements (decided by the user)", "",
              "- Truncated or pruned unary iteration (Q-4): up to 100 Toffolis per walk "
              "(spec 13.1).",
              "- Keeping each alias swap's AND (the paper's remark after Eq. (127)): the swaps' "
              "undo becomes Clifford plus a measured uncompute, w_x + 2 w_b = 19 Toffolis less per "
              "walk for 14 more dirty qubits (ledger rows 2D.1 and 2D.2).",
              "- An SF fix-up one-hot of ob = w_b - 1 = 4 bits in `AngleErase`: 35 instead of 41 "
              "Toffolis per erase, 12 less per walk and per controlled walk, no extra qubits "
              "(ledger row 2E.2).", ""]
    return lines


def large_compiles(report: dict) -> list[str]:
    """The large-compile log: every refused, killed or failed entry of the current code hash
    with its projection, and the count of entries kept as history (other code hashes)."""
    hash_, lines, history = report["code_hash"], [], 0
    for key, entries in report["compiled"].items():
        history += sum(e["code_hash"] != hash_ for e in entries)
        for e in entries:
            if e["code_hash"] == hash_ and e["status"] != "compiled":
                projection = e.get("projection") or {}
                lines.append(f"| `{key}` | {e['status']} | {projection.get('seconds', '—')} | "
                             f"{projection.get('peak_mb', '—')} | "
                             f"{projection.get('reason', e.get('reason', ''))} |")
    out = ["## Large compiles", "", f"Entries of other code hashes kept as history, never shown "
           f"as current measurements: {history}.", ""]
    if lines:
        out += ["| Program | Status | Projected s | Projected MB | Reason |",
                "| --- | --- | ---: | ---: | --- |", *lines, ""]
    return out


def load(out: Path) -> dict:
    path = out / JSON_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--large", choices=LARGE + TEST_SYSTEMS + SMALL)
    parser.add_argument("--rows", nargs="*")
    parser.add_argument("--out", type=Path, default=REPO / "docs")
    parser.add_argument("--compile-one", nargs=2, metavar=("SYSTEM", "ROW"))
    args = parser.parse_args(argv)
    if args.compile_one:
        s = build_system(args.compile_one[0])
        print(PREFIX + json.dumps(compile_counts(s, _find_row(s, args.compile_one[1]))),
              flush=True)
        return 0
    hash_ = code_hash()
    old = load(args.out)
    cache = old.get("compiled", {})
    systems = list(dict.fromkeys(args.systems + ([args.large] if args.large else [])))
    report = {"code_hash": hash_, "environment": environment(), "systems": {}, "log": []}
    errors = []
    for name in sorted(systems, key=lambda n: (SYSTEMS + TEST_SYSTEMS).index(n)):
        s = build_system(name)
        rows = build_rows(s)
        report["log"] += run_system(s, rows, cache, hash_, projected=name == args.large,
                                    only=None if args.rows is None else set(args.rows))
        record, failed = report_system(s, rows, cache, hash_)
        report["systems"][name] = record
        errors += failed
    report["compiled"] = cache
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    (args.out / MD_NAME).write_text(render(report))
    for line in report["log"]:
        print(line)
    for error in errors:
        print("CHECK FAILED:", error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
