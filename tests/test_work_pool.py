"""The walk's shared work pool (spec 10.1 and 10.2): every call of the four assembly rules (the
walks' and the block encodings') gets exactly its required_wires, its work is a prefix of the
pool, and neither the pool nor the comparators' dirty work registers appear where they do not
belong; the junk registers and the unprepares' extra_onehot wires are the start of the angle
wires; WalkLayout.work_size is the largest need, the controlled RefAB's."""

import functools
import inspect
import random

import pennylane as qp
import pytest

from dfthc.compile import dfthc_operator_names
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.walk import (
    FROM_OUTCOMES,
    JUNK_MEASUREMENT,
    _block_encoding,
    _children,
    _controlled_block_encoding,
    _controlled_walk,
    _walk,
    _walk_children,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams, lg
from dfthc.registers import JUNK, WalkLayout

PRESETS = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "mid": lambda: (MID_NAPHTHALENE, load_factors("naphthalene")),
    "femoco": lambda: (FEMOCO54, random_factors(54, 10, 27, 27, n_plus=27, seed=0)),
}
"""FeMoco-54 is classical here: the rules run in tape mode and construct their children (the
constructors check every register); nothing is compiled."""

DIRTY = {"succ_work_x_o": {"OuterPrepare", "OuterUnprepareFromOutcomes"},
         "cmp_work_x_o": {"OuterPrepare", "OuterUnprepareFromOutcomes"},
         "succ_work_b": {"InnerPrepare", "InnerUnprepareFromOutcomes"},
         "cmp_work_b": {"InnerPrepare", "InnerUnprepareFromOutcomes"}}
"""The comparators' work registers, dirty from a prepare to its unprepare (spec 10.2), and the
only classes that may take them."""

AT_LEAST = {"OuterPrepare", "OuterUnprepareFromOutcomes", "InnerPrepare",
            "InnerUnprepareFromOutcomes"}
"""Their ``phase_grad`` is a minimum (``s_u``); the walk passes the whole register."""

LENDS = {cls.__name__: junk for cls, junk in FROM_OUTCOMES.items()}
"""The unprepares from outcomes and the junk register whose start is their ``extra_onehot``."""

RULES = {"block encoding": (_block_encoding, False, 11),
         "controlled block encoding": (_controlled_block_encoding, True, 11),
         "walk": (_walk, False, 2), "controlled walk": (_controlled_walk, True, 2)}
"""Each assembly rule, whether it is controlled, and its number of operator calls."""


@functools.cache
def preset(name):
    p, factors = PRESETS[name]()
    return p, WalkTables.from_factors(factors, p), WalkLayout.from_params(p, qpe_bits=1)


def required(op) -> dict:
    """``required_wires`` of the call: from its compilable arguments, or, for the reflections,
    whose sizes follow from their wires, from the widths of its pattern registers."""
    cls = type(op)
    if cls in (RefT2, ControlledRefT2):
        return cls.required_wires(len(op.b), len(op.m_b))
    if cls in (RefAB, ControlledRefAB):
        return cls.required_wires(len(op.x_o), len(op.m_x_o), len(op.b), len(op.m_b))
    names = inspect.signature(cls.required_wires).parameters
    return cls.required_wires(**{k: v for k, v in op.hyperparameters.items() if k in names})


def calls(rule, p, t, layout, controlled):
    """The rule's operator calls in tape mode, on the layout's groups (``qpe_control`` as
    control); the junk measurements' gates are left out."""
    groups = {name: getattr(layout, name)
              for name in ("outer", "inner", "select", "system", "phase_grad", "work")}
    if controlled:
        groups = {"control": layout.qpe_control} | groups
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(p.as_compilable(), t.as_compilable(), **groups)
    return [op for op in queue.queue if op.name in dfthc_operator_names()]


def check_call(op, layout) -> int:
    """Assert the pool discipline for one call and return its work need."""
    sizes = required(op)
    pool = set(layout.work)
    registers = {name: list(getattr(op, name)) for name in op.wire_argnames}
    assert set(registers) == set(sizes), op.name
    for name, wires in registers.items():
        if name == "phase_grad" and op.name in AT_LEAST:
            assert wires == list(layout.phase_grad) and len(wires) >= sizes[name], op.name
        else:
            assert len(wires) == sizes[name], (op.name, name)
        if name == "work":
            assert wires == list(layout.work)[:len(wires)], op.name
        else:
            assert not pool & set(wires), (op.name, name)
        if name in DIRTY:
            assert op.name in DIRTY[name] and wires == list(getattr(layout, name))
        if name in JUNK:
            assert wires == list(getattr(layout, name)), (op.name, name)
        if name == "extra_onehot":
            assert wires == list(getattr(layout, LENDS[op.name]))[:len(wires)], op.name
    work = set(registers.get("work", ()))
    others = {wire for name, wires in registers.items() if name != "work" for wire in wires}
    dirty = {wire for name in DIRTY for wire in getattr(layout, name)}
    assert not work & others and not work & dirty, op.name
    return sizes.get("work", 0)


@pytest.mark.parametrize("name", PRESETS)
def test_every_call_gets_exactly_its_required_wires(name):
    """Spec 10.2 and master plan Tasks 3.3 and 3.4: in the four assembly rules every child
    receives exactly required_wires (phase_grad of the prepare classes is a minimum: they get the
    whole register; the walks' block encoding gets the whole groups and pool), its work is the
    pool's prefix of that size and disjoint from its other registers, no other register holds a
    pool wire, the dirty comparator work registers go only to their prepare and unprepare (never
    into a work prefix), the junk registers and the unprepares' extra_onehot are the start of
    the angle wires, and every need fits the pool."""
    p, t, layout = preset(name)
    for rule, controlled, count in RULES.values():
        needs = [check_call(op, layout) for op in calls(rule, p, t, layout, controlled)]
        assert len(needs) == count and max(needs) <= len(layout.work)


@pytest.mark.parametrize("name", PRESETS)
def test_the_pool_is_the_controlled_walks_largest_need(name):
    """WalkLayout.work_size is the largest need of the controlled walk's operators below the
    block encoding: the controlled RefAB's w_x + b_k1 + w_b + b_k2 at every preset (44 at
    FeMoco-54, spec 10.3); the uncontrolled walk's RefAB needs one wire less."""
    p, t, layout = preset(name)
    needs = {}
    for rule, controlled, _ in RULES.values():
        for op in calls(rule, p, t, layout, controlled):
            if op.name not in ("DFTHCBlockEncoding", "ControlledDFTHCBlockEncoding"):
                needs[op.name] = max(needs.get(op.name, 0), check_call(op, layout))
    assert max(needs.values()) == needs["ControlledRefAB"] == len(layout.work)
    assert needs["ControlledRefAB"] == WalkLayout.work_size(p) == p.w_x + p.b_k1 + p.w_b + p.b_k2
    assert needs["RefAB"] == needs["ControlledRefAB"] - 1


def own_sizes(cls, args, p) -> dict:
    """A child's own ``required_wires`` from the compilable arguments the rule gives it, or, for
    the reflections (sizes from their wires), from the pattern widths."""
    if cls in (RefT2, ControlledRefT2):
        return cls.required_wires(p.w_b, p.b_k2)
    if cls in (RefAB, ControlledRefAB):
        return cls.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)
    arguments = dict(zip(cls.compilable_argnames, args, strict=True))
    names = inspect.signature(cls.required_wires).parameters
    return cls.required_wires(**{k: v for k, v in arguments.items() if k in names})


def zero_tables(p) -> WalkTables:
    """Tables of ``p``'s shapes with every entry 0: the work needs depend on the shapes only."""
    def rows(n, m):
        return tuple((0,) * m for _ in range(n))

    inner = rows(p.RC, p.B + 1)
    return WalkTables(outer_alt=(0,) * p.X_o, outer_keep=(0,) * p.X_o, inner_alt=inner,
                      inner_keep=inner, inner_sgn=inner, inner_d_sgn=inner,
                      tail_angles=rows(p.N, p.N - 1),
                      sf_angles=tuple(rows(p.B, p.N - 1) for _ in range(p.R)))


CHILDREN = {"OuterPrepare", "InnerPrepare", "InnerSigns", "InnerSelect",
            "InnerUnprepareFromOutcomes", "RefT2", "ControlledRefT2", "InnerSelectDagger",
            "OuterUnprepareFromOutcomes", "DFTHCBlockEncoding", "ControlledDFTHCBlockEncoding",
            "RefAB", "ControlledRefAB"}
"""Every class the four assembly rules call."""


def test_work_size_covers_every_child_at_random_shapes():
    """At 300 random valid shapes, every child of the four assembly rules, taken from the rules'
    own step lists (``_children`` and ``_walk_children``, plain and controlled), is given exactly
    its class's own ``required_wires`` work (and an unprepare from outcomes its own
    ``extra_onehot``, no more than its junk register holds), so no explicit inverse or controlled
    class (``OuterUnprepareFromOutcomes``, ``InnerUnprepareFromOutcomes``, ``InnerSelectDagger``,
    ``ControlledRefT2``, ``ControlledRefAB``) can drift from its forward form unnoticed;
    ``WalkLayout.work_size`` (spec 10.1, provisional until this test) covers every need; and each
    junk register fits in the angle wires and the overflow wires."""
    rng = random.Random(0)
    checked, seen = 0, set()
    while checked < 300:
        N, R, B, C = rng.randint(2, 12), rng.randint(1, 10), rng.randint(1, 30), rng.randint(1, 30)
        b_rot = rng.randint(3, 16)
        values = dict(N=N, R=R, B=B, C=C, n_plus=rng.randint(0, N), b_rot=b_rot,
                      b_k1=rng.randint(1, 16), b_k2=rng.randint(1, 16), s_u=rng.randint(3, b_rot))
        values |= dict(k1=rng.randint(0, lg(N + R * C)), k2=rng.randint(0, lg(B + 1)),
                       ke1=rng.randint(0, lg(N + R * C)), ke2=rng.randint(0, lg(R * C)))
        p = WalkParams(**values)
        if (2**p.ke1 > p.K1 * p.d1 or 2**(p.ke2 + p.w_b) > p.K2 * p.d2
                or 2**p.w_b > (p.N - 1) * p.b_rot + 1):
            continue
        t = zero_tables(p)
        sizes = WalkLayout.register_sizes(p)
        junk = WalkLayout.junk_sizes(p)
        assert max(junk.values()) <= sizes["angles"] + sizes["junk_overflow"], p
        for controlled in (False, True):
            for step in _children(p, t, controlled) + _walk_children(p, t, controlled):
                if step.op == JUNK_MEASUREMENT:
                    assert step.args in JUNK
                    continue
                own = own_sizes(step.op, step.args, p)
                assert own.get("work", 0) == step.work, (p, step.op.__name__)
                assert own.get("extra_onehot", 0) == step.extra_onehot, (p, step.op.__name__)
                if step.op in FROM_OUTCOMES:
                    assert step.extra_onehot <= junk[FROM_OUTCOMES[step.op]], p
                assert step.work <= WalkLayout.work_size(p), (p, step.op.__name__, step.work)
                seen.add(step.op.__name__)
        checked += 1
    assert seen == CHILDREN
