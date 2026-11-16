"""WalkLayout: named wires, groups, the junk aliases, the work pool and the split helpers (spec
section 10)."""

from itertools import pairwise

import jax.numpy as jnp
import pytest
from pennylane.wires import Wires

from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams
from dfthc.registers import GROUPS, JUNK, WalkLayout
from dfthc.testing.walk_sim import FAILS_C3

PRESETS = [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE]


def test_femoco_group_sizes():
    """Spec 10.3 with the junk registers on the angle wires (spec 16, Q-1 variant (b)): 1,102
    wires, 462 fewer than the 1,564 of separate junk registers (72 outer, 390 inner)."""
    layout = WalkLayout.from_params(FEMOCO54)
    sizes = {name: len(getattr(layout, name)) for name in
             ("system", "phase_grad", "outer", "inner", "select", "work")}
    assert sizes == {"system": 108, "phase_grad": 15, "outer": 71, "inner": 65,
                     "select": 799, "work": 44}
    assert layout.num_wires == 1102 == 1564 - 72 - 390


def test_femoco_register_sizes_follow_spec_9_7_and_9_8():
    sizes = WalkLayout.register_sizes(FEMOCO54)
    assert (sizes["succ_work_x_o"], sizes["cmp_work_x_o"]) == (6, 14)
    assert (sizes["succ_work_b"], sizes["cmp_work_b"]) == (2, 14)
    assert (sizes["x_o"], sizes["b"], sizes["r"], sizes["angles"]) == (9, 5, 4, 795)
    assert WalkLayout.junk_sizes(FEMOCO54) == {"junk_x_o": 72, "junk_b": 390}
    assert sizes["junk_overflow"] == 0 and not set(JUNK) & set(sizes)


def test_h2_sto3g_layout():
    """55 wires: the 4 and 7 junk wires share the 4 angle wires and 3 overflow wires."""
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    assert layout.num_wires == 55
    assert len(layout.work) == 8 == WalkLayout.work_size(H2_STO3G_TEST)
    assert len(layout.succ_work_x_o) == 0  # X_o = 4 has odd part 1: a zero-width register
    assert (len(layout.angles), len(layout.junk_overflow)) == (4, 3)


R_1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                 ke1=1, ke2=1)
"""``R = 1``: ``r`` has no wires, so ``inner`` ends with a zero-width register."""

PINNED = {
    "H2_STO3G_TEST": (H2_STO3G_TEST, [0, 2, 4, 6, 7, 7, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 23,
                                      24, 25, 27, 29, 30, 31, 32, 36, 39, 40, 41, 42, 43, 47, 55],
                      {"junk_x_o": (32, 36), "junk_b": (32, 39)}),
    "FAILS_C3": (FAILS_C3, [0, 2, 4, 8, 9, 12, 13, 14, 15, 15, 19, 20, 22, 23, 24, 25, 27, 28, 29,
                            31, 33, 34, 35, 37, 41, 41, 42, 43, 44, 45, 49, 58],
                 {"junk_x_o": (37, 37), "junk_b": (37, 37)}),
    "R_1": (R_1, [0, 2, 4, 6, 7, 7, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 23, 24, 25, 27, 29, 30,
                  31, 31, 34, 37, 38, 39, 40, 41, 44, 52],
            {"junk_x_o": (31, 35), "junk_b": (31, 37)}),
}
"""Per layout: the first wire of every register of :data:`GROUPS` in layout order, then the total;
and the junk aliases. ``H2_STO3G_TEST`` and ``R_1`` have 3 ``junk_overflow`` wires."""


@pytest.mark.parametrize("name", PINNED)
def test_wire_numbers_are_pinned(name):
    params, starts, junk = PINNED[name]
    layout = WalkLayout.from_params(params)
    names = [register for group in GROUPS.values() for register in group]
    bounds = dict(zip(names, pairwise(starts), strict=True))
    for register, (start, stop) in bounds.items():
        wires = getattr(layout, register)
        assert isinstance(wires, Wires) and wires == Wires(range(start, stop)), register
    for group, registers in GROUPS.items():
        wires = getattr(layout, group)
        expected = range(bounds[registers[0]][0], bounds[registers[-1]][1])
        assert isinstance(wires, Wires) and wires == Wires(expected), group
    for register, (start, stop) in junk.items():
        assert getattr(layout, register) == Wires(range(start, stop)), register
    assert layout.num_wires == starts[-1]


def test_work_size_formula():
    # max(w_x + b_k1 + w_b + b_k2, b_rot - 1, w_x + w_b,
    #     lg(l_x) + 2 s_u - 1, lg(l_b) + 2 s_u - 1, 2)
    assert WalkLayout.work_size(FEMOCO54) == max(44, 14, 14, 20, 16, 2) == 44
    assert WalkLayout.work_size(H2_STO3G_TEST) == max(8, 3, 4, 5, 7, 2) == 8
    assert WalkLayout.work_size(MID_NAPHTHALENE) == max(39, 14, 9, 18, 16, 2) == 39


@pytest.mark.parametrize("params", PRESETS)
def test_groups_never_overlap_and_cover_every_wire(params):
    layout = WalkLayout.from_params(params, qpe_bits=3)
    registers = [getattr(layout, name) for group in GROUPS.values() for name in group]
    registers += [layout.qpe_control, layout.qpe_grad]
    labels = [wire for wires in registers for wire in wires]
    assert sorted(labels) == list(range(layout.num_wires))
    groups = [layout.system, layout.outer, layout.inner, layout.select, layout.phase_grad,
              layout.work]
    assert sum(len(g) for g in groups) + 1 + 3 == layout.num_wires
    assert len(Wires.all_wires(groups)) == sum(len(g) for g in groups)


@pytest.mark.parametrize("params", PRESETS)
def test_junk_registers_alias_the_start_of_the_angle_wires(params):
    """Spec 10.2: each junk register is the first wires of angles ‖ junk_overflow, the start of
    the select group, and junk_overflow holds exactly the junk wires beyond the angles (none at
    FeMoco-54, H2_631G_TEST and MID_NAPHTHALENE; 3 at H2_STO3G_TEST); no group register holds a
    junk wire elsewhere."""
    layout = WalkLayout.from_params(params)
    shared = list(layout.angles) + list(layout.junk_overflow)
    junk = WalkLayout.junk_sizes(params)
    assert len(layout.junk_overflow) == max(*junk.values(), len(layout.angles)) - len(
        layout.angles)
    for name, size in junk.items():
        assert list(getattr(layout, name)) == shared[:size] == list(layout.select)[:size]
    others = {wire for group in GROUPS.values() for name in group
              if name not in ("angles", "junk_overflow") for wire in getattr(layout, name)}
    assert not others & set(shared)


@pytest.mark.parametrize("params", PRESETS)
def test_split_group_inverts_the_layout(params):
    layout = WalkLayout.from_params(params)
    for group, names in GROUPS.items():
        wires = getattr(layout, group)
        parts = WalkLayout.split_group(params, group, wires)
        assert list(parts) == list(names)
        assert all(parts[name] == getattr(layout, name) for name in names)
    assert WalkLayout.split_junk(params, layout.select) == {
        name: getattr(layout, name) for name in JUNK}


def test_split_helpers_slice_arrays_as_rules_see_them():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    parts = WalkLayout.split_group(H2_STO3G_TEST, "inner", jnp.asarray(layout.inner.tolist()))
    assert [p.tolist() for p in parts.values()] == [w.tolist() for w in
                                                   (getattr(layout, n) for n in GROUPS["inner"])]
    junk = WalkLayout.split_junk(H2_STO3G_TEST, jnp.asarray(layout.select.tolist()))
    assert [j.tolist() for j in junk.values()] == [layout.junk_x_o.tolist(),
                                                   layout.junk_b.tolist()]


def test_split_helpers_reject_a_wrong_length():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    with pytest.raises(ValueError, match="outer needs 12 wires; got 11"):
        WalkLayout.split_group(H2_STO3G_TEST, "outer", layout.outer[:-1])
    with pytest.raises(ValueError, match="unknown group"):
        WalkLayout.split_group(H2_STO3G_TEST, "angles", layout.angles)
    with pytest.raises(ValueError, match="select needs 11 wires; got 10"):
        WalkLayout.split_junk(H2_STO3G_TEST, layout.select[:-1])


def test_angle_register_is_a_slice_of_angles():
    layout = WalkLayout.from_params(FEMOCO54)
    for k in (0, 1, 52):
        assert layout.angle_register(k) == layout.angles[15 * k:15 * (k + 1)]
    with pytest.raises(ValueError, match="rung 53"):
        layout.angle_register(53)


def test_qpe_registers_only_when_requested():
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=3)
    assert layout.num_wires == 55 + 1 + 3
    assert layout.qpe_control == Wires([55]) and layout.qpe_grad == Wires([56, 57, 58])
    with pytest.raises(AttributeError):
        _ = WalkLayout.from_params(H2_STO3G_TEST).qpe_control
    with pytest.raises(ValueError, match="qpe_bits"):
        WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=-1)
