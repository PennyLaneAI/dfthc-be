"""WalkLayout: named wires, groups, the work pool and the split helper (spec section 10)."""

import jax.numpy as jnp
import pytest
from pennylane.wires import Wires

from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import GROUPS, WalkLayout

PRESETS = [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE]


def test_femoco_group_sizes():
    layout = WalkLayout.from_params(FEMOCO54)
    sizes = {name: len(getattr(layout, name)) for name in
             ("system", "phase_grad", "outer", "inner", "select", "work")}
    assert sizes == {"system": 108, "phase_grad": 15, "outer": 143, "inner": 455,
                     "select": 799, "work": 44}
    assert layout.num_wires == 1564


def test_femoco_register_sizes_follow_spec_9_7_and_9_8():
    sizes = WalkLayout.register_sizes(FEMOCO54)
    assert (sizes["succ_work_x_o"], sizes["cmp_work_x_o"], sizes["junk_x_o"]) == (6, 14, 72)
    assert (sizes["succ_work_b"], sizes["cmp_work_b"], sizes["junk_b"]) == (2, 14, 390)
    assert (sizes["x_o"], sizes["b"], sizes["r"], sizes["angles"]) == (9, 5, 4, 795)


def test_h2_sto3g_layout():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    assert layout.num_wires == 63
    assert len(layout.work) == 8 == WalkLayout.work_size(H2_STO3G_TEST)
    assert len(layout.succ_work_x_o) == 0  # X_o = 4 has odd part 1: a zero-width register


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
def test_split_group_inverts_the_layout(params):
    layout = WalkLayout.from_params(params)
    for group, names in GROUPS.items():
        wires = getattr(layout, group)
        parts = WalkLayout.split_group(params, group, wires)
        assert list(parts) == list(names)
        assert all(parts[name] == getattr(layout, name) for name in names)


def test_split_group_slices_arrays_as_rules_see_them():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    parts = WalkLayout.split_group(H2_STO3G_TEST, "inner", jnp.asarray(layout.inner.tolist()))
    assert [p.tolist() for p in parts.values()] == [w.tolist() for w in
                                                   (getattr(layout, n) for n in GROUPS["inner"])]


def test_split_group_rejects_a_wrong_length():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    with pytest.raises(ValueError, match="outer needs 16 wires; got 15"):
        WalkLayout.split_group(H2_STO3G_TEST, "outer", layout.outer[:-1])
    with pytest.raises(ValueError, match="unknown group"):
        WalkLayout.split_group(H2_STO3G_TEST, "angles", layout.angles)


def test_angle_register_is_a_slice_of_angles():
    layout = WalkLayout.from_params(FEMOCO54)
    for k in (0, 1, 52):
        assert layout.angle_register(k) == layout.angles[15 * k:15 * (k + 1)]
    with pytest.raises(ValueError, match="rung 53"):
        layout.angle_register(53)


def test_qpe_registers_only_when_requested():
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=3)
    assert layout.num_wires == 63 + 1 + 3
    assert layout.qpe_control == Wires([63]) and layout.qpe_grad == Wires([64, 65, 66])
    with pytest.raises(AttributeError):
        _ = WalkLayout.from_params(H2_STO3G_TEST).qpe_control
    with pytest.raises(ValueError, match="qpe_bits"):
        WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=-1)
