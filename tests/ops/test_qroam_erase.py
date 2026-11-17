"""QROAMErase (spec 9.5): counts, rule selection, lightning smoke runs, mcm_parities linearity."""

import dataclasses
import math
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    PENNYLANE_PINS,
    compiled_resources,
    mcm_parities,
    pack_masks,
    packed_parities,
)
from dfthc.ops.qroam import QROAMErase, QROAMRead, _qroam_erase
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program


def random_table(widths, live_ranges, width, seed=11, zero=()):
    """An entry of ``width`` random bits at every live address (one or two registers), zero at
    the addresses in ``zero`` (``range(2**k)``, the first block, gives zero masks)."""
    rng = np.random.default_rng(seed)
    (lo1, hi1), *rest = live_ranges
    addresses = ([(v << widths[1]) | u for v in range(lo1, hi1) for u in range(*rest[0])]
                 if rest else list(range(lo1, hi1)))
    table = []
    for address in addresses:
        value = int(rng.integers(0, 2**width)) * (address not in zero)
        table.append((address, tuple((value >> (32 * w)) & (2**32 - 1)
                                     for w in range(-(-width // 32)))))
    return tuple(table)


def erase_shapes(p):
    """(widths, live ranges, width, block bits, one-hot bits) of the outer and inner erases."""
    return [((p.w_x,), ((0, p.X_o),), p.d1, p.k1, p.ke1),
            ((p.w_x, p.w_b), ((0, p.RC), (0, p.B + 1)), p.d2, p.k2, p.ke2 + p.w_b)]


def layout(shape, control_value=None):
    widths, live, width, k, ob = shape
    erase = QROAMErase.required_wires((), widths, live, width, k, control_value, ob)
    read_work = QROAMRead.required_wires(widths, live, width, k, control_value)["work"]
    sizes = dict(erase, work=max(erase["work"], read_work))
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def make_read(table, shape, control_value, wires):
    widths, live, width, k, _ = shape
    work = wires["work"][:QROAMRead.required_wires(widths, live, width, k,
                                                   control_value)["work"]]
    columns = QROAMRead.columns_from_table(table, address_widths=widths, live_ranges=live,
                                           width=width, block_bits=k)
    return QROAMRead(columns, widths, live, width, k, control_value, wires["address"],
                     wires["target"], wires["junk"], work, wires["control"])


def make_erase(table, shape, control_value, wires):
    widths, live, width, k, ob = shape
    work = wires["work"][:QROAMErase.required_wires((), widths, live, width, k, control_value,
                                                    ob)["work"]]
    return QROAMErase(table, widths, live, width, k, control_value, ob, wires["address"],
                      wires["target"], wires["junk"], work, wires["control"],
                      wires["extra_onehot"])


def expected_cost(shape, control_value):
    widths, live, width, k, ob = shape
    return costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width, ob,
                             control_value is not None)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms: outer T = 22 + 14, M = 96; inner T = 70 + 126, M = 416."""
    outer, inner = erase_shapes(FEMOCO54)
    assert QROAMErase.fixup_high_ranges(*outer[:2], outer[3], outer[4]) == [(0, 21, 5)]
    assert QROAMErase.fixup_high_ranges(*inner[:2], inner[3], inner[4]) == [(0, 68, 7)]
    assert expected_cost(outer, None) == costs.Cost(36, 96)
    assert expected_cost(inner, None) == costs.Cost(196, 416)
    assert QROAMErase.required_wires((), *outer[:3], outer[3], None, outer[4]) == {
        "address": 9, "target": 24, "junk": 72, "work": 4, "control": 0, "extra_onehot": 0}
    assert QROAMErase.required_wires((), *inner[:3], inner[3], None, inner[4]) == {
        "address": 14, "target": 26, "junk": 390, "work": 6, "control": 0, "extra_onehot": 0}


SMALL = erase_shapes(H2_STO3G_TEST) + erase_shapes(H2_631G_TEST)
SMALL_IDS = ["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner"]


@pytest.mark.compiled
@pytest.mark.parametrize("shape", SMALL, ids=SMALL_IDS)
def test_counts_match_the_closed_form(shape):
    """T4 and the T6 signature: M = K width, T = the erase closed form, A = T (every AND of the
    fixup is uncomputed by measurement, and the erase has no plain Toffoli)."""
    _assert_compiled_counts(shape)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("shape", erase_shapes(MID_NAPHTHALENE), ids=["outer", "inner"])
def test_counts_at_mid_naphthalene(shape):
    """T4 at MID_NAPHTHALENE (nightly tier); compiled_resources rejects symbolic counts."""
    _assert_compiled_counts(shape)


def _assert_compiled_counts(shape):
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape)
    resources = compiled_resources(lambda: make_erase(table, shape, None, wires),
                                   num_wires=num_wires)
    expected = expected_cost(shape, None)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == 2**k * width
    assert resources.measured_and_uncomputes == resources.toffoli


def test_fixup_high_ranges_agree_with_the_cost_model():
    """QROAMErase.fixup_high_ranges is the oracle of dfthc.costs' erase ranges: the outer and inner
    unprepare costs equal w + qroam_erase(fixup_high_ranges) + F^dag, also where ke1 < k1 and K1
    does not divide X_o, so the outer erase covers dead positions beyond X_o."""
    shapes = [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE] + [
        dataclasses.replace(H2_STO3G_TEST, N=n, n_plus=1, R=1, C=2, k1=k1, ke1=ke1)
        for n in range(3, 11) for k1 in range(4) for ke1 in range(4)]
    for p in shapes:
        for shape, unprepare, w, L in zip(erase_shapes(p),
                                          (costs.outer_unprepare, costs.inner_unprepare),
                                          (p.w_x, p.w_b), (p.X_o, p.B + 1), strict=True):
            widths, live, width, k, ob = shape
            erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width,
                                      ob, False)
            expected = costs.Cost(w) + erase + costs.flagged_uniform_unprep(L, p.s_u)
            assert unprepare(p) == expected, (p.X_o, p.k1, p.ke1)


@pytest.mark.compiled
def test_erase_is_not_a_second_read_and_its_conditionals_are_hinted(capfd):
    """T6: the chosen rule is the pinned measured erase, its T differs from a re-read, and its
    conditional CZs count their hinted expectation (one per nonzero mask; the first block is
    zero, so its masks emit none)."""
    shape = SMALL[3]
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width, zero=range(2**k))
    wires, num_wires = layout(shape)
    resources = compiled_resources(lambda: make_erase(table, shape, None, wires),
                                   num_wires=num_wires)
    read = costs.qroam_read(*QROAMRead.select_size(widths, live, k, None), k, width)
    assert resources.toffoli != read.toffoli
    masks = QROAMRead.loaded_contents(table, QROAMRead.loaded_set(widths, live, k),
                                      address_widths=widths, live_ranges=live, width=width,
                                      block_bits=k)
    nonzero = sum(1 for m in masks if m)
    assert 0 < nonzero < len(masks)
    assert resources.counts["CZ"] == math.ceil(nonzero / 2)
    qp.analyze(verbose_program(lambda: make_erase(table, shape, None, wires), num_wires),
               level="user")()
    captured = capfd.readouterr()
    assert "_qroam_erase" in chosen_rules(captured.out + captured.err)


@pytest.mark.compiled
def test_counts_with_enable_and_extra_onehot():
    """An enabled erase (c = 1 in U) whose one-hot needs 2**3 - 4 extra wires."""
    shape = ((3, 2), ((0, 2), (0, 3)), 1, 2, 3)
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape, control_value=1)
    assert len(wires["extra_onehot"]) == 4
    resources = compiled_resources(lambda: make_erase(table, shape, 1, wires),
                                   num_wires=num_wires)
    expected = expected_cost(shape, 1)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 4)


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", [qp.adjoint, lambda op: qp.ctrl(op, control=20)],
                         ids=["adjoint", "control"])
def test_erase_has_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rule 5: never adjointed or controlled. Catalyst drops the measuring candidate
    with the expected warning (rule 12), and without another rule the compile fails loudly."""
    shape = SMALL[0]
    widths, live, width, _, _ = shape
    table = random_table(widths, live, width)
    wires, _ = layout(shape)
    key = "Adjoint(QROAMErase)" if wrap is qp.adjoint else "C(QROAMErase)"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(lambda: wrap(make_erase(table, shape, None, wires)), num_wires=21)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the _qroam_erase")]
    assert skipped and {classify_rule_warning(m) for m in skipped} == {"expected"}


def _phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


@pytest.mark.compiled
def test_lightning_smoke_runs_with_five_seeds():
    """Compiled read then erase (k = 2, one-hot bits 2, first block zero) restores a phased
    address state and clears every ancilla for five measurement seeds."""
    shape = ((3,), ((0, 5),), 2, 2, 2)
    widths, live, width, k, _ = shape
    table = random_table(widths, live, width, zero=range(2**k))
    wires, num_wires = layout(shape)
    address = wires["address"]

    def prepared():
        _phased_prepare(address)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps={**FIXED_DECOMPS, **PENNYLANE_PINS})
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def reference():
        prepared()
        return qp.state()

    expected = np.asarray(reference())
    for seed in range(5):
        @qp.qjit(capture=True, seed=seed)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps={**FIXED_DECOMPS, **PENNYLANE_PINS})
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def round_trip():
            prepared()
            make_read(table, shape, None, wires)
            make_erase(table, shape, None, wires)
            return qp.state()

        assert np.allclose(np.asarray(round_trip()), expected, rtol=0, atol=1e-10), seed


def _inner_masks(p, seed=3):
    """The inner erase's masks over its loaded set, for a random table of preset p's shape whose
    first block is zero."""
    widths, live, width, k, _ = erase_shapes(p)[1]
    table = random_table(widths, live, width, seed, zero=range(2**k))
    masks = QROAMRead.loaded_contents(table, QROAMRead.loaded_set(widths, live, k),
                                      address_widths=widths, live_ranges=live, width=width,
                                      block_bits=k)
    return masks, 2**k * width


def test_mcm_parities_capture_core_is_linear_at_the_femoco_inner_erase():
    """Spec 12.5 linearity at the FeMoco-54 inner-erase shape (8,640 masks over 416 outcomes;
    a classical computation, nothing is compiled): f_0 = 0 and f_(x ^ y) = f_x ^ f_y."""
    masks, num_outcomes = _inner_masks(FEMOCO54)
    assert (len(masks), num_outcomes) == (270 * 32, 416)
    packed = pack_masks(masks, num_outcomes)
    rng = np.random.default_rng(12)
    assert not np.asarray(packed_parities(jnp.zeros(num_outcomes, dtype=jnp.uint32), packed)).any()
    for _ in range(100):
        x, y = rng.integers(0, 2, size=(2, num_outcomes))
        fx, fy = (np.asarray(packed_parities(v, packed)) for v in (x, y))
        assert np.array_equal(np.asarray(packed_parities(x ^ y, packed)), fx ^ fy)


def test_mcm_parities_tape_mode_is_linear_at_a_test_inner_erase():
    """The tape implementation (flat MeasurementValues) at the H2_631G_TEST inner erase."""
    masks, num_outcomes = _inner_masks(H2_631G_TEST)
    with qp.queuing.AnnotatedQueue():
        outcomes = [qp.measure(w) for w in range(num_outcomes)]
        parities = mcm_parities(outcomes, masks)

    def evaluate(bits):
        record = {outcomes[i].measurements[0]: int(bits[i]) for i in range(num_outcomes)}
        return np.array([0 if p is False else int(p.concretize(record)) for p in parities])

    zero = [p for p, m in zip(parities, masks, strict=True) if m == 0]
    assert zero and all(p is False for p in zero)
    rng = np.random.default_rng(13)
    assert not evaluate(np.zeros(num_outcomes, dtype=int)).any()
    for _ in range(100):
        x, y = rng.integers(0, 2, size=(2, num_outcomes))
        assert np.array_equal(evaluate(x ^ y), evaluate(x) ^ evaluate(y))


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    shape = SMALL[1]
    widths, live, width, _, _ = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape)
    compiled_resources(lambda: make_erase(table, shape, None, wires), num_wires=num_wires,
                       level="device")


@pytest.mark.parametrize(("shape", "control_value"),
                         [(SMALL[0], None), (((3, 3), ((0, 2), (0, 6)), 3, 2, 3), 1)])
def test_rule_is_valid_in_tape_mode(shape, control_value):
    """T2 in tape mode: the measured outcomes and their parities are MeasurementValues; the first
    block is zero, so the counting twin's zero masks declare no conditional."""
    widths, live, width, k, _ = shape
    table = random_table(widths, live, width, zero=range(2**k))
    wires, _ = layout(shape, control_value)
    qp.testing.assert_valid_decomp_rule(make_erase(table, shape, control_value, wires),
                                        _qroam_erase, check_matrix=False)


def test_required_wires_and_validation():
    assert QROAMErase.required_wires((), (3,), ((0, 5),), 1, 0, None, 2) == {
        "address": 3, "target": 1, "junk": 0, "work": 0, "control": 0, "extra_onehot": 3}
    args = {"table": (), "address_widths": (3,), "live_ranges": ((0, 5),), "width": 1,
            "block_bits": 0, "control_value": None, "onehot_bits": 2, "address": [0, 1, 2],
            "target": [3], "junk": [], "work": [], "control": [], "extra_onehot": [4, 5, 6]}
    QROAMErase(**args)
    with pytest.raises(ValueError, match=r"onehot_bits must lie in \[0, 3\]"):
        QROAMErase(**(args | {"onehot_bits": 4}))
    with pytest.raises(ValueError, match="'extra_onehot' has 2 wires; expected 3"):
        QROAMErase(**(args | {"extra_onehot": [4, 5]}))
