"""Tests for dfthc.compile and dfthc.testing.resources (plan Task 0.2)."""

import itertools
import time
import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from pennylane.core import Operator2
from pennylane.typing import Wire

import dfthc.compile as dc
from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    CompiledResources,
    compiled_resources,
    dfthc_operator_names,
    mcm_parities,
    pack_masks,
    packed_parities,
    register_rule,
    resource_key,
    surviving_dfthc_operators,
)
from dfthc.testing.resources import (
    NOISE_PATTERNS,
    assert_resources,
    chosen_rules,
    classify_rule_warning,
    install_warning_filters,
    verbose_program,
)


class CompileTestLoad(Operator2):
    """Copies ``ctrl`` into ``target``; its pinned adjoint erases ``target`` by measurement."""

    wire_argnames = ("ctrl", "target")

    def __init__(self, ctrl, target):
        super().__init__(ctrl, target)


@qp.register_resources({qp.CNOT: 1})
def _compile_test_load(ctrl, target):
    qp.CNOT([ctrl[0], target[0]])


def _compile_test_unload_resources(base):
    return {qp.H: 1, resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)): 1, qp.Z: 1}


@qp.register_resources(_compile_test_unload_resources)
def _compile_test_unload(base):
    qp.H(base.target[0])
    outcome = qp.measure(base.target[0], reset=True)
    qp.cond(outcome, qp.Z)(base.ctrl[0])


register_rule(CompileTestLoad, _compile_test_load)
register_rule("Adjoint(CompileTestLoad)", _compile_test_unload)


def test_gate_set_matches_spec():
    assert GATE_SET == {
        "PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "Adjoint(S)": 1,
        "CNOT": 1, "CZ": 1, "SWAP": 1, "GlobalPhase": 1, "MidMeasure": 1,
        "Adjoint(TemporaryAND)": 1, "PhaseGradientStatePrep": 1,
        "Toffoli": 1000, "TemporaryAND": 1000,
    }


def test_register_rule_pins_once():
    assert FIXED_DECOMPS["CompileTestLoad"] is _compile_test_load
    assert FIXED_DECOMPS["Adjoint(CompileTestLoad)"] is _compile_test_unload
    with pytest.raises(ValueError, match="already has a pinned rule"):
        register_rule(CompileTestLoad, _compile_test_load)


def test_registered_names_have_only_our_rules():
    for key, rule in FIXED_DECOMPS.items():
        assert list(qp.list_decomps(key)) == [rule], key


@pytest.mark.compiled
def test_pinned_measured_adjoint_is_selected():
    """The T6 pattern: the count signature proves the measured erase rule was chosen."""

    def qfunc():
        qp.H(0)
        CompileTestLoad([0], [1])
        qp.adjoint(CompileTestLoad([0], [1]))

    resources = compiled_resources(qfunc, num_wires=2)
    assert resources.counts == {
        "CNOT": 1, "Hadamard": 2, "MidCircuitMeasure": 1, "PauliX": 1, "PauliZ": 1,
    }
    assert_resources(resources, toffoli=0, measurements=1, measured_and_uncomputes=0)
    assert resources.physical_measurements == 1


@pytest.mark.compiled
def test_verbose_rule_listing_names_the_pinned_rules(capfd):
    """Spec 12.1 T6, checked in Phase 0: the solver's rule listing is capturable."""

    def qfunc():
        qp.H(0)
        CompileTestLoad([0], [1])
        qp.adjoint(CompileTestLoad([0], [1]))

    qp.analyze(verbose_program(qfunc, num_wires=2), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_compile_test_load", "_compile_test_unload"} <= rules


@pytest.mark.compiled
def test_compiled_resources_counts_toffoli_classes():
    resources = compiled_resources(
        lambda: qp.SemiAdder(x_wires=[0, 1, 2], y_wires=[3, 4, 5], work_wires=[6, 7]), num_wires=8
    )
    assert isinstance(resources, CompiledResources)
    assert resources.level == "user"
    assert (resources.toffoli, resources.measured_and_uncomputes) == (2, 2)


@pytest.mark.compiled
def test_compiled_resources_rejects_symbolic_counts():
    def qfunc():
        qp.H(0)
        bound = qp.measure(0)

        @qp.for_loop(bound)
        def loop(_):
            qp.X(1)

        loop()

    with pytest.raises(AssertionError, match="symbolic"):
        compiled_resources(qfunc, num_wires=2)


@pytest.mark.compiled
def test_compiled_resources_rejects_operations_outside_gate_set(monkeypatch):
    monkeypatch.setattr(dc, "GATE_SET", GATE_SET | {"RX": 1})
    with pytest.raises(AssertionError, match="outside the gate set"):
        compiled_resources(lambda: qp.RX(0.3, 0), num_wires=1)


def test_surviving_dfthc_operators_sees_wrapped_names():
    keys = ["OuterPrepare", "Adjoint(OuterPrepare)", "C(RefT2)"]
    names = ["OuterPrepare", "C(OuterPrepare)", "Adjoint(OuterPrepare)", "Adjoint(C(RefT2))",
             "OuterPrepareX", "CNOT", "C(PauliX)", "Adjoint(TemporaryAND)"]
    assert surviving_dfthc_operators(names, keys) == [
        "Adjoint(C(RefT2))", "Adjoint(OuterPrepare)", "C(OuterPrepare)", "OuterPrepare",
    ]


@pytest.mark.parametrize(
    "name", ["CompileTestLoad", "C(CompileTestLoad)", "Adjoint(CompileTestLoad)"]
)
def test_device_level_rejects_surviving_dfthc_operator(monkeypatch, name):
    """T5 at level="device" (L-20): a dfthc operator fails the check bare or wrapped."""
    reported = SimpleNamespace(is_symbolic=False, counts={name: 1, "CNOT": 2}, num_wires=2)
    monkeypatch.setattr(
        qp, "analyze", lambda circuit, level: lambda: SimpleNamespace(resources=reported)
    )
    with pytest.raises(AssertionError, match="dfthc operators reached"):
        compiled_resources(lambda: None, num_wires=2, level="device")


@pytest.mark.compiled
def test_device_level_closure_of_decomposed_program():
    def qfunc():
        qp.H(0)
        CompileTestLoad([0], [1])
        qp.adjoint(CompileTestLoad([0], [1]))

    resources = compiled_resources(qfunc, num_wires=2, level="device")
    assert resources.counts == {
        "CNOT": 1, "Hadamard": 2, "MidCircuitMeasure": 1, "PauliX": 1, "PauliZ": 1,
    }


@pytest.mark.compiled
def test_device_level_record_has_no_toffoli_or_measurement_counts():
    """The device pass rewrites TemporaryAND and its measured adjoint, so T, M and A are unset."""

    def qfunc():
        qp.TemporaryAND([0, 1, 2])
        qp.adjoint(qp.TemporaryAND([0, 1, 2]))

    resources = compiled_resources(qfunc, num_wires=3, level="device")
    assert resources.level == "device"
    assert (resources.toffoli, resources.measurements, resources.measured_and_uncomputes) == (
        None, None, None,
    )
    with pytest.raises(AssertionError, match="needs a level='user' record, got level='device'"):
        assert_resources(resources, toffoli=1, measurements=0)
    with pytest.raises(ValueError, match="needs a level='user' record"):
        _ = resources.physical_measurements


def test_resource_key_pauses_capture():
    qp.capture.enable()
    try:
        assert resource_key(qp.capture.enabled) is False
        key = resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3])))
    finally:
        qp.capture.disable()
    assert key.name == "Adjoint(TemporaryAND)"


def test_pack_masks_layout():
    assert pack_masks([1, 2**32 + 3], 40).tolist() == [[1, 0], [3, 1]]


def test_mask_beyond_outcomes_is_rejected():
    with pytest.raises(ValueError, match="beyond the 3 outcomes"):
        mcm_parities([0, 1, 0], [8])


def test_mcm_parities_zero_masks_need_no_outcomes_under_capture():
    qp.capture.enable()
    try:
        parities = mcm_parities([], [0, 0])
    finally:
        qp.capture.disable()
    assert len(parities) == 2 and all(parity is False for parity in parities)


def _reference(masks, outcome_bits):
    value = sum(int(bit) << i for i, bit in enumerate(outcome_bits))
    return [bin(mask & value).count("1") % 2 for mask in masks]


def test_mcm_parities_tape_mode_exhaustive():
    """All 2^5 outcome vectors: f equals the GF(2) reference, so f_0 = 0 and f is linear."""
    masks = [0b00000, 0b00001, 0b10110, 0b11111, 0b01000, 0b10001, 0b00110]
    with qp.queuing.AnnotatedQueue():
        outcomes = [qp.measure(wire) for wire in range(5)]
        parities = mcm_parities(outcomes, masks)
    assert parities[0] is False
    for bits in itertools.product([0, 1], repeat=5):
        record = {outcomes[i].measurements[0]: bits[i] for i in range(5)}
        got = [0 if p is False else int(p.concretize(record)) for p in parities]
        assert got == _reference(masks, bits), bits


def test_mcm_parities_tape_mode_wide_masks_are_fast():
    """300 outcomes with an all-ones mask: one flat MeasurementValue, not a 300-deep XOR chain."""
    rng = np.random.default_rng(300)
    num_outcomes = 300
    masks = [(1 << num_outcomes) - 1] + [
        sum(int(bit) << i for i, bit in enumerate(rng.integers(0, 2, num_outcomes)))
        for _ in range(2)
    ]
    start = time.perf_counter()
    with qp.queuing.AnnotatedQueue():
        outcomes = [qp.measure(wire) for wire in range(num_outcomes)]
        parities = mcm_parities(outcomes, masks)
    for _ in range(3):
        bits = rng.integers(0, 2, size=num_outcomes)
        record = {outcomes[i].measurements[0]: int(bits[i]) for i in range(num_outcomes)}
        assert [int(p.concretize(record)) for p in parities] == _reference(masks, bits)
    assert time.perf_counter() - start < 1.0


def test_mcm_parities_capture_core_is_linear():
    rng = np.random.default_rng(1234)
    num_outcomes = 70
    masks = [int(m) for m in rng.integers(0, 2**62, size=12)] + [0, 2**69 + 1]
    packed = pack_masks(masks, num_outcomes)
    zero = packed_parities(jnp.zeros(num_outcomes, dtype=jnp.uint32), packed)
    assert np.all(np.asarray(zero) == 0)
    for _ in range(100):
        xi, eta = rng.integers(0, 2, size=(2, num_outcomes))
        f_xi = np.asarray(packed_parities(xi, packed))
        f_eta = np.asarray(packed_parities(eta, packed))
        f_sum = np.asarray(packed_parities(xi ^ eta, packed))
        assert np.array_equal(f_sum, f_xi ^ f_eta)
        assert f_xi.tolist() == _reference(masks, xi)


@pytest.mark.compiled
def test_mcm_parities_in_compiled_circuit():
    """Deterministic outcomes (1, 1, 0) drive conditional X gates through capture-mode parities.

    The outcomes are not a palindrome, so reading mask bits in reverse order gives (0, 1, 1, 0).
    """
    dev = qp.device("lightning.qubit", wires=7)
    masks = [0b001, 0b011, 0b101, 0b000]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(dev)
    def circuit():
        qp.X(0)
        qp.X(1)
        outcomes = [qp.measure(wire, reset=True) for wire in range(3)]
        for target, parity in zip(range(3, 7), mcm_parities(outcomes, masks), strict=True):
            if parity is not False:
                qp.cond(parity, qp.X)(target)
        return qp.probs(wires=[3, 4, 5, 6])

    probs = np.asarray(circuit())
    expected = np.zeros(16)
    expected[0b1010] = 1.0  # parities (1, 0, 1, 0) on wires 3..6, big-endian
    assert np.allclose(probs, expected)


@pytest.mark.compiled
def test_mcm_parities_in_compiled_circuit_across_two_words():
    """40 outcomes measured on one reused wire span two 32-bit words; only bits 0, 33, 39 are 1.

    Reading mask bits in reverse order would give (0, 0, 0, 0, 1).
    """
    num_outcomes, ones = 40, {0, 33, 39}
    masks = [1 << 33, 1 << 34, (1 << 39) | 1, (1 << 1) | (1 << 38), 1 << 39]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("lightning.qubit", wires=1 + len(masks)))
    def circuit():
        outcomes = []
        for i in range(num_outcomes):
            if i in ones:
                qp.X(0)
            outcomes.append(qp.measure(0, reset=True))
        for target, parity in enumerate(mcm_parities(outcomes, masks), start=1):
            qp.cond(parity, qp.X)(target)
        return qp.probs(wires=list(range(1, 1 + len(masks))))

    probs = np.asarray(circuit())
    expected = np.zeros(2 ** len(masks))
    expected[0b10001] = 1.0  # parities (1, 0, 0, 0, 1) on wires 1..5, big-endian
    assert np.allclose(probs, expected)


def test_noise_shaped_warning_naming_a_dfthc_operator_is_recorded():
    """The filters keep noise-shaped messages, so the fixture classifies them before dropping."""
    assert "CompileTestLoad" in dfthc_operator_names()
    named = ("Excluded the _x_workers decomposition rule for CompileTestLoad since "
             "--decompose-lowering cannot work with multiple registers yet")
    plain = named.replace("CompileTestLoad", "Toffoli")
    assert NOISE_PATTERNS[0].search(named) and NOISE_PATTERNS[0].search(plain)
    with warnings.catch_warnings(record=True) as records:
        install_warning_filters()
        warnings.warn(RuleLoweringWarning(named), stacklevel=1)
        warnings.warn(RuleLoweringWarning(plain), stacklevel=1)
    assert [str(record.message) for record in records] == [named, plain]
    assert [classify_rule_warning(str(record.message)) for record in records] == ["dfthc", "noise"]


@pytest.mark.parametrize("rule", ["_toffoli_elbow", "_ppm"])
def test_classify_rule_warning(rule):
    names = {"OuterPrepare"}
    noise = (f"Excluded the {rule} decomposition rule for Toffoli since "
             "--decompose-lowering cannot work with multiple registers yet")
    assert classify_rule_warning(noise, names) == "noise"
    assert classify_rule_warning(
        "Excluded the use_allocation(decompose_mcx_two_controls_elbows) decomposition rule for "
        "MultiControlledX since --decompose-lowering cannot work with multiple registers yet", names
    ) == "noise"
    assert classify_rule_warning(noise.replace("Toffoli", "OuterPrepare"), names) == "dfthc"
    skipped = "Skipped the _outer rule for Adjoint(OuterPrepare)"
    assert classify_rule_warning(skipped, names) == "dfthc"
    assert classify_rule_warning("Something new about CNOT", names) == "other"


def test_assert_resources_reports_all_mismatches():
    resources = CompiledResources(toffoli=3, measurements=1, measured_and_uncomputes=2,
                                  counts={}, num_wires=4)
    assert_resources(resources, toffoli=3, measurements=1, measured_and_uncomputes=2)
    with pytest.raises(AssertionError, match="'toffoli': 4"):
        assert_resources(resources, toffoli=4, measurements=0)
