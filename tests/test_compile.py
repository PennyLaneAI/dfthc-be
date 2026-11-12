"""Tests for dfthc.compile and dfthc.testing.resources (plan Task 0.2)."""

import itertools
import re
import time
import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from catalyst import QJIT
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError
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


class CompileTestMeasureOnly(Operator2):
    """Test-local operator whose only rule measures; never pinned through ``register_rule``."""

    wire_argnames = ("target", "aux")

    def __init__(self, target, aux):
        super().__init__(target, aux)


def _compile_test_measure_only_resources(target, aux):
    return {qp.H: 1, resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)): 1, qp.Z: 1}


@qp.register_resources(_compile_test_measure_only_resources)
def _compile_test_measure_only(target, aux):
    qp.H(target[0])
    outcome = qp.measure(target[0], reset=True)
    qp.cond(outcome, qp.Z)(aux[0])


qp.add_decomps(CompileTestMeasureOnly, _compile_test_measure_only)


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
def test_compiled_resources_compiles_the_program_once(monkeypatch):
    """Task 0.2c: one ``QJIT.aot_compile`` per resource count, the one inside ``qp.analyze``.

    A zero-argument ``qp.qjit`` function is AOT-compiled to a linked binary at decoration (L-27),
    and ``qp.analyze`` then compiles the program again; the binary is never used. Counted on
    ``main`` (before the fix): two calls, one from the decorator and one from ``qp.analyze``.
    """
    calls = []
    original = QJIT.aot_compile

    def spy(self, *args, **kwargs):
        calls.append(self.compile_options.target)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(QJIT, "aot_compile", spy)
    resources = compiled_resources(lambda: qp.H(0), num_wires=1)
    assert calls == ["mlir_opt"]
    assert resources.counts == {"Hadamard": 1}
    assert resources.num_wires == 1


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


_SUFFIX = (
    ": it contains a mid-circuit measurement, which is not supported with adjoint or control "
    "regions."
)
_ANGLE_DUMP = (
    "{}{wires:82}{shape = [4, 2, 2, 2, 2, 4, 2, 2, 1, 1], "
    "tables = [[[0, [2]], [1, [18]]], [[0, [48]], [1, [59]]]]}"
)
_TABLE_DUMP = (
    "{}{high:2,junk:5,low:1,target:5,work:13}{block_bits = 1 : i64, high_ranges = [[2, 0, 4]], "
    "table = [[0, [2]], [1, [18]]], width = 5 : i64}"
)
# The three wrapper forms of Task 0.9's recorded texts (spikes/results/k1_small.json), tables
# shortened.
_ANGLE_RULE = "Skipped the _proto_angle_erase decomposition rule for "
_TABLE_RULE = "Skipped the _proto_table_erase decomposition rule for "
EXPECTED_ADJOINT = f"{_ANGLE_RULE}Adjoint(ProtoAngleErase){_ANGLE_DUMP}{_SUFFIX}"
EXPECTED_CONTROL = f"{_TABLE_RULE}C(ProtoTableErase){_TABLE_DUMP}{_SUFFIX}"
EXPECTED_CONTROL_ADJOINT = f"{_ANGLE_RULE}C(Adjoint(ProtoAngleErase)){_ANGLE_DUMP}{_SUFFIX}"
BOTH = {"ProtoAngleErase", "ProtoTableErase"}


@pytest.mark.parametrize(
    ("message", "names"),
    [
        (EXPECTED_ADJOINT, {"ProtoAngleErase"}),
        (EXPECTED_CONTROL, {"ProtoTableErase"}),
        (EXPECTED_CONTROL_ADJOINT, {"ProtoAngleErase"}),
        (EXPECTED_CONTROL_ADJOINT, ["ProtoAngleErase"]),
        (EXPECTED_ADJOINT.replace("ProtoAngleErase", "CompileTestLoad"), None),
        (EXPECTED_CONTROL_ADJOINT.replace("ProtoAngleErase", "CompileTestLoad"), None),
        (
            "Skipped the _rule decomposition rule for Adjoint(C(Adjoint(ProtoAngleErase)))"
            + "{}{wires:1}{}" + _SUFFIX,
            {"ProtoAngleErase"},
        ),
    ],
)
def test_classify_expected_measuring_rule_warning(message, names):
    assert classify_rule_warning(message, names) == "expected"


def test_classify_expected_accepts_the_dfthc_names_by_default():
    assert "CompileTestLoad" in dfthc_operator_names()
    message = EXPECTED_CONTROL.replace("ProtoTableErase", "CompileTestLoad")
    assert classify_rule_warning(message) == "expected"
    assert classify_rule_warning(message, {"OtherOperator"}) == "other"


@pytest.mark.parametrize(
    ("message", "names"),
    [
        # An unwrapped operator, and unknown wrapper names.
        (EXPECTED_CONTROL.replace("C(ProtoTableErase)", "ProtoTableErase"), {"ProtoTableErase"}),
        (EXPECTED_CONTROL.replace("C(ProtoTableErase)", "Pow(ProtoTableErase)"),
         {"ProtoTableErase"}),
        # Unbalanced wrappers.
        (EXPECTED_CONTROL_ADJOINT.replace("ProtoAngleErase))", "ProtoAngleErase)"),
         {"ProtoAngleErase"}),
        (EXPECTED_CONTROL_ADJOINT.replace("ProtoAngleErase))", "ProtoAngleErase)))"),
         {"ProtoAngleErase"}),
        (EXPECTED_ADJOINT.replace("Adjoint(ProtoAngleErase)", "Adjoint(ProtoAngleErase"),
         {"ProtoAngleErase"}),
        # The suffix without its final period, or with text after it.
        (EXPECTED_ADJOINT.removesuffix("."), {"ProtoAngleErase"}),
        (EXPECTED_ADJOINT + " Also ProtoAngleErase.", {"ProtoAngleErase"}),
        # A second name of `names`: before the suffix, in the rule name, in an attribute dump.
        (EXPECTED_CONTROL.replace("{}{high:2", "{}{ProtoAngleErase:2"), BOTH),
        (EXPECTED_CONTROL.replace("_proto_table_erase", "x.ProtoAngleErase"), BOTH),
        (EXPECTED_CONTROL.replace("table = [[0, [2]]", "table = ProtoAngleErase, [[0, [2]]"), BOTH),
        # A different reason after the colon.
        (EXPECTED_ADJOINT.replace(
            "which is not supported with adjoint or control regions",
            "which cannot be placed in a control region"), {"ProtoAngleErase"}),
        (EXPECTED_ADJOINT.replace("it contains a mid-circuit measurement", "it is slow"),
         {"ProtoAngleErase"}),
        # A different message shape that names the operator.
        ("Skipped the _rule rule for Adjoint(ProtoAngleErase)", {"ProtoAngleErase"}),
        (EXPECTED_ADJOINT.replace("Skipped the", "Excluded the"), {"ProtoAngleErase"}),
    ],
)
def test_classify_other_messages_naming_a_dfthc_operator_stay_dfthc(message, names):
    assert classify_rule_warning(message, names) == "dfthc"


def test_classify_expected_needs_a_whole_word_operator_name():
    message = EXPECTED_ADJOINT.replace("Adjoint(ProtoAngleErase)", "Adjoint(MyProtoAngleErase)")
    assert classify_rule_warning(message, {"ProtoAngleErase"}) == "other"


def test_classify_rule_warning_rejects_a_bare_string_of_names():
    with pytest.raises(TypeError, match="names"):
        classify_rule_warning(EXPECTED_ADJOINT, names="ProtoAngleErase")


def test_classify_rule_warning_accepts_a_generator_of_names():
    assert classify_rule_warning(EXPECTED_ADJOINT, (n for n in ["ProtoAngleErase"])) == "expected"


# The real texts of PennyLane's own controlled adjoint TemporaryAND candidate, as printed by a
# few-qubit compile of qp.ctrl(qp.SemiAdder(x_wires=[0, 1], y_wires=[2, 3], work_wires=[4]),
# control=5) with CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 (Task 0.2b). They name "rule for"
# after "decomposition", and give a different reason than the measuring-rule warning.
NOISE_ADJOINT_AND = (
    "Skipped the _adjoint_temporary_and decomposition rule for "
    "C(Adjoint(TemporaryAND)){control_values:[tensor<2xi1>]}{wires:3}{}: it contains a "
    "mid-circuit measurement, which cannot be placed in a control region."
)
NOISE_ADJOINT_AND_PPM = NOISE_ADJOINT_AND.replace(
    "_adjoint_temporary_and ", "_adjoint_temporary_and_ppm "
)


@pytest.mark.parametrize("message", [NOISE_ADJOINT_AND, NOISE_ADJOINT_AND_PPM])
@pytest.mark.parametrize("names", [None, {"ProtoAngleErase"}])
def test_classify_adjoint_temporary_and_text_as_noise(message, names):
    assert classify_rule_warning(message, names) == "noise"


def test_classify_adjoint_temporary_and_old_text_form_is_still_noise():
    old = NOISE_ADJOINT_AND.replace(" decomposition rule", " rule")
    assert classify_rule_warning(old, {"ProtoAngleErase"}) == "noise"


def test_classify_adjoint_temporary_and_text_naming_a_dfthc_operator_stays_dfthc():
    message = NOISE_ADJOINT_AND.replace("{wires:3}", "{ProtoAngleErase:3}")
    assert classify_rule_warning(message, {"ProtoAngleErase"}) == "dfthc"


def test_expected_warning_is_recorded_and_neither_fails_nor_is_re_emitted():
    """The filters keep the warning; the autouse fixture classifies it "expected" and drops it.

    The warning is raised at test level, outside any recorder of our own, so the autouse fixture
    sees it: a "dfthc" classification would fail this test at teardown, and a re-emission would
    show up in pytest's warnings summary (the output stays pristine).
    """
    message = EXPECTED_ADJOINT.replace("ProtoAngleErase", "CompileTestLoad")
    with warnings.catch_warnings(record=True) as records:
        install_warning_filters()
        warnings.warn(RuleLoweringWarning(message), stacklevel=1)
    assert [str(record.message) for record in records] == [message]
    assert classify_rule_warning(str(records[0].message)) == "expected"
    warnings.warn(RuleLoweringWarning(message), stacklevel=1)


@pytest.mark.compiled
@pytest.mark.parametrize(
    ("wrapper", "key"),
    [
        (lambda op: qp.adjoint(op), "Adjoint(CompileTestMeasureOnly)"),
        (lambda op: qp.ctrl(op, control=2), "C(CompileTestMeasureOnly)"),
    ],
    ids=["adjoint", "control"],
)
def test_modified_operator_without_a_remaining_rule_fails_to_compile(wrapper, key):
    """The safety net of spec 7.4 rule 12: Catalyst drops the measuring candidate of the wrapped
    operator with an "expected" warning, and without another rule the compile fails loudly."""
    pins = {**FIXED_DECOMPS, "CompileTestMeasureOnly": _compile_test_measure_only}
    with warnings.catch_warnings(record=True) as records:
        install_warning_filters()
        with pytest.raises(CompileError, match="Decomposition rule not found for operator '"
                           + re.escape(key)):
            compiled_resources(
                lambda: wrapper(CompileTestMeasureOnly([0], [1])), num_wires=3, fixed_decomps=pins
            )
    skipped = [
        str(record.message) for record in records
        if issubclass(record.category, RuleLoweringWarning)
        and str(record.message).startswith("Skipped the _compile_test_measure_only")
    ]
    assert skipped, "Catalyst no longer reports the dropped measuring rule"
    assert {classify_rule_warning(message, {"CompileTestMeasureOnly"}) for message in skipped} == {
        "expected"
    }
    assert all(f"for {key}{{" in message for message in skipped)


def test_mcm_parities_tape_mode_keeps_the_processing_function_of_each_outcome():
    """[~m0, m1] over all four inputs, with deferred measurements: parity (1 - a) ^ b."""

    def outcomes_of():
        m0, m1 = qp.measure(0), qp.measure(1)
        return [~m0, m1]

    dev = qp.device("default.qubit", wires=3)

    @qp.qnode(dev, mcm_method="deferred")
    def circuit(a, b):
        qp.BasisState(np.array([a, b]), wires=[0, 1])
        qp.cond(mcm_parities(outcomes_of(), [0b11])[0], qp.X)(2)
        return qp.probs(wires=2)

    for a, b in itertools.product([0, 1], repeat=2):
        expected = np.zeros(2)
        expected[(1 - a) ^ b] = 1.0
        assert np.allclose(circuit(a, b), expected), (a, b)


def test_mcm_parities_tape_mode_sample_works_under_one_shot():
    """qp.sample(op=parity) needs a one-measurement-per-outcome MeasurementValue."""

    def outcomes_of():
        m0, m1 = qp.measure(0), qp.measure(1)
        return [~m0, m1]

    dev = qp.device("default.qubit", wires=2)

    @qp.set_shots(5)
    @qp.qnode(dev, mcm_method="one-shot")
    def circuit(a, b):
        qp.BasisState(np.array([a, b]), wires=[0, 1])
        measured = outcomes_of()
        return qp.sample(op=mcm_parities(measured, [0b11])[0])

    for a, b in itertools.product([0, 1], repeat=2):
        assert np.asarray(circuit(a, b)).ravel().tolist() == [(1 - a) ^ b] * 5, (a, b)


def test_mcm_parities_tape_mode_rejects_an_outcome_with_two_measurements():
    with qp.queuing.AnnotatedQueue():
        m0, m1, m2 = qp.measure(0), qp.measure(1), qp.measure(2)
        composite = m0 + m1
        assert len(composite.measurements) == 2
        with pytest.raises(ValueError, match="outcome 1"):
            mcm_parities([m2, composite], [0b10])
        # An unchosen composite outcome is not read, and a zero mask needs no outcome.
        assert mcm_parities([m2, composite], [0b01])[0].measurements == m2.measurements
        assert mcm_parities([m2, composite], [0]) == [False]


def test_assert_resources_reports_all_mismatches():
    resources = CompiledResources(toffoli=3, measurements=1, measured_and_uncomputes=2,
                                  counts={}, num_wires=4)
    assert_resources(resources, toffoli=3, measurements=1, measured_and_uncomputes=2)
    with pytest.raises(AssertionError, match="'toffoli': 4"):
        assert_resources(resources, toffoli=4, measurements=0)
