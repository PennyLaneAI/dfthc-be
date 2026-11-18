# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlaggedUniformPrep (spec 9.2): exact uniformity, the failure weight, round trips and counts."""

import re

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.encoded import uniform_failure_probability
from dfthc.ops.uniform import FlaggedUniformPrep, _flagged_uniform_prep, amplitude_angles
from dfthc.params import lg, odd_part
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation


def _layout(L, s, grad=None):
    """Registers target, succ, succ_work, phase_grad, work, in that order of wires."""
    sizes = FlaggedUniformPrep.required_wires(L, s)
    sizes["phase_grad"] = s if grad is None else grad
    names = ["target", "succ", "succ_work", "phase_grad", "work"]
    wires, start = {}, 0
    for name in names:
        wires[name] = list(range(start, start + sizes[name]))
        start += sizes[name]
    return wires, start


def _prep(L, s, wires):
    return FlaggedUniformPrep(L, s, wires["target"], wires["succ"], wires["succ_work"],
                              wires["phase_grad"], wires["work"])


def _simulate(L, s, grad=None, round_trip=False):
    wires, num_wires = _layout(L, s, grad)

    def qfunc():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        _prep(L, s, wires)
        if round_trip:
            qp.adjoint(_prep(L, s, wires))

    tape = decompose_for_simulation(qfunc)
    return SparseSimulator(num_wires).run(tape.operations), wires


def _value(state, register, row):
    return sum(int(state.bits(w)[row]) << (len(register) - 1 - i) for i, w in enumerate(register))


def _without_gradient(state, wires):
    """Amplitudes with the catalysed phase-gradient factor divided out, keyed by the other wires.

    Asserts that the state is (something) x (the phase-gradient state) with work in |0>.
    """
    grad = wires["phase_grad"]
    b = len(grad)
    others = wires["target"] + wires["succ"] + wires["succ_work"]
    amplitudes = {}
    for row, amp in enumerate(state.amps):
        assert not any(state.bits(w)[row] for w in wires["work"])
        y = _value(state, grad, row)
        key = _value(state, others, row)
        amplitudes.setdefault(key, []).append(amp * np.sqrt(2**b) * np.exp(2j * np.pi * y / 2**b))
    for key, values in amplitudes.items():
        assert len(values) == 2**b and np.allclose(values, values[0], rtol=0, atol=1e-12), key
    return {key: values[0] for key, values in amplitudes.items()}, len(others)


def _expected_target_amplitudes(L, s):
    """Spec 9.2's amplitudes on t in [0, 2**lg(L)) for l > 1, with k recomputed from the spec
    formula: H on target, phase on top < l, H on top, phase on top = 0, H on top."""
    odd, n = odd_part(L), lg(odd_part(L))
    angle = np.arccos(1 - 2 ** (odd.bit_length() - 1) / odd)
    k = int(np.floor(angle / (2 * np.pi) * 2**s)) % 2**s
    phase = np.exp(2j * np.pi * k / 2**s)
    h = np.ones((1, 1))
    for _ in range(n):
        h = np.kron(h, [[1, 1], [1, -1]]) / np.sqrt(2)
    top = h @ (np.where(np.arange(2**n) < odd, phase, 1) / np.sqrt(2**n))
    top[0] *= phase
    eta = lg(L) - n
    return np.repeat(h @ top, 2**eta) / np.sqrt(2**eta)


@pytest.mark.parametrize("s", [3, 7])
@pytest.mark.parametrize("L", [3, 5, 6, 7])
def test_succ_part_is_exactly_uniform(L, s):
    """T3 (sparse simulator): succ = 1 exactly on t < L with equal amplitudes, succ = 0 only on
    t >= L, and the failure weight equals uniform_failure_probability (spec 8.5). The gradient
    has s + 1 wires, so the rotations must add into its top s wires. Every amplitude, phase
    included, equals spec 9.2's."""
    state, wires = _simulate(L, s, grad=s + 1)
    amplitudes, num_others = _without_gradient(state, wires)
    shift = num_others - lg(L)
    succ_shift = shift - 1
    good, failure = [], 0.0
    observed = np.zeros(2 ** lg(L), dtype=complex)
    for key, amp in amplitudes.items():
        t, succ = key >> shift, (key >> succ_shift) & 1
        assert succ == int(t < L), (t, succ)
        assert observed[t] == 0, t
        observed[t] = amp
        if t < L:
            good.append(amp)
        else:
            failure += abs(amp) ** 2
    assert len(good) == L and np.allclose(good, good[0], rtol=0, atol=1e-12)
    assert np.allclose(observed, _expected_target_amplitudes(L, s), rtol=0, atol=1e-12)
    assert abs(failure - uniform_failure_probability(L, s)) < 1e-12
    assert abs(L * abs(good[0]) ** 2 + failure - 1) < 1e-12


@pytest.mark.parametrize(("L", "s"), [(3, 3), (5, 7), (6, 3), (7, 7), (8, 3)])
def test_adjoint_round_trip_restores_every_register(L, s):
    """The prepare followed by its auto-derived adjoint returns every register to |0>."""
    state, wires = _simulate(L, s, round_trip=True)
    amplitudes, _ = _without_gradient(state, wires)
    assert list(amplitudes) == [0] and abs(amplitudes[0] - 1) < 1e-12


def _lightning(num_wires, body, returns):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return np.asarray(circuit())


@pytest.mark.compiled
@pytest.mark.parametrize("L", [5, 8])
def test_compiled_distribution_on_lightning(L):
    """T3 on the compiled path (s = 3): P(t, succ) on lightning.qubit; L = 8 is the l = 1
    branch (Hadamards and X(succ))."""
    s = 3
    wires, num_wires = _layout(L, s)

    def body():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        _prep(L, s, wires)

    probs = _lightning(num_wires, body,
                       lambda: qp.probs(wires=wires["target"] + wires["succ"]))
    probs = probs.reshape(2 ** lg(L), 2)
    failure = uniform_failure_probability(L, s)
    for t in range(2 ** lg(L)):
        if t < L:
            assert abs(probs[t, 1] - (1 - failure) / L) < 1e-10 and probs[t, 0] < 1e-10
        else:
            assert probs[t, 1] < 1e-10
    assert abs(probs[L:, 0].sum() - failure) < 1e-10


@pytest.mark.compiled
def test_compiled_round_trip_on_lightning():
    L, s = 7, 3
    wires, num_wires = _layout(L, s)

    def body():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        _prep(L, s, wires)
        qp.adjoint(_prep(L, s, wires))

    grad = np.exp(-2j * np.pi * np.arange(2**s) / 2**s) / np.sqrt(2**s)
    before = 2 ** (lg(L) + 1 + 2)
    after = 2 ** len(wires["work"])
    expected = np.kron(np.kron(np.eye(before)[0], grad), np.eye(after)[0])
    assert np.allclose(_lightning(num_wires, body, qp.state), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("L", "s", "grad"), [(324, 7, 15), (28, 7, 15), (5, 7, 7), (7, 7, 7),
                                              (12, 7, 7), (8, 7, 7)])
@pytest.mark.parametrize("adjoint", [False, True], ids=["prepare", "unprepare"])
def test_toffoli_count_matches_the_closed_form(L, s, grad, adjoint):
    """T4: 3n + 2s - 5 and 2n + 2s - 4 for l > 1, 0 for l = 1; FeMoco widths are table-free."""
    wires, num_wires = _layout(L, s, grad)

    def program():
        op = _prep(L, s, wires)
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=num_wires)
    expected = (costs.flagged_uniform_unprep if adjoint else costs.flagged_uniform_prep)(L, s)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)


def test_femoco_values_of_the_closed_form():
    assert (costs.flagged_uniform_prep(324, 7).toffoli,
            costs.flagged_uniform_unprep(324, 7).toffoli) == (30, 24)
    assert (costs.flagged_uniform_prep(28, 7).toffoli,
            costs.flagged_uniform_unprep(28, 7).toffoli) == (18, 16)


@pytest.mark.compiled
def test_prepare_and_unprepare_uncompute_every_and():
    """A prepare/unprepare pair measures every AND it computes (spec 9.24, A = T per pair)."""
    L, s = 12, 4
    wires, num_wires = _layout(L, s)

    def program():
        _prep(L, s, wires)
        qp.adjoint(_prep(L, s, wires))

    resources = compiled_resources(program, num_wires=num_wires)
    expected = costs.flagged_uniform_prep(L, s) + costs.flagged_uniform_unprep(L, s)
    assert resources.toffoli == resources.measured_and_uncomputes == expected.toffoli


@pytest.mark.compiled
def test_pinned_rule_is_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the solver names the pinned rule for the prepare and its adjoint-lowered instance for
    the auto-derived adjoint; T5 at level="device" (L-20)."""
    wires, num_wires = _layout(6, 3)

    def program():
        _prep(6, 3, wires)
        qp.adjoint(_prep(6, 3, wires))

    qp.analyze(verbose_program(program, num_wires=num_wires), level="user")()
    captured = capfd.readouterr()
    rules = chosen_rules(captured.out + captured.err)
    assert "_flagged_uniform_prep" in rules
    assert any(re.fullmatch(r"_flagged_uniform_prep_\d+", rule) for rule in rules), rules
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize(("L", "s"), [(5, 3), (12, 4), (8, 3)])
def test_rule_is_valid(L, s):
    wires, _ = _layout(L, s)
    qp.testing.assert_valid_decomp_rule(_prep(L, s, wires), _flagged_uniform_prep,
                                        check_matrix=False)


def test_amplitude_angles():
    assert amplitude_angles(324, 7) == (27, 27)
    assert amplitude_angles(28, 7) == (22, 22)
    assert amplitude_angles(5, 3) == (1, 1)
    assert amplitude_angles(8, 7) == amplitude_angles(1, 3) == (0, 0)


def test_required_wires():
    assert FlaggedUniformPrep.required_wires(324, 7) == {
        "target": 9, "succ": 1, "succ_work": 6, "phase_grad": 7, "work": 20}
    assert FlaggedUniformPrep.required_wires(8, 3) == {
        "target": 3, "succ": 1, "succ_work": 0, "phase_grad": 3, "work": 0}
    assert odd_part(324) == 81 and lg(81) == 7


@pytest.mark.parametrize(("L", "s", "change", "match"), [
    (0, 3, {}, r"L must lie in \[1, None\]"),
    (5, 2, {}, r"s must lie in \[3, None\]"),
    (5, 3, {"phase_grad": 2}, "'phase_grad' has 2 wires; expected at least 3"),
    (5, 3, {"succ_work": 1}, "'succ_work' has 1 wires; expected 2"),
])
def test_constructor_validates(L, s, change, match):
    sizes = FlaggedUniformPrep.required_wires(max(L, 1), max(s, 3)) | change
    wires = {name: list(range(100 * i, 100 * i + size))
             for i, (name, size) in enumerate(sizes.items())}
    with pytest.raises(ValueError, match=match):
        FlaggedUniformPrep(L, s, **wires)
