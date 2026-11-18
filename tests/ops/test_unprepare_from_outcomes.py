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

"""OuterUnprepareFromOutcomes and InnerUnprepareFromOutcomes (spec 9.7, 9.8 and 16, Q-1 variant
(b)): the round trip after the junk measurement, the order of the junk outcomes, a caller inside
another operator's rule, counts, rule selection, checks.

A pair is a prepare, ``H`` and ``qp.measure(q, reset=True)`` on every junk qubit right after it,
and the unprepare from those outcomes; the unprepare's ``extra_onehot`` is the first wires of the
measured junk register, as the block encoding lends them. The tables are random. The certificates
over every outcome are in tests/ops/test_certificates_prepare.py (nightly).
"""

import re
import warnings
from typing import NamedTuple

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError
from pennylane.core import Operator2
from pennylane.ops import Conditional
from pennylane.typing import AbstractArray, Wire

from dfthc import costs
from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    PENNYLANE_PINS,
    compiled_resources,
    register_rule,
    resource_key,
)
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerUnprepareFromOutcomes,
    OuterPrepare,
    OuterUnprepareFromOutcomes,
    _inner_unprepare_from_outcomes,
    _outer_unprepare_from_outcomes,
    abstract_junk_outcomes,
)
from dfthc.ops.qroam import QROAMErase
from dfthc.params import (
    FEMOCO54,
    H2_631G_TEST,
    H2_STO3G_TEST,
    MID_NAPHTHALENE,
    WalkParams,
    lg,
    odd_part,
)
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation


def shape(N, R, B, C, b_k1, b_k2, k1, k2, ke1, ke2):
    """b_rot = 5: the select's one-hot fits up to B = 6 at N = 3 (spec 8.3)."""
    return WalkParams(N=N, R=R, B=B, C=C, n_plus=1, b_rot=5, b_k1=b_k1, b_k2=b_k2, s_u=3, k1=k1,
                      k2=k2, ke1=ke1, ke2=ke2)


OUTER = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST,
         "x-o-5": shape(2, 3, 1, 1, 3, 2, 1, 1, 1, 1),
         "x-o-7-k0-b1": shape(3, 2, 1, 2, 1, 2, 0, 1, 2, 1)}
"""Both test presets (extra_onehot 0); X_o = 5 (l = 5: the uniform preparation fails, with a
phase); X_o = 7 with k1 = 0 (no junk: an empty outcome array) and b_k1 = 1 (no comparator
work)."""
INNER = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST,
         "6-2": shape(3, 2, 5, 1, 2, 2, 1, 2, 1, 1),
         "7-0": shape(3, 2, 6, 1, 2, 2, 1, 0, 1, 0),
         "small": WalkParams(N=2, R=1, B=1, C=1, n_plus=1, b_rot=3, b_k1=1, b_k2=1, s_u=3, k1=0,
                             k2=1, ke1=0, ke2=0)}
"""Both test presets (extra_onehot 1); rows [0, 2) of a 3-bit x_o with (B + 1, k2) = (6, 2)
(dead positions, failed b, l = 3); (B + 1, k2) = (7, 0) (no junk, l = 7); the small shape of the
caller-inside-a-rule tests (R = 1: no rank bits; 20 wires)."""


class Pair(NamedTuple):
    """A prepare and its unprepare from outcomes on one wire layout."""

    params: WalkParams
    prepare: type
    unprepare: type
    args: tuple
    prepare_wires: dict
    unprepare_wires: dict
    junk: str
    """The prepare's junk register."""
    measured: int
    """K d: every qubit of the read's target and junk."""
    inputs: list
    """The read-only input wires (``x_o`` of the inner pair)."""
    num_wires: int


def layout(sizes):
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def lend(unprepare, prepare_wires, junk, structure):
    """The unprepare's registers: the prepare's, and ``extra_onehot`` from the start of the
    measured junk register."""
    sizes = unprepare.required_wires(*structure)
    wires = {name: prepare_wires[name] for name in sizes if name != "extra_onehot"}
    return wires | {"extra_onehot": prepare_wires[junk][:sizes["extra_onehot"]]}


def outer_structure(p):
    return (p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)


def inner_structure(p):
    return (p.N, p.R, p.B, p.C, p.b_k2, p.k2, p.ke2, p.s_u)


def outer(p, seed=5):
    rng = np.random.default_rng(seed)
    args = (p.X_o, tuple(int(a) for a in rng.integers(0, p.X_o, p.X_o)),
            tuple(int(k) for k in rng.integers(0, 2**p.b_k1, p.X_o)), p.b_k1, p.k1, p.ke1, p.s_u)
    wires, num_wires = layout(OuterPrepare.required_wires(*outer_structure(p)))
    return Pair(p, OuterPrepare, OuterUnprepareFromOutcomes, args, wires,
                lend(OuterUnprepareFromOutcomes, wires, "junk_x_o", outer_structure(p)),
                "junk_x_o", p.K1 * p.d1, [], num_wires)


def inner(p, seed=5):
    rng = np.random.default_rng(seed)
    size = (p.RC, p.B + 1)
    columns = [rng.integers(0, p.B + 1, size), rng.integers(0, 2**p.b_k2, size),
               rng.integers(0, 2, size), rng.integers(0, 2, size)]
    tables = tuple(tuple(tuple(int(v) for v in row) for row in column) for column in columns)
    args = (p.N, p.R, p.B, p.C, *tables, p.b_k2, p.k2, p.ke2, p.s_u)
    wires, num_wires = layout(InnerPrepare.required_wires(*inner_structure(p)))
    return Pair(p, InnerPrepare, InnerUnprepareFromOutcomes, args, wires,
                lend(InnerUnprepareFromOutcomes, wires, "junk_b", inner_structure(p)), "junk_b",
                p.K2 * p.d2, wires["x_o"], num_wires)


PAIRS = {f"outer-{name}": (outer, p) for name, p in OUTER.items()} | {
    f"inner-{name}": (inner, p) for name, p in INNER.items()}


def pair(key, seed=5):
    build, p = PAIRS[key]
    return build(p, seed)


def measure_x(wires, postselect=None):
    """``H`` and ``qp.measure(q, reset=True)`` on every wire, postselected on the bits of
    ``postselect`` if it is given."""
    outcomes = []
    for i, wire in enumerate(wires):
        qp.H(wire)
        outcomes.append(qp.measure(wire, reset=True,
                                   postselect=None if postselect is None else postselect[i]))
    return outcomes


def round_trip(pr, order=lambda outcomes: outcomes, postselect=None):
    """The phase gradient, the prepare, the junk measurement (postselected on ``postselect`` if
    it is given) and the unprepare from the outcomes, passed in the order ``order`` gives."""
    def qfunc():
        qp.PhaseGradientStatePrep(pr.prepare_wires["phase_grad"])
        pr.prepare(*pr.args, **pr.prepare_wires)
        outcomes = measure_x(pr.prepare_wires[pr.junk], postselect)
        pr.unprepare(order(outcomes), *pr.args, **pr.unprepare_wires)

    return qfunc


def random_input_state(num_wires, inputs, seed=11):
    """A random complex amplitude on every basis value of ``inputs``, every other wire |0>."""
    rng = np.random.default_rng(seed)
    size = 2 ** len(inputs)
    amps = rng.normal(size=size) + 1j * rng.normal(size=size)
    keys = []
    for v in range(size):
        value = sum(((v >> (len(inputs) - 1 - i)) & 1) << (num_wires - 1 - wire)
                    for i, wire in enumerate(inputs))
        keys.append(SparseState.basis(num_wires, value).keys[0])
    return SparseState(num_wires, np.array(keys), amps / np.linalg.norm(amps))


def sorted_terms(state):
    order = np.lexsort(state.keys.T[::-1])
    return state.keys[order], state.amps[order]


def restores(num_wires, gradient_wires, inputs, operations, schedule) -> bool:
    """Whether ``operations`` return a random complex state of ``inputs``, every other register
    |0> and the phase gradient unchanged, under the forced ``schedule``, all of it consumed
    (strict mode: every measurement has probability 1/2, every AND precondition holds)."""
    initial = random_input_state(num_wires, inputs)
    gradient = decompose_for_simulation(
        lambda: qp.PhaseGradientStatePrep(gradient_wires)).operations
    expected = sorted_terms(SparseSimulator(num_wires).run(gradient, initial_state=initial))
    simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
    keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
    assert simulator.policy.position == len(schedule)
    return (np.array_equal(keys, expected[0])
            and np.allclose(amps, expected[1], rtol=0, atol=1e-12))


def schedules(num, seed=3):
    """The alternating and a seeded schedule of ``num`` outcomes."""
    return {"alternating": [i % 2 for i in range(num)],
            "seeded": np.random.default_rng(seed).integers(0, 2, num).tolist()}


def basis_schedules(num):
    """The zero schedule and every unit schedule of ``num`` outcomes (spec 12.5)."""
    return [[0] * num] + [[int(i == j) for j in range(num)] for i in range(num)]


@pytest.mark.parametrize("key", list(PAIRS))
def test_round_trip_after_the_junk_measurement(key):
    """T3 (sparse simulator, strict): prepare, X measurement of the junk, unprepare from the
    outcomes return a random state of the inputs (x_o for the inner pair), every other register
    |0> and the phase gradient unchanged, under the alternating and a seeded schedule of all K d
    measurements, junk first; an all-ones schedule would see only each address's mask parity
   . With more than one junk outcome, the outcomes passed in reversed order fail
    the round trip under the zero or some unit schedule: the fixup reads them in the junk
    register's wire order, the order of QROAMRead.junk_layout's slots."""
    pr = pair(key)
    operations = decompose_for_simulation(round_trip(pr)).operations
    assert sum(op.name == "MidMeasure" for op in operations) == pr.measured
    grad = pr.prepare_wires["phase_grad"]
    for kind, outcomes in schedules(pr.measured).items():
        assert restores(pr.num_wires, grad, pr.inputs, operations, outcomes), kind
    if len(pr.prepare_wires[pr.junk]) > 1:
        wrong = decompose_for_simulation(round_trip(pr, lambda o: o[::-1])).operations
        assert not all(restores(pr.num_wires, grad, pr.inputs, wrong, outcomes)
                       for outcomes in basis_schedules(pr.measured))


def lightning(num_wires, body, seed):
    @qp.qjit(capture=True, seed=seed)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps={**FIXED_DECOMPS, **PENNYLANE_PINS})
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return qp.state()

    return np.asarray(circuit())


def expected_state(num_wires, gradient_wires, inputs=()):
    """|+>^len(inputs) on ``inputs``, the phase-gradient state, |0> elsewhere."""
    state = np.ones(1)
    for wire in range(num_wires):
        if wire == gradient_wires[0]:
            b = len(gradient_wires)
            state = np.kron(state, np.exp(-2j * np.pi * np.arange(2**b) / 2**b) / np.sqrt(2**b))
        elif wire in gradient_wires[1:]:
            continue
        else:
            state = np.kron(state, [2**-0.5, 2**-0.5] if wire in inputs else [1, 0])
    return state


@pytest.mark.compiled
def test_lightning_round_trip_with_two_seeds():
    """The compiled round trip at H2_STO3G_TEST's outer shape (19 wires): every register returns
    to |0> and the phase gradient is unchanged, for two measurement seeds (smoke runs: the
    certificates prove every outcome on the tape path, and the postselected test below fixes the
    junk outcomes of the compiled path)."""
    pr = pair("outer-sto3g")
    expected = expected_state(pr.num_wires, pr.prepare_wires["phase_grad"])
    for seed in range(2):
        state = lightning(pr.num_wires, round_trip(pr), seed)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), seed


@pytest.mark.compiled
@pytest.mark.parametrize("key", ["outer-sto3g", "inner-small"])
def test_lightning_round_trip_with_postselected_junk(key):
    """The compiled path of the junk outcomes, deterministically: the junk measurements are
    postselected on the unit pattern 1, 0, ..., 0 (analytic lightning.qubit honours postselection
    under capture and renormalizes), so the unprepare receives a traced bool array of known
    content. The inputs (x_o of the inner pair in |+>) come back, every other register |0> and the
    phase gradient unchanged. The pattern is neither zero nor a palindrome, so junk outcomes that
    the capture branch of the canonicalization reverses or drops leave a phase, which the sampled
    lightning tests see only for some seeds."""
    pr = pair(key)
    pattern = [1] + [0] * (len(pr.prepare_wires[pr.junk]) - 1)

    def body():
        for wire in pr.inputs:
            qp.H(wire)
        round_trip(pr, postselect=pattern)()

    expected = expected_state(pr.num_wires, pr.prepare_wires["phase_grad"], pr.inputs)
    assert np.allclose(lightning(pr.num_wires, body, 0), expected, rtol=0, atol=1e-10)


# ------------------------------------------------- a caller inside another operator's rule


class InnerRoundTripProbe(Operator2):
    """Test operator: its rule calls ``InnerPrepare``,
    X-measures ``junk_b`` and calls ``InnerUnprepareFromOutcomes`` with the outcomes, as the block
    encoding's rule does. ``wires`` is ``InnerPrepare``'s registers, concatenated in its
    ``wire_argnames`` order."""

    compilable_argnames = ("args",)
    wire_argnames = ("wires",)

    def __init__(self, args, wires):
        super().__init__(args, wires)


def _probe_registers(args, wires) -> dict:
    sizes = InnerPrepare.required_wires(*args[:4], *args[8:])
    registers, start = {}, 0
    for name, size in sizes.items():
        registers[name] = wires[start:start + size]
        start += size
    return registers


def _probe_resources(args, wires):
    prepare = InnerPrepare.required_wires(*args[:4], *args[8:])
    unprepare = InnerUnprepareFromOutcomes.required_wires(*args[:4], *args[8:])
    junk = prepare["junk_b"]
    outcomes = abstract_junk_outcomes(junk)
    return {
        resource_key(lambda: InnerPrepare(*args, **{k: Wire[n] for k, n in prepare.items()})): 1,
        qp.H: junk, resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)): junk,
        resource_key(lambda: InnerUnprepareFromOutcomes(
            outcomes, *args, **{k: Wire[n] for k, n in unprepare.items()})): 1}


@qp.register_resources(_probe_resources)
def _inner_round_trip_probe(args, wires):
    registers = _probe_registers(args, wires)
    InnerPrepare(*args, **registers)
    outcomes = measure_x(registers["junk_b"])
    InnerUnprepareFromOutcomes(outcomes, *args, **lend(InnerUnprepareFromOutcomes, registers,
                                                       "junk_b", args[:4] + args[8:]))


register_rule(InnerRoundTripProbe, _inner_round_trip_probe)


def probe(key):
    pr = pair(key)
    flat = [wire for register in pr.prepare_wires.values() for wire in register]
    return pr, flat


@pytest.mark.compiled
@pytest.mark.parametrize("key", ["inner-sto3g", "inner-small"])
def test_caller_inside_another_rule_is_valid_and_counted(key):
    """The junk outcomes cross an operator boundary inside another rule (K-10 part 2 in
    production form): T2 of the probe's rule in tape mode, where the outcomes are an object array
    of MeasurementValues, and its compiled T, M and A, where they are a traced bool array: T of
    the prepare and the unprepare, all K2 d2 measurements, and A the prepare's T less the ANDs
    its unprepare uncomputes (comparator and uniform preparation) plus the unprepare's: the
    round trip's T, no Toffoli."""
    pr, flat = probe(key)
    op = InnerRoundTripProbe(pr.args, flat)
    qp.testing.assert_valid_decomp_rule(op, _inner_round_trip_probe, check_matrix=False)
    resources = compiled_resources(lambda: InnerRoundTripProbe(pr.args, flat), pr.num_wires)
    p = pr.params
    prepare = costs.inner_prepare(p).toffoli
    expected = prepare + costs.inner_unprepare_from_outcomes(p).toffoli
    assert (resources.toffoli, resources.measurements) == (expected, p.K2 * p.d2)
    prepare_and = prepare - p.b_k2 - _dirty(p.B + 1)
    assert resources.measured_and_uncomputes == prepare_and + expected_and_uncomputes(pr)
    assert resources.measured_and_uncomputes == expected
    assert resources.counts.get("Toffoli", 0) == 0


def test_caller_inside_another_rule_restores_every_outcome():
    """The probe on the tape path (strict sparse simulator) at the small shape restores a random
    complex state of x_o, every other register |0> and the phase gradient, under the zero and
    every unit schedule of its K2 d2 = 8 outcomes: by GF(2) linearity, every outcome (spec 12.5);
    every conditional is a PauliZ or CZ on a linear predicate of the outcomes."""
    pr, flat = probe("inner-small")

    def qfunc():
        qp.PhaseGradientStatePrep(pr.prepare_wires["phase_grad"])
        InnerRoundTripProbe(pr.args, flat)

    operations = decompose_for_simulation(qfunc).operations
    assert any(isinstance(op, Conditional) for op in operations)
    assert not nonlinear_conditionals(operations)
    for schedule in basis_schedules(pr.measured):
        assert restores(pr.num_wires, pr.prepare_wires["phase_grad"], pr.inputs, operations,
                        schedule), schedule


@pytest.mark.compiled
def test_caller_inside_another_rule_on_lightning():
    """The compiled probe at the small shape (20 wires), x_o in |++>: x_o comes back in |++>,
    every other register in |0> and the phase gradient unchanged, for two measurement seeds."""
    pr, flat = probe("inner-small")

    def body():
        for wire in pr.inputs:
            qp.H(wire)
        qp.PhaseGradientStatePrep(pr.prepare_wires["phase_grad"])
        InnerRoundTripProbe(pr.args, flat)

    expected = expected_state(pr.num_wires, pr.prepare_wires["phase_grad"], pr.inputs)
    for seed in range(2):
        assert np.allclose(lightning(pr.num_wires, body, seed), expected, rtol=0,
                           atol=1e-10), seed


# --------------------------------------------------------------------------- resources


def _dirty(L):
    n = lg(odd_part(L))
    return n - 1 if n else 0


def expected_and_uncomputes(pr) -> int:
    """A of the unprepare from outcomes, as the baseline unprepare's (spec 9.24): the swaps'
    w, the comparator's b_k, the fixup's T and the adjoint uniform preparation's
    3(n - 1) + 2(s_u - 1) (n = lg l, none if l = 1)."""
    p = pr.params
    if pr.prepare is OuterPrepare:
        L, b_k, ob, swaps = p.X_o, p.b_k1, p.ke1, p.w_x
        read = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1)
    else:
        L, b_k, ob, swaps = p.B + 1, p.b_k2, p.ke2 + p.w_b, p.w_b
        read = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2)
    widths, live, width, k = read.values()
    fixup = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width, ob,
                              False)
    n = lg(odd_part(L))
    return swaps + b_k + fixup.toffoli + (3 * (n - 1) + 2 * (p.s_u - 1) if n else 0)


def closed_form(pr) -> costs.Cost:
    if pr.prepare is OuterPrepare:
        return costs.outer_unprepare_from_outcomes(pr.params)
    return costs.inner_unprepare_from_outcomes(pr.params)


def alone(pr):
    """The unprepare alone, with constant junk outcomes (a bool array under capture)."""
    junk = np.zeros(len(pr.prepare_wires[pr.junk]), dtype=bool)
    return lambda: pr.unprepare(junk, *pr.args, **pr.unprepare_wires)


def assert_counts(pr):
    resources = compiled_resources(alone(pr), pr.num_wires)
    expected = closed_form(pr)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == pr.measured // (pr.params.K1 if pr.prepare is OuterPrepare
                                                     else pr.params.K2)
    assert resources.measured_and_uncomputes == expected_and_uncomputes(pr)
    assert resources.counts.get("Toffoli", 0) == 0


@pytest.mark.compiled
@pytest.mark.parametrize("key", ["outer-sto3g", "outer-631g", "outer-x-o-5", "inner-sto3g",
                                 "inner-631g", "inner-6-2"])
def test_counts_match_the_closed_forms(key):
    """T4 and the T6 signature, each class alone: T equals the baseline unprepare's (dfthc.costs),
    M is the target's d (not the K d of the baseline: the junk was measured by the caller), and A
    is the baseline unprepare's."""
    assert_counts(pair(key))


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("build", [outer, inner], ids=["outer", "inner"])
def test_counts_at_mid_naphthalene(build):
    """T4 at MID_NAPHTHALENE with the naphthalene tables (nightly tier): outer T 38, M 21; inner
    T 53, M 23."""
    p = MID_NAPHTHALENE
    pr = build(p)
    t = WalkTables.from_factors(load_factors("naphthalene"), p)
    args = ((p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u) if build is outer
            else (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn,
                  p.b_k2, p.k2, p.ke2, p.s_u))
    assert_counts(pr._replace(args=args))


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 on the closed forms, nothing compiled: T 69 and 217 as the baseline unprepares,
    M 24 and 26 (the targets; the block encoding measures the 72 and 390 junk qubits); the
    registers are the prepares' without the junk, with extra_onehot 0 (2**4 <= 24) and 102
    (2**7 - 26) wires."""
    p = FEMOCO54
    assert costs.outer_unprepare_from_outcomes(p) == costs.Cost(69, 24)
    assert costs.inner_unprepare_from_outcomes(p) == costs.Cost(217, 26)
    assert costs.outer_unprepare(p).measurements == 24 + 72
    assert costs.inner_unprepare(p).measurements == 26 + 390
    assert OuterUnprepareFromOutcomes.junk_size(p.X_o, p.b_k1, p.k1) == 72
    assert InnerUnprepareFromOutcomes.junk_size(p.N, p.R, p.B, p.C, p.b_k2, p.k2) == 390
    outer_sizes = OuterUnprepareFromOutcomes.required_wires(*outer_structure(p))
    assert outer_sizes == {"x_o": 9, "succ_x_o": 1, "succ_work_x_o": 6, "sigma_1": 1,
                           "m_x_o": 15, "less_than_x_o": 1, "cmp_work_x_o": 14, "alt_x_o": 9,
                           "keep_x_o": 15, "phase_grad": 7, "work": 20, "extra_onehot": 0}
    inner_sizes = InnerUnprepareFromOutcomes.required_wires(*inner_structure(p))
    assert inner_sizes == {"x_o": 9, "b": 5, "succ_b": 1, "succ_work_b": 2, "sigma_0": 1,
                           "m_b": 15, "less_than_b": 1, "cmp_work_b": 14, "alt_b": 5,
                           "keep_b": 15, "sgn": 1, "d_sgn": 1, "r": 4, "phase_grad": 7,
                           "work": 16, "extra_onehot": 102}


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: in a program with both classes alone, the solver names both pinned rules and no
    QROAMErase rule (the erase is split, spec 16); T5 at level="device" (L-20)."""
    outer_pair, inner_pair = pair("outer-x-o-5"), pair("inner-sto3g")

    def program():
        alone(outer_pair)()
        alone(inner_pair)()

    num_wires = max(outer_pair.num_wires, inner_pair.num_wires)
    qp.analyze(verbose_program(program, num_wires=num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_outer_unprepare_from_outcomes", "_inner_unprepare_from_outcomes"} <= rules, rules
    assert "_qroam_erase" not in rules
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize("key", ["outer-sto3g", "outer-x-o-5", "outer-x-o-7-k0-b1",
                                 "inner-sto3g", "inner-6-2", "inner-7-0"])
def test_rules_are_valid(key):
    """T2 in tape mode, with junk outcomes measured before the operator, in every branch of the
    hand-written resources: l = 1 (sto3g), l > 1, no junk and no comparator work (k = 0,
    b_k1 = 1), extra_onehot of 0 (outer) and 1 or more (inner) wires."""
    pr = pair(key)
    with qp.queuing.AnnotatedQueue():
        outcomes = measure_x(pr.prepare_wires[pr.junk])
    rule = (_outer_unprepare_from_outcomes if pr.unprepare is OuterUnprepareFromOutcomes
            else _inner_unprepare_from_outcomes)
    qp.testing.assert_valid_decomp_rule(pr.unprepare(outcomes, *pr.args, **pr.unprepare_wires),
                                        rule, check_matrix=False)


def test_required_wires_and_junk_sizes_by_hand():
    """At H2_STO3G_TEST: the prepares' registers without the junk; extra_onehot 0 for the outer
    one-hot (2**1 <= d1 = 4) and 1 for the inner (2**(1 + 2) - d2 = 8 - 7); 4 and 7 outcomes."""
    p = H2_STO3G_TEST
    assert OuterUnprepareFromOutcomes.required_wires(*outer_structure(p)) == {
        "x_o": 2, "succ_x_o": 1, "succ_work_x_o": 0, "sigma_1": 1, "m_x_o": 2,
        "less_than_x_o": 1, "cmp_work_x_o": 1, "alt_x_o": 2, "keep_x_o": 2, "phase_grad": 3,
        "work": 1, "extra_onehot": 0}
    assert InnerUnprepareFromOutcomes.required_wires(*inner_structure(p)) == {
        "x_o": 2, "b": 2, "succ_b": 1, "succ_work_b": 1, "sigma_0": 1, "m_b": 2,
        "less_than_b": 1, "cmp_work_b": 1, "alt_b": 2, "keep_b": 2, "sgn": 1, "d_sgn": 1,
        "r": 1, "phase_grad": 3, "work": 7, "extra_onehot": 1}
    assert OuterUnprepareFromOutcomes.junk_size(p.X_o, p.b_k1, p.k1) == 4
    assert InnerUnprepareFromOutcomes.junk_size(p.N, p.R, p.B, p.C, p.b_k2, p.k2) == 7


@pytest.mark.parametrize(("key", "change", "match"), [
    ("outer-x-o-5", {"junk_outcomes": "short"}, r"need one outcome per junk qubit \(6\), got 5"),
    ("outer-x-o-5", {"junk_outcomes": "booleans"},
     "in tape mode every junk outcome must be a MeasurementValue; entry 0 is"),
    ("outer-x-o-5", {"junk_outcomes": "abstract"}, r"junk_outcomes must have shape \(6,\), got"),
    ("outer-x-o-5", {"extra_onehot": [99]}, "'extra_onehot' has 1 wires; expected 0"),
    ("outer-x-o-5", {"ke1": 3, "k1": 0}, "needs 8 wires, more than the 6 target and junk wires"),
    ("inner-sto3g", {"junk_outcomes": "short"}, r"need one outcome per junk qubit \(7\), got 6"),
    ("inner-sto3g", {"extra_onehot": []}, "'extra_onehot' has 0 wires; expected 1"),
    ("inner-sto3g", {"r": []}, "'r' has 0 wires; expected 1"),
    ("inner-sto3g", {"ke2": 2}, r"ke2 must lie in \[0, 1\]"),
], ids=["outer-short", "outer-booleans", "outer-abstract", "outer-extra-onehot", "outer-onehot",
        "inner-short", "inner-extra-onehot", "inner-r", "inner-ke2"])
def test_constructors_validate(key, change, match):
    """Spec 9.0: the junk outcomes must be one MeasurementValue per junk qubit in tape mode (or an
    array of their shape), the registers exactly required_wires, and the compilable arguments in
    the prepare's domains, the one-hot fitting in the reset target and junk."""
    pr = pair(key)
    names = pr.unprepare.compilable_argnames
    arguments = dict(zip(names, pr.args, strict=True)) | pr.unprepare_wires
    with qp.queuing.AnnotatedQueue():
        outcomes = measure_x(pr.prepare_wires[pr.junk])
    junk = {"short": outcomes[:-1], "booleans": [False] * len(outcomes),
            "abstract": AbstractArray((len(outcomes) - 1,), bool), None: outcomes}
    arguments |= {name: value for name, value in change.items() if name != "junk_outcomes"}
    with pytest.raises(ValueError, match=match):
        pr.unprepare(junk[change.get("junk_outcomes")], **arguments)


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", ["adjoint", "control"])
def test_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rules 5 and 6: nothing adjoints or controls an unprepare from outcomes. Wrapped
    anyway, Catalyst drops the measuring candidates with the expected warning (rule 12), and the
    compile fails loudly."""
    pr = pair("inner-sto3g")
    key = ("Adjoint(InnerUnprepareFromOutcomes)" if wrap == "adjoint"
           else "C(InnerUnprepareFromOutcomes)")

    def program():
        op = alone(pr)()
        return qp.adjoint(op) if wrap == "adjoint" else qp.ctrl(op, control=pr.num_wires)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(program, pr.num_wires + 1)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}
