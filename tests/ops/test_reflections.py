"""The reflections ref_t2 and ref_ab (spec 9.19, 9.20): a multi-controlled Z (PennyLane's
MultiControlledX with static control values) with GlobalPhase(pi), controlled with Z on the
control. The exact matrix with the global sign, the action at the walk's
widths with the control in |0>, |1> and |+>, counts and work sizes."""

import math

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.reflections import ref_ab, ref_t2, reflection_work
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import WalkLayout

PRESETS = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST, "mid": MID_NAPHTHALENE,
           "femoco-width": FEMOCO54}
HELPERS = {
    "ref_t2": (ref_t2, ("succ_x_o", "b", "m_b", "sigma_0"), {"succ_x_o"}, costs.ref_t2),
    "ref_ab": (ref_ab, ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0"), set(), costs.ref_ab),
}
"""Each helper, its pattern registers in argument order, the registers with pattern value 1, and
its closed form."""


def pattern_sizes(kind, p=None) -> dict[str, int]:
    """The widths of the helper's pattern registers: the walk's at ``p`` (WalkLayout's), else 1."""
    sizes = WalkLayout.register_sizes(p) if p is not None else {}
    return {name: sizes.get(name, 1) for name in HELPERS[kind][1]}


def split(sizes, controlled, wires) -> dict:
    """``wires`` in order: the control first if ``controlled``, then the pattern registers."""
    registers, start = {}, int(controlled)
    if controlled:
        registers["control"] = wires[0]
    for name, size in sizes.items():
        registers[name], start = list(wires[start:start + size]), start + size
    return registers


def layout(sizes, controlled, lent=1) -> tuple[dict, int]:
    """The helper's arguments on consecutive wires (the work, then ``lent`` extra_work wires,
    last; the walk lends one) and the number of wires."""
    n = sum(sizes.values()) + controlled
    work = reflection_work(sum(sizes.values()), controlled)
    wires = split(sizes, controlled, list(range(n))) | {"work": list(range(n, n + work))}
    if lent:
        wires["extra_work"] = list(range(n + work, n + work + lent))
    return wires, n + work + lent


def reflection_diagonal(sizes, ones):
    """Diagonal of 2 Pi - I over the pattern registers (wire order, big-endian): +1 on the one
    basis state with the registers in ``ones`` all ones and the others all zeros, -1 elsewhere."""
    target = "".join(("1" if name in ones else "0") * n for name, n in sizes.items())
    return np.where(np.arange(2 ** len(target)) == int(target, 2), 1.0, -1.0)


def choi_matrix(num_system, apply, num_work):
    """The matrix of ``apply`` on wires 0 .. num_system - 1, from one compiled lightning run.

    Each system wire is entangled with a reference wire (H, CNOT), so the output amplitude of
    |y>_system |x>_reference |0>_work is U[y, x] / sqrt(2^n); work must come back to |0>.
    """
    system = list(range(num_system))
    reference = list(range(num_system, 2 * num_system))
    work = list(range(2 * num_system, 2 * num_system + num_work))

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=2 * num_system + num_work))
    def circuit():
        for s, r in zip(system, reference, strict=True):
            qp.H(r)
            qp.CNOT([r, s])
        apply(system, work)
        return qp.state()

    state = np.asarray(circuit()).reshape(2**num_system, 2**num_system, 2**num_work)
    assert np.allclose(state[:, :, 1:], 0, rtol=0, atol=1e-10)
    return state[:, :, 0] * np.sqrt(2**num_system)


@pytest.mark.compiled
@pytest.mark.parametrize("controlled", [False, True], ids=["plain", "controlled"])
@pytest.mark.parametrize("kind", HELPERS)
def test_exact_matrix_with_the_global_sign(kind, controlled):
    """T3: at unit register widths the compiled matrix is exactly 2 Pi - I, global sign included,
    and controlled |0><0| (x) I + |1><1| (x) (2 Pi - I), where a wrong global sign of the
    uncontrolled reflection would show as a relative phase; the work and the lent extra_work
    wire return to |0>."""
    lent = 1
    helper, _, ones, _ = HELPERS[kind]
    sizes = pattern_sizes(kind)
    n_w = sum(sizes.values())
    num_work = reflection_work(n_w, controlled)

    def apply(system, work):
        helper(**split(sizes, controlled, system), work=work[:num_work],
               extra_work=work[num_work:])

    matrix = choi_matrix(n_w + controlled, apply, num_work + lent)
    diagonal = reflection_diagonal(sizes, ones)
    expected = np.concatenate([np.ones(2**n_w), diagonal]) if controlled else diagonal
    assert np.allclose(matrix, np.diag(expected), rtol=0, atol=1e-10)


def phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


def phased_amplitudes(n):
    """The state phased_prepare makes on n wires, wire 0 the most significant bit."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return (1j ** bits[:, ::2].sum(axis=1) * (-1.0) ** (bits[:, :-1] * bits[:, 1:]).sum(axis=1)
            / math.sqrt(2**n))


@pytest.mark.compiled
@pytest.mark.parametrize("preset", ["sto3g", "631g"])
@pytest.mark.parametrize("kind", HELPERS)
def test_action_at_the_walks_widths(kind, preset):
    """T3 on the compiled path (lightning) at the walk's register widths of both test presets,
    one program with the control's state as a runtime argument: on a phased state of every
    pattern register (every amplitude nonzero, phases not uniform) the uncontrolled helper gives
    exactly (2 Pi - I) applied, global sign included; controlled, the control in |0> leaves the
    state unchanged, in |1> applies 2 Pi - I, and in |+> gives (|0> psi + |1> (2 Pi - I) psi) /
    sqrt 2, which detects a wrong phase on the controlled branch (spec 3.3); work ends in |0>."""
    helper, pattern, ones, _ = HELPERS[kind]
    sizes = pattern_sizes(kind, PRESETS[preset])
    psi = phased_amplitudes(sum(sizes.values()))
    reflected = reflection_diagonal(sizes, ones) * psi

    def run(controlled):
        wires, num_wires = layout(sizes, controlled)
        pattern_wires = [wire for name in pattern for wire in wires[name]]

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit(mode):
            if controlled:
                qp.cond(mode == 1, qp.X)(wires["control"])
                qp.cond(mode == 2, qp.H)(wires["control"])
            phased_prepare(pattern_wires)
            helper(**wires)
            return qp.state()

        clean = np.eye(2 ** (len(wires["work"]) + len(wires["extra_work"])))[0]
        return circuit, clean

    circuit, clean = run(False)
    assert np.allclose(np.asarray(circuit(0)), np.kron(reflected, clean), rtol=0, atol=1e-10)
    circuit, clean = run(True)
    zero, one = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    expected = {0: np.kron(zero, psi), 1: np.kron(one, reflected),
                2: (np.kron(zero, psi) + np.kron(one, reflected)) / math.sqrt(2)}
    for mode, state in expected.items():
        assert np.allclose(np.asarray(circuit(mode)), np.kron(state, clean), rtol=0,
                           atol=1e-10), mode


COUNT_CASES = [pytest.param(kind, controlled, preset,
                            id=f"{kind}-{'controlled' if controlled else 'plain'}-{preset}")
               for kind in HELPERS for controlled in (False, True) for preset in PRESETS]


@pytest.mark.compiled
@pytest.mark.parametrize(("kind", "controlled", "preset"), COUNT_CASES)
def test_counts(kind, controlled, preset):
    """T4 compiled at both test presets, MID_NAPHTHALENE and FeMoco-54's register widths (the
    reflections are table-free, so per commit), with the one extra_work wire the walk lends:
    T = n_w - 2, controlled n_w - 1, against dfthc.costs.ref_t2 and ref_ab, M = 0, all
    TemporaryAND with measured uncomputes and no Toffoli (the MultiControlledX's core included),
    so A = T; the sign is one GlobalPhase uncontrolled and one PauliZ on the control
    controlled."""
    helper, _, _, cost = HELPERS[kind]
    p = PRESETS[preset]
    wires, num_wires = layout(pattern_sizes(kind, p), controlled)
    resources = compiled_resources(lambda: helper(**wires), num_wires)
    expected = cost(p, controlled).toffoli
    assert (resources.toffoli, resources.measurements) == (expected, 0)
    assert (resources.counts.get("Toffoli", 0), resources.counts["TemporaryAND"]) == (0, expected)
    assert resources.measured_and_uncomputes == expected
    assert (resources.counts.get("GlobalPhase", 0), resources.counts.get("PauliZ", 0)) == (
        (0, 1) if controlled else (1, 0))


@pytest.mark.parametrize("controlled", [False, True], ids=["plain", "controlled"])
@pytest.mark.parametrize("kind", HELPERS)
def test_at_least_two_controls_need_the_lent_wire(kind, controlled):
    """GATE_SET has no Toffoli, so a reflection whose MultiControlledX has at least two controls
    raises ValueError without its extra_work wire (it would have no decomposition path, L-18),
    and one with fewer queues without it: n_w = 1 has no MultiControlledX, and the controlled
    n_w = 1 and the plain n_w = 2 have one control."""
    helper = HELPERS[kind][0]
    wires, _ = layout(pattern_sizes(kind), controlled, lent=0)
    with pytest.raises(ValueError, match="needs one zeroed extra_work wire"):
        with qp.queuing.AnnotatedQueue():
            helper(**wires)
    for n_w in (1, 2) if not controlled else (1,):
        with qp.queuing.AnnotatedQueue() as queue:
            ref_ab([0], [], [], [], [], list(range(1, n_w)), [],
                   control=n_w if controlled else None)
        assert queue.queue


def test_femoco_counts_and_work_on_the_closed_form():
    """FeMoco-54 (spec 9.24): ref_t2 20 and 21 controlled, ref_ab 44 and 45 controlled; the work
    is n_w - 3, controlled n_w - 2, and the controlled ref_ab's 44 wires are the walk's largest
    need, WalkLayout.work_size."""
    p = FEMOCO54
    assert [costs.ref_t2(p, False), costs.ref_t2(p, True)] == [costs.Cost(20), costs.Cost(21)]
    assert [costs.ref_ab(p, False), costs.ref_ab(p, True)] == [costs.Cost(44), costs.Cost(45)]
    works = [reflection_work(sum(pattern_sizes(kind, p).values()), controlled)
             for kind in HELPERS for controlled in (False, True)]
    assert works == [19, 20, 43, 44]
    assert works[-1] == WalkLayout.work_size(p)
    assert [reflection_work(n_w, False) for n_w in (1, 3, 4)] == [0, 0, 1]
    assert [reflection_work(n_w, True) for n_w in (1, 2, 3)] == [0, 0, 1]


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" (L-20) for both helpers, plain and controlled, at H2_STO3G_TEST."""
    calls = [layout(pattern_sizes(kind, H2_STO3G_TEST), controlled) + (HELPERS[kind][0],)
             for kind in HELPERS for controlled in (False, True)]

    def program():
        for wires, _, helper in calls:
            helper(**wires)

    compiled_resources(program, max(num for _, num, _ in calls), level="device")
