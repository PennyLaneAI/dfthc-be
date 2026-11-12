"""Spike K-10: measurement-valued arrays as dynamic Operator2 arguments, in both modes.

Three arrays cross an operator boundary (spec section 16, K-10):
1. parities: the caller X-measures a read's target and junk, computes the fixup parities with
   mcm_parities and passes them as one dynamic array to ProtoParityFixup;
2. junk outcomes: the caller X-measures only the junk and passes the outcome array to
   ProtoUnprepareFromOutcomes (shaped like InnerUnprepareFromOutcomes, Q-1 variant (b)), which
   measures the target and computes the parities from the concatenated outcomes;
3. earlier_bits: a QPERound-like operator receives a boolean array of shape (t,), with no entry
   set in round j = t-1 and set entries in later rounds.
Each part is simulated at a small size in both modes (lightning.qubit under capture; the tape
path with deferred measurements on default.qubit) and must be exact. Parts 1 and 2 also compile
at two mid sizes with the FeMoco-54 inner-erase widths (416 outcomes, 390 of them junk) over 544
and 1088 addresses, with non-symbolic counts; the growth exponent of each part's compile time
between the two sizes is recorded. Every warning of each check is recorded and classified, and a
warning that names a prototype operator fails that check (spec 7.4, rule 12). Writes
spikes/results/k10.json; exit status 1 on failure.

The FeMoco-54 shape (8640 addresses) is not compiled (master plan section 12.9): the timings of the
plan's dry-run are quoted in spikes/results/k10_femoco_dryrun.json.

Run: CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 .venv/bin/python spikes/k10_measurement_arrays.py
(add --small-only to skip the four mid-size compiles, about 80 s together)
"""

import json
import math
import sys
import time
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.transforms import _tape_decompose
from pennylane.typing import AbstractArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototypes import (  # noqa: E402
    ProtoTableRead,
    _Count,
    _Emit,
    _measure_x,
    classify_warnings,
    counts_to_resources,
    from_words,
    loaded_masks,
    nested_iterate,
    to_words,
)

from dfthc.compile import (  # noqa: E402
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    mcm_parities,
    register_rule,  # noqa: E402
)


def as_dynamic_array(values):
    """Canonicalize a list of measurement values for a dynamic argument (L-23)."""
    if isinstance(values, AbstractArray):
        return values
    if qp.capture.enabled():
        return jnp.stack([jnp.asarray(v, dtype=bool) for v in values])
    return np.asarray(values, dtype=object)


def _fixup_leaves(emit, registers, work, parity_of):
    """Plain fixup: a conditional Z on the flag of every address, in iteration order."""
    widths = [len(w) for w, _, _ in registers]
    index = {"next": 0}

    def leaf(flag, values):
        parity = parity_of(index["next"])
        index["next"] += 1
        if parity is not False:
            emit.cond_z(parity, flag)

    nested_iterate(emit, registers, work, leaf)
    del widths


class _EmitCond(_Emit):
    def cond_z(self, parity, flag):
        qp.cond(parity, qp.Z)(flag)


class _CountCond(_Count):
    def cond_z(self, parity, flag):
        self.counts["PauliZ"] += 1


def _fixup_counts(register_ranges, num_measured):
    counter = _CountCond()
    dummy = list(range(10_000))
    registers = [(dummy[:w], lo, hi) for w, lo, hi in register_ranges]
    _fixup_leaves(counter, registers, dummy[:64], lambda a: True)
    counter.counts["Hadamard"] += num_measured
    counter.counts["MidMeasure"] += num_measured
    return counter.counts


class ProtoParityFixup(Operator2):
    """Part 1: applies (-1)^parities[a] on the a-th address of ``register_ranges``."""

    dynamic_argnames = ("parities",)
    compilable_argnames = ("register_ranges",)
    wire_argnames = ("address", "work")

    def __init__(self, parities, register_ranges, address, work):
        super().__init__(as_dynamic_array(parities), register_ranges, address, work)


def _split(address, register_ranges):
    registers, start = [], 0
    for width, lo, hi in register_ranges:
        registers.append((address[start:start + width], lo, hi))
        start += width
    return registers


def _parity_fixup_resources(parities, register_ranges, address, work):
    return counts_to_resources(_fixup_counts(register_ranges, 0))


@qp.register_resources(_parity_fixup_resources)
def _proto_parity_fixup(parities, register_ranges, address, work):
    _fixup_leaves(_EmitCond(), _split(address, register_ranges), work, lambda a: parities[a])


register_rule(ProtoParityFixup, _proto_parity_fixup)


class ProtoUnprepareFromOutcomes(Operator2):
    """Part 2: X-measures ``target``, then fixes phases from (target ‖ junk_outcomes).

    ``masks`` holds one tuple of 32-bit words per address (compilable integers must fit in
    int64, L-24).
    """

    dynamic_argnames = ("junk_outcomes",)
    compilable_argnames = ("masks", "register_ranges")
    wire_argnames = ("address", "target", "work")

    def __init__(self, junk_outcomes, masks, register_ranges, address, target, work):
        super().__init__(as_dynamic_array(junk_outcomes), masks, register_ranges, address,
                         target, work)


def _unprepare_resources(junk_outcomes, masks, register_ranges, address, target, work):
    counts = _fixup_counts(register_ranges, len(target))
    counts["PauliZ"] = sum(1 for m in masks if from_words(m))
    return counts_to_resources(counts)


@qp.register_resources(_unprepare_resources)
def _proto_unprepare(junk_outcomes, masks, register_ranges, address, target, work):
    target_outcomes = [_measure_x(q) for q in target]
    junk = [junk_outcomes[i] for i in range(junk_outcomes.shape[0])]
    parities = mcm_parities(target_outcomes + junk, [from_words(m) for m in masks])
    _fixup_leaves(_EmitCond(), _split(address, register_ranges), work, lambda a: parities[a])


register_rule(ProtoUnprepareFromOutcomes, _proto_unprepare)


class ProtoRound(Operator2):
    """Part 3: X on ``control`` for round t-1, then X conditioned on every earlier bit l > j."""

    dynamic_argnames = ("earlier_bits",)
    compilable_argnames = ("j", "num_bits")
    wire_argnames = ("control",)

    def __init__(self, j, num_bits, earlier_bits, control):
        super().__init__(j, num_bits, as_dynamic_array(earlier_bits), control)


def _round_resources(j, num_bits, earlier_bits, control):
    return {qp.X: int(j == num_bits - 1) + (num_bits - 1 - j)}


@qp.register_resources(_round_resources)
def _proto_round(j, num_bits, earlier_bits, control):
    if j == num_bits - 1:
        qp.X(control[0])
    for later in range(j + 1, num_bits):
        qp.cond(earlier_bits[later], qp.X)(control[0])


register_rule(ProtoRound, _proto_round)

# --------------------------------------------------------------------------- small instances

WIDTH, BLOCK_BITS = 2, 1
HIGH_RANGES = ((2, 0, 3),)
TABLE = tuple((a, to_words(v, WIDTH)) for a, v in enumerate([1, 2, 3, 1, 2]))
HIGH, LOW, TARGET, JUNK, WORK = [0, 1], [2], [3, 4], [5, 6], [7, 8, 9, 10]
MASK_INTS = tuple(m for _, m in loaded_masks(TABLE, HIGH_RANGES, BLOCK_BITS, WIDTH))
MASKS = tuple(to_words(m, WIDTH * 2**BLOCK_BITS) for m in MASK_INTS)
FIXUP_RANGES = ((2, 0, 3), (1, 0, 2))  # high part, then the low (block) bit


def part1_small():
    for wire in HIGH + LOW:
        qp.H(wire)
    ProtoTableRead(TABLE, HIGH_RANGES, BLOCK_BITS, WIDTH, HIGH, LOW, TARGET, JUNK, WORK)
    outcomes = [_measure_x(q) for q in TARGET + JUNK]
    ProtoParityFixup(mcm_parities(outcomes, MASK_INTS), FIXUP_RANGES, HIGH + LOW, WORK)


def part2_small():
    for wire in HIGH + LOW:
        qp.H(wire)
    ProtoTableRead(TABLE, HIGH_RANGES, BLOCK_BITS, WIDTH, HIGH, LOW, TARGET, JUNK, WORK)
    junk_outcomes = [_measure_x(q) for q in JUNK]
    ProtoUnprepareFromOutcomes(junk_outcomes, MASKS, FIXUP_RANGES, HIGH + LOW, TARGET, WORK)


NUM_BITS, CONTROL, OUT = 4, [0], [1, 2, 3, 4]
EXPECTED_BITS = (0, 0, 1, 1)  # m_3 = 1, m_2 = m_3, m_1 = m_3 ^ m_2, m_0 = m_3 ^ m_2 ^ m_1


def part3_small():
    if qp.capture.enabled():
        bits = jnp.zeros(NUM_BITS, dtype=bool)
    else:
        bits = np.full(NUM_BITS, False, dtype=object)
    for j in reversed(range(NUM_BITS)):
        ProtoRound(j, NUM_BITS, bits, CONTROL)
        outcome = qp.measure(CONTROL[0], reset=True)
        if qp.capture.enabled():
            bits = bits.at[j].set(jnp.asarray(outcome, dtype=bool))
        else:
            bits = bits.copy()
            bits[j] = outcome
        qp.cond(outcome, qp.X)(OUT[j])


def restored_address_checks(qfunc):
    """Exact restoration of |+++> on the address in both modes, ancillas back in |0>."""
    plus = np.full(8, 1 / np.sqrt(8))
    expected = np.zeros(2**11, dtype=complex)
    expected[[a << 8 for a in range(8)]] = 1 / np.sqrt(8)
    capture_ok = []
    for seed in range(3):
        @qp.qjit(capture=True, seed=seed)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=11))
        def circuit():
            qfunc()
            return qp.state()

        capture_ok.append(bool(np.allclose(np.asarray(circuit()), expected, atol=1e-10)))
    qp.decomposition.enable_graph()
    with qp.queuing.AnnotatedQueue() as queue:
        qfunc()
        qp.density_matrix(wires=HIGH + LOW)
        qp.probs(wires=TARGET + JUNK + WORK)
    [tape], _ = _tape_decompose(qp.tape.QuantumScript.from_queue(queue), gate_set=GATE_SET,
                                fixed_decomps=dict(FIXED_DECOMPS))
    [deferred], _ = qp.defer_measurements(tape)
    rho, probs = qp.devices.DefaultQubit().execute(deferred)
    tape_ok = bool(np.allclose(rho, np.outer(plus, plus), atol=1e-10) and np.isclose(probs[0], 1))
    return {"capture_seeds_ok": capture_ok, "tape_ok": tape_ok,
            "passed": all(capture_ok) and tape_ok}


def round_bits_checks():
    expected = np.zeros(16)
    expected[int("".join(map(str, EXPECTED_BITS)), 2)] = 1.0

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=5))
    def circuit():
        part3_small()
        return qp.probs(wires=OUT)

    capture_ok = bool(np.allclose(np.asarray(circuit()), expected))
    qp.decomposition.enable_graph()
    with qp.queuing.AnnotatedQueue() as queue:
        part3_small()
        qp.probs(wires=OUT)
    [tape], _ = _tape_decompose(qp.tape.QuantumScript.from_queue(queue), gate_set=GATE_SET,
                                fixed_decomps=dict(FIXED_DECOMPS))
    [deferred], _ = qp.defer_measurements(tape)
    tape_ok = bool(np.allclose(qp.devices.DefaultQubit().execute(deferred), expected))
    return {"capture_ok": capture_ok, "tape_ok": tape_ok, "passed": capture_ok and tape_ok}


# ------------------------------------------------- mid sizes with the FeMoco-54 widths

TARGET_BITS, JUNK_BITS = 26, 390                   # d_2 and (K_2 - 1) d_2 at FeMoco-54
SHAPES = {                                         # (rows, row bits, b positions, b bits)
    "mid544": (17, 5, 32, 5),
    "mid1088": (34, 6, 32, 5),                     # an eighth of FeMoco-54's 270 x 32
}


def shape_compile(part, shape):
    rows, row_bits, b_positions, b_bits = SHAPES[shape]
    ranges = ((row_bits, 0, rows), (b_bits, 0, b_positions))
    num_addresses = rows * b_positions
    rng = np.random.default_rng(10)
    num_outcomes = TARGET_BITS + JUNK_BITS
    masks = tuple(int.from_bytes(rng.bytes(52), "little") for _ in range(num_addresses))
    address = list(range(row_bits + b_bits))
    target = list(range(14, 14 + TARGET_BITS))
    junk = list(range(40, 40 + JUNK_BITS))
    work = list(range(430, 445))

    def qfunc():
        if part == 1:
            outcomes = [_measure_x(q) for q in target + junk]
            ProtoParityFixup(mcm_parities(outcomes, masks), ranges, address, work)
        else:
            junk_outcomes = [_measure_x(q) for q in junk]
            mask_words = tuple(to_words(m, num_outcomes) for m in masks)
            ProtoUnprepareFromOutcomes(junk_outcomes, mask_words, ranges, address, target, work)

    start = time.perf_counter()
    resources = compiled_resources(qfunc, num_wires=445)
    seconds = round(time.perf_counter() - start, 1)
    return {"seconds": seconds, "measurements": resources.measurements,
            "conditional_z_upper_bound": resources.counts.get("PauliZ", 0),
            "num_addresses": num_addresses, "passed": resources.measurements == num_outcomes}


def growth_exponents(results):
    """Per part, d log(seconds) / d log(addresses) between the two mid sizes."""
    exponents = {}
    for part in (1, 2):
        a, b = (results.get(f"part{part}_{shape}_compile", {}) for shape in SHAPES)
        if a.get("passed") and b.get("passed"):
            exponents[f"part{part}"] = round(
                math.log(b["seconds"] / a["seconds"])
                / math.log(b["num_addresses"] / a["num_addresses"]), 2)
    return exponents


PROTOTYPE_NAMES = {
    "ProtoParityFixup", "ProtoUnprepareFromOutcomes", "ProtoRound", "ProtoTableRead",
}


def run_check(check):
    """Run one check and record its warnings; a warning that names a prototype fails it."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            result = check()
        except Exception as exc:  # noqa: BLE001
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    result["warnings"] = classify_warnings(records, PROTOTYPE_NAMES)
    if result["warnings"]["prototype"]:
        result["passed"] = False
    return result


def console_line(result):
    """The result with its warnings as counts per class; k10.json keeps the texts."""
    counts = {kind: value if isinstance(value, int) else len(value)
              for kind, value in result["warnings"].items()}
    return json.dumps({**result, "warnings": counts}, default=str)


def main():
    checks = {
        "part1_small": lambda: restored_address_checks(part1_small),
        "part2_small": lambda: restored_address_checks(part2_small),
        "part3_small": round_bits_checks,
    }
    if "--small-only" not in sys.argv:
        for shape in SHAPES:
            for part in (1, 2):
                checks[f"part{part}_{shape}_compile"] = (
                    lambda part=part, shape=shape: shape_compile(part, shape))
    results = {}
    for name, check in checks.items():
        results[name] = run_check(check)
        print(name, console_line(results[name]), flush=True)
    passed = all(r.get("passed") for r in results.values())
    if "--small-only" not in sys.argv:
        results["compile_time_growth_exponents"] = growth_exponents(results)
        print("compile_time_growth_exponents", json.dumps(results["compile_time_growth_exponents"]))
    results["verdict"] = "pass" if passed else "fail"
    out = Path(__file__).parent / "results" / "k10.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("verdict:", results["verdict"])
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
