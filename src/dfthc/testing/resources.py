"""Assertions on compiled resources and the rule-lowering warning policy.

Spec sections 7.1 (warning noise), 7.4 rule 12 (warning policy) and 7.6 (accounting).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable

import pennylane as qp
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.passes import graph_decomposition

from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    PENNYLANE_PINS,
    CompiledResources,
    compiled_resources,
    dfthc_operator_names,
)

_EXCLUDED_RULE = (
    r"(?:_(?:\w*_)?ppm|_toffoli_elbow|_cphase_to_single_phase|\w*_workers?\w*"
    r"|use_allocation\(\w+\))"
)
NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^Excluded the {_EXCLUDED_RULE} decomposition rule for \S+ since "
        r"--decompose-lowering cannot work with multiple registers yet$"
    ),
    re.compile(
        r"^Skipped the _adjoint_temporary_and\w* (?:decomposition )?rule for "
        r"C\(Adjoint\(TemporaryAND\)\)"
    ),
)
"""RuleLoweringWarning texts that PennyLane's own rules print on every compile (spec 7.1)."""

UNRESOLVED_BRANCH_MESSAGE = (
    r"Resource analysis was unable to determine the branch of a conditional or switch statement"
)
"""qp.analyze UserWarning for an unhinted conditional. PennyLane's SemiAdder rule triggers it on
every compile, so it is filtered; exact T4 counts and the erase-hint tests guard our own code."""


_EXPECTED_SHAPE = re.compile(
    r"^Skipped the \S+ decomposition rule for "
    r"(?P<wrapped>(?P<open>(?:(?:Adjoint|C)\()+)(?P<name>\w+)(?P<close>\)+))"
    r"(?:\{[^{}]*\})*"
    r": it contains a mid-circuit measurement, which is not supported with adjoint or control "
    r"regions\.\Z"
)
"""Catalyst's drop of a measuring rule for a modified operator (spec 7.4 rule 12). Nested
``Adjoint(...)``/``C(...)`` wrappers around one operator name, then only attribute dumps; the
parentheses are balanced if the wrapper opens equal the closes (checked in the classifier)."""


def classify_rule_warning(message: str, names: Iterable[str] | None = None) -> str:
    """Return ``"expected"``, ``"dfthc"``, ``"noise"`` or ``"other"`` for a RuleLoweringWarning.

    ``"expected"`` is exactly Catalyst's "Skipped the <rule> decomposition rule for <wrapped op>:
    it contains a mid-circuit measurement, which is not supported with adjoint or control
    regions." where ``<wrapped op>`` is nested ``Adjoint(...)``/``C(...)`` around an operator in
    ``names`` and that operator is the only name of ``names`` in the message. Every other message
    naming a dfthc operator is ``"dfthc"``, even if it also matches a noise pattern. ``names``
    defaults to the dfthc operator names of the pinned rules; a bare ``str`` raises ``TypeError``.
    """
    if isinstance(names, str):
        raise TypeError(f"names must be an iterable of operator names, not the string {names!r}")
    names = dfthc_operator_names() if names is None else frozenset(names)
    named = {name for name in names if re.search(rf"\b{re.escape(name)}\b", message)}
    match = _EXPECTED_SHAPE.match(message)
    if (
        match
        and named == {match["name"]}
        and match["open"].count("(") == len(match["close"])
    ):
        return "expected"
    if named:
        return "dfthc"
    if any(pattern.search(message) for pattern in NOISE_PATTERNS):
        return "noise"
    return "other"


def install_warning_filters() -> None:
    """Show every RuleLoweringWarning; the test fixture drops the known PennyLane noise after
    classification."""
    warnings.filterwarnings("always", category=RuleLoweringWarning)
    warnings.filterwarnings("ignore", message=UNRESOLVED_BRANCH_MESSAGE, category=UserWarning)


def verbose_program(qfunc: Callable[[], None], num_wires: int):
    """The compile stack of spec 7.2 with the solver's rule listing forwarded to stderr (T6).

    Only ``qjit(verbose=True)`` forwards the compiler output. Run the returned program (or
    ``qp.analyze`` it) under pytest's ``capfd`` and pass the captured text to :func:`chosen_rules`.
    """

    @qp.qjit(capture=True, verbose=True)
    @graph_decomposition(gate_set=GATE_SET, fixed_decomps={**FIXED_DECOMPS, **PENNYLANE_PINS},
                         verbose=True)
    @qp.qnode(qp.device("null.qubit", wires=num_wires))
    def program():
        qfunc()
        return qp.probs(wires=[0])

    return program


def chosen_rules(listing: str) -> list[str]:
    """Rule names from the lines ``Chosen Rule: __builtin_<rule>`` of a verbose listing."""
    prefix = "Chosen Rule: __builtin_"
    return [line.split(prefix, 1)[1].strip() for line in listing.splitlines() if prefix in line]


def captured_targets(qfunc: Callable[[], None], num_wires: int) -> list[str]:
    """The rule targets the Catalyst frontend converts while :func:`compiled_resources` compiles
    ``qfunc`` once, one name per converted target, in order: the operator name with its modifiers
    (``Adjoint(MajoranaGate)``, ``C(PauliZ)``), without the attribute dump that starts at the first
    ``{`` (spec 12.1, T6: the inventory test).

    A port of Task 0.9d's capture instrument (``spikes/k1_whole_program.capture_instrument``): it
    wraps Catalyst's private ``catalyst.from_plxpr.qfunc_interpreter._convert_decomp_target_spec``
    for this one compile and restores it afterwards. The frontend converts every target it builds
    rules for, chosen or not, so the list holds every distributed ``Adjoint``/``C`` candidate the
    compile captured. The hook is private, which makes this a compatibility canary at every
    nightly bump: a renamed hook raises ``AttributeError``, and callers assert that the program's
    own top-level target is in the list, so a hook that no longer sees targets fails instead of
    passing. The private module is imported here, so a renamed module fails only the callers.
    """
    import catalyst.from_plxpr.qfunc_interpreter as qfunc_interpreter

    convert = qfunc_interpreter._convert_decomp_target_spec
    names: list[str] = []

    def recording(interpreter, target_spec):
        names.append(str(target_spec.target_id).split("{", 1)[0])
        return convert(interpreter, target_spec)

    qfunc_interpreter._convert_decomp_target_spec = recording
    try:
        compiled_resources(qfunc, num_wires)
    finally:
        qfunc_interpreter._convert_decomp_target_spec = convert
    return names


LEAF_ADJOINTS = frozenset({"Adjoint(FlaggedUniformPrep)", "Adjoint(GivensRotation)",
                           "Adjoint(MajoranaGate)", "Adjoint(PhaseGradientRotation)"})
"""The only modified dfthc targets a walk program may capture: the auto-derived leaf adjoints of
spec 7.4 rule 6 (the uniform preparation's and, inside it, its rotations'; the Givens rung's; the
Majorana gate's)."""

TEMPLATE_ADJOINTS = frozenset({"Adjoint(SemiAdder)", "Adjoint(LeftQuantumComparator)",
                               "Adjoint(LeftClassicalComparator)"})
"""The PennyLane templates those leaves and the unprepares adjoint: the rotations' and Givens
rungs' adders and the comparators (spec 7.4 rule 4, K-7)."""


def modified_targets(names: Iterable[str], bases: Iterable[str]) -> set[str]:
    """The ``Adjoint(...)`` and ``C(...)`` targets of ``names`` (any nesting, ``2C(...)``
    included) whose operator is one of ``bases``."""
    bases = set(bases)
    return {name for name in names if "(" in name and dfthc_operator_names([name]) & bases}


def inventory(names: Iterable[str]) -> set[str]:
    """Spec 12.1 T6: the modified targets of ``names`` (from :func:`captured_targets`) whose
    operator is a dfthc operator (every name with a pinned rule: the composites, ``QROAMRead`` and
    the leaves) or a template of :data:`TEMPLATE_ADJOINTS`. For a walk program it must equal
    ``LEAF_ADJOINTS | TEMPLATE_ADJOINTS``, and for a controlled-walk or QPE program that plus
    its controlled walk's target (``C(DFTHCWalk)``, the toy's ``C(ToyWalk)``), whose rule is
    pinned, and, for the DFTHC walk, ``C(DFTHCBlockEncoding)``, where Catalyst's generic
    controlled candidate of ``C(DFTHCWalk)`` stops (its base rule measures): no other modified
    composite or read."""
    templates = {name[len("Adjoint("):-1] for name in TEMPLATE_ADJOINTS}
    return modified_targets(names, dfthc_operator_names() | templates)


def assert_resources(
    resources: CompiledResources, *, toffoli: int, measurements: int,
    measured_and_uncomputes: int | None = None,
) -> None:
    """Assert exact T, M and optionally A, with every mismatch in one message.

    T, M and A exist only in a ``level="user"`` record; any other record fails the assertion.
    """
    if resources.level != "user":
        raise AssertionError(
            f"assert_resources needs a level='user' record, got level={resources.level!r}"
        )
    expected = {"toffoli": toffoli, "measurements": measurements}
    if measured_and_uncomputes is not None:
        expected["measured_and_uncomputes"] = measured_and_uncomputes
    actual = {name: getattr(resources, name) for name in expected}
    if actual != expected:
        raise AssertionError(
            f"resources differ: expected {expected}, got {actual}; counts {resources.counts}"
        )
