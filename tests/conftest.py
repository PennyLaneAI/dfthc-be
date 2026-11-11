"""Shared pytest configuration (spec sections 7.1 and 12.6)."""

import warnings

import catalyst  # noqa: F401  # importing Catalyst reads CATALYST_SILENCE_RULE_LOWERING_WARNINGS
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

# Second line of defence (spec 7.1): if the variable was not exported before Python started,
# Catalyst installed an "ignore" filter for RuleLoweringWarning; this filter takes precedence.
warnings.filterwarnings("always", category=RuleLoweringWarning)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """`release` implies `slow`; runs before `-m` deselection (spec section 12.6)."""
    for item in items:
        if item.get_closest_marker("release") and not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)
