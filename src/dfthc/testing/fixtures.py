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

"""Loaders for the dfthcbe reference fixtures (spec sections 5.2 and 8.6, check 4).

``scripts/make_dfthcbe_fixtures.py`` writes the fixtures in ``.venv-ref``; tests read them here.
A missing fixture skips the test with :data:`SKIP_REASON`, so checks 1 and 2 of spec 8.6 and the
end-to-end simulation still run where ``.venv-ref`` cannot be built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "dfthcbe"
SKIP_REASON = "dfthcbe fixtures not generated: run scripts/make_dfthcbe_fixtures.py in .venv-ref"


def require_fixture(name: str) -> Path:
    """The path of fixture ``name``; skips the calling test with :data:`SKIP_REASON` if missing."""
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.skip(SKIP_REASON)
    return path


def load_dfthcbe_fixture(name: str) -> dict | np.ndarray:
    """A fixture's content: a dict for ``.json`` and ``.npz`` files, an array for ``.npy``."""
    path = require_fixture(name)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".npz":
        with np.load(path) as data:
            return {key: np.array(data[key]) for key in data.files}
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"unknown fixture format: {name}")
