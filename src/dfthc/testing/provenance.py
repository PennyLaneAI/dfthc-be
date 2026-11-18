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

"""Code hash and environment record for cached compiles (spec 13.3).

:func:`code_hash` decides whether a compile cached in ``docs/resources.json`` is a current
measurement: it hashes the syntax trees, without docstrings (comments are not in a syntax tree),
of the package's modules and of ``scripts/resource_report.py``, which builds the cached programs
and selects the cache's entries, and the lock file. So a documentation edit keeps the hash and any
code or toolchain change gives a new one. ``dfthc/testing/`` is excluded: no compiled program
depends on it.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import platform
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOCK_FILE = "requirements-nightly.lock"
REPORT_SCRIPT = "scripts/resource_report.py"
DISTRIBUTIONS = ("pennylane", "pennylane_catalyst", "pennylane_lightning", "jax", "jaxlib", "numpy")


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the docstring of the module and of every class and function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def hashed_files(root: Path = REPO) -> list[Path]:
    """The modules :func:`code_hash` covers: ``src/dfthc/**/*.py`` without ``dfthc/testing/``,
    then the report script."""
    package = root / "src" / "dfthc"
    return sorted(path for path in package.rglob("*.py")
                  if "testing" not in path.relative_to(package).parts) + [root / REPORT_SCRIPT]


def code_hash(root: Path = REPO) -> str:
    """SHA-256 over the docstring-free syntax trees of :func:`hashed_files` (each preceded by its
    path relative to ``root``) and the bytes of the lock file."""
    digest = hashlib.sha256()
    for path in hashed_files(root):
        tree = _strip_docstrings(ast.parse(path.read_text(), filename=str(path)))
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())
    digest.update((root / LOCK_FILE).read_bytes())
    return digest.hexdigest()


def environment(root: Path = REPO) -> dict[str, str]:
    """Python, platform, the toolchain's package versions and the lock file's SHA-256."""
    record = {"python": platform.python_version(), "platform": platform.platform()}
    for name in DISTRIBUTIONS:
        try:
            record[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record[name] = "not installed"
    lock = root / LOCK_FILE
    record["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() \
        else "missing"
    return record

