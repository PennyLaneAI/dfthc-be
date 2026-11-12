"""Generate the dfthcbe reference fixtures of spec 8.6, check 4. Runs only in ``.venv-ref``.

Usage, from the repository root (environment recipe: docs/agents/HANDOFF.md, section 3.4):

    .venv-ref/bin/python scripts/make_dfthcbe_fixtures.py [--out tests/fixtures/dfthcbe]

Writes into ``--out``:

- ``tables_h2_sto3g_test.json``, ``tables_h2_631g_test.json``: dfthcbe's alias and angle tables at
  the bits of ``H2_STO3G_TEST`` and ``H2_631G_TEST``, keyed by the ``WalkTables`` field names;
- ``encoded_h2_sto3g_test.npz``: dfthcbe's ``<0|BE|0>`` of ``H2_STO3G_TEST`` by exact tensor
  contraction with ``exact_erase=True`` (``block``), its ``Lambda`` (``lambda_tilde``) and the
  ``H_SA`` of its ``encoded_factors`` (``h_sa``);
- ``encoded_x_o_fails.npz``: the same three arrays at ``X_O_FAILS``, a shape where the uniform
  preparation of ``x_o`` fails (``X_o = 5``), which neither H2 preset exercises, with the factors
  ``random_factors(2, 3, 1, 1, n_plus=1, seed=3)`` (dfthcbe's ``x_o_fails`` shape);
- ``femoco_block_toffolis.json``: dfthcbe's Toffolis per block at ``FEMOCO54`` (``k_e1 = 4``,
  ``k_e2 = 2``, ``s = 7``), from its symbolic call graph, so no FeMoco table is built;
- ``provenance.json``: the dfthcbe commit, package versions, parameters and inputs. Three fields
  pin the dfthcbe source: ``dfthcbe_source_url``, the immutable codeload tarball of the commit;
  ``dfthcbe_source_sha256``, the digest of that tree that the environment recipe verifies (SHA-256
  of ``pyproject.toml`` and the package's ``.py`` files, as ``shasum -a 256`` lists them); and
  ``dfthcbe_installed_sha256``, computed here from the imported package (SHA-256 over its ``.py``
  files sorted by path relative to the package directory, each fed as the POSIX relative path, a
  zero byte and the file's bytes), so the fixture records what was actually imported. ``inputs``
  gives, per encoded fixture, the source of its factors and :func:`factor_digest` of the arrays
  handed to dfthcbe; the tests recompute the digest from the arrays they feed to our code.

The heavy imports (Qualtran, dfthcbe) happen inside the functions, so the test suite can import
this module's constants in the main environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dfthc.data.factors import random_factors  # noqa: E402
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, WalkParams  # noqa: E402

DFTHCBE_COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"
DFTHCBE_SOURCE_URL = f"https://codeload.github.com/Covestro-HPQC/dfthcbe/tar.gz/{DFTHCBE_COMMIT}"
DFTHCBE_SOURCE_SHA256 = "802221370175be502cde773d1c369f02c77a0d167d72ddffc1fb2d5df7df3c27"
DATA = REPO / "data" / "dfthc"
KEYS = ("U", "W", "w_B", "w_plus", "u_plus", "w_minus", "u_minus")
X_O_FAILS = WalkParams(N=2, R=3, B=1, C=1, n_plus=1, b_rot=6, b_k1=3, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
X_O_FAILS_FACTORS = "random_factors(2, 3, 1, 1, n_plus=1, seed=3)"
TABLE_FIXTURES = {
    "tables_h2_sto3g_test.json": (H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"),
    "tables_h2_631g_test.json": (H2_631G_TEST, "h2_631g_R2_B2_C2"),
}
ENCODED_FIXTURE = "encoded_h2_sto3g_test.npz"
X_O_FAILS_FIXTURE = "encoded_x_o_fails.npz"
ENCODED_FIXTURES = {
    ENCODED_FIXTURE: (H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"),
    X_O_FAILS_FIXTURE: (X_O_FAILS, X_O_FAILS_FACTORS),
}
"""Encoded fixture -> (parameters, the factors' source: a vendored system or ``random_factors``)."""
FEMOCO_FIXTURE = "femoco_block_toffolis.json"
PROVENANCE = "provenance.json"
FIXTURES = (*TABLE_FIXTURES, *ENCODED_FIXTURES, FEMOCO_FIXTURE, PROVENANCE)
FEMOCO_BLOCKS = ("outer_prepare", "outer_unprepare", "inner_prepare", "inner_unprepare",
                 "inner_select", "inner_select_adjoint", "ref_t2", "ref_ab", "block_encoding",
                 "controlled_block_encoding", "walk", "controlled_walk")


def factor_digest(arrays: dict) -> str:
    """SHA-256 of factor arrays: for each of :data:`KEYS` in order, the key name, the shape, the
    dtype string and the C-contiguous bytes. Pure NumPy, so the tests can recompute it."""
    digest = hashlib.sha256()
    for key in KEYS:
        array = np.asarray(arrays[key])
        for part in (key, str(array.shape), array.dtype.str):
            digest.update(part.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def installed_digest() -> str:
    """SHA-256 of the imported ``dfthcbe`` package: its ``.py`` files sorted by relative path,
    each fed as the POSIX relative path, a zero byte and the file's bytes."""
    import dfthcbe

    root = Path(dfthcbe.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _factors(source: str) -> dict:
    """The seven arrays dfthcbe's ``from_factors`` takes, of a vendored system or of
    :data:`X_O_FAILS_FACTORS`."""
    if source == X_O_FAILS_FACTORS:
        factors = random_factors(2, 3, 1, 1, n_plus=1, seed=3)
        return {key: np.array(getattr(factors, key)) for key in KEYS}
    with np.load(DATA / source / "factors.npz") as data:
        return {key: np.array(data[key]) for key in KEYS}


def block_encoding(params, source: str, exact_erase: bool):
    """dfthcbe's ``DFTHCBlockEncoding.from_factors`` at the bits and block sizes of ``params``."""
    from dfthcbe import DFTHCBlockEncoding

    return DFTHCBlockEncoding.from_factors(
        **_factors(source), b_rot=params.b_rot, b_k1=params.b_k1, b_k2=params.b_k2,
        k_1=params.k1, k_2=params.k2, k_e1=params.ke1, k_e2=params.ke2, s=params.s_u,
        exact_erase=exact_erase,
    )


def tables(params, system: str) -> dict:
    be = block_encoding(params, system, exact_erase=False)
    o, i, s = be.outer, be.inner, be.select
    return {"outer_alt": o.alt, "outer_keep": o.keep, "inner_alt": i.alt, "inner_keep": i.keep,
            "inner_sgn": i.sgn, "inner_d_sgn": i.d_sgn, "tail_angles": s.tail_data,
            "sf_angles": s.angle_data}


def signal_block(be) -> np.ndarray:
    """``<0|be|0>`` on the ancillas: a ``4**N`` matrix on ``(psi_up, psi_dn)``.

    A port of ``_signal_block`` in dfthcbe's ``tests/test_block_encoding.py``.
    """
    from dfthcbe.block_encoding import INNER_ANCILLAS, OUTER_ANCILLAS
    from qualtran import BloqBuilder, QAny
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState

    num_orb, b_rot = be.select.num_orb, be.select.b_rot
    bb = BloqBuilder()
    psi = {name: bb.add_register(name, QAny(num_orb)) for name in ("psi_up", "psi_dn")}
    width = {name: be.signature.get_left(name).total_bits()
             for name in OUTER_ANCILLAS + INNER_ANCILLAS}
    grad = PhaseGradientState(b_rot)
    out = bb.add_d(be, **psi, phase_grad=bb.add(grad),
                   **{name: bb.add(IntState(0, n)) for name, n in width.items()})
    bb.add(grad.adjoint(), phase_grad=out.pop("phase_grad"))
    for name, n in width.items():
        bb.add(IntEffect(0, n), val=out.pop(name))
    return bb.finalize(**out).tensor_contract()


def encoded(params, source: str) -> dict:
    from dfthcbe import hamiltonian_matrix, walk_e_sos, walk_integrals

    be = block_encoding(params, source, exact_erase=True)
    enc = be.encoded_factors()
    h_sa = (hamiltonian_matrix(*walk_integrals(**enc))
            - walk_e_sos(enc["W"], enc["w_B"], enc["w_minus"]) * np.eye(4**params.N))
    with warnings.catch_warnings():
        # exact_erase replaces each measured erase by its read's inverse and says so each time
        warnings.filterwarnings("ignore", message="QROAMCleanHetero", category=UserWarning)
        block = signal_block(be)
    return {"block": block, "lambda_tilde": np.array(be.Lambda), "h_sa": h_sa}


def femoco_toffolis() -> dict[str, int]:
    """Toffolis per block (``Toffoli + CSwap + And + T/4`` of Qualtran's ``QECGatesCost``, as in
    dfthcbe's tests) of the symbolic walk, evaluated at ``FEMOCO54``."""
    import sympy
    from dfthcbe import DFTHCWalk
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    p = FEMOCO54
    symbols = sympy.symbols("N R B C b_rot b_k1 b_k2 k_1 k_2 k_e1 k_e2", positive=True,
                            integer=True)
    walk = DFTHCWalk.symbolic(*symbols, p.s_u)
    values = dict(zip(symbols, (p.N, p.R, p.B, p.C, p.b_rot, p.b_k1, p.b_k2, p.k1, p.k2, p.ke1,
                                p.ke2), strict=True))

    def toffoli(bloq) -> int:
        cost = get_cost_value(bloq, QECGatesCost())
        total = sympy.sympify(cost.toffoli + cost.cswap + cost.and_bloq + cost.t / 4)
        return int(total.subs(values))

    be = walk.block_encoding
    bloqs = (be.outer, be.outer.adjoint(), be.inner, be.inner.adjoint(), be.select,
             be.select.adjoint(), be.ref_t2, walk.ref_ab, be, be.controlled(), walk,
             walk.controlled())
    return {name: toffoli(bloq) for name, bloq in zip(FEMOCO_BLOCKS, bloqs, strict=True)}


def provenance() -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("dfthcbe", "qualtran",
                                                                     "numpy", "sympy")}
    presets = {"H2_STO3G_TEST": H2_STO3G_TEST, "H2_631G_TEST": H2_631G_TEST, "FEMOCO54": FEMOCO54,
               "X_O_FAILS": X_O_FAILS}
    return {
        "dfthcbe_commit": DFTHCBE_COMMIT,
        "dfthcbe_source_url": DFTHCBE_SOURCE_URL,
        "dfthcbe_source_sha256": DFTHCBE_SOURCE_SHA256,
        "dfthcbe_installed_sha256": installed_digest(),
        "versions": versions | {"python": platform.python_version()},
        "parameters": {name: dataclasses.asdict(p) for name, p in presets.items()},
        "systems": {name: system for name, (_, system) in TABLE_FIXTURES.items()}
        | {name: source for name, (_, source) in ENCODED_FIXTURES.items()},
        "inputs": {name: {"factors": source, "sha256": factor_digest(_factors(source))}
                   for name, (_, source) in ENCODED_FIXTURES.items()},
        "encoded_block": "exact tensor contraction of DFTHCBlockEncoding.from_factors(..., "
                         "exact_erase=True) with every ancilla pinned to |0>",
        "femoco_counts": "DFTHCWalk.symbolic call graph evaluated at FEMOCO54; Toffolis = "
                         "Toffoli + CSwap + And + T/4 of QECGatesCost",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "fixtures" / "dfthcbe")
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)
    for name, (params, system) in TABLE_FIXTURES.items():
        (out / name).write_text(json.dumps(tables(params, system)) + "\n")
    for name, (params, source) in ENCODED_FIXTURES.items():
        np.savez(out / name, **encoded(params, source))
    (out / FEMOCO_FIXTURE).write_text(json.dumps(femoco_toffolis(), indent=2) + "\n")
    (out / PROVENANCE).write_text(json.dumps(provenance(), indent=2) + "\n")
    for name in FIXTURES:
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
