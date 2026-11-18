#!/usr/bin/env bash
# Build .venv-ref (Qualtran 0.7.0 and dfthcbe at the pinned commit) and write the dfthcbe
# reference fixtures into tests/fixtures/dfthcbe. Run from anywhere: bash scripts/make_venv_ref.sh
# VENV_REF overrides the environment's location (default: <repo>/.venv-ref).
#
# dfthcbe comes from the codeload tarball of the commit, checked by a SHA-256 digest of its
# sources; nothing is cloned. The recipe avoids two sandbox failures of a plain install: pip
# verifies certificates only with the macOS keychain's certificates (PIP_CERT), and no directory
# named venv may be created in the workspace, which jedi (through IPython) would need.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PIN=930255020cec30172e0a7d17c0071d9476a687d1
REF="$REPO/.scratch/dfthcbe-ref"
VENV_REF="${VENV_REF:-$REPO/.venv-ref}"
[ -d "$REF" ] || { mkdir -p "$REF" && curl -sSfL \
    "https://codeload.github.com/Covestro-HPQC/dfthcbe/tar.gz/$PIN" \
    | tar -xz -C "$REF" --strip-components=1; }
# SHA-256 of pyproject.toml and the package sources at $PIN (xargs fails in the sandbox).
test "$(cd "$REF" && shasum -a 256 pyproject.toml $(find dfthcbe -name '*.py' | LC_ALL=C sort) \
    | shasum -a 256 | cut -d' ' -f1)" \
    = 802221370175be502cde773d1c369f02c77a0d167d72ddffc1fb2d5df7df3c27
cd "$REPO"
export PIP_NO_CACHE_DIR=1 TMPDIR="$REPO/.scratch/tmp"
mkdir -p "$TMPDIR"
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain \
    /Library/Keychains/System.keychain > "$TMPDIR/keychain-certs.pem"
export PIP_CERT="$TMPDIR/keychain-certs.pem"
[ -d "$VENV_REF" ] || python3.12 -m venv "$VENV_REF"
"$VENV_REF/bin/pip" install --upgrade pip
# Qualtran 0.7.0 without its notebook dependencies (jedi ships a typeshed/stdlib/venv/ directory).
"$VENV_REF/bin/pip" install "attrs>=23.2.0" "cachetools>=5.3" networkx "numpy<3.0,>=1.26" \
    sympy "cirq-core~=1.4" fxpmath galois pydot quimb "typing_extensions>=4.10.0" mpmath \
    protobuf scipy pytest decorator ipython-pygments-lexers matplotlib-inline pexpect \
    prompt_toolkit pygments stack_data traitlets
"$VENV_REF/bin/pip" install --no-deps qualtran==0.7.0 ipython ipywidgets comm \
    widgetsnbextension jupyterlab_widgets
# pip builds inside the directory it installs, so install a copy and keep $REF pristine.
rm -rf "$TMPDIR/dfthcbe-src"
cp -R "$REF" "$TMPDIR/dfthcbe-src"
"$VENV_REF/bin/pip" install --no-deps "$TMPDIR/dfthcbe-src"
mkdir -p tests/fixtures/dfthcbe
# The lock names the immutable tarball instead of the local copy, so it is portable.
"$VENV_REF/bin/pip" freeze \
    | sed "s|^dfthcbe @ file://.*|dfthcbe @ https://codeload.github.com/Covestro-HPQC/dfthcbe/tar.gz/$PIN|" \
    > tests/fixtures/dfthcbe/requirements-ref.lock
MPLCONFIGDIR="$TMPDIR" "$VENV_REF/bin/python" scripts/make_dfthcbe_fixtures.py
