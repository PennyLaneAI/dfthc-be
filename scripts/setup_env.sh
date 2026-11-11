#!/usr/bin/env bash
# Create the canonical environment of spec section 7.1 in .venv at the repository root.
#
# Expected noise, both benign:
#   * the second install prints "ERROR: ... pennylane-catalyst ... requires pennylane==<dev N>,
#     but you have pennylane <dev M>"; overriding the pin is intended;
#   * every compile later prints "ld: warning: object file ... was built for newer 'macOS' version".
# Never use `uv pip`: it resolves pennylane-catalyst from PyPI and installs the stable release.
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

if [[ -e .venv ]]; then
    echo "setup_env.sh: .venv exists; remove it first (rm -rf .venv) to recreate it" >&2
    exit 2
fi

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Catalyst nightly first (the user's `catame` alias), then the PennyLane nightly on top:
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane-catalyst --upgrade
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane --upgrade
pip install pytest pytest-xdist ruff          # from PyPI, without the TestPyPI index
pip freeze > requirements-nightly.lock
