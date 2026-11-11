#!/usr/bin/env bash
# Create the canonical environment of spec section 7.1 in .venv at the repository root.
#
# Expected noise, both benign:
#   * the second install prints "ERROR: ... pennylane-catalyst ... requires pennylane==<dev N>,
#     but you have pennylane <dev M>"; overriding the pin is intended;
#   * every compile later prints "ld: warning: object file ... was built for newer 'macOS' version".
# Never use `uv pip`: it resolves pennylane-catalyst from PyPI and installs the stable release.
set -euo pipefail

# The environment always goes into the main checkout, also when this runs from a linked worktree:
# the directory that holds the common .git is the main checkout (as in scripts/test.sh).
common_git_dir="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --path-format=absolute --git-common-dir)"
main_checkout="$(dirname "$common_git_dir")"
cd "$main_checkout"

if [[ -e .venv ]]; then
    echo "setup_env.sh: $main_checkout/.venv exists; remove it first (rm -rf .venv) to recreate it" >&2
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
