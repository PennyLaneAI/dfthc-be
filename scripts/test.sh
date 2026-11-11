#!/usr/bin/env bash
# Run pytest with the canonical environment (spec sections 7.1 and 12.6).
#
# Catalyst reads CATALYST_SILENCE_RULE_LOWERING_WARNINGS once, when it is first imported, so the
# variable must be exported before Python starts. Works from the main checkout and from every
# git worktree: the shared .venv lives next to the main checkout's .git directory.
#
# Load limits (master plan section 12.9): 2 xdist workers unless DFTHC_TEST_WORKERS says
# otherwise (0 runs serially), at lowered priority. nice needs setpriority(), which the agent
# sandbox forbids; there nice only warns and runs the command unchanged, so the run gets the
# utility QoS clamp of taskpolicy instead, which children inherit.
set -euo pipefail
export CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0

worktree_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
common_git_dir="$(git -C "$worktree_root" rev-parse --path-format=absolute --git-common-dir)"
venv="${DFTHC_VENV:-$(dirname "$common_git_dir")/.venv}"

if [[ ! -x "$venv/bin/python" ]]; then
    echo "test.sh: no Python at $venv/bin/python; run scripts/setup_env.sh in the main checkout" >&2
    exit 2
fi

if [[ -z "$(nice -n 10 true 2>&1)" ]]; then
    lower_priority="nice -n 10"
elif taskpolicy -c utility true 2>/dev/null; then
    lower_priority="taskpolicy -c utility"
else
    echo "test.sh: cannot lower the priority (neither nice nor taskpolicy works)" >&2
    lower_priority=""
fi

cd "$worktree_root"
# $lower_priority is unquoted on purpose: it is a command prefix of several words.
exec $lower_priority "$venv/bin/python" -m pytest -n "${DFTHC_TEST_WORKERS:-2}" "$@"
