#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x .venv_vision/bin/python ]]; then
    exec .venv_vision/bin/python scripts/bootstrap.py "$@"
fi
exec python3 scripts/bootstrap.py "$@"
