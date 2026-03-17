#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  _py="${PYTHON_BIN}"
elif command -v python3.10 >/dev/null 2>&1; then
  _py="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  _py="python3"
elif command -v python >/dev/null 2>&1; then
  _py="python"
else
  echo "No Python interpreter found. Install Python 3.10+ and re-run." >&2
  exit 1
fi

PYTHON_BIN="${_py}"
DEFAULT_VENV_BASE="${HOME}/.venvs"
VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_BASE}/bread-core}"
KERNEL_NAME="${KERNEL_NAME:-bread-core}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-bread-core}"

case "${VENV_DIR}" in
  "${REPO_ROOT}"/*|"${REPO_ROOT}")
    if [[ "${ALLOW_IN_REPO_VENV:-0}" != "1" ]]; then
      echo "Refusing to create virtualenv inside repository: ${VENV_DIR}" >&2
      echo "Use default (~/.venvs/...) or set VENV_DIR outside repo." >&2
      echo "If you really want this, set ALLOW_IN_REPO_VENV=1." >&2
      exit 1
    fi
    ;;
esac

"${PYTHON_BIN}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    raise SystemExit(
        f"Unsupported Python {major}.{minor} for core setup. Use Python 3.10+."
    )
PY

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install --upgrade -r "${REPO_ROOT}/requirements-core.txt"
python - <<'PY'
import importlib

missing = []
for mod in ("plotly",):
    try:
        importlib.import_module(mod)
    except Exception as exc:  # pragma: no cover - setup-time guard
        missing.append((mod, exc))

if missing:
    mods = ", ".join(m for m, _ in missing)
    raise SystemExit(f"Missing required modules after install: {mods}")
PY
python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${KERNEL_DISPLAY}"

echo "Core environment ready: ${VENV_DIR} (kernel: ${KERNEL_NAME})"
