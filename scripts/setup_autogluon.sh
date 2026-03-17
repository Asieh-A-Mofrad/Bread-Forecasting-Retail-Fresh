#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  _py="${PYTHON_BIN}"
elif command -v python3.10 >/dev/null 2>&1; then
  _py="python3.10"
elif command -v python3.11 >/dev/null 2>&1; then
  _py="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  _py="python3"
elif command -v python >/dev/null 2>&1; then
  _py="python"
else
  echo "No Python interpreter found. Install Python 3.10 or 3.11 and re-run." >&2
  exit 1
fi

PYTHON_BIN="${_py}"
DEFAULT_VENV_BASE="${HOME}/.venvs"
VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_BASE}/bread-autogluon}"
KERNEL_NAME="${KERNEL_NAME:-bread-autogluon}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-bread-autogluon}"

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
if major != 3 or minor not in (10, 11):
    raise SystemExit(
        f"Unsupported Python {major}.{minor} for AutoGluon setup. "
        "Use Python 3.10 or 3.11 (set PYTHON_BIN=/path/to/python3.10)."
    )
PY

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install --upgrade -r "${REPO_ROOT}/requirements-autogluon.txt"
python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${KERNEL_DISPLAY}"

echo "AutoGluon environment ready: ${VENV_DIR} (kernel: ${KERNEL_NAME})"
