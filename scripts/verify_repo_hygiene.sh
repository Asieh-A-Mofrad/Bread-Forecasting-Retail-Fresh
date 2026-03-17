#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

venv_count=$(find . -maxdepth 2 -type d -name '.venv*' | wc -l | tr -d ' ')
pycache_count=$(
  find . \
    -type d \( -name '.venv*' -o -name '.git' \) -prune -o \
    -type d -name '__pycache__' -print | wc -l | tr -d ' '
)
ckpt_count=$(
  find . \
    -type d \( -name '.venv*' -o -name '.git' \) -prune -o \
    -type d -name '.ipynb_checkpoints' -print | wc -l | tr -d ' '
)

echo "Repository hygiene check:"
echo "  .venv* dirs           : ${venv_count}"
echo "  __pycache__ dirs      : ${pycache_count}"
echo "  .ipynb_checkpoints dirs: ${ckpt_count}"

if [[ "${venv_count}" -gt 0 || "${pycache_count}" -gt 0 || "${ckpt_count}" -gt 0 ]]; then
  echo ""
  echo "Generated artifacts detected. Run:"
  echo "  bash scripts/cleanup_notebook_artifacts.sh --remove-venvs"
  exit 2
fi

echo "No generated artifact folders detected."
