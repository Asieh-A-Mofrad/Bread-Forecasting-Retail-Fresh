#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}"
MODE="apply"
REMOVE_VENVS="false"

for arg in "$@"; do
  case "${arg}" in
    --dry-run|-n)
      MODE="dry"
      ;;
    --remove-venvs)
      REMOVE_VENVS="true"
      ;;
    *)
      TARGET_DIR="${arg}"
      ;;
  esac
done

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Target directory not found: ${TARGET_DIR}" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

checkpoint_list="${tmp_dir}/checkpoint_dirs.txt"
conflict_list="${tmp_dir}/conflict_files.txt"
stray_list="${tmp_dir}/stray_checkpoints.txt"
pycache_list="${tmp_dir}/pycache_dirs.txt"
venv_list="${tmp_dir}/venv_dirs.txt"
touch "${venv_list}"

find "${TARGET_DIR}" \
  -type d \( -name '.venv*' -o -name '.git' \) -prune -o \
  -type d -name '.ipynb_checkpoints' -print > "${checkpoint_list}" 2>/dev/null || true
find "${TARGET_DIR}" \
  -type d \( -name '.venv*' -o -name '.git' \) -prune -o \
  -type f \( \
  -name '*-checkpoint 2.ipynb' -o \
  -name '*-checkpoint [0-9]*.ipynb' -o \
  -name '*-fpg*.ipynb' -o \
  -name '* (conflicted copy)*.ipynb' \
\) -print > "${conflict_list}" 2>/dev/null || true
find "${TARGET_DIR}" \
  -type d \( -name '.venv*' -o -name '.git' \) -prune -o \
  -type f -name '*-checkpoint.ipynb' ! -path '*/.ipynb_checkpoints/*' -print > "${stray_list}" 2>/dev/null || true
find "${TARGET_DIR}" \
  -type d \( -name '.venv*' -o -name '.git' \) -prune -o \
  -type d -name '__pycache__' -print > "${pycache_list}" 2>/dev/null || true
if [[ "${REMOVE_VENVS}" == "true" ]]; then
  find "${TARGET_DIR}" -maxdepth 2 -type d -name '.venv*' > "${venv_list}" 2>/dev/null || true
fi

echo "Target: ${TARGET_DIR}"
echo "Mode: ${MODE}"
echo "Remove in-repo venvs: ${REMOVE_VENVS}"

echo ""
echo "[1/5] .ipynb_checkpoints directories"
sed '/^$/d' "${checkpoint_list}" || true

echo ""
echo "[2/5] Notebook conflict/duplicate copies"
sed '/^$/d' "${conflict_list}" || true

echo ""
echo "[3/5] Stray checkpoint notebooks outside checkpoint folders"
sed '/^$/d' "${stray_list}" || true

echo ""
echo "[4/5] __pycache__ directories"
sed '/^$/d' "${pycache_list}" || true

echo ""
echo "[5/5] In-repo .venv* directories (only with --remove-venvs)"
sed '/^$/d' "${venv_list}" || true

if [[ "${MODE}" == "dry" ]]; then
  echo ""
  echo "Dry-run only. No files deleted."
  exit 0
fi

while IFS= read -r d; do
  [[ -n "${d}" ]] && rm -rf "${d}"
done < "${checkpoint_list}"

while IFS= read -r f; do
  [[ -n "${f}" ]] && rm -f "${f}"
done < "${conflict_list}"

while IFS= read -r f; do
  [[ -n "${f}" ]] && rm -f "${f}"
done < "${stray_list}"

while IFS= read -r d; do
  [[ -n "${d}" ]] && rm -rf "${d}"
done < "${pycache_list}"

while IFS= read -r d; do
  [[ -n "${d}" ]] && rm -rf "${d}"
done < "${venv_list}"

echo ""
echo "Cleanup complete."
