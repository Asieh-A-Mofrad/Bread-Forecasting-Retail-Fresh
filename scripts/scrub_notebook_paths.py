#!/usr/bin/env python3
"""Scrub local absolute paths from notebook outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import nbformat as nbf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrub local absolute paths from notebook outputs.")
    p.add_argument(
        "--path",
        default="notebooks",
        help="Notebook file or directory to scrub (default: notebooks)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which notebooks would be updated without writing changes.",
    )
    return p.parse_args()


def scrub_in_place(val, replacements):
    if isinstance(val, str):
        for old, new in replacements:
            val = val.replace(old, new)
        return val
    if isinstance(val, list):
        for i, v in enumerate(val):
            val[i] = scrub_in_place(v, replacements)
        return val
    if isinstance(val, dict):
        for k in list(val.keys()):
            val[k] = scrub_in_place(val[k], replacements)
        return val
    return val


def scrub_notebook(path: Path, replacements, dry_run: bool) -> bool:
    nb = nbf.read(path, as_version=4)
    original = nbf.writes(nb)

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "outputs" in cell:
            scrub_in_place(cell["outputs"], replacements)
        if "metadata" in cell:
            scrub_in_place(cell["metadata"], replacements)

    updated = nbf.writes(nb)
    if original != updated:
        if not dry_run:
            nbf.write(nb, path)
        return True
    return False


def main() -> None:
    args = parse_args()
    target = Path(args.path)
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[1]
    home = Path.home()
    replacements = [
        (str(repo_root), "<REPO_ROOT>"),
        (str(home), "~"),
    ]

    notebooks = [target] if target.suffix == ".ipynb" else list(target.rglob("*.ipynb"))
    if not notebooks:
        print("No notebooks found.")
        return

    changed = 0
    for nb_path in notebooks:
        if scrub_notebook(nb_path, replacements, args.dry_run):
            changed += 1
            print(f"Scrubbed: {nb_path}")

    if args.dry_run:
        print(f"Dry run: {changed} notebooks would be updated.")
    else:
        print(f"Done. Updated {changed} notebooks.")


if __name__ == "__main__":
    main()
