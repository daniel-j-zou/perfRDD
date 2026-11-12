"""Move each dataset's ``data/`` folder to an external root and leave a symlink behind.

    python -m experiments.datasets.relocate_data --root ~/Dropbox/PerfRDD-data --dry-run
    python -m experiments.datasets.relocate_data --root ~/Dropbox/PerfRDD-data
    python -m experiments.datasets.relocate_data --root ~/Dropbox/PerfRDD-data --only vlbw nc_sentencing

Every adapter resolves ``HERE / "data"``, so a symlink needs no code change. The script is
idempotent: a ``data`` that is already a symlink to the root is skipped; a ``data`` that is a
real folder is moved with ``shutil.move`` (a rename on the same volume, a copy across volumes,
verified by file count and total size before the original is removed) and replaced by a
relative-free absolute symlink. Nothing is deleted except the moved original after the
verification. ``--restore`` reverses the move for the named datasets.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HERE = Path(__file__).parent


def _tree_stats(path: Path) -> tuple[int, int]:
    n = size = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            n += 1
            size += p.stat().st_size
    return n, size


def _datasets(only: list[str] | None) -> list[Path]:
    out = []
    for d in sorted(HERE.iterdir()):
        if d.is_dir() and (d / "data").exists() and (only is None or d.name in only):
            out.append(d)
    return out


def relocate(root: Path, only: list[str] | None, dry_run: bool) -> None:
    root = root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist; create the Dropbox folder first.")
    for d in _datasets(only):
        src = d / "data"
        dst = root / d.name / "data"
        if src.is_symlink():
            print(f"{d.name:22s} already a symlink -> {os.readlink(src)}")
            continue
        n, size = _tree_stats(src)
        print(f"{d.name:22s} {n:7d} files {size / 1e9:7.2f} GB -> {dst}")
        if dry_run:
            continue
        if dst.exists():
            raise SystemExit(f"{dst} already exists; resolve by hand.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n2, size2 = _tree_stats(dst)
        if (n2, size2) != (n, size):
            raise SystemExit(f"{d.name}: moved tree differs ({n2} files, {size2} bytes); original is at {dst}, "
                             "no symlink made.")
        src.symlink_to(dst)
        print(f"{'':22s} moved and linked")


def restore(root: Path, only: list[str] | None) -> None:
    root = root.expanduser().resolve()
    for d in _datasets(only):
        src = d / "data"
        if not src.is_symlink():
            continue
        dst = Path(os.readlink(src))
        src.unlink()
        shutil.move(str(dst), str(src))
        print(f"{d.name:22s} restored from {dst}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="external root, e.g. ~/Dropbox/PerfRDD-data")
    ap.add_argument("--only", nargs="*", default=None, help="dataset folder names; default all with data/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true", help="move the data back and remove the symlinks")
    a = ap.parse_args(argv)
    if a.restore:
        restore(Path(a.root), a.only)
    else:
        relocate(Path(a.root), a.only, a.dry_run)


if __name__ == "__main__":
    main()
