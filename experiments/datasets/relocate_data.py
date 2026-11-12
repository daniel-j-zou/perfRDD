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

Archive mode (rclone, no desktop app):

    python -m experiments.datasets.relocate_data --archive --remote dropbox:PerfRDD-data --only vlbw
    python -m experiments.datasets.relocate_data --fetch   --remote dropbox:PerfRDD-data --only vlbw

``--archive`` runs ``rclone copy`` of ``<dataset>/data`` to ``<remote>/<dataset>/data`` and then
``rclone check`` (sizes and hashes); it never deletes the local copy. ``--fetch`` copies a
dataset back. Removing a local copy after a verified archive is a separate, explicit step:
``--remove-local --only <dataset>`` refuses unless ``rclone check`` passes first.
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


import subprocess


def _rclone(*args: str) -> int:
    cmd = ["rclone", *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _check(remote: str, d: Path) -> bool:
    """True when the remote copy matches the local data folder (rclone check, sizes and hashes)."""
    return _rclone("check", str(d / "data"), f"{remote}/{d.name}/data", "--one-way") == 0


def archive(remote: str, only: list[str] | None) -> None:
    for d in _datasets(only):
        n, size = _tree_stats(d / "data")
        print(f"{d.name:22s} {n:7d} files {size / 1e9:7.2f} GB -> {remote}/{d.name}/data", flush=True)
        if _rclone("copy", str(d / "data"), f"{remote}/{d.name}/data", "--progress", "--transfers", "4") != 0:
            raise SystemExit(f"{d.name}: rclone copy failed")
        print(f"{'':22s} {'verified' if _check(remote, d) else 'CHECK FAILED'}", flush=True)


def fetch(remote: str, only: list[str] | None) -> None:
    for name in only or []:
        d = HERE / name
        if _rclone("copy", f"{remote}/{name}/data", str(d / "data"), "--progress", "--transfers", "4") != 0:
            raise SystemExit(f"{name}: rclone copy failed")
        print(f"{name:22s} fetched")


def remove_local(remote: str, only: list[str]) -> None:
    for d in _datasets(only):
        if not _check(remote, d):
            raise SystemExit(f"{d.name}: remote does not match local; nothing removed")
        shutil.rmtree(d / "data")
        (d / "data").mkdir()
        (d / "data" / "ARCHIVED.txt").write_text(
            f"Data archived at {remote}/{d.name}/data. Restore with\n"
            f"python -m experiments.datasets.relocate_data --fetch --remote {remote.split('/')[0]}/"
            f"{'/'.join(remote.split('/')[1:])} --only {d.name}\n")
        print(f"{d.name:22s} local copy removed (verified archive)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="external root for the move modes, e.g. ~/Dropbox/PerfRDD-data")
    ap.add_argument("--only", nargs="*", default=None, help="dataset folder names; default all with data/")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true", help="move the data back and remove the symlinks")
    ap.add_argument("--remote", default="dropbox:PerfRDD-data", help="rclone remote path for the archive modes")
    ap.add_argument("--archive", action="store_true", help="rclone copy data/ to the remote and verify")
    ap.add_argument("--fetch", action="store_true", help="rclone copy data/ back from the remote (needs --only)")
    ap.add_argument("--remove-local", action="store_true", help="delete a verified local copy (needs --only)")
    a = ap.parse_args(argv)
    if a.archive:
        return archive(a.remote, a.only)
    if a.fetch:
        return fetch(a.remote, a.only)
    if a.remove_local:
        if not a.only:
            raise SystemExit("--remove-local needs --only")
        return remove_local(a.remote, a.only)
    if a.root is None:
        raise SystemExit("--root is needed for the move modes")
    if a.restore:
        restore(Path(a.root), a.only)
    else:
        relocate(Path(a.root), a.only, a.dry_run)


if __name__ == "__main__":
    main()
