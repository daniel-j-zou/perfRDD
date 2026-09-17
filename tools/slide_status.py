#!/usr/bin/env python3
"""Advisory status for the shared Beamer deck. Never blocks anything.

Prints the frames marked ``% FINAL``, protected ``% SECTION FINAL`` ranges
(read-only per COLLABORATION.md), and the most recent ``DECK CLAIM`` lines from
the research log. This is a plain text scan with no frame-structure parsing,
so it is safe to run on the deck in any state (including mid-edit or with
syntax errors). It only reads; it never edits, stages, or commits.

Usage:  python3 code/tools/slide_status.py
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]  # PerfRDD/
SLIDES = ROOT / "manuscript" / "prelim" / "slides.tex"
LOG = ROOT / "code" / "RESEARCH_LOG.md"

_FRAME_TITLE = re.compile(r"\\begin\{frame\}\s*(?:\[[^\]]*\])?\s*\{(.*?)\}")
_SECTION_FINAL_START = re.compile(r"^\s*%\s*SECTION FINAL:\s*(.+?)\s*$")
_SECTION_FINAL_END = re.compile(r"^\s*%\s*END SECTION FINAL:\s*(.+?)\s*$")


def final_frames() -> list[tuple[int, str]]:
    if not SLIDES.exists():
        return []
    found: list[tuple[int, str]] = []
    for lineno, line in enumerate(SLIDES.read_text(errors="replace").splitlines(), 1):
        if "% FINAL" in line and r"\begin{frame}" in line:
            m = _FRAME_TITLE.search(line)
            found.append((lineno, m.group(1) if m else "(untitled frame)"))
    return found


def final_sections() -> list[tuple[int, int, str]]:
    if not SLIDES.exists():
        return []
    lines = SLIDES.read_text(errors="replace").splitlines()
    found: list[tuple[int, int, str]] = []
    open_section: tuple[int, str] | None = None
    for lineno, line in enumerate(lines, 1):
        start = _SECTION_FINAL_START.match(line)
        if start:
            if open_section is not None:
                # Keep this advisory and recoverable if a marker is malformed.
                found.append((open_section[0], lineno, open_section[1]))
            open_section = (lineno, start.group(1))
            continue
        end = _SECTION_FINAL_END.match(line)
        if end and open_section is not None:
            found.append((open_section[0], lineno, open_section[1]))
            open_section = None
    if open_section is not None:
        found.append((open_section[0], len(lines), open_section[1]))
    return found


def deck_claims(limit: int = 8) -> list[str]:
    if not LOG.exists():
        return []
    # RESEARCH_LOG.md is newest-first, so the first matches are the most recent.
    hits = [ln.strip() for ln in LOG.read_text(errors="replace").splitlines()
            if "DECK CLAIM" in ln]
    return hits[:limit]


def main() -> None:
    ff = final_frames()
    print(f"% FINAL (read-only) frames in {SLIDES.name}: {len(ff)}")
    for lineno, title in ff:
        print(f"  L{lineno}: {title}")

    sections = final_sections()
    print(f"\n% SECTION FINAL (read-only) ranges in {SLIDES.name}: {len(sections)}")
    for start, end, name in sections:
        print(f"  L{start}-L{end}: {name}")

    claims = deck_claims()
    print(f"\nRecent DECK CLAIM lines in {LOG.name}: {len(claims)}")
    for claim in claims:
        print(f"  {claim}")

    print("\n(Advisory only — this never blocks commits and never edits files.)")


if __name__ == "__main__":
    main()
