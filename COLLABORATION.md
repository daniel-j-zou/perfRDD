# Claude–Codex collaboration protocol

This is the canonical, tracked protocol for agents working on PerfRDD. Agent-specific
instruction files should point here rather than maintain competing versions.

## Shared sources of truth

- `RESEARCH_LOG.md` records verified findings and methodological decisions.
- `../manuscript/TODO.md` records current tasks, ownership, and completion status.
- `../manuscript/CHANGELOG.md` is the human-readable edit trail of the paper: one entry
  per manuscript push describing what changed, where, and why, so the authors can follow
  updates without reading diffs.
- Reproducible code and durable computational documentation live in this repository.
- Paper text, bibliography, figures, and the tracked task board live in `../manuscript/`.
- `../work/` and `../outputs/` are local scratch and review directories. Anything needed
  by another agent must be copied into a tracked repository and linked from the log.

## Starting and handing off work

1. Read the newest entries in `RESEARCH_LOG.md` and the current manuscript task board.
2. Fetch the affected repository and inspect its branch, upstream divergence, and working
   tree. Preserve unrelated local changes; a dirty tree is not permission to rewrite them.
3. Record meaningful verified findings and decisions at the top of `RESEARCH_LOG.md`,
   dated and signed `Claude` or `Codex`. Never edit or delete another agent's entry; add a
   follow-up entry if a result changes.
4. When adding, closing, or materially changing a task, update `../manuscript/TODO.md` and
   the research log together.
5. Report reproducible commands, inputs, outputs, and commit identifiers when relevant.

## Git and publication policy

- After completing and verifying an authorized task, commit and push the affected
  repository by default so the other agent and the author can see the result. This
  **includes the manuscript / Overleaf repository**: the author wants paper updates pushed
  automatically. If the author explicitly says to keep work local or hold a push, follow
  that instruction.
- Every manuscript push must add a `../manuscript/CHANGELOG.md` entry (newest first,
  dated, signed, with the commit hash) summarizing the change in plain language.
- Before a manuscript push, compile the paper locally and confirm it builds without new
  errors or undefined references. The local TeX Live can build it (see the toolchain note
  in `RESEARCH_LOG.md`); do not push LaTeX you have not compiled.
- **Fetch immediately before every push.** If the push is rejected as non-fast-forward,
  fetch, rebase your task commit(s) onto the updated upstream, rerun the relevant checks,
  and only then push. Never force-push or overwrite the other agent's commits.
- Stage only files belonging to the task. Do not include unrelated local changes in a
  cleanup or handoff commit.
- Do not force-push, rewrite shared history, discard another contributor's work, or push
  credentials, private data, generated secrets, or known-broken results.
- Manuscript claims and reported numbers must remain traceable to verified code and data.

## Avoiding collisions on shared files

Both agents push by default, so simultaneous edits to the same file are the main hazard.

- Before starting a task, claim it on `../manuscript/TODO.md`: mark it in progress with an
  owner and the files/sections you expect to touch, e.g.
  `_(owner: Claude · files: prefRDD.tex §trim · since 2026-08-31)_`. Clear the claim when
  the task is pushed.
- If a file you need is claimed in progress by the other agent, coordinate through the log
  rather than editing it concurrently.
- For anything beyond a small, self-contained edit to a file the other agent may also be
  in, work on a short-lived task branch and fast-forward it onto the mainline when the
  task is verified and pushed.

## Slide-deck collaboration (`manuscript/prelim/slides.tex`)

The deck is a single file that the author edits live through Overleaf, so its `master`
history moves on its own. Coordinate with in-file markers plus a logged branch claim.
There is **no lock registry or blocking Git hook** — an earlier tool-based lock was
removed because it added three-commit ceremony, crashed on ordinary source, and failed
silently. Finalization markers are advisory protocol, reported by the status helper,
and must remain lightweight and grep-able.

**Author markers (in `slides.tex`).**
- `% FINAL` on the same line as a frame's `\begin{frame}{...}` means the author has
  finalized that frame. Treat it as **read-only**: do not change its content, title,
  layout, or the marker. Edit it only after the author names that slide and asks for a
  revision. Find them with `rg -n '% FINAL' manuscript/prelim/slides.tex`.
- `% SECTION FINAL: <name>` followed later by `% END SECTION FINAL: <name>` marks an
  author-finalized section. Treat every line in that range as **read-only**, including
  commented-out frames and supporting prose. Only the author may revise the section;
  an agent must ask the author to remove the marker or explicitly name the requested
  revision before editing it. Find protected ranges with
  `python3 code/tools/slide_status.py`.
- `% TODO: ...` (a LaTeX comment, so it does not render) on or just below a frame's
  `\begin{frame}` line is an author edit request for that frame. Address it and delete the
  marker in the same commit. Do not add work to a `% FINAL` frame on a `% TODO`'s behalf
  without the author. (Bracketed `[TODO: ...]` also works but renders on the slide, so the
  comment form is preferred.)

**Agent deck claims.** Before any deck change beyond a single-frame `% TODO` fix:
1. Append a claim to `RESEARCH_LOG.md`, e.g.
   `## <date> - DECK CLAIM: Claude - <scope>; branch slides/<topic> (open)`.
2. Do the work on that short-lived `slides/<topic>` branch and compile it:
   `latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscript/prelim slides.tex`.
3. Fetch, rebase the branch onto the updated `master`, recompile, fast-forward `master`,
   and push.
4. Append a one-line release follow-up to `RESEARCH_LOG.md` (`DECK CLAIM ... released`).

Before claiming, check for an open claim (`rg 'DECK CLAIM' code/RESEARCH_LOG.md | head`); if
another agent's is open, coordinate in the log rather than editing concurrently. A
single-frame `% TODO` fix needs no branch, but still fetch immediately before pushing.

**The author always wins.** The author's live Overleaf edits land on `master` continuously.
Fetch right before merging; if they touched your frames, rebase and re-apply. Never
force-push or discard their edits.

**Optional advisory check.** `python3 code/tools/slide_status.py` lists the current
`% FINAL` frames, `% SECTION FINAL` ranges, and the most recent deck claims. It is advisory
only — a plain text scan that never blocks a commit and never edits anything.

## Drafting manuscript prose

When writing or rewriting manuscript sections (Introduction, Discussion, application
narrative, and similar author-facing prose), follow the author's staged workflow rather
than drafting whole sections unprompted:

1. **Plan first.** Propose a paragraph-by-paragraph outline: for each paragraph, one line
   on what it should accomplish (its job in the argument), not the prose itself. Present
   the plan and wait for the author's approval or revisions.
2. **Write on request.** Only after the author approves the plan (and asks) do you draft
   the prose, matching *Biometrika* house style and the level and voice of the closest
   prior papers (e.g. Mukherjee, Banerjee & Ritov). Keep it terse; avoid contents-of-the-
   paper listing.
3. **Draft and finalize one paragraph at a time.** Provide the draft for a given
   paragraph, then iterate with the author to finalize it before moving on.

Additional standing rules for this workflow:
- Never invent citations. Use only keys already in `../manuscript/references.bib`; where a
  referenced work is not yet in the bibliography, name it in prose or leave a marked
  citation gap and defer it to a dedicated citation-gathering pass — do not fabricate a
  key or entry.
- The `\section*{To discuss.}` block at the top of `prelim/prelim.tex` is the author's
  working notes for advisor discussion, not manuscript prose. Do not treat it as draft
  text or rewrite it into the paper.
