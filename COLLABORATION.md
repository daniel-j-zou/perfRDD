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
