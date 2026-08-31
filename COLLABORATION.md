# Claude–Codex collaboration protocol

This is the canonical, tracked protocol for agents working on PerfRDD. Agent-specific
instruction files should point here rather than maintain competing versions.

## Shared sources of truth

- `RESEARCH_LOG.md` records verified findings and methodological decisions.
- `../manuscript/TODO.md` records current tasks, ownership, and completion status.
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
  repository by default so the other agent and the author can see the result. If the
  author explicitly says to keep work local or hold a push, follow that instruction.
- Stage only files belonging to the task. Do not include unrelated local changes in a
  cleanup or handoff commit.
- Do not force-push, rewrite shared history, discard another contributor's work, or push
  credentials, private data, generated secrets, or known-broken results.
- If upstream changed, integrate it safely before pushing and rerun relevant checks.
- Manuscript claims and reported numbers must remain traceable to verified code and data.
