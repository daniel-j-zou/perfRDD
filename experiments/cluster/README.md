# Running simulations on Great Lakes

How to run this repository's simulations on the University of Michigan Great Lakes cluster
(ARC, Slurm), written while setting it up with the author on 2026-11-15 (Claude). The
September runs (Codex) used the older scripts in this folder; their results are in
`../../../outputs/` (outside this repo), with a `provenance*.txt` per task.

## Access

- **VPN.** Off campus, connect the UMich VPN first ("UMVPN – Only U-M Traffic" in macOS
  System Settings, or Cisco Secure Client). Without it ssh times out.
- **Login.** `ssh greatlakes` (the author's shell alias `ssh1`). Password plus Okta: every
  interactive session needs both (ARC policies page). File-transfer sessions need only the
  password.
- **SSH multiplexing does not work any more (2026-11-15).** The author's `~/.ssh/config` has
  `ControlMaster auto` for `greatlakes`, which is how agents ran commands in September. Now
  every extra session on a shared connection is closed right after the login banner (exit
  254), sftp included, so agents cannot run `ssh greatlakes '<command>'` on their own.
- **How an agent works instead.** The author logs in once inside a `screen` session in the
  Claude app's Terminal panel:

  ```bash
  screen -S gl ssh -o ControlMaster=no -o ControlPath=none greatlakes
  ```

  The agent pastes commands into that session (`screen -S gl -X readbuf <file>` then
  `screen -S gl -X paste .`) and reads the output with `screen -S gl -X hardcopy -h <file>`,
  between marker lines it prints before and after each command. The agent never sees the
  password, and the author sees every command in the tab. A helper that does this is in the
  session notes of 2026-11-15; copy it into a repo if it is needed again.

## Accounts, storage and rules

- Slurm accounts (2026-11-15): `stats_dept1`, `stats_dept2` (no per-user caps; the author's
  choice for this project), `lsa1` (24 CPUs, 120 GB), `lsa2` (2 CPUs), `lsa3` (36 CPUs,
  180 GB, for `largemem`). Jobs are billed to the account.
- Billing is by the share of a node a job takes: the larger of its CPU share and its memory
  share. A standard node has 36 cores and 180 GB, so asking for up to 5 GB per CPU costs no
  more than the CPUs. `my_account_billing -v stats_dept1` shows spending (needs a tty);
  in `sacct`, billing-minutes / 1e7 is dollars.
- Login nodes: 2 cores and 4 GB per user; for git, editing, `sbatch`, `squeue` and
  imports only. Anything heavier goes through Slurm, including small tests (`debug`
  partition: one job, 8 CPUs, 40 GB, 4 hours at most).
- Storage: `/home` is 80 GB and not backed up; `/scratch/stats_dept_root` deletes files
  untouched for 60 days. Copy results back to the Mac.

## Code and Python

- Code: a clone of `https://github.com/daniel-j-zou/perfRDD` at `~/src/perfRDD` (public, so
  `git fetch` needs no login). Update with
  `GIT_TERMINAL_PROMPT=0 git fetch -q origin && git merge --ff-only origin/main`.
- Python: `~/envs/perfrdd`, a venv on the cluster's conda Python 3.8.20 with numpy 1.24.4,
  scipy 1.10.1 and matplotlib 3.7.5, the versions of the local `srsd` environment. The
  cluster's own `~/.conda/envs/srsd` is a different environment (it has PyTorch and no
  matplotlib); do not use it for this project.

## Job pattern

- One array task per shard of replications (`--rep-start a --reps b` in
  `ds_four_role_coverage.py`); seeds depend only on (seed, n, rep), so merged shards equal one
  run. Each task writes to its own directory, with a `provenance.txt`.
- Size tasks from a short test job: measure seconds per fit and peak memory (`sacct
  --format=Elapsed,MaxRSS,TotalCPU`), then ask for about 1.5 times the measured time and
  memory. Short tasks (under an hour) start sooner because the scheduler can backfill them.
- One BLAS thread per worker, `--workers` equal to `--cpus-per-task`, `srun --cpu-bind=cores`.
- A merge job with `--dependency=afterok:<array id>` joins the shards and writes the summary.

## Section 5 Table 2 (bootstrap coverage, nonlinear design)

`ds_section5_coverage_array.sbatch` and `ds_section5_coverage_merge.sbatch`; see the header of
the array script for the submit commands and the run log below for what was submitted.
