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
  password, and the author sees every command in the tab. The helper is `gl_screen.sh`
  (`gl_screen.sh '<command>' [timeout_seconds]`); run it outside the command sandbox. Long
  waits (`while squeue ... | grep -q .; do sleep 30; done`) keep the tab busy, so tell the
  author not to type there meanwhile. Raise the scrollback first
  (`screen -S gl -X scrollback 20000`) before printing long output.
- **Copying results back through the logged-in session** (no second login; used 2026-11-16
  for the 67 MB archive of the 2026-11-15 runs). On the cluster, pack the run directories
  (`tar czf ~/xfer_runs.tgz -C experiments/runs <dirs>`; leave out shard copies that the
  merged files repeat), note `sha256sum`, then split the encoded archive and list the parts'
  md5:

  ```bash
  mkdir ~/xfer && cd ~/xfer && base64 -w 76 ~/xfer_runs.tgz | split -l 40000 -d -a 3 - part_ && md5sum part_*
  ```

  Save that md5 list locally (one `md5  part_NNN` per line), raise the scrollback
  (`screen -S gl -X scrollback 100000`; a 40,000-line part wraps to about 80,000 rows in
  the 48-column tab), and run, outside the sandbox,
  `experiments/cluster/gl_fetch_parts.sh <md5 list> '~/xfer' <local dir>`. Each part takes
  about 8 seconds (about 280 KB/s) and is checked against its md5; a rerun skips parts
  already copied. Then `cat part_0* | base64 -D > xfer_runs.tgz`, compare `shasum -a 256`
  with the cluster's, list it with `tar tzf`, and unpack into `experiments/runs/` with
  `tar xkzf` (`-k` never overwrites).
- **Copying results back with a second login.** The author can also copy with the transfer
  host, which needs the password and Duo but no Okta push:

  ```bash
  scp -r <uniqname>@greatlakes-xfer.arc-ts.umich.edu:src/perfRDD/experiments/runs/<run> experiments/runs/
  ```

  Globus (endpoint `umich#greatlakes`) is the alternative for very large transfers.
- **Sending a file up through the logged-in session** (no second login; used 2026-11-15 for
  the 10.9 MB taxi parquet, 5 seconds): encode locally with `base64 -b 76 -i <file> -o
  <file>.b64` and append Ctrl-D (`printf '\004' >> <file>.b64`); paste
  `stty -echo; base64 -d > <dest>.part; stty echo; echo XFER_RECV_END` into the session,
  then paste the encoded file; wait for `XFER_RECV_END`, compare `sha256sum` with
  `shasum -a 256` locally, and rename `.part`.
- **Always give `screen -X readbuf` and `hardcopy` absolute paths.** Relative paths are
  resolved from the directory screen started in. A failed `readbuf` leaves the old buffer,
  and `paste .` then re-sends the previous command (on 2026-11-15 this resubmitted a rerun
  and a merge job, which had to be cancelled).
- **Wait before reading a `hardcopy` file.** `screen -X hardcopy` returns before screen has
  written the file, so a script that reads it at once can see a half-written copy (on
  2026-11-16 this made half the parts wait out a 5-minute timeout). Pause a second first.

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

## Run log

| Date | Job | What | Result |
|---|---|---|---|
| 2026-11-15 | 66189298 | test, debug partition | failed: cluster `srsd` env has no matplotlib; `~/envs/perfrdd` made |
| 2026-11-15 | 66189320 | test, debug, 2 CPUs, 2 reps x 10 fits per n | 17 s; 0.095 / 0.245 / 0.82 s per fit; MaxRSS 190 MB; billing 5009 (2 CPUs) |
| 2026-11-15 | 66189342 | Table 2 array, 60 tasks x 4 CPUs, commit 499e8bb, out `experiments/runs/ds_four_role_coverage_nl_999` | submitted |
| 2026-11-15 | 66189353 | merge, afterok:66189342 | cancelled: tasks 44 and 57 (n = 64k) on slow nodes (gl3311, gl3369) would time out |
| 2026-11-15 | 66189654 | non-Gaussian t5/mixture, nonlinear effect, 8 CPUs | 5 min 42 s; out `experiments/runs/four_role_nongaussian_nl` |
| 2026-11-15 | 66191050 | rerun of array tasks 44, 57 (afterany:66189342), resumes from their saved rows | not needed: both tasks had finished (about 1014 s per rep); the rerun found every rep done |
| 2026-11-15 | 66191051 | merge, afterok:66191050 | done 22:13; 500 / 500 / 200 data sets, 999 draws; summary in `displayed_summary.json` |
| 2026-11-15 | 66191101, 66191102 | rerun and merge resubmitted by a failed relative-path `readbuf` | rerun found every rep done; merge cancelled; merged files checked equal to the shards |
| 2026-11-15 | 66191143 | taxi bootstrap test, debug, 2 CPUs, all-hours sample, out `experiments/runs/taxi_boot_test` | setup 2.7 min, 61 s per draw per CPU, peak 5.5 GB |
| 2026-11-15 | 66191339 | taxi robustness bootstrap array, 40 tasks x 8 CPUs, 3 GB/CPU, commit 6f46d92, out `experiments/runs/taxi_robust_boot` | every task 20 min or less; about 90 CPU-hours |
| 2026-11-15 | 66191340 | taxi merge, afterok:66191339 | 999 of 999 draws in all 9 rows; intervals in the 2026-11-15 research-log entry |
| 2026-11-16 | none | copy back through the tab (`gl_fetch_parts.sh`) | 67,010,654-byte archive, sha256 matched; unpacked into local `experiments/runs/` (git-ignored) |
| 2026-11-16 | 66197214 | Table 2 at 64k, replications 200-499: 75 tasks x 4 CPUs, one round each, commit 8a5b464, out `experiments/runs/ds_four_role_coverage_nl_999_n64k500` | 772-1140 s (median 827); about 70 CPU-hours; 40 tasks started within two minutes |
| 2026-11-16 | 66197215 | merge, afterok:66197214, `R64=500` | 500 / 500 / 500 data sets; copied back (21 MB archive, sha256 matched) |
| 2026-11-17 | none | login screen session ended overnight ("[screen is terminating]"); one login node refused port 22 on the first retry | author logged in again; cluster copies `~/xfer_64k.tgz` and `~/xfer/` deleted (author's approval, 2026-11-16) |
| 2026-11-17 | 66267571 | non-Gaussian coverage test, debug, 4 CPUs, 4 data sets x 19 draws, t5 at 4k/16k/64k and mixture at 256k, commit f88b5d1, out `experiments/runs/ds_ng_cov_test` (gl3053) | 75 s; 0.061 / 0.175 / 0.62 / 2.52 s per fit; MaxRSS 317 MB; about 590 CPU-hours for the full run |
| 2026-11-17 | 66267585 | non-Gaussian coverage, `SET=long` (mixture 256k), 125 tasks x 4 CPUs, `--time=01:50:00`, commit f88b5d1, out `experiments/runs/ds_nongaussian_coverage_999` | submitted; 60 tasks running within a minute |
| 2026-11-17 | 66267586 | non-Gaussian coverage, `SET=short` (t5 and mixture at 4k/16k/64k), 330 tasks x 4 CPUs, 50 min limit | submitted |
| 2026-11-17 | 66267587 | merge, afterok:66267585:66267586 | submitted |
| 2026-11-17 | 66267585 | (result) | all 125 completed; 2328-3011 s (median 2574) for one round of 4 data sets at 256k |
| 2026-11-17 | 66267586 | (result) | all 330 completed; 182-1131 s (median 668) |
| 2026-11-17 | 66267587 | merge, afterok:66267585:66267586 | done; 500 distinct data sets in all 7 cells, 999 draws, no failed draws; about 600 CPU-hours in all |
| 2026-11-17 | none | copy back through the tab (`gl_fetch_parts.sh`, 41 parts) | 91,240,593-byte archive (merged rows, summaries, logs, shard provenance; shard rows left on the cluster), sha256 matched; unpacked into local `experiments/runs/ds_nongaussian_coverage_999/`, archive kept in its `transfer/`; summaries recomputed locally from the rows agree to 1e-6 |
| 2026-11-17 | 66272450 | R^2/treated-share coverage test, debug, 8 CPUs, commit 3a63926, out `experiments/runs/ds_r2bal_test` | r2_50_bal equals Table 2's rows exactly (difference 0); 0.37-0.6 s per fit at 64k, 1.4-2.1 s at 256k; MaxRSS 580 MB |
| 2026-11-17 | 66272512 | R^2/treated-share coverage, `SET=long` (10 designs at 256k), 630 tasks x 8 CPUs, `--time=02:00:00`, commit 3a63926, out `experiments/runs/ds_r2_balance_coverage_999` | submitted 03:20 |
| 2026-11-17 | 66272513 | the same, `SET=short` (10 designs at 4k/16k/64k), 850 tasks x 4 CPUs, 75 min limit | submitted |
| 2026-11-17 | 66272514 | merge, afterok:66272512:66272513 | submitted |
| 2026-11-17 | 66272616, 66272617 | taxi bootstrap resampling days / day-hours (`taxi_cluster_boot.py`), 8 CPUs x 3 GB, 999 draws each, out `experiments/runs/taxi_cluster_boot` | submitted |
| 2026-11-17 | 66272733 | taxi_like test, debug, 8 CPUs, n = 633,616 | 8.0-8.2 s per fit; MaxRSS 1.4 GB |
| 2026-11-17 | 66272837, 66272838 | taxi_like coverage (R^2 0.925, 11% treated, eps 0.05): `SET=taxi_big` (633,616; 63 tasks x 8 CPUs, 2 GB/CPU, 6 h) and `SET=taxi_small` (64k, 256k; 126 tasks x 8 CPUs, 2 h); commit d34400a from the worktree `~/src/perfRDD_b` (so the queued tasks above keep commit 3a63926's code), out `~/src/perfRDD/experiments/runs/ds_r2bal_taxi_like_999` | submitted |
| 2026-11-17 | 66272839 | merge, afterok:66272837:66272838, `DESIGNS=taxi_like NS="64000 256000 633616"` | submitted |
| 2026-11-17 | 66272616, 66272617 | (result) | 32 min each; days [$12.42, $13.28], day-hours [$12.53, $13.17]; copied back (sha256 matched) |
| 2026-11-17 | 66272512, 66272513, 66272514 | (result) | all 630 + 850 tasks completed, merge done 07:03; 40 cells x 500 data sets x 999 draws, no failed draws, about 3,500 CPU-hours; summaries and the 844 MB archive of merged rows copied back (sha256 matched; summaries recomputed locally agree to 1e-6); shard rows left on the cluster |
| 2026-11-17 | 66274792 | interim summary from shard rows (`ds_r2_balance_peek.py`), debug | monitoring only |
| 2026-11-17 | 66276359 | packing the merged rows (tar + gzip, base64 parts), debug, 2 CPUs | 2.6 min; 371 parts, about 8 s each through the tab |
| 2026-11-17 | 66272837, 66272838, 66272839 | (result) | taxi_like: 633,616 tasks ran 2.2-3.5 h; merge done about 08:20; copied back (63 MB, sha256 matched) |
| 2026-11-17 | 66277857, 66277858 | location-scale row on the day / day-hour draws, run from `~/src/perfRDD_b` | failed in 3 s: the worktree has no taxi data (git-ignored), so every worker died loading it |
| 2026-11-17 | 66278466, 66278467 | the same, rerun from `~/src/perfRDD` (updated to 3aca0c1 once no task was queued) | 44 and 40 min; day: location-scale [$11.82, $12.96], paired difference [-$0.78, -$0.15]; day-hour: [$12.02, $12.81], [-$0.72, -$0.19]; copied back (sha256 matched) |
| 2026-11-18 | none | login screen session had ended; author logged in again (tab "great lakes login") | scratch check for the author: `/scratch/stats_dept_root/stats_dept1/zoudj` and `stats_dept2/zoudj` are empty (one entry each); all PerfRDD runs are under `/home` |
| 2026-11-18 | 66381766 | review-ablation test (`ds_review_ablations.py`), debug, 8 CPUs, commit dd7bbcd, out `experiments/runs/ds_review_test` | 69 s; 2.3-2.7 s per fit at 256k, 0.6 s at 64k; MaxRSS 837 MB with 4 workers |
| 2026-11-18 | 66381844 | review ablations, `SET=long` (11 designs at 256k), 693 tasks x 8 CPUs, 2 GB/CPU, `--time=03:00:00`, commit dd7bbcd, out `experiments/runs/ds_review_ablations_999` | submitted |
| 2026-11-18 | 66381846 | the same, `SET=short` (16k, 64k), 869 tasks x 4 CPUs, 90 min limit | submitted |
| 2026-11-18 | 66381847 | merge, afterok:66381844:66381846 | submitted; about 5,200 CPU-hours expected in all |

Lesson (2026-11-15): seconds per replication at n = 64k ranged from 796 to 1637 across
standard nodes, a factor of two, so a time limit of 1.7 times the median was too tight for
the slowest nodes. Allow about 2.5 times the median, or make shards one round long.

Lesson (2026-11-17): a second worktree keeps queued tasks on their commit, but it has none
of the git-ignored inputs (the taxi parquet, saved runs); jobs that read them must run from
the main clone or be given absolute paths.
