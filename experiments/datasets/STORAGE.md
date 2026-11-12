# Raw-data storage (2026-11-11)

The datasets' `data/` folders are git-ignored and held 59 GB on the laptop (disk at 99%).
The archive lives in the author's Michigan Dropbox, reached through `rclone` (Homebrew,
v1.76), remote name `dropbox`, folder `PerfRDD-data/<dataset>/data`, mirroring the tree.
The desktop app is not installed; nothing syncs in the background.

Commands (run from `code/`):

```bash
# archive a dataset's data/ (copy, then rclone check of sizes and hashes; never deletes)
python -m experiments.datasets.relocate_data --archive --only vlbw
# bring it back
python -m experiments.datasets.relocate_data --fetch --only vlbw
# remove a local copy after a verified archive (explicit; refuses if the check fails);
# leaves data/ARCHIVED.txt with the fetch command
python -m experiments.datasets.relocate_data --remove-local --only vlbw
```

Authorization: `rclone config create dropbox dropbox` opened the browser once and the author
clicked Allow (2026-11-11); the token sits in `~/.config/rclone/rclone.conf`, not in the repo.
Archived so far: see the table kept at the end of this file.

| Dataset | Archived | Local copy |
|---|---|---|
