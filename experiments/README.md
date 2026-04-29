# Experiments

A unified harness for applying RDD methodologies to multiple real-world
datasets. Every dataset exposes the same minimal contract (`adapter.py` →
`RDDSample`); a registry auto-discovers them; a runner applies any
method-as-a-function to whichever datasets have data on this machine.

## Layout

```
experiments/
├── _core/                       # the framework (don't edit unless changing the contract)
│   ├── sample.py                #   RDDSample dataclass — the contract
│   ├── registry.py              #   auto-discovers datasets/*/adapter.py
│   └── runner.py                #   run_all(method_fn) helper
├── datasets/
│   ├── gpa/                     # academic probation (Lindo et al.)
│   ├── hmda/                    # 2024 mortgage, dual-threshold (build via notebook)
│   ├── taxi/                    # NYC TLC 2009 default tips (Haggag-Paci)
│   ├── oulad/                   # first-TMA pass cutoff (UK Open University)
│   ├── nhanes/                  # HbA1c diabetic threshold
│   ├── lending_club/            # FICO floor (Kaggle auth needed)
│   ├── nlsy/                    # AFQT → wages (manual extract)
│   └── mimic/                   # ICU severity score (PhysioNet DUA needed)
├── methods/                     # functions of RDDSample → result
│   └── summary.py               #   trivial reference implementation
├── scripts/
│   └── run_all.py               #   CLI: python -m experiments.scripts.run_all
└── tests/                       # python -m unittest discover -s experiments/tests
```

A dataset folder typically looks like:

```
datasets/<name>/
├── adapter.py         # required: load() -> RDDSample
├── README.md          # required: Q, X, Y, threshold, source
├── data/
│   ├── raw/           # gitignored — actual data files
│   └── processed/     # gitignored — cached intermediates
├── analysis/          # optional: dataset-specific scripts
├── notes/             # optional: notebooks, exploration
└── figures/           # optional: outputs of analysis scripts
```

Only `adapter.py` and `README.md` are required. Anything else under
`datasets/<name>/` is your private workspace; the framework doesn't care.

## Running

```bash
# from the repo root
python -m unittest discover -s experiments/tests          # tests
python -m experiments.scripts.run_all                     # all datasets, summary method
python -m experiments.scripts.run_all --only gpa hmda     # subset
python -m experiments.scripts.run_all --method summary --out runs/today.json
```

Datasets whose data files aren't present locally are *skipped* with a
warning rather than failing — so a fresh clone with just the GPA CSV will
still run.

---

## How to add a new dataset

1. **Create the folder.**
   ```
   experiments/datasets/<name>/
   ├── __init__.py              # empty
   ├── adapter.py
   ├── README.md
   └── data/raw/                # data goes here, gitignored
   ```

2. **Write `adapter.py`.** It must export `load() -> RDDSample`. Minimal
   skeleton:

   ```python
   from pathlib import Path
   import numpy as np
   import pandas as pd
   from experiments._core.sample import RDDSample

   DATA = Path(__file__).parent / "data" / "raw" / "main.csv"
   X_COLS = ["x1", "x2", ...]

   def load() -> RDDSample:
       if not DATA.exists():
           raise FileNotFoundError(
               f"{DATA} missing. See README for download instructions."
           )
       df = pd.read_csv(DATA)
       return RDDSample(
           Q=df["score"].to_numpy(dtype=float),
           X=df[X_COLS].to_numpy(dtype=float),
           Y=df["outcome"].to_numpy(dtype=float),
           threshold=0.0,                       # scalar; or tuple for multi
           name="<name>",
           feature_names=list(X_COLS),
           description="...",
           citation="...",
       )
   ```

   * For **multi-threshold** designs, `Q` has shape `(n, k)` and `threshold`
     is a length-`k` tuple. The default treatment rule is "above the
     threshold on every running variable".
   * For **non-default treatment rules** (e.g. `Q < threshold`), pass a
     callable `treatment_rule=lambda Q: ...` that returns a 0/1 ndarray.
     See `experiments/datasets/gpa/adapter.py` for an example.

3. **Write `README.md`.** Use the table-of-fields format the existing
   datasets use: Q, threshold, treatment, X, Y, n, citation, plus how to
   download/build the data.

4. **Add a `download.py`** if the data is fetchable programmatically. It
   should be idempotent (skip if data already present) and write into
   `data/raw/`.

5. **Verify it loads.**
   ```bash
   python -c "from experiments._core.registry import load; print(load('<name>').summary())"
   python -m unittest discover -s experiments/tests
   ```

That's it — the registry picks it up automatically.

## How to add a new method

1. Create `experiments/methods/<method>.py` exporting a function of the
   same name (or `run`):

   ```python
   from experiments._core.sample import RDDSample

   def my_method(sample: RDDSample) -> dict:
       D = sample.D
       # ... your estimator ...
       return {"phi_hat": ..., "alpha": ...}
   ```

2. Run it across everything:

   ```bash
   python -m experiments.scripts.run_all --method my_method
   ```

For dataset-specific or one-off analysis, just write a script under
`datasets/<name>/analysis/` that imports `from experiments.datasets.<name>.adapter
import load` (or reads the raw CSV directly) — there's no requirement to
go through `methods/`.

---

## Data, git, and storage

The default policy is:

| File class | In git? | Notes |
|---|---|---|
| `adapter.py`, `download.py`, READMEs | yes | small text |
| `*.dta`, `*.csv` (small, ≤ ~100 MB) | git-LFS | already configured in `.gitattributes` |
| `data/raw/` and `data/processed/` directories | **no** (gitignored) | except for `.dvc` pointer files |
| `runs/`, `figures/` | gitignored | regenerable outputs |
| DUA-restricted data (e.g. MIMIC, restricted-use NLSY) | **never** | `download.py` should error with manual instructions |

If any single dataset grows beyond a few hundred MB, switch from git-LFS
to [DVC](https://dvc.org):

```bash
dvc init --subdir
cd experiments/datasets/<name>
dvc add data/raw/big_file.parquet
git add data/raw/big_file.parquet.dvc data/raw/.gitignore
```

Set the DVC remote to a folder you control (local drive, Google Drive, S3).

---

## The contract (what `RDDSample` guarantees)

```python
@dataclass
class RDDSample:
    Q: np.ndarray                     # (n,) single | (n, k) multi
    X: np.ndarray                     # (n, p) covariates
    Y: np.ndarray                     # (n,) continuous outcome
    threshold: float | tuple          # scalar | per-Q-column
    name: str
    feature_names: list[str]
    description: str = ""
    citation: str = ""
    treatment_rule: Callable | None   # default: AND of (Q > threshold)
    extras: dict                      # weights, fold ids, raw df, etc.

    @property
    def D(self) -> np.ndarray         # 0/1 treatment indicator
    def summary(self) -> dict
```

All adapters validate length / shape consistency in `__post_init__`, so a
malformed adapter fails fast at `load()` time rather than midway through a
method.
