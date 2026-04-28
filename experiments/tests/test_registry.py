"""Registry + runner integration tests.

These tests verify that whatever datasets are present (i.e. their data files
exist) can be loaded and produce a valid RDDSample.

Datasets whose data is absent on this machine (FileNotFoundError) are
SKIPPED, not failed — that mirrors the runner's behavior. This keeps the
test suite green on a fresh clone where only some datasets have data.
"""
from __future__ import annotations

import unittest

from experiments._core.registry import iter_available, list_datasets, load
from experiments._core.runner import run_all
from experiments._core.sample import RDDSample
from experiments.methods.summary import summary


class TestRegistry(unittest.TestCase):
    def test_lists_known_datasets(self):
        names = list_datasets()
        # Every folder we scaffolded with an adapter.py should be discovered.
        for expected in ("gpa", "hmda", "lending_club", "nlsy", "taxi"):
            self.assertIn(expected, names, f"{expected} missing from registry")

    def test_load_returns_rddsample_when_data_present(self):
        any_loaded = False
        for name in list_datasets():
            try:
                sample = load(name)
            except (FileNotFoundError, NotImplementedError):
                continue
            self.assertIsInstance(sample, RDDSample, f"{name} returned wrong type")
            self.assertEqual(sample.n, len(sample.Y), f"{name} n mismatch")
            self.assertEqual(sample.X.shape[0], sample.n, f"{name} X rows")
            self.assertEqual(len(sample.Q), sample.n, f"{name} Q len")
            any_loaded = True
        # If literally no dataset loaded, that means no data is on this machine.
        # We don't fail on that — but we do print a warning so it's visible.
        if not any_loaded:
            print("[warn] no datasets had local data; this is fine on a fresh clone")


class TestRunner(unittest.TestCase):
    def test_run_all_with_summary_method(self):
        results = run_all(summary)
        # Whatever loads should have a summary dict.
        for name, payload in results.items():
            self.assertIn("n", payload)
            self.assertIn("threshold", payload)
            self.assertGreater(payload["n"], 0, f"{name} reported n=0")

    def test_run_all_only_filter(self):
        results = run_all(summary, only=["gpa"])
        self.assertTrue(set(results).issubset({"gpa"}))


class TestIterAvailable(unittest.TestCase):
    def test_iter_yields_pairs(self):
        for name, sample in iter_available():
            self.assertIsInstance(name, str)
            self.assertIsInstance(sample, RDDSample)
            break  # exercising the generator once is enough


if __name__ == "__main__":
    unittest.main()
