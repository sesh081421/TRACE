"""Keep validation-only outputs separate from completed test evaluations."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trace_tf_target import summarize_seed_sweep as summary
from trace_tf_target import sweep


class ResultModeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_result(self, seed, test=True):
        path = self.root / "BM" / f"seed_{seed}" / f"TRACE_BM_seed_{seed}_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {"data": "GSE150774_RAW/BM", "seed": seed, "validation_only": not test},
            "best_threshold": 0.5,
            "best_validation_metrics": {metric: 0.6 for metric in summary.METRICS},
            "test_metrics": {metric: 0.7 for metric in summary.METRICS} if test else None,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def summarize(self, *arguments):
        with patch.object(sys, "argv", ["summarize", "--result_dir", str(self.root), *arguments]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                summary.main()

    def test_validation_only_has_actionable_default_error(self):
        self.write_result(8, test=False)
        with self.assertRaisesRegex(SystemExit, "--metric_split validation"):
            self.summarize()
        self.assertFalse((self.root / "seed_sweep_raw.csv").exists())

    def test_explicit_validation_summary(self):
        self.write_result(8, test=False)
        self.summarize("--metric_split", "validation")
        raw = pd.read_csv(self.root / "seed_sweep_validation_raw.csv")
        self.assertEqual(raw.metric_split.tolist(), ["validation"])
        self.assertEqual(raw.AUPRC.tolist(), [0.6])
        self.assertFalse((self.root / "seed_sweep_raw.csv").exists())

    def test_mixed_directory_does_not_mix_endpoints(self):
        self.write_result(8, test=False)
        self.write_result(18)
        self.summarize()
        self.summarize("--metric_split", "validation")
        test = pd.read_csv(self.root / "seed_sweep_raw.csv")
        validation = pd.read_csv(self.root / "seed_sweep_validation_raw.csv")
        self.assertEqual(test.seed.tolist(), [18])
        self.assertEqual(test.AUPRC.tolist(), [0.7])
        self.assertEqual(validation.seed.tolist(), [8, 18])

    def test_missing_and_partial_metrics_are_not_complete(self):
        path = self.root / "partial_metrics.json"
        self.assertFalse(sweep.has_test_results(path))
        for contents in ("{", "null", "[]", '{"test_metrics": {}}'):
            path.write_text(contents, encoding="utf-8")
            self.assertFalse(sweep.has_test_results(path))

    def test_sweep_reruns_validation_only(self):
        self.write_result(8, test=False)
        args = ["sweep", "--datasets", "GSE150774_RAW/BM", "--seeds", "8",
                "--result_dir", str(self.root), "--skip_existing"]
        with patch.object(sys, "argv", args), patch.object(sweep.subprocess, "run") as run:
            with redirect_stdout(io.StringIO()):
                sweep.main()
            run.assert_called_once()

    def test_sweep_skips_completed_test_run(self):
        self.write_result(8)
        args = ["sweep", "--datasets", "GSE150774_RAW/BM", "--seeds", "8",
                "--result_dir", str(self.root), "--skip_existing"]
        with patch.object(sys, "argv", args), patch.object(sweep.subprocess, "run") as run:
            with redirect_stdout(io.StringIO()):
                sweep.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
