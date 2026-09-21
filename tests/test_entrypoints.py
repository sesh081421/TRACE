"""Exercise checkout entry points without an editable install or caller PYTHONPATH."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CheckoutEntryPointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="trace entrypoint ")
        self.addCleanup(self.directory.cleanup)
        self.cwd = Path(self.directory.name)
        self.env = os.environ.copy()
        self.env.pop("PYTHONPATH", None)
        self.env.update(
            PYTHON_BIN=sys.executable,
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_VISIBLE_DEVICES="-1",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )

    def run_command(self, command):
        return subprocess.run(command, cwd=self.cwd, env=self.env,
                              text=True, capture_output=True, timeout=120)

    def test_python_wrappers_from_another_directory(self):
        for name in ["run_trace.py", "run_sweep.py", "summarize.py"]:
            with self.subTest(script=name):
                result = self.run_command([sys.executable, str(ROOT / "scripts" / name), "--help"])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_shell_examples_support_help_without_training(self):
        for name in ["run_quick_check.sh", "run_all_datasets.sh"]:
            with self.subTest(script=name):
                result = self.run_command(["bash", str(ROOT / "examples" / name), "--help"])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)
                self.assertNotIn("Epoch 1/", result.stdout)

    def test_sweep_child_imports_package_before_loading_missing_data(self):
        # The missing prior deliberately stops the child before any training.
        result = self.run_command([
            sys.executable, str(ROOT / "scripts/run_sweep.py"),
            "--datasets", "GSE150774_RAW/BM", "--seeds", "8", "--epochs", "1",
            "--data_root", str(self.cwd / "missing_data"),
            "--result_dir", str(self.cwd / "results"),
        ])
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("ModuleNotFoundError", combined)
        self.assertNotIn("No module named", combined)
        self.assertIn("FileNotFoundError", combined)
        self.assertIn("tflink.h5", combined)
        self.assertFalse((self.cwd / "results").exists())


if __name__ == "__main__":
    unittest.main()
