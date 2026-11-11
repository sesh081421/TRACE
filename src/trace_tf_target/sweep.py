"""Multi-dataset runner for the released TRACE model."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DATASETS = [
    ("GSE150774_RAW/BM", False),
    ("GSE150774_RAW/UCB", False),
    ("pbmc_granulocytes", False),
    ("lymph", False),
    ("hESC", True),
    ("hHep", True),
]

SHORT_NAMES = {
    "GSE150774_RAW/BM": "BM",
    "GSE150774_RAW/UCB": "UCB",
    "pbmc_granulocytes": "pbmc",
    "lymph": "lymphoma",
    "hESC": "hESC",
    "hHep": "hHep",
}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def has_test_results(path: Path) -> bool:
    """A validation-only or interrupted run must not skip a full sweep run."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    metrics = payload.get("test_metrics")
    return isinstance(metrics, dict) and bool(metrics)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the released TRACE model across datasets and seeds."
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="8,18,28,38,48")
    parser.add_argument("--data_root", default="./dataset")
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_neg_ratio", type=int, default=10)
    parser.add_argument("--validation_neg_ratio", type=int, default=1)
    parser.add_argument("--eval_seed", type=int, default=20260821)
    parser.add_argument("--eval_background_root", default=None)
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.datasets == "all":
        selected = DATASETS
    else:
        wanted = set(parse_csv(args.datasets))
        selected = [item for item in DATASETS if item[0] in wanted]
        missing = wanted - {name for name, _ in selected}
        if missing:
            raise SystemExit(f"Unknown dataset(s): {sorted(missing)}")

    env = os.environ.copy()
    # A script-local sys.path change is not inherited by the child interpreter.
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [package_root, env.get("PYTHONPATH", "")] if part
    )
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for dataset, benchmark in selected:
        short_name = SHORT_NAMES[dataset]
        for seed in parse_int_csv(args.seeds):
            metrics_path = (
                Path(args.result_dir)
                / short_name
                / f"seed_{seed}"
                / f"TRACE_{short_name}_seed_{seed}_metrics.json"
            )
            if args.skip_existing and has_test_results(metrics_path):
                print(f"Skipping existing dataset={dataset} seed={seed}")
                continue

            command = [
                sys.executable,
                "-m",
                "trace_tf_target.cli",
                "--data",
                dataset,
                "--benchmark",
                str(benchmark).lower(),
                "--data_root",
                args.data_root,
                "--result_dir",
                args.result_dir,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
                "--eval_neg_ratio",
                str(args.eval_neg_ratio),
                "--validation_neg_ratio",
                str(args.validation_neg_ratio),
                "--eval_seed",
                str(args.eval_seed),
            ]
            if args.eval_background_root is not None:
                command.extend(["--eval_background_root", args.eval_background_root])
            print(f"Running dataset={dataset} seed={seed}")
            print("Command:", " ".join(command))
            subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
