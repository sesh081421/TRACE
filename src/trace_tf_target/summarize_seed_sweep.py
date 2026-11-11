import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

METRICS = [
    "AUROC",
    "AUPRC",
    "F1",
    "MCC",
    "PerTF_AUPRC",
    "TF_P@10",
    "TF_P@30",
    "TF_P@50",
    "TF_P@100",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize multi-seed TRACE results.")
    parser.add_argument("--result_dir", default="./results", help="Base result directory.")
    parser.add_argument("--raw_csv", default=None, help="Optional output path for raw run table.")
    parser.add_argument("--summary_csv", default=None, help="Optional output path for aggregated summary table.")
    parser.add_argument(
        "--metric_split", choices=["test", "validation"], default="test",
        help="Summarize test metrics (default) or best validation metrics; never mix the two.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    metric_files = sorted(result_dir.glob("**/*_metrics.json"))
    if not metric_files:
        raise SystemExit(f"No metrics files found under {result_dir}")

    rows = []
    for metric_file in metric_files:
        payload = json.loads(metric_file.read_text(encoding="utf-8"))
        config = payload["config"]
        metric_key = "test_metrics" if args.metric_split == "test" else "best_validation_metrics"
        metrics = payload.get(metric_key)
        if metrics is None:
            print(f"Skipping {metric_file}: no {args.metric_split} metrics.", file=sys.stderr)
            continue
        if not isinstance(metrics, dict) or not metrics:
            raise SystemExit(f"Invalid {metric_key} in {metric_file}")
        row = {
            "dataset": config["data"],
            "seed": config["seed"],
            "metric_split": args.metric_split,
            "metrics_file": str(metric_file),
            "best_threshold": payload["best_threshold"],
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric)
        rows.append(row)

    if not rows:
        hint = " Use --metric_split validation for validation-only runs." if args.metric_split == "test" else ""
        raise SystemExit(f"No {args.metric_split} metrics found under {result_dir}.{hint}")

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    summary_rows = []
    for dataset, group in raw_df.groupby("dataset", sort=True):
        row = {"dataset": dataset, "metric_split": args.metric_split, "n_runs": int(len(group))}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = math.nan
                row[f"{metric}_std"] = math.nan
                row[f"{metric}_ci95"] = math.nan
                continue
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)

    prefix = "seed_sweep" if args.metric_split == "test" else "seed_sweep_validation"
    raw_csv = Path(args.raw_csv) if args.raw_csv else result_dir / f"{prefix}_raw.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else result_dir / f"{prefix}_summary.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved raw runs: {raw_csv}")
    print(f"Saved summary: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
