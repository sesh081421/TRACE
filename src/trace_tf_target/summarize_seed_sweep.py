import argparse
import json
import math
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
        metrics = payload["test_metrics"]
        row = {
            "dataset": config["data"],
            "seed": config["seed"],
            "metrics_file": str(metric_file),
            "best_threshold": payload["best_threshold"],
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric)
        rows.append(row)

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    summary_rows = []
    for dataset, group in raw_df.groupby("dataset", sort=True):
        row = {"dataset": dataset, "n_runs": int(len(group))}
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

    raw_csv = Path(args.raw_csv) if args.raw_csv else result_dir / "seed_sweep_raw.csv"
    summary_csv = Path(args.summary_csv) if args.summary_csv else result_dir / "seed_sweep_summary.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved raw runs: {raw_csv}")
    print(f"Saved summary: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
