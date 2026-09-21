"""Public CLI for the released TRACE model."""

import argparse
import sys

from . import train


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train TRACE: Transcription-factor-aware Regulatory Architecture "
            "for Candidate-Edge prioritization."
        )
    )
    parser.add_argument("--data", required=True, help="Configured dataset name.")
    parser.add_argument(
        "--benchmark",
        choices=["true", "false"],
        default=None,
        help="Override BEELINE-style input handling.",
    )
    parser.add_argument("--data_root", default="./dataset")
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument("--prior_filename", default="tflink.h5")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_neg_ratio", type=int, default=10)
    parser.add_argument("--validation_neg_ratio", type=int, default=1)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=20260821,
        help="Fixed candidate-background seed, independent of the optimization seed.",
    )
    parser.add_argument(
        "--eval_background_root",
        default=None,
        help="Optional directory containing or receiving frozen evaluation candidates.",
    )
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001331512668153493)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_head", default="3,3")
    parser.add_argument("--hidden_dim", default="128,64")
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--topo_thr", type=float, default=0.15)
    parser.add_argument("--refresh_eval", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    forwarded = [
        "trace-train",
        "--data",
        args.data,
        "--data_root",
        args.data_root,
        "--result_dir",
        args.result_dir,
        "--prior_filename",
        args.prior_filename,
        "--seed",
        str(args.seed),
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
        "--neg_ratio",
        str(args.neg_ratio),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--num_head",
        args.num_head,
        "--hidden_dim",
        args.hidden_dim,
        "--output_dim",
        str(args.output_dim),
        "--topk",
        str(args.topk),
        "--topo_thr",
        str(args.topo_thr),
        "--attention_mode",
        "topology_only",
        "--encoder_type",
        "gat",
        "--decoder",
        "bilinear",
        "--selection_metric",
        "AUPRC",
        "--train_neg_sampling",
        "global",
        "--loss_type",
        "legacy_focal",
        "--exclude_eval_candidates_from_train_negatives",
    ]
    if args.benchmark is not None:
        forwarded.extend(["--benchmark", args.benchmark])
    if args.refresh_eval:
        forwarded.append("--refresh_eval")
    if args.eval_background_root is not None:
        forwarded.extend(["--eval_background_root", args.eval_background_root])

    sys.argv = forwarded
    train.main()


if __name__ == "__main__":
    main()
