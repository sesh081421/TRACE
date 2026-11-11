import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

if __package__ in {None, ''}:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch_sparse import SparseTensor

from trace_tf_target.early_stopping import EarlyStopping
from trace_tf_target.model import TRACE
from trace_tf_target.topology_controls import bipartite_degree_preserving_rewire
from trace_tf_target.graph_null import load_or_create_training_null
from trace_tf_target.utils import (
    SHORT_NAMES,
    add_bidirectional_edges,
    apply_prior_robustness,
    audit_eval_benchmarks,
    build_eval_benchmarks,
    classification_loss_with_logits,
    evaluate_predictions,
    generate_negative_edges_train,
    get_prior_network,
    infer_benchmark_mode,
    load_expression,
    load_expression_rna_only,
    load_or_prepare_dataset_artifacts,
    optimize_probability_threshold,
    parse_int_list,
    positive_edge_set,
    prepare_feature_matrix,
    set_seed,
    str2bool,
)


class EdgeDataset(Dataset):
    def __init__(self, edges: torch.Tensor):
        self.edges = edges.t().contiguous()

    def __len__(self):
        return self.edges.shape[0]

    def __getitem__(self, index):
        return self.edges[index]


def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    pos_logits = logits[labels > 0.5]
    neg_logits = logits[labels <= 0.5]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return logits.new_tensor(0.0)
    return torch.relu(margin - pos_logits[:, None] + neg_logits[None, :]).mean()


def rewire_topology_source_preserving(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    adj: SparseTensor,
    num_nodes: int,
    seed: int,
):
    """Randomize topology destinations while preserving each source degree and weight multiset."""
    row_cpu = row.detach().cpu().long()
    col_cpu = col.detach().cpu().long()
    weight_cpu = weight.detach().cpu()
    if row_cpu.numel() == 0:
        raise ValueError("Cannot rewire an empty derived topology graph.")

    direct_neighbors = [set() for _ in range(num_nodes)]
    direct_row, direct_col, _ = adj.coo()
    for src, dst in zip(direct_row.detach().cpu().tolist(), direct_col.detach().cpu().tolist()):
        direct_neighbors[int(src)].add(int(dst))
        direct_neighbors[int(dst)].add(int(src))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rewired_col = torch.empty_like(col_cpu)
    rewired_weight = torch.empty_like(weight_cpu)
    source_degrees = []

    sources, counts = torch.unique_consecutive(row_cpu, return_counts=True)
    offset = 0
    for src, count in zip(sources.tolist(), counts.tolist()):
        degree = int(count)
        positions = torch.arange(offset, offset + degree)
        offset += degree
        original_targets = set(int(item) for item in col_cpu[positions].tolist())
        forbidden = direct_neighbors[int(src)] | original_targets | {int(src)}
        candidates = [node for node in range(num_nodes) if node not in forbidden]
        if len(candidates) < degree:
            raise ValueError(
                f"Insufficient rewiring candidates for source {src}: "
                f"need {degree}, found {len(candidates)}."
            )
        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        selected = candidate_tensor[torch.randperm(len(candidates), generator=generator)[:degree]]
        weight_order = torch.randperm(degree, generator=generator)
        rewired_col[positions] = selected
        rewired_weight[positions] = weight_cpu[positions][weight_order]
        source_degrees.append(degree)

    original_edges = set(zip(row_cpu.tolist(), col_cpu.tolist()))
    rewired_edges = set(zip(row_cpu.tolist(), rewired_col.tolist()))
    direct_edges = set()
    for src, neighbors in enumerate(direct_neighbors):
        direct_edges.update((src, dst) for dst in neighbors)

    audit = {
        "rewire_method": "source_degree_preserving",
        "rewire_seed": int(seed),
        "source_degree_preserved": True,
        "target_degree_preserved": False,
        "weight_multiset_preserved_per_source": True,
        "original_edge_overlap": len(original_edges & rewired_edges),
        "direct_edge_overlap": len(direct_edges & rewired_edges),
        "source_count": len(source_degrees),
        "edge_count": int(row_cpu.numel()),
    }
    return (
        row_cpu.to(row.device),
        rewired_col.to(col.device),
        rewired_weight.to(weight.device),
        audit,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRACE TF->Target training script.")
    parser.add_argument("--data", required=True, help="Dataset name.")
    parser.add_argument("--benchmark", type=str2bool, default=None, help="Override benchmark mode.")
    parser.add_argument("--data_root", default="./dataset", help="Data root directory.")
    parser.add_argument(
        "--split_manifest_root",
        default=None,
        help=(
            "Optional root containing versioned split_<id>/<dataset>/ train, "
            "validation, and test CSV files. The original data split is unchanged."
        ),
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=None,
        help="Versioned split identifier used with --split_manifest_root.",
    )
    parser.add_argument("--result_dir", default="./results", help="Directory for models and metrics.")
    parser.add_argument("--prior_filename", default="tflink.h5", help="Prior network file name.")
    parser.add_argument("--lr", type=float, default=0.0001331512668153493)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on the selected validation metric.")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument(
        "--train_neg_sampling",
        choices=["batch_local", "global", "mixed"],
        default="global",
        help="Training negative sampling pool: batch-local targets, global gene targets, or a 50/50 mixed pool.",
    )
    parser.add_argument("--eval_neg_ratio", type=int, default=10)
    parser.add_argument(
        "--validation_neg_ratio",
        type=int,
        default=None,
        help="Validation negative ratio. Defaults to --eval_neg_ratio.",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=20260821,
        help="Fixed benchmark seed, independent of the optimization seed.",
    )
    parser.add_argument(
        "--eval_background_root",
        default=None,
        help="Optional root for versioned evaluation backgrounds; dataset names are appended.",
    )
    parser.add_argument(
        "--allow_eval_negative_overlap",
        action="store_true",
        help="Permit legacy validation/test negative overlap. Never use for reported results.",
    )
    parser.add_argument(
        "--exclude_eval_candidates_from_train_negatives",
        action="store_true",
        help=(
            "Exclude every validation/test candidate pair from dynamic training-negative "
            "sampling and record a pair-identity audit."
        ),
    )
    parser.add_argument("--num_head", default="3,3", help="Two comma-separated attention head counts.")
    parser.add_argument("--hidden_dim", default="128,64", help="Two comma-separated hidden dimensions.")
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument("--neg_slope", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--topo_thr", type=float, default=0.15)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.0)
    parser.add_argument("--ranking_loss_margin", type=float, default=1.0)
    parser.add_argument(
        "--loss_type",
        choices=["legacy_focal", "bce", "focal"],
        default="legacy_focal",
        help="Classification loss. legacy_focal preserves earlier experiment behavior.",
    )
    parser.add_argument("--focal_alpha", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--cross_attn_lse_tau", type=float, default=0.7)
    parser.add_argument(
        "--prior_robustness_mode",
        choices=[
            "none",
            "topology_edge_dropout",
            "topology_source_rewire",
            "train_positive_subsample",
        ],
        default="none",
        help=(
            "Controlled prior perturbation. Topology modes keep supervision fixed; "
            "train_positive_subsample reduces both supervision and topology edges."
        ),
    )
    parser.add_argument(
        "--prior_robustness_fraction",
        type=float,
        default=0.0,
        help="Fraction of train-prior edges removed or rewired for robustness analysis.",
    )
    parser.add_argument(
        "--prior_robustness_seed",
        type=int,
        default=None,
        help="Perturbation seed. Defaults to the model seed for paired experiments.",
    )
    parser.add_argument(
        "--attention_mode",
        choices=["dual", "regulatory_only", "topology_only", "feature_only"],
        default="topology_only",
        help="Encoder branch setting for ablation.",
    )
    parser.add_argument(
        "--topology_graph_variant",
        choices=["derived", "direct", "rewired", "bipartite_rewired", "train_only_rewired"],
        default="derived",
        help=(
            "Topology-attention graph: derived 2-hop/Jaccard, direct train edges, "
            "a source-degree-rewired derived graph, held-out-excluding "
            "bipartite_rewired, or the label-blind train_only_rewired null."
        ),
    )
    parser.add_argument(
        "--topology_positive_scope",
        choices=["train_only", "all_splits"],
        default="train_only",
        help=(
            "Positive edges used to construct the topology source graph. all_splits is "
            "a deliberately leaky diagnostic and must never be reported as a valid model."
        ),
    )
    parser.add_argument(
        "--topology_rewire_seed",
        type=int,
        default=None,
        help="Seed for topology rewiring. Defaults to the model seed.",
    )
    parser.add_argument(
        "--topology_rewire_swaps_per_edge",
        type=float,
        default=10.0,
        help="Accepted double-edge swaps per train edge for bipartite rewiring.",
    )
    parser.add_argument(
        "--topology_rewire_max_attempt_factor",
        type=int,
        default=50,
        help="Maximum attempted swaps per requested swap for bipartite rewiring.",
    )
    parser.add_argument(
        "--topology_null_cache", default=None,
        help="Shared graph CSV for train_only_rewired; reuse one graph seed/cache across optimization seeds.",
    )
    parser.add_argument(
        "--disable_jaccard_bias",
        action="store_true",
        help="Keep derived topology edges but fix their learned Jaccard attention coefficient to zero.",
    )
    parser.add_argument(
        "--encoder_type",
        choices=["gat", "sparse_transformer"],
        default="gat",
        help="Encoder backbone. sparse_transformer keeps the train-derived topology graph but replaces GAT-style attention.",
    )
    parser.add_argument(
        "--decoder",
        choices=["hybrid", "bilinear", "bilinear_bias", "calibrated_mlp", "role_bilinear", "local_cross_attn", "local_cross_attn_lse"],
        default="bilinear",
        help="Edge decoder setting for ablation.",
    )
    parser.add_argument(
        "--feature_mode",
        choices=["default", "rna_only", "pca"],
        default="default",
        help="Use default, RNA-only, or fixed-dimensional PCA expression features.",
    )
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--pca_seed", type=int, default=20260819)
    parser.add_argument(
        "--selection_metric",
        choices=["F1", "AUPRC", "AUROC", "MCC", "PerTF_AUPRC"],
        default="AUPRC",
        help="Validation metric used for best checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Save best validation metrics and skip test evaluation (for hyperparameter tuning).",
    )
    parser.add_argument("--refresh_eval", action="store_true", help="Rebuild validation/test benchmarks.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.num_head = parse_int_list(args.num_head, expected_len=2, name="num_head")
    args.hidden_dim = parse_int_list(args.hidden_dim, expected_len=2, name="hidden_dim")
    args.benchmark = infer_benchmark_mode(args.data, args.benchmark)
    if not 0.0 <= args.focal_alpha <= 1.0:
        parser.error("--focal_alpha must be between 0 and 1.")
    if args.focal_gamma < 0.0:
        parser.error("--focal_gamma must be non-negative.")
    if (args.split_manifest_root is None) != (args.split_id is None):
        parser.error("--split_manifest_root and --split_id must be provided together.")
    if args.split_id is not None and args.split_id < 0:
        parser.error("--split_id must be non-negative.")
    if args.prior_robustness_seed is None:
        args.prior_robustness_seed = args.seed
    if args.topology_rewire_seed is None:
        args.topology_rewire_seed = args.seed
    if args.encoder_type == "sparse_transformer" and args.attention_mode != "topology_only":
        parser.error("--encoder_type sparse_transformer requires --attention_mode topology_only.")
    if args.attention_mode == "feature_only" and args.topology_graph_variant != "derived":
        parser.error("--attention_mode feature_only does not use --topology_graph_variant.")
    if args.topology_graph_variant != "derived" and args.attention_mode not in {"dual", "topology_only"}:
        parser.error("Non-derived topology variants require a topology attention branch.")
    if (
        args.topology_graph_variant in {"bipartite_rewired", "train_only_rewired"}
        and args.prior_robustness_mode != "none"
    ):
        parser.error(
            "Bipartite rewiring requires "
            "--prior_robustness_mode none."
        )
    if args.topology_null_cache is not None and args.topology_graph_variant != "train_only_rewired":
        parser.error("--topology_null_cache requires --topology_graph_variant train_only_rewired.")

    if args.topology_positive_scope == "all_splits":
        if args.prior_robustness_mode != "none":
            parser.error("--topology_positive_scope all_splits requires --prior_robustness_mode none.")
        if args.topology_graph_variant != "derived":
            parser.error("--topology_positive_scope all_splits requires --topology_graph_variant derived.")
        if args.attention_mode not in {"dual", "topology_only"}:
            parser.error(
                "--topology_positive_scope all_splits requires a topology attention branch."
            )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_net = get_prior_network(args.data_root, args.prior_filename)
    if args.feature_mode == "rna_only":
        x = load_expression_rna_only(args.data_root, args.data, args.benchmark)
    else:
        x = load_expression(args.data_root, args.data, args.benchmark)
    artifacts = load_or_prepare_dataset_artifacts(args.data_root, args.data, x, prior_net, args.seed)

    split_manifest_dir = None
    if args.split_manifest_root is not None:
        short_name = SHORT_NAMES.get(args.data, args.data.replace("/", "_"))
        split_manifest_dir = (
            Path(args.split_manifest_root) / f"split_{args.split_id}" / short_name
        )
        split_paths = {
            "train": split_manifest_dir / "train_data.csv",
            "val": split_manifest_dir / "validation_data.csv",
            "test": split_manifest_dir / "test_data.csv",
        }
        missing = [str(path) for path in split_paths.values() if not path.exists()]
        if missing:
            parser.error(f"Missing versioned split files: {missing}")
        artifacts.update(split_paths)
        print(f"[Split Manifest] split_id={args.split_id} path={split_manifest_dir}")

    short_name = SHORT_NAMES.get(args.data, args.data.replace("/", "_"))
    eval_output_dir = split_manifest_dir
    if args.eval_background_root is not None:
        eval_output_dir = Path(args.eval_background_root)
        if args.split_id is not None:
            eval_output_dir = eval_output_dir / f"split_{args.split_id}"
        eval_output_dir = eval_output_dir / short_name

    val_eval_file, test_eval_file = build_eval_benchmarks(
        data_root=args.data_root,
        data_type=args.data,
        tf_file=artifacts["tf"],
        tg_file=artifacts["tg"],
        train_file=artifacts["train"],
        val_file=artifacts["val"],
        test_file=artifacts["test"],
        neg_ratio=args.eval_neg_ratio,
        seed=args.eval_seed,
        refresh=args.refresh_eval,
        output_dir=eval_output_dir,
        validation_neg_ratio=args.validation_neg_ratio,
        require_disjoint_negatives=not args.allow_eval_negative_overlap,
    )

    tf_df = pd.read_csv(artifacts["tf"])
    tg_df = pd.read_csv(artifacts["tg"])
    full_train_df = pd.read_csv(artifacts["train"])
    val_df = pd.read_csv(artifacts["val"])
    test_df = pd.read_csv(artifacts["test"])
    val_eval_df = pd.read_csv(val_eval_file)
    test_eval_df = pd.read_csv(test_eval_file)
    all_positive_edges = (
        positive_edge_set(full_train_df)
        | positive_edge_set(val_df)
        | positive_edge_set(test_df)
    )
    evaluation_background_audit = audit_eval_benchmarks(
        val_eval_df,
        test_eval_df,
        all_positive_edges,
        require_disjoint_negatives=not args.allow_eval_negative_overlap,
    )
    validation_candidate_edges = {
        (int(tf), int(target))
        for tf, target in val_eval_df[["TF", "Target"]].to_numpy()
    }
    test_candidate_edges = {
        (int(tf), int(target))
        for tf, target in test_eval_df[["TF", "Target"]].to_numpy()
    }

    train_df, topology_df, prior_robustness_audit = apply_prior_robustness(
        train_df=full_train_df,
        val_df=val_df,
        test_df=test_df,
        mode=args.prior_robustness_mode,
        fraction=args.prior_robustness_fraction,
        seed=args.prior_robustness_seed,
        gene_pool=[int(item) for item in tg_df["index"].tolist()],
    )
    topology_source_audit = {
        "positive_scope": args.topology_positive_scope,
        "deliberately_leaky": args.topology_positive_scope == "all_splits",
        "train_positive_edges": int(len(full_train_df)),
        "validation_positive_edges": int(len(val_df)),
        "test_positive_edges": int(len(test_df)),
        "heldout_positive_edges_added": 0,
    }
    if args.topology_positive_scope == "all_splits":
        def positive_rows(frame: pd.DataFrame) -> pd.DataFrame:
            if "Label" in frame.columns:
                frame = frame.loc[frame["Label"] == 1]
            result = frame[["TF", "Target"]].copy()
            result["Label"] = 1
            return result

        heldout_edges = positive_edge_set(val_df) | positive_edge_set(test_df)
        topology_df = (
            pd.concat(
                [positive_rows(full_train_df), positive_rows(val_df), positive_rows(test_df)],
                ignore_index=True,
            )
            .drop_duplicates(["TF", "Target"])
            .reset_index(drop=True)
        )
        topology_source_audit["heldout_positive_edges_added"] = int(
            len(positive_edge_set(topology_df) & heldout_edges)
        )
        print(
            "[INVALID DIAGNOSTIC] Validation/test positives are intentionally included "
            "in topology construction; supervision and evaluation labels remain unchanged."
        )
    topology_source_audit["topology_source_edges"] = int(len(topology_df))

    bipartite_rewire_audit = None
    if args.topology_graph_variant == "bipartite_rewired":
        warnings.warn(
            "bipartite_rewired excludes held-out positives. "
            "Use train_only_rewired for a graph null constructed from training edges only.",
            RuntimeWarning,
        )
        topology_df, bipartite_rewire_audit = bipartite_degree_preserving_rewire(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=args.topology_rewire_seed,
            swaps_per_edge=args.topology_rewire_swaps_per_edge,
            max_attempt_factor=args.topology_rewire_max_attempt_factor,
        )
    elif args.topology_graph_variant == "train_only_rewired":
        topology_df, bipartite_rewire_audit = load_or_create_training_null(
            train_df, args.topology_rewire_seed, args.topology_rewire_swaps_per_edge,
            args.topology_rewire_max_attempt_factor, args.topology_null_cache,
        )
        # Diagnostic only: held-out labels do not accept, reject, or regenerate the graph.
        bipartite_rewire_audit["posthoc_heldout_positive_overlap"] = len(
            positive_edge_set(topology_df) & (positive_edge_set(val_df) | positive_edge_set(test_df))
        )
    print(
        "[Prior Robustness] "
        f"mode={prior_robustness_audit['mode']} "
        f"requested_fraction={prior_robustness_audit['requested_fraction']:.3f} "
        f"supervision={prior_robustness_audit['supervision_edges']}/"
        f"{prior_robustness_audit['original_train_edges']} "
        f"topology={prior_robustness_audit['topology_edges']}/"
        f"{prior_robustness_audit['original_train_edges']} "
        f"topology_added={prior_robustness_audit['topology_added_edges']}"
    )

    feature_array, feature_audit = prepare_feature_matrix(
        x,
        feature_mode=args.feature_mode,
        pca_components=args.pca_components,
        pca_seed=args.pca_seed,
    )
    feature_audit["input_modalities"] = (
        ["RNA", "GAS"]
        if args.data in {"pbmc_granulocytes", "lymph"}
        and not args.benchmark and args.feature_mode != "rna_only"
        else ["RNA"]
    )
    feature_audit["input_policy"] = "pbmc_lymphoma_rna_gas_v2"
    print(
        "[Features] "
        f"mode={feature_audit['mode']} "
        f"input_shape={feature_audit['input_shape']} "
        f"output_shape={feature_audit['output_shape']}"
    )
    feature = torch.from_numpy(feature_array).to(torch.float32).to(device)
    tf_idx = torch.tensor(tf_df["index"].to_numpy(np.int64), dtype=torch.long)
    train_edges = torch.tensor(train_df[["TF", "Target"]].to_numpy(np.int64), dtype=torch.long).t().contiguous()
    topology_edges = torch.tensor(
        topology_df[["TF", "Target"]].to_numpy(np.int64),
        dtype=torch.long,
    ).t().contiguous()

    train_adj_edges = add_bidirectional_edges(topology_edges, tf_idx)
    adj = SparseTensor(
        row=train_adj_edges[0],
        col=train_adj_edges[1],
        sparse_sizes=(feature.shape[0], feature.shape[0]),
    ).to(device)

    model = TRACE(
        in_dim=feature.shape[1],
        h1_dim=args.hidden_dim[0],
        h2_dim=args.hidden_dim[1],
        out_dim=args.output_dim,
        num_head1=args.num_head[0],
        num_head2=args.num_head[1],
        neg_slope=args.neg_slope,
        dropout=args.dropout,
        device=device,
        num_nodes=feature.shape[0],
        attention_mode=args.attention_mode,
        decoder=args.decoder,
        encoder_type=args.encoder_type,
        cross_attn_lse_tau=args.cross_attn_lse_tau,
    ).to(device)

    topology_ablation_audit = {
        "attention_mode": args.attention_mode,
        "graph_variant": args.topology_graph_variant,
        "jaccard_bias_enabled": (
            args.attention_mode in {"dual", "topology_only"}
            and args.topology_graph_variant != "direct"
            and not args.disable_jaccard_bias
        ),
        "edge_count": 0,
        "topology_source": topology_source_audit,
    }
    if bipartite_rewire_audit is not None:
        topology_ablation_audit.update(bipartite_rewire_audit)
    if args.attention_mode in {"dual", "topology_only"}:
        topology_layers = list(model.Conv1) + list(model.Conv2)
        if args.topology_graph_variant == "direct":
            direct_row, direct_col, _ = adj.coo()
            topology_layers[0].topo_row = direct_row
            topology_layers[0].topo_col = direct_col
            topology_layers[0].topo_w = torch.zeros(
                direct_row.numel(), dtype=feature.dtype, device=device
            )
            topology_layers[0].topo_ready = True
            topology_ablation_audit["jaccard_bias_enabled"] = False
        else:
            topology_layers[0].precompute_topo_jaccard_edges(
                adj=adj,
                topk=args.topk,
                thr=args.topo_thr,
                undirected=True,
                exclude_1hop=True,
                device=device,
            )
            if args.topology_graph_variant == "rewired":
                (
                    topology_layers[0].topo_row,
                    topology_layers[0].topo_col,
                    topology_layers[0].topo_w,
                    rewire_audit,
                ) = rewire_topology_source_preserving(
                    row=topology_layers[0].topo_row,
                    col=topology_layers[0].topo_col,
                    weight=topology_layers[0].topo_w,
                    adj=adj,
                    num_nodes=feature.shape[0],
                    seed=args.topology_rewire_seed,
                )
                topology_ablation_audit.update(rewire_audit)

        for layer in topology_layers[1:]:
            layer.topo_row = topology_layers[0].topo_row
            layer.topo_col = topology_layers[0].topo_col
            layer.topo_w = topology_layers[0].topo_w
            layer.topo_ready = True
        if args.disable_jaccard_bias or args.topology_graph_variant == "direct":
            for layer in topology_layers:
                layer.jaccard_beta.data.zero_()
                layer.jaccard_beta.requires_grad_(False)
        topology_ablation_audit["edge_count"] = int(topology_layers[0].topo_row.numel())

    print(
        "[Matched Ablation] "
        f"attention={topology_ablation_audit['attention_mode']} "
        f"graph={topology_ablation_audit['graph_variant']} "
        f"edges={topology_ablation_audit['edge_count']} "
        f"jaccard_bias={topology_ablation_audit['jaccard_bias_enabled']}"
    )

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    tf_set = set(int(item) for item in tf_df["index"].tolist())
    global_target_pool = [int(item) for item in tg_df["index"].tolist()]
    forbidden_train_set = (
        positive_edge_set(full_train_df)
        | positive_edge_set(val_df)
        | positive_edge_set(test_df)
    )
    if args.exclude_eval_candidates_from_train_negatives:
        forbidden_train_set |= validation_candidate_edges | test_candidate_edges
    sampled_training_negatives: set[tuple[int, int]] = set()
    sampled_training_negative_draws = 0
    train_loader = DataLoader(EdgeDataset(train_edges), batch_size=args.batch_size, shuffle=True)

    val_edges = torch.tensor(val_eval_df[["TF", "Target"]].to_numpy(np.int64), dtype=torch.long, device=device).t().contiguous()
    val_labels = torch.tensor(val_eval_df["Label"].to_numpy(np.float32), dtype=torch.float32, device=device)
    test_edges = torch.tensor(test_eval_df[["TF", "Target"]].to_numpy(np.int64), dtype=torch.long, device=device).t().contiguous()
    test_labels = torch.tensor(test_eval_df["Label"].to_numpy(np.float32), dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_score = -float("inf")
    best_threshold = 0.5
    best_val_result = None

    total_batches = len(train_loader)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch_edges in enumerate(train_loader, start=1):
            pos_edges = batch_edges.long().t().contiguous().to(device)
            batch_nodes = pos_edges.unique()
            combined_edges, combined_labels = generate_negative_edges_train(
                pos_edges=pos_edges,
                batch_nodes=batch_nodes,
                forbidden_train_set=forbidden_train_set,
                tf_set=tf_set,
                neg_ratio=args.neg_ratio,
                seed=args.seed + epoch + num_batches,
                mode=args.train_neg_sampling,
                global_target_pool=global_target_pool,
            )
            if args.exclude_eval_candidates_from_train_negatives:
                sampled_negative_array = (
                    combined_edges[:, combined_labels == 0]
                    .detach()
                    .cpu()
                    .t()
                    .numpy()
                )
                sampled_training_negative_draws += int(len(sampled_negative_array))
                sampled_training_negatives.update(
                    (int(tf), int(target)) for tf, target in sampled_negative_array
                )

            shuffle_indices = torch.randperm(combined_edges.shape[1], device=device)
            combined_edges = combined_edges[:, shuffle_indices]
            combined_labels = combined_labels[shuffle_indices]

            optimizer.zero_grad()
            logits = model(feature, adj, combined_edges)
            loss = classification_loss_with_logits(
                logits,
                combined_labels,
                loss_type=args.loss_type,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
            )
            if args.ranking_loss_weight > 0:
                rank_loss = pairwise_ranking_loss(logits, combined_labels, args.ranking_loss_margin)
                loss = loss + args.ranking_loss_weight * rank_loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1
            avg_running_loss = running_loss / num_batches
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_batches:
                sys.stdout.write(
                    f"\r  [Train] batch {batch_idx}/{total_batches} loss={avg_running_loss:.4f}"
                )
                sys.stdout.flush()

        sys.stdout.write("\n")
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(feature, adj, val_edges)
            epoch_threshold, best_f1, best_auc, best_aupr, best_mcc = optimize_probability_threshold(
                val_labels,
                val_logits,
            )
            val_result = evaluate_predictions(val_edges, val_labels, val_logits, tf_idx, threshold=epoch_threshold)
            selection_score = float(val_result[args.selection_metric])
            early_stopping(selection_score, model)

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_threshold = epoch_threshold
            best_state = copy.deepcopy(model.state_dict())
            best_val_result = copy.deepcopy(val_result)

        avg_loss = running_loss / max(1, num_batches)
        print(f"  [Train Summary] loss={avg_loss:.4f}")
        print(
            f"  [Val Summary] AUC={val_result['AUROC']:.3f}, AUPRC={val_result['AUPRC']:.3f}, "
            f"F1={val_result['F1']:.3f}, MCC={val_result['MCC']:.3f}, "
            f"PerTF_AUPRC={val_result['PerTF_AUPRC']:.3f}, "
            f"TF_P@10={val_result['TF_P@10']:.3f}, TF_P@30={val_result['TF_P@30']:.3f}, TF_P@50={val_result['TF_P@50']:.3f}, TF_P@100={val_result['TF_P@100']:.3f}"
        )
        print(
            f"  [Threshold Search] best_threshold={epoch_threshold:.3f}, "
            f"AUC={best_auc:.3f}, AUPRC={best_aupr:.3f}, MCC={best_mcc:.3f}, F1={best_f1:.3f}"
        )
        print(
            f"  [Best So Far] selection_metric={args.selection_metric}, "
            f"best_val_score={best_val_score:.3f}, applied_test_threshold={best_threshold:.3f}"
        )

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(best_state)
    model.eval()

    train_negative_pair_audit = {
        "exclude_eval_candidates_from_train_negatives": bool(
            args.exclude_eval_candidates_from_train_negatives
        ),
        "validation_candidate_count": int(len(validation_candidate_edges)),
        "test_candidate_count": int(len(test_candidate_edges)),
        "validation_test_candidate_overlap": int(
            len(validation_candidate_edges & test_candidate_edges)
        ),
        "evaluation_candidate_union_count": int(
            len(validation_candidate_edges | test_candidate_edges)
        ),
        "sampled_training_negative_draws": int(sampled_training_negative_draws),
        "sampled_training_negative_unique_pairs": int(len(sampled_training_negatives)),
        "training_negative_validation_candidate_overlap": int(
            len(sampled_training_negatives & validation_candidate_edges)
        ),
        "training_negative_test_candidate_overlap": int(
            len(sampled_training_negatives & test_candidate_edges)
        ),
    }
    train_negative_pair_audit["strict_pair_identity_passed"] = bool(
        args.exclude_eval_candidates_from_train_negatives
        and train_negative_pair_audit[
            "training_negative_validation_candidate_overlap"
        ]
        == 0
        and train_negative_pair_audit["training_negative_test_candidate_overlap"]
        == 0
    )
    if (
        args.exclude_eval_candidates_from_train_negatives
        and not train_negative_pair_audit["strict_pair_identity_passed"]
    ):
        raise RuntimeError(
            "Strict train/evaluation pair-identity audit failed: "
            f"{train_negative_pair_audit}"
        )

    if best_val_result is None:
        raise RuntimeError("Training completed without a validation result.")

    base_result_dir = Path(args.result_dir)
    short_name = SHORT_NAMES.get(args.data, args.data.replace("/", "_"))
    result_dir = base_result_dir / short_name / f"seed_{args.seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = result_dir / f"TRACE_{short_name}_seed_{args.seed}_metrics.json"

    if args.validation_only:
        metrics_payload = {
            "config": vars(args),
            "best_threshold": best_threshold,
            "best_validation_metrics": best_val_result,
            "validation_eval_file": str(val_eval_file),
            "test_eval_file": str(test_eval_file),
            "test_metrics": None,
            "prior_robustness": prior_robustness_audit,
            "topology_ablation": topology_ablation_audit,
            "feature_audit": feature_audit,
            "evaluation_background": evaluation_background_audit,
            "train_negative_pair_audit": train_negative_pair_audit,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        print(
            f"Validation-only: best {args.selection_metric}="
            f"{best_val_result[args.selection_metric]:.6f}"
        )
        print(f"Saved metrics: {metrics_path}")
        return

    with torch.no_grad():
        test_logits = model(feature, adj, test_edges)
        test_result = evaluate_predictions(test_edges, test_labels, test_logits, tf_idx, threshold=best_threshold)

    print(
        f"Test: AUC={test_result['AUROC']:.3f}, AUPRC={test_result['AUPRC']:.3f}, "
        f"F1={test_result['F1']:.3f}, MCC={test_result['MCC']:.3f}, "
        f"PerTF_AUPRC={test_result['PerTF_AUPRC']:.3f}, "
        f"TF_P@10={test_result['TF_P@10']:.3f}, TF_P@30={test_result['TF_P@30']:.3f}, TF_P@50={test_result['TF_P@50']:.3f}, TF_P@100={test_result['TF_P@100']:.3f}"
    )

    model_path = result_dir / (
        f"TRACE_{short_name}_seed_{args.seed}_AUC_{test_result['AUROC']:.2f}_"
        f"AUPRC_{test_result['AUPRC']:.2f}_PerTF_{test_result['PerTF_AUPRC']:.2f}.pt"
    )
    metrics_path = result_dir / f"TRACE_{short_name}_seed_{args.seed}_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "best_threshold": best_threshold,
            "best_validation_metrics": best_val_result,
            "test_metrics": test_result,
            "prior_robustness": prior_robustness_audit,
            "topology_ablation": topology_ablation_audit,
            "feature_audit": feature_audit,
            "evaluation_background": evaluation_background_audit,
            "train_negative_pair_audit": train_negative_pair_audit,
        },
        model_path,
    )

    metrics_payload = {
        "config": vars(args),
        "best_threshold": best_threshold,
        "best_validation_metrics": best_val_result,
        "validation_eval_file": str(val_eval_file),
        "test_eval_file": str(test_eval_file),
        "test_metrics": test_result,
        "prior_robustness": prior_robustness_audit,
        "topology_ablation": topology_ablation_audit,
        "feature_audit": feature_audit,
        "evaluation_background": evaluation_background_audit,
        "train_negative_pair_audit": train_negative_pair_audit,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
