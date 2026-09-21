import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from google.protobuf import message_factory as protobuf_message_factory
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if not hasattr(protobuf_message_factory.MessageFactory, "GetPrototype"):
    protobuf_message_factory.MessageFactory.GetPrototype = staticmethod(protobuf_message_factory.GetMessageClass)

try:
    import muon as mu
except ModuleNotFoundError:  # pragma: no cover
    mu = None


DATASET_FILES = {
    "pbmc_granulocytes": "pbmc10k_integrated.h5mu",
    "GSE150774_RAW/UCB": "UCB_preprocessed.h5ad",
    "GSE150774_RAW/BM": "BM_preprocessed.h5ad",
    "lymph": "lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5mu",
}

SHORT_NAMES = {
    "pbmc_granulocytes": "pbmc",
    "GSE150774_RAW/UCB": "UCB",
    "GSE150774_RAW/BM": "BM",
    "lymph": "lymphoma",
    "hESC": "hESC",
    "hHep": "hHep",
}

BENCHMARK_DATASETS = {"hESC", "hHep"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "1"}:
        return True
    if lowered in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_int_list(value: str, expected_len: int, name: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"{name} must contain exactly {expected_len} comma-separated integers."
        )
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain integers only.") from exc


def infer_benchmark_mode(data_type: str, benchmark_flag: bool | None) -> bool:
    if benchmark_flag is None:
        return data_type in BENCHMARK_DATASETS
    return benchmark_flag


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sc.settings.seed = seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_sc_data(data_root: str, data: str, filename: str):
    full_path = os.path.join(data_root, "scRNA_ATAC", data, filename)
    if filename.endswith("h5mu"):
        if mu is None:
            raise ModuleNotFoundError("muon is required to load .h5mu inputs.")
        scdata = mu.read(full_path)
        return scdata.mod["rna"]
    if filename.endswith(".csv"):
        return sc.read_csv(full_path)
    if filename.endswith(".h5ad"):
        return sc.read_h5ad(full_path)
    raise ValueError(f"Unsupported file format: {filename}")


def load_gas(data_root: str, data: str, filename: str = "GAS_gene_cell_score.h5", save_fixed: bool = True, key: str = "data") -> pd.DataFrame:
    path = os.path.join(data_root, "scRNA_ATAC", data, filename)
    try:
        gas = pd.read_hdf(path)
        gas.columns = gas.columns.str.replace(r"\\.", "-", regex=True)
        return gas
    except (ValueError, OSError, KeyError, ImportError):
        pass

    with h5py.File(path, "r") as handle:
        if "matrix" in handle:
            matrix = handle["matrix"][:]
        elif "data" in handle:
            matrix = handle["data"][:]
        else:
            raise ValueError("No dataset named 'matrix' or 'data' found in HDF5 file.")
        genes = handle["gene_names"][:]
        cells = handle["cell_names"][:]

    genes = [item.decode() if isinstance(item, bytes) else item for item in genes]
    cells = [item.decode() if isinstance(item, bytes) else item for item in cells]
    array = np.array(matrix)
    if len(genes) == array.shape[0] and len(cells) == array.shape[1]:
        gas_df = pd.DataFrame(array, index=genes, columns=cells)
    else:
        gas_df = pd.DataFrame(array, index=cells, columns=genes).T
    gas_df.columns = gas_df.columns.str.replace(r"\\.", "-", regex=True)

    if save_fixed:
        try:
            gas_df.to_hdf(path, key=key, mode="w", format="fixed")
        except ImportError:
            pass
    return gas_df


def integrate_rna_gas(x: pd.DataFrame, gas: pd.DataFrame) -> pd.DataFrame:
    gas_aligned = gas.reindex(index=x.index, columns=x.columns, fill_value=0)
    return x + gas_aligned


def get_filtered_x(rna) -> pd.DataFrame:
    keep_idx = rna.var["highly_variable"].to_numpy()
    filtered_idx = rna.var.index[keep_idx]
    matrix = rna.X[:, keep_idx]
    if isinstance(matrix, sp.csr_matrix):
        matrix = matrix.todense()
    return pd.DataFrame(matrix, index=rna.obs.index, columns=filtered_idx).T


def get_exp_data(data_root: str, data: str, filename: str) -> pd.DataFrame:
    rna = get_sc_data(data_root, data, filename)
    x = get_filtered_x(rna)
    if filename.endswith("h5mu"):
        gas = load_gas(data_root, data)
        x = integrate_rna_gas(x, gas)
    return x


def get_exp_data_rna_only(data_root: str, data: str, filename: str) -> pd.DataFrame:
    rna = get_sc_data(data_root, data, filename)
    return get_filtered_x(rna)


def load_expression(data_root: str, data_type: str, benchmark: bool) -> pd.DataFrame:
    if benchmark:
        benchmark_root = f"BEELINE/inputs/scRNA-Seq/{data_type}"
        rna = get_sc_data(data_root, benchmark_root, "ExpressionData.csv").transpose()
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.01, max_mean=4, min_disp=0.5)
        if int(np.sum(rna.var["highly_variable"])) < 3000:
            sc.pp.highly_variable_genes(rna, flavor="seurat", n_top_genes=3000)
        return get_filtered_x(rna)
    return get_exp_data(data_root, data_type, DATASET_FILES[data_type])


def load_expression_rna_only(data_root: str, data_type: str, benchmark: bool) -> pd.DataFrame:
    if benchmark:
        benchmark_root = f"BEELINE/inputs/scRNA-Seq/{data_type}"
        rna = get_sc_data(data_root, benchmark_root, "ExpressionData.csv").transpose()
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, min_mean=0.01, max_mean=4, min_disp=0.5)
        if int(np.sum(rna.var["highly_variable"])) < 3000:
            sc.pp.highly_variable_genes(rna, flavor="seurat", n_top_genes=3000)
        return get_filtered_x(rna)
    return get_exp_data_rna_only(data_root, data_type, DATASET_FILES[data_type])


def standardize(x: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(x.values.T).T


def prepare_feature_matrix(
    x: pd.DataFrame,
    feature_mode: str = "default",
    pca_components: int = 128,
    pca_seed: int = 20260819,
) -> tuple[np.ndarray, dict[str, object]]:
    """Standardize gene-by-cell features and optionally fix their dimension."""
    feature = standardize(x).astype(np.float32, copy=False)
    audit: dict[str, object] = {
        "mode": feature_mode,
        "input_shape": [int(feature.shape[0]), int(feature.shape[1])],
        "output_shape": [int(feature.shape[0]), int(feature.shape[1])],
        "pca_components_requested": None,
        "pca_components_used": None,
        "pca_seed": None,
        "pca_explained_variance_ratio_sum": None,
    }
    if feature_mode != "pca":
        return feature, audit

    components = min(int(pca_components), int(min(feature.shape)))
    if components < 1:
        raise ValueError("PCA feature mode requires a non-empty expression matrix.")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=int(pca_seed),
    )
    feature = pca.fit_transform(feature).astype(np.float32, copy=False)
    audit.update(
        output_shape=[int(feature.shape[0]), int(feature.shape[1])],
        pca_components_requested=int(pca_components),
        pca_components_used=int(components),
        pca_seed=int(pca_seed),
        pca_explained_variance_ratio_sum=float(
            pca.explained_variance_ratio_.sum()
        ),
    )
    return feature, audit


def filter_tflist(data_root: str, data_type: str, x: pd.DataFrame, prior_net: pd.DataFrame, tg_file: str = "sctglist.csv", filename: str = "tflist.csv") -> None:
    save_root = os.path.join(data_root, "tfs", data_type)
    os.makedirs(save_root, exist_ok=True)
    filtered_prior = prior_net[prior_net["TF"].isin(x.index)]
    tf = filtered_prior["TF"].drop_duplicates()
    tf_df = pd.DataFrame({"TF": tf, "index": range(len(tf))})
    tg = pd.read_csv(os.path.join(save_root, tg_file))
    tg_lookup = tg.set_index("Gene")["index"].to_dict()
    tf_df["index"] = tf_df["TF"].map(tg_lookup)
    tf_df.to_csv(os.path.join(save_root, filename), index=False)


def filter_tglist(data_root: str, data_type: str, x: pd.DataFrame, prior_net: pd.DataFrame, filename: str = "sctglist.csv") -> None:
    save_root = os.path.join(data_root, "tfs", data_type)
    os.makedirs(save_root, exist_ok=True)
    scgenes = pd.Series(x.index)
    tg = pd.concat([prior_net["TF"], prior_net["Target"], scgenes]).drop_duplicates()
    pd.DataFrame({"Gene": tg, "index": range(len(tg))}).to_csv(os.path.join(save_root, "tglist.csv"), index=False)
    pd.DataFrame({"Gene": scgenes, "index": range(len(scgenes))}).to_csv(os.path.join(save_root, filename), index=False)


def get_prior_network(data_root: str, filename: str = "tflink.h5") -> pd.DataFrame:
    path = os.path.join(data_root, "prior_network", filename)
    if filename.endswith(".h5"):
        prior_net = pd.read_hdf(path)
    else:
        prior_net = pd.read_csv(path)
    prior_net.columns = ["TF", "Target"]
    prior_net["Label"] = 1
    prior_net.reset_index(drop=True, inplace=True)
    return prior_net


def convert_prior_to_idx_net(data_root: str, data_type: str, tf: pd.DataFrame, tg: pd.DataFrame, prior_net: pd.DataFrame, filename: str = "prior_net_idx.csv") -> None:
    save_root = os.path.join(data_root, "prior_network", data_type)
    os.makedirs(save_root, exist_ok=True)
    tf_lookup = tf.set_index("TF")["index"].to_dict()
    tg_lookup = tg.set_index("Gene")["index"].to_dict()
    prior_copy = prior_net.copy()
    prior_copy["tf_index"] = prior_copy["TF"].map(tf_lookup)
    prior_copy["tg_index"] = prior_copy["Target"].map(tg_lookup)
    prior_net_idx = prior_copy[["tf_index", "tg_index"]].dropna().astype(np.int64)
    prior_net_idx.columns = ["TF", "Target"]
    prior_net_idx["Label"] = 1
    prior_net_idx.to_csv(os.path.join(save_root, filename), index=False)


def get_prior_net_idx(data_root: str, data_type: str, filename: str = "prior_net_idx.csv") -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_root, "prior_network", data_type, filename))


def split_train_val_test(data_root: str, data_type: str, prior_net_idx: pd.DataFrame, seed: int) -> None:
    save_root = os.path.join(data_root, "prior_network", data_type)
    os.makedirs(save_root, exist_ok=True)
    train_data, remaining_data = train_test_split(prior_net_idx, test_size=0.2, random_state=seed, shuffle=True)
    validation_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=seed, shuffle=True)
    train_data.to_csv(os.path.join(save_root, "train_data.csv"), index=False)
    validation_data.to_csv(os.path.join(save_root, "validation_data.csv"), index=False)
    test_data.to_csv(os.path.join(save_root, "test_data.csv"), index=False)


def load_or_prepare_dataset_artifacts(data_root: str, data_type: str, x: pd.DataFrame, prior_net: pd.DataFrame, seed: int) -> dict[str, Path]:
    tf_path = Path(data_root) / "tfs" / data_type / "tflist.csv"
    tg_path = Path(data_root) / "tfs" / data_type / "sctglist.csv"
    prior_idx_path = Path(data_root) / "prior_network" / data_type / "prior_net_idx.csv"
    train_path = Path(data_root) / "prior_network" / data_type / "train_data.csv"
    val_path = Path(data_root) / "prior_network" / data_type / "validation_data.csv"
    test_path = Path(data_root) / "prior_network" / data_type / "test_data.csv"

    if not (tf_path.exists() and tg_path.exists()):
        filter_tglist(data_root, data_type, x, prior_net)
        filter_tflist(data_root, data_type, x, prior_net)
    if not prior_idx_path.exists():
        convert_prior_to_idx_net(data_root, data_type, pd.read_csv(tf_path), pd.read_csv(tg_path), prior_net)
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        split_train_val_test(data_root, data_type, get_prior_net_idx(data_root, data_type), seed)

    return {
        "tf": tf_path,
        "tg": tg_path,
        "prior_idx": prior_idx_path,
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }


def positive_edge_set(df: pd.DataFrame) -> set[tuple[int, int]]:
    return {(int(tf), int(tg)) for tf, tg in zip(df["TF"], df["Target"])}


def labeled_edge_set(df: pd.DataFrame, label: int) -> set[tuple[int, int]]:
    selected = df.loc[df["Label"].astype(int) == label, ["TF", "Target"]]
    return {(int(tf), int(tg)) for tf, tg in selected.itertuples(index=False, name=None)}



def apply_prior_robustness(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mode: str,
    fraction: float,
    seed: int,
    gene_pool: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build supervision and topology edge sets for prior robustness experiments."""
    supported_modes = {
        "none",
        "topology_edge_dropout",
        "topology_source_rewire",
        "train_positive_subsample",
    }
    if mode not in supported_modes:
        raise ValueError(f"Unsupported prior robustness mode: {mode}")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("prior robustness fraction must satisfy 0 <= fraction < 1")
    if mode == "none" and fraction != 0.0:
        raise ValueError("prior robustness fraction must be 0 when mode=none")
    if mode != "none" and fraction <= 0.0:
        raise ValueError(f"prior robustness fraction must be > 0 when mode={mode}")

    edge_columns = ["TF", "Target", "Label"]
    full_train = train_df[edge_columns].drop_duplicates(["TF", "Target"]).copy()
    full_train[["TF", "Target"]] = full_train[["TF", "Target"]].astype(np.int64)
    full_train["Label"] = 1
    supervision = full_train.copy()
    topology = full_train.copy()
    rng = np.random.default_rng(seed)

    if mode == "topology_edge_dropout":
        remove_count = min(
            len(full_train) - 1,
            max(1, int(np.ceil(len(full_train) * fraction))),
        )
        remove_idx = rng.choice(len(full_train), size=remove_count, replace=False)
        keep_mask = np.ones(len(full_train), dtype=bool)
        keep_mask[remove_idx] = False
        topology = full_train.iloc[keep_mask].reset_index(drop=True)

    elif mode == "topology_source_rewire":
        known_positive = (
            positive_edge_set(full_train)
            | positive_edge_set(val_df)
            | positive_edge_set(test_df)
        )
        original_edges = list(
            zip(full_train["TF"].astype(int), full_train["Target"].astype(int))
        )
        edges = original_edges.copy()
        target_change_count = min(
            len(edges) - 1,
            max(1, int(np.ceil(len(edges) * fraction))),
        )

        if gene_pool is None:
            gene_pool = sorted(
                set(full_train["Target"].astype(int))
                | set(val_df["Target"].astype(int))
                | set(test_df["Target"].astype(int))
            )
        candidate_targets = np.asarray(sorted({int(item) for item in gene_pool}))
        if candidate_targets.size == 0:
            raise ValueError("gene_pool must contain at least one target candidate")
        candidate_target_set = set(int(item) for item in candidate_targets)
        target_degree = full_train.groupby("Target").size().to_dict()
        target_weight_lookup = {
            int(target): float(np.sqrt(target_degree.get(int(target), 0) + 1))
            for target in candidate_targets
        }

        source_to_indices: dict[int, list[int]] = {}
        for edge_idx, (source, _) in enumerate(original_edges):
            source_to_indices.setdefault(source, []).append(edge_idx)
        known_by_source: dict[int, set[int]] = {}
        for source, target in known_positive:
            known_by_source.setdefault(source, set()).add(target)

        capacities = {}
        allocations = {}
        for source, indices in source_to_indices.items():
            unavailable = known_by_source.get(source, set()) | {source}
            unknown_count = len(candidate_target_set - unavailable)
            capacity = min(len(indices), unknown_count)
            capacities[source] = capacity
            allocations[source] = min(
                capacity,
                int(np.floor(len(indices) * fraction)),
            )

        remaining = target_change_count - sum(allocations.values())
        spares = {
            source: capacities[source] - allocations[source]
            for source in allocations
            if capacities[source] > allocations[source]
        }
        total_spare = sum(spares.values())
        if total_spare < remaining:
            raise RuntimeError(
                "Requested source-preserving rewiring exceeds the candidate-space capacity: "
                f"requested={target_change_count}, maximum={sum(allocations.values()) + total_spare}."
            )
        if remaining > 0:
            raw_additions = {
                source: remaining * spare / total_spare
                for source, spare in spares.items()
            }
            additions = {
                source: min(spares[source], int(np.floor(raw_additions[source])))
                for source in spares
            }
            leftover = remaining - sum(additions.values())
            tie_break = {source: float(rng.random()) for source in spares}
            ranked_sources = sorted(
                spares,
                key=lambda source: (
                    raw_additions[source] - additions[source],
                    tie_break[source],
                ),
                reverse=True,
            )
            for source in ranked_sources:
                if leftover <= 0:
                    break
                if additions[source] < spares[source]:
                    additions[source] += 1
                    leftover -= 1
            if leftover != 0:
                raise RuntimeError(f"Could not allocate {leftover} rewired edges.")
            for source, addition in additions.items():
                allocations[source] += addition

        changed_count = 0
        for source, edge_indices in source_to_indices.items():
            source_change_count = allocations[source]
            if source_change_count <= 0:
                continue
            unavailable = known_by_source.get(source, set()) | {source}
            available_targets = np.asarray(
                sorted(candidate_target_set - unavailable),
                dtype=np.int64,
            )
            weights = np.asarray(
                [target_weight_lookup[int(target)] for target in available_targets],
                dtype=float,
            )
            weights /= weights.sum()
            selected_edges = rng.choice(
                edge_indices,
                size=source_change_count,
                replace=False,
            )
            selected_targets = rng.choice(
                available_targets,
                size=source_change_count,
                replace=False,
                p=weights,
            )
            for edge_idx, target in zip(selected_edges, selected_targets):
                edges[int(edge_idx)] = (source, int(target))
            changed_count += source_change_count

        if changed_count != target_change_count:
            raise RuntimeError(
                f"Expected {target_change_count} rewired edges, generated {changed_count}."
            )
        topology = pd.DataFrame(edges, columns=["TF", "Target"])
        topology["Label"] = 1

    elif mode == "train_positive_subsample":
        retained_groups = []
        for _, group in full_train.groupby("TF", sort=True):
            keep_count = max(1, int(np.ceil(len(group) * (1.0 - fraction))))
            keep_positions = rng.choice(len(group), size=keep_count, replace=False)
            retained_groups.append(group.iloc[np.sort(keep_positions)])
        supervision = (
            pd.concat(retained_groups, ignore_index=True)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        topology = supervision.copy()

    original_edges = positive_edge_set(full_train)
    supervision_edges = positive_edge_set(supervision)
    topology_edges = positive_edge_set(topology)
    original_source_degree = full_train.groupby("TF").size().sort_index()
    original_target_degree = full_train.groupby("Target").size().sort_index()
    topology_source_degree = topology.groupby("TF").size().sort_index()
    topology_target_degree = topology.groupby("Target").size().sort_index()
    degree_index = original_target_degree.index.union(topology_target_degree.index)
    original_target_aligned = original_target_degree.reindex(degree_index, fill_value=0)
    topology_target_aligned = topology_target_degree.reindex(degree_index, fill_value=0)
    if original_target_aligned.nunique() < 2 or topology_target_aligned.nunique() < 2:
        target_degree_spearman = float(original_target_aligned.equals(topology_target_aligned))
    else:
        target_degree_spearman = float(
            original_target_aligned.corr(topology_target_aligned, method="spearman")
        )

    audit = {
        "mode": mode,
        "requested_fraction": float(fraction),
        "seed": int(seed),
        "original_train_edges": int(len(original_edges)),
        "supervision_edges": int(len(supervision_edges)),
        "topology_edges": int(len(topology_edges)),
        "supervision_removed_edges": int(len(original_edges - supervision_edges)),
        "topology_removed_edges": int(len(original_edges - topology_edges)),
        "topology_added_edges": int(len(topology_edges - original_edges)),
        "realized_supervision_removal_fraction": float(
            len(original_edges - supervision_edges) / max(1, len(original_edges))
        ),
        "realized_topology_change_fraction": float(
            max(
                len(original_edges - topology_edges),
                len(topology_edges - original_edges),
            )
            / max(1, len(original_edges))
        ),
        "original_tf_sources": int(full_train["TF"].nunique()),
        "supervision_tf_sources": int(supervision["TF"].nunique()),
        "topology_tf_sources": int(topology["TF"].nunique()),
        "topology_source_degree_preserved": bool(
            original_source_degree.equals(topology_source_degree)
        ),
        "topology_target_degree_preserved": bool(
            original_target_degree.equals(topology_target_degree)
        ),
        "topology_target_degree_spearman": target_degree_spearman,
    }
    return supervision.reset_index(drop=True), topology.reset_index(drop=True), audit


def generate_negative_edges_train(
    pos_edges: torch.Tensor,
    batch_nodes: torch.Tensor,
    forbidden_train_set: set[tuple[int, int]],
    tf_set: set[int],
    neg_ratio: int,
    seed: int,
    mode: str = "batch_local",
    global_target_pool: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos_edges.device
    if mode not in {"batch_local", "global", "mixed"}:
        raise ValueError(f"Unsupported train negative sampling mode: {mode}")

    src_pool = sorted({int(item) for item in pos_edges[0].tolist() if int(item) in tf_set})
    local_target_pool = sorted({int(item) for item in batch_nodes.tolist()})
    if global_target_pool is None:
        global_target_pool = local_target_pool
    global_target_pool = sorted({int(item) for item in global_target_pool})
    if not src_pool:
        raise ValueError("No TF source nodes found in the current batch.")
    if not local_target_pool:
        raise ValueError("No target candidates found in the current batch.")
    if mode in {"global", "mixed"} and not global_target_pool:
        raise ValueError("No global target candidates were provided.")

    rng = random.Random(seed)
    num_negs = int(neg_ratio * pos_edges.shape[1])
    neg_set: set[tuple[int, int]] = set()

    def is_valid(tf_idx: int, tg_idx: int) -> bool:
        return tf_idx != tg_idx and (tf_idx, tg_idx) not in forbidden_train_set and (tf_idx, tg_idx) not in neg_set

    def sample_from_pool(target_pool: list[int], needed: int, allow_short: bool = False) -> int:
        if needed <= 0:
            return 0
        start_size = len(neg_set)
        max_attempts = max(1000, needed * 100)
        attempts = 0
        while len(neg_set) - start_size < needed and attempts < max_attempts:
            attempts += 1
            tf_idx = rng.choice(src_pool)
            tg_idx = rng.choice(target_pool)
            if is_valid(tf_idx, tg_idx):
                neg_set.add((tf_idx, tg_idx))

        if len(neg_set) - start_size < needed:
            remaining = [
                (tf_idx, tg_idx)
                for tf_idx in src_pool
                for tg_idx in target_pool
                if is_valid(tf_idx, tg_idx)
            ]
            take = min(needed - (len(neg_set) - start_size), len(remaining))
            if take > 0:
                neg_set.update(rng.sample(remaining, take))

        sampled = len(neg_set) - start_size
        if sampled < needed and not allow_short:
            raise ValueError(
                "Unable to sample enough training negatives: "
                f"mode={mode}, requested={needed}, sampled={sampled}, "
                f"src_count={len(src_pool)}, target_count={len(target_pool)}"
            )
        return sampled

    if mode == "batch_local":
        sample_from_pool(local_target_pool, num_negs)
    elif mode == "global":
        sample_from_pool(global_target_pool, num_negs)
    else:
        local_target_count = num_negs // 2
        sample_from_pool(local_target_pool, local_target_count, allow_short=True)
        sample_from_pool(global_target_pool, num_negs - len(neg_set))

    neg_edges = torch.tensor(sorted(neg_set), dtype=torch.long, device=device).t().contiguous()
    pos_labels = torch.ones(pos_edges.shape[1], dtype=torch.float32, device=device)
    neg_labels = torch.zeros(neg_edges.shape[1], dtype=torch.float32, device=device)
    return torch.cat([pos_edges, neg_edges], dim=1), torch.cat([pos_labels, neg_labels], dim=0)


def sample_eval_negatives(tf_list: list[int], gene_list: list[int], forbidden_pos_set: set[tuple[int, int]], num_negatives: int, seed: int) -> pd.DataFrame:
    tf_set = set(tf_list)
    gene_set = set(gene_list)
    self_loop_count = len(tf_set & gene_set)
    forbidden_in_space = sum(
        1
        for tf_idx, gene_idx in forbidden_pos_set
        if tf_idx in tf_set and gene_idx in gene_set and tf_idx != gene_idx
    )
    candidate_capacity = len(tf_set) * len(gene_set) - self_loop_count - forbidden_in_space
    if num_negatives > candidate_capacity:
        raise ValueError(
            "Requested more unique evaluation negatives than available: "
            f"requested={num_negatives}, available={candidate_capacity}, "
            f"tf_count={len(tf_set)}, gene_count={len(gene_set)}. "
            "Use a smaller --eval_neg_ratio for this dataset."
        )

    rng = random.Random(seed)
    negatives: set[tuple[int, int]] = set()
    max_attempts = max(10000, num_negatives * 100)
    attempts = 0
    while len(negatives) < num_negatives:
        attempts += 1
        if attempts > max_attempts:
            remaining = [
                (tf_idx, gene_idx)
                for tf_idx in tf_list
                for gene_idx in gene_list
                if tf_idx != gene_idx
                and (tf_idx, gene_idx) not in forbidden_pos_set
                and (tf_idx, gene_idx) not in negatives
            ]
            needed = num_negatives - len(negatives)
            negatives.update(rng.sample(remaining, needed))
            break
        tf_idx = rng.choice(tf_list)
        gene_idx = rng.choice(gene_list)
        if tf_idx == gene_idx:
            continue
        if (tf_idx, gene_idx) in forbidden_pos_set:
            continue
        negatives.add((tf_idx, gene_idx))
    neg_df = pd.DataFrame(sorted(negatives), columns=["TF", "Target"])
    neg_df["Label"] = 0
    return neg_df


def save_full_edges_eval(out_file: Path, positive_df: pd.DataFrame, forbidden_pos_set: set[tuple[int, int]], tf_list: list[int], gene_list: list[int], neg_ratio: int, seed: int) -> None:
    positive_df = positive_df[["TF", "Target", "Label"]].copy()
    negative_df = sample_eval_negatives(tf_list, gene_list, forbidden_pos_set, int(len(positive_df) * neg_ratio), seed)
    full_df = pd.concat([positive_df, negative_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_file, index=False)


def eval_benchmark_filename(split: str, neg_ratio: int) -> str:
    suffix = "" if neg_ratio == 1 else f"_neg{neg_ratio}"
    return f"{split}_tf_target_eval{suffix}.csv"


def audit_eval_benchmarks(
    val_eval_df: pd.DataFrame,
    test_eval_df: pd.DataFrame,
    all_positive_edges: set[tuple[int, int]],
    require_disjoint_negatives: bool = True,
) -> dict[str, int | bool]:
    val_neg_df = val_eval_df.loc[val_eval_df["Label"].astype(int) == 0]
    test_neg_df = test_eval_df.loc[test_eval_df["Label"].astype(int) == 0]
    val_negatives = labeled_edge_set(val_eval_df, 0)
    test_negatives = labeled_edge_set(test_eval_df, 0)
    val_positives = labeled_edge_set(val_eval_df, 1)
    test_positives = labeled_edge_set(test_eval_df, 1)

    audit: dict[str, int | bool] = {
        "validation_positive_count": len(val_positives),
        "validation_negative_count": len(val_negatives),
        "test_positive_count": len(test_positives),
        "test_negative_count": len(test_negatives),
        "validation_negative_duplicates": len(val_neg_df) - len(val_negatives),
        "test_negative_duplicates": len(test_neg_df) - len(test_negatives),
        "validation_test_positive_overlap": len(val_positives & test_positives),
        "validation_test_negative_overlap": len(val_negatives & test_negatives),
        "validation_negative_known_positive_overlap": len(
            val_negatives & all_positive_edges
        ),
        "test_negative_known_positive_overlap": len(
            test_negatives & all_positive_edges
        ),
        "validation_negative_self_loops": sum(tf == target for tf, target in val_negatives),
        "test_negative_self_loops": sum(tf == target for tf, target in test_negatives),
        "require_disjoint_negatives": require_disjoint_negatives,
    }
    invalid_keys = [
        "validation_negative_duplicates",
        "test_negative_duplicates",
        "validation_test_positive_overlap",
        "validation_negative_known_positive_overlap",
        "test_negative_known_positive_overlap",
        "validation_negative_self_loops",
        "test_negative_self_loops",
    ]
    if require_disjoint_negatives:
        invalid_keys.append("validation_test_negative_overlap")
    failures = {key: audit[key] for key in invalid_keys if audit[key] != 0}
    if failures:
        raise RuntimeError(f"Invalid evaluation benchmark: {failures}")
    return audit


def build_eval_benchmarks(
    data_root: str,
    data_type: str,
    tf_file: Path,
    tg_file: Path,
    train_file: Path,
    val_file: Path,
    test_file: Path,
    neg_ratio: int,
    seed: int,
    refresh: bool,
    output_dir: Path | None = None,
    validation_neg_ratio: int | None = None,
    require_disjoint_negatives: bool = True,
) -> tuple[Path, Path]:
    tf_list = pd.read_csv(tf_file)["index"].astype(int).tolist()
    gene_list = pd.read_csv(tg_file)["index"].astype(int).tolist()
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    eval_root = output_dir or (Path(data_root) / "prior_network" / data_type)
    val_neg_ratio = neg_ratio if validation_neg_ratio is None else validation_neg_ratio
    val_eval_file = eval_root / eval_benchmark_filename("validation", val_neg_ratio)
    test_eval_file = eval_root / eval_benchmark_filename("test", neg_ratio)

    all_forbidden = positive_edge_set(train_df) | positive_edge_set(val_df) | positive_edge_set(test_df)

    if require_disjoint_negatives:
        regenerate = refresh or not (val_eval_file.exists() and test_eval_file.exists())
        if regenerate:
            save_full_edges_eval(
                val_eval_file, val_df, all_forbidden, tf_list, gene_list, val_neg_ratio, seed
            )
            val_eval_df = pd.read_csv(val_eval_file)
            test_forbidden = all_forbidden | labeled_edge_set(val_eval_df, 0)
            save_full_edges_eval(
                test_eval_file,
                test_df,
                test_forbidden,
                tf_list,
                gene_list,
                neg_ratio,
                seed + 1,
            )
    else:
        if refresh or not val_eval_file.exists():
            save_full_edges_eval(
                val_eval_file, val_df, all_forbidden, tf_list, gene_list, val_neg_ratio, seed
            )
        if refresh or not test_eval_file.exists():
            save_full_edges_eval(
                test_eval_file, test_df, all_forbidden, tf_list, gene_list, neg_ratio, seed
            )

    audit_eval_benchmarks(
        pd.read_csv(val_eval_file),
        pd.read_csv(test_eval_file),
        all_forbidden,
        require_disjoint_negatives=require_disjoint_negatives,
    )
    return val_eval_file, test_eval_file


def add_bidirectional_edges(edge_idx: torch.Tensor, tflist: torch.Tensor) -> torch.Tensor:
    bidir_idx = torch.isin(edge_idx[1], tflist)
    bidir_edges = edge_idx[:, bidir_idx].flip(0)
    return torch.unique(torch.cat([edge_idx, bidir_edges], dim=1), dim=1)


def focal_loss_with_logits(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), labels, reduction="mean")
    pt = torch.exp(-bce)
    return alpha * (1 - pt) ** gamma * bce


def standard_focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits = logits.reshape(-1)
    labels = labels.reshape(-1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = labels * alpha + (1.0 - labels) * (1.0 - alpha)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def classification_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    focal_alpha: float = 0.5,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    if loss_type == "legacy_focal":
        return focal_loss_with_logits(logits, labels)
    if loss_type == "bce":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1), labels.reshape(-1), reduction="mean"
        )
    if loss_type == "focal":
        return standard_focal_loss_with_logits(
            logits,
            labels,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def evaluate_predictions(edge_index: torch.Tensor, y_true: torch.Tensor, logits: torch.Tensor, tf_list: torch.Tensor, threshold: float = 0.5, k_list: tuple[int, ...] = (10, 30, 50, 100)) -> dict[str, float]:
    edge_np = edge_index.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy().astype(int)
    logit_np = logits.detach().cpu().numpy()
    prob_np = torch.sigmoid(logits.detach()).cpu().numpy()
    tf_set = set(int(item) for item in tf_list.detach().cpu().tolist())

    src, _ = edge_np
    results: dict[str, float] = {}
    try:
        results["AUROC"] = roc_auc_score(y_true_np, logit_np)
    except ValueError:
        results["AUROC"] = float("nan")
    results["AUPRC"] = average_precision_score(y_true_np, logit_np)

    preds = (prob_np > threshold).astype(np.int64)
    results["F1"] = f1_score(y_true_np, preds)
    results["MCC"] = matthews_corrcoef(y_true_np, preds)

    tf_mask = np.isin(src, list(tf_set))
    tf_groups = {}
    per_tf_auprc = []
    for tf_idx in np.unique(src[tf_mask]):
        mask = src == tf_idx
        y_tf = y_true_np[mask]
        s_tf = logit_np[mask]
        if y_tf.sum() == 0:
            continue
        tf_groups[int(tf_idx)] = (y_tf, s_tf)
        per_tf_auprc.append(average_precision_score(y_tf, s_tf))

    results["PerTF_AUPRC"] = float(np.mean(per_tf_auprc)) if per_tf_auprc else float("nan")

    max_k = max(k_list)
    fixed_tf_ids = [tf_idx for tf_idx, (y_tf, _) in tf_groups.items() if len(y_tf) >= max_k]
    results["TF_SetSize"] = int(len(fixed_tf_ids))
    results["TF_MaxK"] = int(max_k)

    for k in k_list:
        tf_precision = []
        for tf_idx in fixed_tf_ids:
            y_tf, s_tf = tf_groups[tf_idx]
            order_tf = np.argsort(-s_tf)
            sorted_tf_labels = y_tf[order_tf]
            tf_precision.append(sorted_tf_labels[:k].mean())
        results[f"TF_Count@{k}"] = int(len(fixed_tf_ids))
        results[f"TF_P@{k}"] = float(np.mean(tf_precision)) if tf_precision else float("nan")

    return results


def optimize_probability_threshold(y_true: torch.Tensor, logits: torch.Tensor) -> tuple[float, float, float, float, float]:
    y_true_np = y_true.detach().cpu().numpy().astype(int)
    logit_np = logits.detach().cpu().numpy()
    prob_np = torch.sigmoid(logits.detach()).cpu().numpy()

    best_threshold = 0.5
    best_score = -1.0
    best_auc = roc_auc_score(y_true_np, logit_np)
    best_aupr = average_precision_score(y_true_np, logit_np)
    best_mcc = float("nan")

    for threshold in np.linspace(0.1, 0.9, 81):
        preds = (prob_np > threshold).astype(np.int64)
        f1 = f1_score(y_true_np, preds)
        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)
            best_mcc = matthews_corrcoef(y_true_np, preds)

    return best_threshold, best_score, best_auc, best_aupr, best_mcc
