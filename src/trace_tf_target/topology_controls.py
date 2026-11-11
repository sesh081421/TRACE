from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy import sparse


HEURISTIC_METHODS = (
    "target_indegree",
    "preferential_attachment",
    "common_neighbors",
    "jaccard",
    "adamic_adar",
)


def _edge_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(source), int(target))
        for source, target in frame[["TF", "Target"]].to_numpy(np.int64)
    }


def _degree_counts(edges: list[tuple[int, int]], position: int) -> dict[int, int]:
    counts: dict[int, int] = {}
    for edge in edges:
        node = int(edge[position])
        counts[node] = counts.get(node, 0) + 1
    return counts


def _degree_digest(counts: dict[int, int]) -> str:
    payload = "\n".join(f"{node}:{counts[node]}" for node in sorted(counts))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def bipartite_degree_preserving_rewire(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    swaps_per_edge: float = 10.0,
    max_attempt_factor: int = 50,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Degree-preserving control that excludes held-out positive pairs."""
    graph, audit = _bipartite_rewire(
        train_df, _edge_set(val_df) | _edge_set(test_df), seed,
        swaps_per_edge, max_attempt_factor,
    )
    audit.update(heldout_labels_consulted=True,
                 null_definition="held-out-positive-excluding bipartite rewire")
    return graph, audit


def training_only_degree_preserving_rewire(
    train_df: pd.DataFrame,
    seed: int,
    swaps_per_edge: float = 10.0,
    max_attempt_factor: int = 50,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Label-blind graph null: this API cannot access held-out labels."""
    graph, audit = _bipartite_rewire(train_df, set(), seed, swaps_per_edge, max_attempt_factor)
    audit.pop("held_out_positive_overlap")
    audit.update(heldout_labels_consulted=False,
                 null_definition="training-only bipartite degree-preserving graph",
                 self_rule="no new self pairs; original self pairs may remain or be removed")
    return graph, audit


def _bipartite_rewire(train_df, held_out_positive, seed, swaps_per_edge, max_attempt_factor):
    if swaps_per_edge <= 0:
        raise ValueError("swaps_per_edge must be positive")
    if max_attempt_factor < 1:
        raise ValueError("max_attempt_factor must be at least 1")

    train = (
        train_df[["TF", "Target"]]
        .astype(np.int64)
        .drop_duplicates()
        .sort_values(["TF", "Target"])
        .reset_index(drop=True)
    )
    if len(train) < 2:
        raise ValueError("At least two unique train edges are required for rewiring.")

    original_edges = [
        (int(source), int(target))
        for source, target in train[["TF", "Target"]].to_numpy(np.int64)
    ]
    original_set = set(original_edges)
    original_self_edges = {
        edge for edge in original_set if edge[0] == edge[1]
    }
    edges = original_edges.copy()
    edge_set = set(edges)

    rng = np.random.default_rng(int(seed))
    edge_count = len(edges)
    target_swaps = max(1, int(math.ceil(edge_count * float(swaps_per_edge))))
    max_attempts = max(1000, target_swaps * int(max_attempt_factor))
    accepted_swaps = 0
    attempts = 0

    while accepted_swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        first = int(rng.integers(edge_count))
        second = int(rng.integers(edge_count - 1))
        if second >= first:
            second += 1

        source_a, target_a = edges[first]
        source_b, target_b = edges[second]
        if source_a == source_b or target_a == target_b:
            continue

        swapped_a = (source_a, target_b)
        swapped_b = (source_b, target_a)
        if swapped_a[0] == swapped_a[1] or swapped_b[0] == swapped_b[1]:
            continue
        if swapped_a in held_out_positive or swapped_b in held_out_positive:
            continue
        if swapped_a in edge_set or swapped_b in edge_set:
            continue

        edge_set.remove(edges[first])
        edge_set.remove(edges[second])
        edges[first] = swapped_a
        edges[second] = swapped_b
        edge_set.add(swapped_a)
        edge_set.add(swapped_b)
        accepted_swaps += 1

    if accepted_swaps < target_swaps:
        raise RuntimeError(
            "Bipartite rewiring did not reach the requested mixing target: "
            f"accepted={accepted_swaps}, requested={target_swaps}, "
            f"attempts={attempts}, max_attempts={max_attempts}."
        )

    source_before = _degree_counts(original_edges, 0)
    source_after = _degree_counts(edges, 0)
    target_before = _degree_counts(original_edges, 1)
    target_after = _degree_counts(edges, 1)
    original_overlap = len(set(edges) & original_set)
    rewired_self_edges = {edge for edge in set(edges) if edge[0] == edge[1]}
    new_self_edges = rewired_self_edges - original_self_edges
    held_out_overlap = len(set(edges) & held_out_positive)
    duplicate_count = len(edges) - len(set(edges))
    if source_before != source_after or target_before != target_after:
        raise RuntimeError("Bipartite rewiring changed a source or target degree.")
    if new_self_edges or held_out_overlap or duplicate_count:
        raise RuntimeError(
            "Invalid bipartite rewire: "
            f"new_self_edges={len(new_self_edges)}, held_out_overlap={held_out_overlap}, "
            f"duplicates={duplicate_count}."
        )

    rewired = pd.DataFrame(edges, columns=["TF", "Target"])
    rewired["Label"] = 1
    audit: dict[str, object] = {
        "rewire_method": "bipartite_double_edge_swap",
        "rewire_seed": int(seed),
        "swaps_per_edge": float(swaps_per_edge),
        "requested_swaps": int(target_swaps),
        "accepted_swaps": int(accepted_swaps),
        "attempts": int(attempts),
        "rewire_input_edge_count": int(edge_count),
        "source_degree_preserved": True,
        "target_degree_preserved": True,
        "source_degree_sha256": _degree_digest(source_after),
        "target_degree_sha256": _degree_digest(target_after),
        "original_edge_overlap": int(original_overlap),
        "original_edge_overlap_fraction": float(original_overlap / edge_count),
        "changed_edge_count": int(edge_count - original_overlap),
        "changed_edge_fraction": float(1.0 - original_overlap / edge_count),
        "held_out_positive_overlap": 0,
        "original_self_edge_count": int(len(original_self_edges)),
        "remaining_original_self_edge_count": int(len(rewired_self_edges)),
        "removed_original_self_edge_count": int(
            len(original_self_edges - rewired_self_edges)
        ),
        "new_self_edge_count": 0,
        "duplicate_edge_count": 0,
    }
    return rewired, audit


def _stable_tie_values(
    tf_index: int,
    targets: np.ndarray,
    seed: int,
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        values ^= np.uint64(int(tf_index) + 1) * np.uint64(0x9E3779B97F4A7C15)
        values ^= np.uint64(int(seed)) * np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return values.astype(np.float64) / float(np.iinfo(np.uint64).max)


class TopologyHeuristicScorer:
    """Score TF-target pairs using only the train-positive graph."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        num_nodes: int,
        tie_seed: int = 20260818,
    ) -> None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        train = (
            train_df[["TF", "Target"]]
            .astype(np.int64)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        source = train["TF"].to_numpy(np.int64)
        target = train["Target"].to_numpy(np.int64)
        if (
            source.size
            and (
                source.min() < 0
                or target.min() < 0
                or source.max() >= num_nodes
                or target.max() >= num_nodes
            )
        ):
            raise ValueError("Train edge index falls outside num_nodes.")

        self.num_nodes = int(num_nodes)
        self.tie_seed = int(tie_seed)
        self.source_outdegree = np.bincount(source, minlength=num_nodes).astype(float)
        self.target_indegree = np.bincount(target, minlength=num_nodes).astype(float)

        undirected_source = np.concatenate([source, target])
        undirected_target = np.concatenate([target, source])
        data = np.ones(undirected_source.size, dtype=np.float64)
        adjacency = sparse.csr_matrix(
            (data, (undirected_source, undirected_target)),
            shape=(num_nodes, num_nodes),
        )
        adjacency.sum_duplicates()
        adjacency.data[:] = 1.0
        self.adjacency = adjacency
        self.graph_degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        self._common_cache: dict[int, np.ndarray] = {}
        self._adamic_cache: dict[int, np.ndarray] = {}

        inv_log_degree = np.zeros(num_nodes, dtype=float)
        valid = self.graph_degree > 1
        inv_log_degree[valid] = 1.0 / np.log(self.graph_degree[valid])
        self.inv_log_degree = inv_log_degree
        self.audit = {
            "train_positive_only": True,
            "train_edge_count": int(len(train)),
            "num_nodes": int(num_nodes),
            "source_count": int(np.count_nonzero(self.source_outdegree)),
            "target_count": int(np.count_nonzero(self.target_indegree)),
            "undirected_graph_edges": int(adjacency.nnz),
            "tie_break": "fixed splitmix64 jitter",
            "tie_seed": int(tie_seed),
        }

    def _structural_rows(self, tf_index: int) -> tuple[np.ndarray, np.ndarray]:
        tf_index = int(tf_index)
        if tf_index not in self._common_cache:
            source_row = self.adjacency.getrow(tf_index)
            common = np.asarray((source_row @ self.adjacency).toarray()).reshape(-1)
            weighted_source = source_row.multiply(self.inv_log_degree)
            adamic = np.asarray((weighted_source @ self.adjacency).toarray()).reshape(-1)
            self._common_cache[tf_index] = common
            self._adamic_cache[tf_index] = adamic
        return self._common_cache[tf_index], self._adamic_cache[tf_index]

    def score(
        self,
        method: str,
        tf_index: int,
        targets: np.ndarray,
    ) -> np.ndarray:
        if method not in HEURISTIC_METHODS:
            raise ValueError(f"Unsupported heuristic method: {method}")
        targets = np.asarray(targets, dtype=np.int64)
        if method == "target_indegree":
            base = self.target_indegree[targets]
        elif method == "preferential_attachment":
            base = self.source_outdegree[int(tf_index)] * self.target_indegree[targets]
        else:
            common, adamic = self._structural_rows(int(tf_index))
            if method == "common_neighbors":
                base = common[targets]
            elif method == "adamic_adar":
                base = adamic[targets]
            else:
                denominator = (
                    self.graph_degree[int(tf_index)]
                    + self.graph_degree[targets]
                    - common[targets]
                )
                base = np.divide(
                    common[targets],
                    denominator,
                    out=np.zeros_like(denominator, dtype=float),
                    where=denominator > 0,
                )

        base = np.asarray(base, dtype=np.float64)
        tie = _stable_tie_values(int(tf_index), targets, self.tie_seed)
        scale = max(1.0, float(np.max(np.abs(base))) if base.size else 1.0) * 1e-12
        return base + tie * scale
