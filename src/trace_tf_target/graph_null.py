"""Frozen training-only graph nulls; never takes validation or test labels."""

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .topology_controls import training_only_degree_preserving_rewire


def canonical_edges(frame):
    if "Label" in frame and not (frame.Label == 1).all():
        raise ValueError("Graph source must contain positive training rows only.")
    return frame[["TF", "Target"]].astype(np.int64).drop_duplicates().sort_values(
        ["TF", "Target"]).reset_index(drop=True)


def edge_digest(frame):
    return hashlib.sha256(canonical_edges(frame).to_numpy(dtype="<i8").tobytes()).hexdigest()


def validate_null(train, graph):
    if len(graph) != len(canonical_edges(graph)):
        raise ValueError("Graph null contains duplicate pairs.")
    train, graph = canonical_edges(train), canonical_edges(graph)
    if len(train) != len(graph):
        raise ValueError("Graph null edge count changed.")
    for key in ("TF", "Target"):
        if not train.groupby(key).size().equals(graph.groupby(key).size()):
            raise ValueError(f"Graph null changed {key} degrees.")
    allowed_self = set(train.loc[train.TF == train.Target, "TF"])
    if set(graph.loc[graph.TF == graph.Target, "TF"]) - allowed_self:
        raise ValueError("Graph null introduced new self pairs.")


def load_or_create_training_null(train, seed, swaps, max_attempt_factor, cache=None):
    settings = {"train_edge_digest": edge_digest(train), "graph_seed": int(seed),
                "swaps_per_edge": float(swaps), "max_attempt_factor": int(max_attempt_factor)}

    def generate():
        graph, audit = training_only_degree_preserving_rewire(train, seed, swaps, max_attempt_factor)
        validate_null(train, graph)
        graph = canonical_edges(graph)
        graph["Label"] = 1
        return graph, dict(audit, **settings, graph_edge_digest=edge_digest(graph))

    if cache is None:
        return generate()
    path = Path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = path.with_suffix(".json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if metadata.exists():
            graph = pd.read_csv(path)
            audit = json.loads(metadata.read_text())
            if any(audit.get(key) != value for key, value in settings.items()):
                raise ValueError("Frozen graph null configuration mismatch.")
            validate_null(train, graph)
            if edge_digest(graph) != audit.get("graph_edge_digest") or audit.get("heldout_labels_consulted") is not False:
                raise ValueError("Frozen graph null identity or construction mismatch.")
            return graph, audit
        if path.exists():
            raise ValueError("Graph CSV exists without a completed manifest; inspect before resuming.")
        graph, audit = generate()
        temporary = path.with_suffix(".csv.tmp")
        graph.to_csv(temporary, index=False)
        temporary.replace(path)
        temporary = metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(audit, indent=2) + "\n")
        temporary.replace(metadata)
        return graph, audit
