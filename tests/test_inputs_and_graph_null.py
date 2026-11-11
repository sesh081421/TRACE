"""Check model feature inputs and the label-blind graph control."""

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trace_tf_target import utils
from trace_tf_target.graph_null import canonical_edges, load_or_create_training_null, validate_null
from trace_tf_target.topology_controls import (
    bipartite_degree_preserving_rewire,
    training_only_degree_preserving_rewire,
)


class InputPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        folder = self.root / "scRNA_ATAC" / "toy"
        folder.mkdir(parents=True)
        self.path = folder / "GAS_gene_cell_score.h5"
        self.rna = pd.DataFrame([[1, 2], [3, 5]], index=["A", "B"],
                                columns=["AAAC-1", "TTTC-1"], dtype=np.float32)

    def test_pandas_gas_normalizes_literal_dots_without_writing(self):
        gas = self.rna.astype(np.float64)
        gas.columns = ["AAAC.1", "TTTC-1"]
        gas.to_hdf(self.path, key="df")
        original = self.path.read_bytes()
        loaded = utils.load_gas(str(self.root), "toy", save_fixed=False)
        pd.testing.assert_frame_equal(loaded, self.rna.astype(np.float64))
        self.assertEqual(self.path.read_bytes(), original)

    def test_raw_hdf_fallback_normalizes_literal_dots(self):
        with h5py.File(self.path, "w") as handle:
            handle["matrix"] = self.rna.to_numpy()
            handle["gene_names"] = np.array([b"A", b"B"])
            handle["cell_names"] = np.array([b"AAAC.1", b"TTTC-1"])
        original = self.path.read_bytes()
        loaded = utils.load_gas(str(self.root), "toy", save_fixed=False)
        pd.testing.assert_frame_equal(loaded, self.rna)
        self.assertEqual(self.path.read_bytes(), original)

    def test_pbmc_integrates_gas_after_barcode_normalization(self):
        folder = self.root / "scRNA_ATAC/pbmc_granulocytes"
        folder.mkdir()
        gas = pd.DataFrame([[2., 0.], [0., 8.]], index=self.rna.index,
                           columns=["AAAC.1", "TTTC.1"])
        gas.to_hdf(folder / "GAS_gene_cell_score.h5", key="df")
        with patch.object(utils, "get_sc_data"), patch.object(utils, "get_filtered_x", return_value=self.rna):
            loaded = utils.load_expression(str(self.root), "pbmc_granulocytes", False)
        gas.columns = self.rna.columns
        pd.testing.assert_frame_equal(loaded, self.rna + gas)
        self.assertFalse(np.array_equal(loaded.values, self.rna.values))
        np.testing.assert_array_equal(utils.prepare_feature_matrix(loaded)[0],
                                      utils.prepare_feature_matrix(self.rna + gas)[0])

    def test_explicit_rna_only_mode_does_not_load_gas(self):
        for dataset in ["pbmc_granulocytes", "lymph"]:
            with self.subTest(dataset=dataset):
                with patch.object(utils, "get_sc_data"), patch.object(utils, "get_filtered_x", return_value=self.rna):
                    with patch.object(utils, "load_gas", side_effect=AssertionError("RNA-only must not load GAS")):
                        loaded = utils.load_expression_rna_only(str(self.root), dataset, False)
                pd.testing.assert_frame_equal(loaded, self.rna)

    def test_rna_h5ad_inputs_do_not_load_gas(self):
        for dataset in ["GSE150774_RAW/BM", "GSE150774_RAW/UCB"]:
            with self.subTest(dataset=dataset):
                with patch.object(utils, "get_sc_data"), patch.object(utils, "get_filtered_x", return_value=self.rna):
                    with patch.object(utils, "load_gas", side_effect=AssertionError("RNA input must not load GAS")):
                        loaded = utils.load_expression(str(self.root), dataset, False)
                pd.testing.assert_frame_equal(loaded, self.rna)

    def test_pbmc_missing_gas_is_not_silently_ignored(self):
        with patch.object(utils, "get_sc_data"), patch.object(utils, "get_filtered_x", return_value=self.rna):
            with self.assertRaises(FileNotFoundError):
                utils.load_expression(str(self.root), "pbmc_granulocytes", False)

    def test_lymphoma_keeps_gas_and_does_not_rewrite_input(self):
        gas = self.rna.astype(np.float64) * 2
        with patch.object(utils, "get_sc_data"), patch.object(utils, "get_filtered_x", return_value=self.rna):
            with patch.object(utils, "load_gas", return_value=gas) as load:
                loaded = utils.get_exp_data(str(self.root), "lymph", "input.h5mu")
        load.assert_called_once_with(str(self.root), "lymph", save_fixed=False)
        pd.testing.assert_frame_equal(loaded, self.rna + gas)

    def test_zero_cell_or_gene_overlap_fails(self):
        for gas in (self.rna.rename(columns=lambda cell: cell + "x"),
                    self.rna.rename(index=lambda gene: gene + "x")):
            with self.assertRaisesRegex(ValueError, "no shared cells or genes"):
                utils.integrate_rna_gas(self.rna, gas)

    def test_partial_matching_keeps_legacy_zero_fill(self):
        gas = pd.DataFrame([[7.0]], index=["A"], columns=["AAAC-1"])
        with self.assertWarnsRegex(RuntimeWarning, "1/2 cells and 1/2 genes"):
            loaded = utils.integrate_rna_gas(self.rna, gas)
        legacy = self.rna + gas.reindex(index=self.rna.index, columns=self.rna.columns, fill_value=0)
        pd.testing.assert_frame_equal(loaded, legacy)

    def test_normalization_collision_fails_integration(self):
        gas = self.rna.copy()
        gas.columns = ["AAAC.1", "AAAC-1"]
        gas.to_hdf(self.path, key="df")
        loaded = utils.load_gas(str(self.root), "toy", save_fixed=False)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            utils.integrate_rna_gas(self.rna, loaded)


class TrainingOnlyGraphTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"TF": [0, 0, 1, 1, 2, 2, 3, 3],
                                   "Target": [4, 5, 5, 6, 6, 7, 7, 4], "Label": 1})
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "graph.csv"

    def test_training_only_api_has_no_heldout_inputs(self):
        self.assertEqual(list(inspect.signature(training_only_degree_preserving_rewire).parameters),
                         ["train_df", "seed", "swaps_per_edge", "max_attempt_factor"])
        graph, audit = training_only_degree_preserving_rewire(self.train, 8)
        validate_null(self.train, graph)
        self.assertIs(audit["heldout_labels_consulted"], False)
        self.assertNotIn("held_out_positive_overlap", audit)
        self.assertEqual(audit["new_self_edge_count"], 0)
        repeated, _ = training_only_degree_preserving_rewire(self.train, 8)
        pd.testing.assert_frame_equal(graph, repeated)

    def test_legacy_and_label_blind_controls_are_distinct(self):
        graph, _ = training_only_degree_preserving_rewire(self.train, 8)
        original = set(self.train[["TF", "Target"]].itertuples(index=False, name=None))
        novel = set(graph[["TF", "Target"]].itertuples(index=False, name=None)) - original
        self.assertTrue(novel)
        heldout = pd.DataFrame([sorted(novel)[0]], columns=["TF", "Target"])
        legacy, audit = bipartite_degree_preserving_rewire(
            self.train, heldout, heldout.iloc[:0], seed=8)
        self.assertIs(audit["heldout_labels_consulted"], True)
        self.assertFalse(set(legacy[["TF", "Target"]].itertuples(index=False, name=None)) &
                         set(heldout.itertuples(index=False, name=None)))
        validate_null(self.train, legacy)

    def test_cache_reuses_same_graph_and_rejects_setting_changes(self):
        graph, audit = load_or_create_training_null(self.train, 8, 10, 50, self.cache)
        repeated, repeated_audit = load_or_create_training_null(self.train, 8, 10, 50, self.cache)
        pd.testing.assert_frame_equal(graph, repeated)
        self.assertEqual(audit, repeated_audit)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            load_or_create_training_null(self.train, 18, 10, 50, self.cache)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            load_or_create_training_null(self.train.iloc[:-1], 8, 10, 50, self.cache)

    def test_cache_rejects_heldout_excluding_manifest(self):
        load_or_create_training_null(self.train, 8, 10, 50, self.cache)
        path = self.cache.with_suffix(".json")
        payload = json.loads(path.read_text())
        payload["heldout_labels_consulted"] = True
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "identity or construction mismatch"):
            load_or_create_training_null(self.train, 8, 10, 50, self.cache)

    def test_cache_rejects_modified_pairs(self):
        graph, _ = load_or_create_training_null(self.train, 8, 10, 50, self.cache)
        graph.iloc[:-1].to_csv(self.cache, index=False)
        with self.assertRaisesRegex(ValueError, "edge count changed"):
            load_or_create_training_null(self.train, 8, 10, 50, self.cache)

    def test_input_must_be_positive_and_no_new_self_pairs(self):
        invalid = self.train.copy()
        invalid.loc[0, "Label"] = 0
        with self.assertRaisesRegex(ValueError, "positive training rows"):
            canonical_edges(invalid)
        train = pd.DataFrame({"TF": [0, 0, 1, 1, 2, 2, 3, 3],
                              "Target": [0, 2, 2, 3, 3, 0, 1, 4], "Label": 1})
        graph, _ = training_only_degree_preserving_rewire(train, 8)
        validate_null(train, graph)
        self.assertTrue(set(graph.loc[graph.TF == graph.Target, "TF"]) <= {0})


if __name__ == "__main__":
    unittest.main()
