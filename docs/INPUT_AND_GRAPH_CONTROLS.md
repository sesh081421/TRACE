# Input And Graph Options

## Feature Inputs

| Datasets | Default feature input |
| --- | --- |
| BM, UCB, hESC, hHep | RNA |
| PBMC, lymphoma | RNA+GAS |

For PBMC and lymphoma, place `GAS_gene_cell_score.h5` beside the expression
H5MU file. The loader aligns GAS to RNA by gene and cell identifiers, replaces
literal dots in GAS barcodes with hyphens, and adds the aligned matrices before
gene-wise standardization. Unmatched GAS entries are zero-filled.

Partial alignment produces a warning. Zero shared genes or cells, or duplicate
identifiers after normalization, raises an error. Training does not rewrite
the source GAS file. The direct training CLI's `--feature_mode rna_only` uses
RNA without loading GAS; `--feature_mode pca` additionally reduces the default
features to `--pca_components` dimensions.

TRACE uses the training TFLink prior for supervision and graph construction.
PPI data are not required by TRACE.

## Optional Graph Controls

Use the direct training CLI for these `--topology_graph_variant` options:

| Option | Construction |
| --- | --- |
| `derived` | Default topology derived from training-positive edges |
| `bipartite_rewired` | Degree-preserving swaps that exclude held-out positives |
| `train_only_rewired` | Degree-preserving swaps using training edges only |

For `train_only_rewired`, held-out labels do not determine swap acceptance.
The graph preserves training source outdegrees and target indegrees, excludes
duplicate pairs and new self-pairs, and leaves supervised labels unchanged.
Set `--topology_rewire_seed`, `--topology_rewire_swaps_per_edge`, and
`--topology_rewire_max_attempt_factor` to control generation.

Use `--topology_null_cache ./graph_nulls/BM/graph.csv` to save and reuse a
training-only graph. Use a separate cache per dataset. The loader verifies the
training edges, graph contents, and generation settings before reuse. Keep the
graph seed fixed when varying only the optimization seed.
