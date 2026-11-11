# Data preparation

Set `--data_root` to the directory containing prepared expression inputs and
the TFLink prior. The examples and prepared archives use `./dataset`.
For downloadable ZIP archives and extraction commands, see
[Dataset Setup](../dataset/README.md). Extract an archive at the repository
root, not inside `dataset/`, since it already contains that directory.

| Dataset | Command-line dataset ID | Source |
| --- | --- | --- |
| Bone marrow (BM) | `GSE150774_RAW/BM` | [GEO GSE150774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150774), GSM4558616–GSM4558618 |
| Umbilical cord blood (UCB) | `GSE150774_RAW/UCB` | [GEO GSE150774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150774), GSM4558614–GSM4558615 |
| PBMC | `pbmc_granulocytes` | [10x Genomics PBMC Multiome](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0) |
| Lymphoma | `lymph` | [10x Genomics lymphoma Multiome](https://www.10xgenomics.com/datasets/fresh-frozen-lymph-node-with-b-cell-lymphoma-14-k-sorted-nuclei-1-standard-2-0-0) |
| Embryonic stem cells (hESC) | `hESC` | [BEELINE](https://doi.org/10.1038/s41592-019-0690-6) |
| Mature hepatocytes (hHep) | `hHep` | [BEELINE](https://doi.org/10.1038/s41592-019-0690-6) |

Use [TFLink](https://doi.org/10.1093/database/baac083) for the human regulatory
prior. See the manuscript's Methods and Supplementary Note 1 for preprocessing,
identifier harmonization, and the reference TF set.

```text
dataset/
  prior_network/
    tflink.h5
  scRNA_ATAC/
    GSE150774_RAW/BM/BM_preprocessed.h5ad
    GSE150774_RAW/UCB/UCB_preprocessed.h5ad
    pbmc_granulocytes/pbmc10k_integrated.h5mu
    pbmc_granulocytes/GAS_gene_cell_score.h5
    lymph/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5mu
    lymph/GAS_gene_cell_score.h5
    BEELINE/inputs/scRNA-Seq/hESC/ExpressionData.csv
    BEELINE/inputs/scRNA-Seq/hHep/ExpressionData.csv
```

- **TFLink:** Store a two-column table ordered as regulator gene and target
  gene. The loader accepts a pandas HDF5 table or a CSV selected with
  `--prior_filename`. Match the gene identifiers to the expression matrices.
- **BM and UCB:** Prepare AnnData files with cells in rows, genes in columns, and
  a Boolean `var["highly_variable"]` mask. The loader uses the selected values
  in `X`.
- **PBMC and lymphoma:** Use a MuData file with an `rna` modality and an adjacent
  gene-by-cell activity HDF5 file. Use matching gene names and cell barcodes.
  The loader aligns gene activity to RNA, fills unmatched entries with zero,
  and adds the matrices before gene-wise standardization.
- **hESC and hHep:** Use the BEELINE gene-by-cell expression CSVs. Benchmark mode
  transposes them for normalization, log transformation, and variable-gene
  selection.

On first use, the training tools write TF and gene indices, positive splits,
and sampled evaluation candidates under `dataset/tfs/` and `dataset/prior_network/`.
Keep the indexed files with the run and reuse the same data root for paired
comparisons. Use separate roots for independent split-generation jobs.
