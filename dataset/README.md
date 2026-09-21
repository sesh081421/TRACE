# Dataset Setup

The [TRACE code repository](https://github.com/sesh081421/TRACE) does not bundle
large expression matrices or TF-target prior networks. The separate prepared-data
location is the `dataset/` subfolder of this
[Google Drive folder](https://drive.google.com/drive/folders/1WmWXiD0VyNyD_Bd8IkFWRRWi5N1LAsDp?usp=drive_link).
Data are uploaded separately from the code release as ZIP archives. Download
and extract a ZIP locally; downloading individual unpacked files is not required.

## Download And Extract

- `TRACE_example_data_BM.zip` (92.7 MB): BM, sufficient for the quick example.
- `TRACE_preprocessed_data.zip` (3.51 GB): all six datasets, including BM;
  approximately 9.6 GB after extraction.

Choose one archive. Both contain the shared TFLink prior and the corresponding
gene/TF index, positive split and frozen evaluation files. From the **TRACE
repository root**, extract the BM archive with:

```bash
unzip ../TRACE_example_data_BM.zip -d .
```

For all datasets, use `TRACE_preprocessed_data.zip` instead. Adjust the archive
path to its actual download location. Each ZIP already contains `dataset/`, so
do not add another nested `dataset/` directory. With a desktop archive manager,
place the extracted `dataset/` folder directly inside `TRACE/`. The resulting
prior path must be `TRACE/dataset/prior_network/tflink.h5`. Run the examples
from `TRACE/`, not from inside this data directory. Another input directory can
be supplied with `--data_root`.

## Prior Network

The default prior path is:

```text
dataset/prior_network/tflink.h5
```

The HDF5 table must provide source-TF and target-gene columns that can be
interpreted as `TF` and `Target`. Gene symbols must use the same naming
convention as the expression input.

## Expression Inputs

TRACE's configured datasets use the following paths:

```text
dataset/scRNA_ATAC/GSE150774_RAW/BM/BM_preprocessed.h5ad
dataset/scRNA_ATAC/GSE150774_RAW/UCB/UCB_preprocessed.h5ad
dataset/scRNA_ATAC/pbmc_granulocytes/pbmc10k_integrated.h5mu
dataset/scRNA_ATAC/pbmc_granulocytes/GAS_gene_cell_score.h5
dataset/scRNA_ATAC/lymph/lymph_node_lymphoma_14k_filtered_feature_bc_matrix.h5mu
dataset/scRNA_ATAC/lymph/GAS_gene_cell_score.h5
dataset/scRNA_ATAC/BEELINE/inputs/scRNA-Seq/hESC/ExpressionData.csv
dataset/scRNA_ATAC/BEELINE/inputs/scRNA-Seq/hHep/ExpressionData.csv
```

AnnData inputs must contain genes in `var_names`. Non-BEELINE inputs use a
`var['highly_variable']` mask. MuData inputs must contain an `rna`
modality. PBMC and lymphoma additionally use the adjacent gene-activity-score
HDF5 file.

## Preserved Or Generated Files

The prepared archives include existing indexed TF/gene tables and positive
splits. If they are absent, the loader creates them by intersecting expression
genes with the prior network. The expected paths are:

```text
dataset/tfs/<dataset>/tflist.csv
dataset/tfs/<dataset>/sctglist.csv
dataset/prior_network/<dataset>/prior_net_idx.csv
dataset/prior_network/<dataset>/train_data.csv
dataset/prior_network/<dataset>/validation_data.csv
dataset/prior_network/<dataset>/test_data.csv
dataset/prior_network/<dataset>/validation_tf_target_eval.csv
dataset/prior_network/<dataset>/test_tf_target_eval.csv
dataset/prior_network/<dataset>/test_tf_target_eval_neg10.csv
```

Keep these files when repeating a run so the same indexed candidate universe,
positive split, and evaluation candidates are reused. Alternatively, pass
`--eval_background_root <path>` to place the generated evaluation candidates
outside the input tree.

The archives use the verified disjoint candidate files, placed in the default
paths above. `TRACE_DATA_MANIFEST.csv` records their source and destination
paths. Do not pass `--refresh_eval` when preserving these supplied candidates.
The unsuffixed test file is the 1:1 test set; the default CLI uses the 1:10 file.

The released CLI uses a fixed 1:1 validation background and a fixed 1:10 test
background. These backgrounds are mutually exclusive and omit every annotated
positive. Their complete candidate union is also excluded from dynamic
training-negative sampling. The metrics JSON records a post-training audit of
these conditions.

Before a full sweep, verify one dataset from the **repository root** with:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_trace.py \
  --data GSE150774_RAW/BM --benchmark false \
  --data_root ./dataset --result_dir ./results_quick \
  --epochs 3 --seed 8
```

The same check is available as `CUDA_VISIBLE_DEVICES=0 bash examples/run_quick_check.sh`.
The small archive cannot run `--datasets all`; download the full archive first.
Data and regulatory resources retain their original terms and are not
relicensed under the code's MIT license.
