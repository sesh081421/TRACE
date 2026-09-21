# TRACE

Official implementation of **TRACE** (**T**ranscription-factor-aware
**R**egulatory **A**rchitecture for **C**andidate-**E**dge prioritization), a
topology-aware neural model for directed transcription factor (TF)-target edge
inference from single-cell gene features and a TF-target prior network.

TRACE restricts source nodes to known TFs and ranks candidate target genes. It
is intended for directed TF-target inference rather than unrestricted or
undirected gene-gene association prediction.

**Code:** [sesh081421/TRACE](https://github.com/sesh081421/TRACE)  
**Prepared data:** [Google Drive](https://drive.google.com/drive/folders/1WmWXiD0VyNyD_Bd8IkFWRRWi5N1LAsDp?usp=drive_link), in the `dataset/` subfolder.

## Method Summary

TRACE intersects the regulatory prior with measured genes and splits retained
positive edges into training, validation, and test sets. It then builds a
topology graph using training positives only:

1. construct the training TF-target adjacency;
2. identify two-hop candidate gene pairs;
3. remove direct one-hop edges;
4. weight candidates by neighborhood Jaccard similarity; and
5. retain edges passing the top-k and similarity thresholds.

Gene features and the derived graph are processed by a two-layer topology-aware
graph-attention encoder. A bilinear decoder scores ordered TF-target embedding
pairs. Evaluation candidates are excluded from dynamic training-negative
sampling, and the resulting pair-identity audit is saved with each run.

## Repository Layout

```text
src/trace_tf_target/    Model, data loading, training, and metrics
scripts/run_trace.py    Single-dataset entry point
scripts/run_sweep.py    Multi-dataset and multi-seed runner
scripts/summarize.py    Aggregate completed runs
dataset/                Data placement instructions
examples/               Ready-to-run command examples
```

Datasets, regulatory resources, checkpoints, and manuscript result files are
not bundled in this code-only repository.

## Installation

The reference environment uses Python 3.10, PyTorch 2.2.1, and CUDA 12.1.

```bash
git clone https://github.com/sesh081421/TRACE.git
cd TRACE

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The PyG extension wheels in `requirements.txt` target CUDA 12.1. For a
different CUDA version or CPU-only execution, install matching PyTorch, PyG,
`torch-scatter`, and `torch-sparse` builds before installing TRACE.

## Data

Open the shared [Google Drive folder](https://drive.google.com/drive/folders/1WmWXiD0VyNyD_Bd8IkFWRRWi5N1LAsDp?usp=drive_link)
and select its `dataset/` subfolder. Data uploads are managed separately from
this code repository. Data are distributed as ZIP archives, not as individual
uncompressed files. Download one archive:

| Archive | Download size | Contents |
|---|---:|---|
| `TRACE_example_data_BM.zip` | 92.7 MB | BM inputs for the quick example |
| `TRACE_preprocessed_data.zip` | 3.51 GB | All six configured datasets, including BM |

Both include the relevant gene/TF indices, TFLink prior, positive-edge splits
and frozen evaluation candidates. The full archive includes BM, so downloading
both is unnecessary. It requires approximately 9.6 GB when extracted.

From the repository root, extract the downloaded archive, adjusting its path
to the download location. Run **one** of these commands:

```bash
unzip ../TRACE_example_data_BM.zip -d .
```

```bash
unzip ../TRACE_preprocessed_data.zip -d .
```

The archives already contain a `dataset/` directory. Do not extract them into
`dataset/` again, which would create `dataset/dataset/`. When extracting with a
desktop archive manager, place the resulting `dataset/` folder directly inside
`TRACE/`. Run the training commands from `TRACE/`, not from inside `dataset/`.
Another data location can be supplied with `--data_root`. The expected layout is:

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

See [dataset/README.md](dataset/README.md) for formats, generated split files,
and input assumptions.

## Run TRACE

After installing the environment and extracting the BM or full data archive,
run the existing quick example from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/run_quick_check.sh
```

This runs three training epochs on BM. It checks execution, not final paper
performance. The full archive is required for `examples/run_all_datasets.sh`.

Run one dataset and one optimization seed:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_trace.py \
  --data GSE150774_RAW/BM \
  --benchmark false \
  --data_root ./dataset \
  --result_dir ./results \
  --seed 8
```

Run all six configured datasets and five seeds:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep.py \
  --datasets all \
  --seeds 8,18,28,38,48 \
  --data_root ./dataset \
  --result_dir ./results \
  --skip_existing
```

Run a short smoke check after preparing one dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_trace.py \
  --data GSE150774_RAW/BM \
  --benchmark false \
  --data_root ./dataset \
  --result_dir ./results_quick \
  --epochs 3 \
  --seed 8
```

The same commands are installed as `trace-train`, `trace-sweep`, and
`trace-summarize`.

Use a different `--result_dir` when changing the evaluation ratio or training
configuration. In this release, `--skip_existing` checks whether the metrics
file exists; it does not validate the saved configuration against the new command.

## Outputs

Each run writes a checkpoint and metrics JSON:

```text
results/
  BM/
    seed_8/
      TRACE_BM_seed_8_AUC_..._AUPRC_..._PerTF_....pt
      TRACE_BM_seed_8_metrics.json
```

The JSON records the complete configuration, selected threshold, validation
and test metrics, candidate-background audit, and training-negative
pair-identity audit. Aggregate completed runs with:

```bash
python scripts/summarize.py --result_dir ./results
```

This writes `seed_sweep_raw.csv` and `seed_sweep_summary.csv`.

## Model and Protocol

- Encoder: two topology-aware graph-attention layers
- Decoder: bilinear ordered TF-target scorer
- Output dimension: `64`
- Topology filtering: top-k `50`, Jaccard threshold `0.15`
- Training negative ratio: `1:1`, sampled globally
- Validation background: fixed `1:1`
- Test background: fixed `1:10`
- Candidate-background seed: `20260821`
- Checkpoint criterion: validation AUPRC
- Learning rate: `0.0001331512668153493`
- Dropout: `0.1`
- Maximum epochs: `100`, early-stopping patience `10`

Optimization seeds affect model fitting, while the evaluation seed fixes the
candidate background. Validation and test negatives are mutually exclusive,
exclude all annotated positives, and are also forbidden during training-negative
sampling.

## Data and Code Availability

Public source datasets and regulatory resources are cited in the manuscript.
The TRACE code repository is [https://github.com/sesh081421/TRACE](https://github.com/sesh081421/TRACE)
and contains the model, environment specifications, data-layout documentation,
and training/evaluation commands. The separate distribution location for
prepared inputs is the `dataset/` subfolder of the shared
[Google Drive folder](https://drive.google.com/drive/folders/1WmWXiD0VyNyD_Bd8IkFWRRWi5N1LAsDp?usp=drive_link).
The prepared archives include the inputs, prior, indexed gene/TF lists,
positive-edge splits and frozen evaluation candidates described above.

This code release and its example-data archives do not include trained
checkpoints, per-run manuscript metrics, figure source tables, or the complete
manuscript analysis workflow. No archival DOI is assigned in this release.

## License

TRACE is released under the MIT License. See [LICENSE](LICENSE). Source data
and regulatory resources remain subject to their original terms.

## Citation

The manuscript citation will be added after publication. Until then, cite the
repository and report the release tag or commit used.
