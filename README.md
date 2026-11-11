# TRACE

TRACE (Transcription-factor-aware Regulatory Architecture for Candidate-Edge
prioritization) prioritizes directed transcription factor (TF)–target
interactions from single-cell gene profiles and a training-derived regulatory
graph. A two-layer graph attention encoder aggregates neighborhoods derived from
training TFLink interactions, and an ordered bilinear decoder scores candidate
pairs. Scores support network-wide ranking and target shortlists for individual
TFs.

This repository accompanies *TRACE: graph-informed learning for network-wide
regulatory interaction prioritization* by Sanghun Sel, Subin Lee, and Giltae Song.

## Installation

Use Python 3.10 with the dependencies in
[`requirements.txt`](requirements.txt). This file pins PyTorch 2.2.1 and
CUDA 12.1 builds of the PyTorch Geometric extension packages.

```bash
git clone https://github.com/sesh081421/TRACE.git
cd TRACE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The editable install enables `trace-train`, `trace-sweep`, and
`trace-summarize`. The `scripts/` and `examples/` entry points can also run
directly from a checkout without that step, once dependencies are installed.

For CPU-only use, choose matching CPU builds of PyTorch and the PyTorch
Geometric extension packages instead of the CUDA-specific packages in the
requirements file. See the [baseline guide](docs/EXTERNAL_BASELINES.md) for
comparison-method setup.

## Prepared Data

Download `TRACE_example_data_BM.zip` for BM, or `TRACE_preprocessed_data.zip`
for all six datasets, from the `dataset/` subfolder of the shared
[Google Drive folder](https://drive.google.com/drive/folders/1WmWXiD0VyNyD_Bd8IkFWRRWi5N1LAsDp?usp=drive_link).
From the repository root, extract one archive, adjusting its download path:

```bash
unzip ../TRACE_example_data_BM.zip -d .
```

The archive already contains `dataset/`. Do not extract it into `dataset/`
again. The prior must be at `dataset/prior_network/tflink.h5`.
See [dataset setup](dataset/README.md) for sizes, formats, and the full layout.

## Train TRACE

[Prepare the expression matrices and TFLink prior](docs/DATA.md), then run
the following example from the repository root. It trains one model on the
bone marrow dataset and reports validation metrics on a balanced candidate set.

```bash
CUDA_VISIBLE_DEVICES=0 python src/trace_tf_target/train.py \
  --data GSE150774_RAW/BM --benchmark false \
  --data_root ./dataset --result_dir ./Result_trace_example \
  --epochs 100 --batch_size 128 --patience 10 --seed 8 \
  --eval_neg_ratio 1 --attention_mode topology_only \
  --encoder_type gat --decoder bilinear --loss_type legacy_focal \
  --selection_metric AUPRC --train_neg_sampling global --neg_ratio 1 \
  --exclude_eval_candidates_from_train_negatives \
  --lr 0.00013315 --weight_decay 0.0 --dropout 0.1 \
  --output_dim 64 --topk 50 --topo_thr 0.15 --validation_only
```

The run writes its best validation metrics to
`Result_trace_example/BM/seed_8/`. To save a model checkpoint, run without
`--validation_only`. The CLI name `AUPRC` refers to pooled, non-interpolated
average precision (AP). Run `python src/trace_tf_target/train.py --help` for
command options.

To summarize the validation-only output without treating it as test performance:

```bash
python scripts/summarize.py --result_dir ./Result_trace_example --metric_split validation
```

For a three-epoch execution check that also saves a checkpoint and test metrics:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/run_quick_check.sh
python scripts/summarize.py --result_dir ./results_quick
```

The quick example uses validation 1:1 and test 1:10; it is an execution check,
not a reproduction of final paper performance. Set `PYTHON_BIN` to select a
different Python interpreter. Use separate result directories for different
protocols or hyperparameters.

## Evaluation

The [evaluation guide](docs/REPRODUCIBILITY.md) describes how to reuse dataset
artifacts across runs and distinguishes the example above from the paper's
fixed-candidate benchmark, complete-candidate evaluation, and source-TF holdout.
See the [model preset](configs/README.md) for model and optimizer settings.
These protocol descriptions do not imply that all manuscript analyses can be
regenerated from this model-only release; baseline and source-TF-holdout runners
are not bundled here.

| Path | Contents |
| --- | --- |
| [`src/trace_tf_target/`](src/trace_tf_target/) | Model, training, evaluation, and control code |
| [`configs/`](configs/) | TRACE model preset |
| [`docs/`](docs/) | Data preparation, evaluation protocols, and baseline setup |
| [`scripts/`](scripts/) | Single-run, multi-dataset, and result-summary entry points |
| [`tests/`](tests/) | Entry-point checks |

## License

TRACE is released under the [MIT License](LICENSE). Refer to the original
licenses for datasets and third-party methods.
