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
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For CPU-only use, choose matching CPU builds of PyTorch and the PyTorch
Geometric extension packages instead of the CUDA-specific packages in the
requirements file. See the [baseline guide](docs/EXTERNAL_BASELINES.md) for
comparison-method setup.

## Train TRACE

[Prepare the expression matrices and TFLink prior](docs/DATA.md), then run
the following example from the repository root. It trains one model on the
bone marrow dataset and reports validation metrics on a balanced candidate set.

```bash
CUDA_VISIBLE_DEVICES=0 python src/trace_tf_target/train.py \
  --data GSE150774_RAW/BM --benchmark false \
  --data_root ./data --result_dir ./Result_trace_example \
  --epochs 100 --batch_size 128 --patience 10 --seed 8 \
  --eval_neg_ratio 1 --attention_mode topology_only \
  --encoder_type gat --decoder bilinear --loss_type legacy_focal \
  --selection_metric AUPRC --train_neg_sampling global --neg_ratio 1 \
  --lr 0.00013315 --weight_decay 0.0 --dropout 0.1 \
  --output_dim 64 --topk 50 --topo_thr 0.15 --validation_only
```

The run writes its best validation metrics to
`Result_trace_example/BM/seed_8/`. To save a model checkpoint, run without
`--validation_only`. The CLI name `AUPRC` refers to pooled, non-interpolated
average precision (AP). Run `python src/trace_tf_target/train.py --help` for
command options.

## Evaluation

The [evaluation guide](docs/REPRODUCIBILITY.md) describes how to reuse dataset
artifacts across runs and distinguishes the example above from the paper's
fixed-candidate benchmark, complete-candidate evaluation, and source-TF holdout.
See the [model preset](configs/README.md) for model and optimizer settings.

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
