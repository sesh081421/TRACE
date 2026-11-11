# TRACE

**Transcription-factor-aware Regulatory Architecture for Candidate-Edge prioritization**

TRACE scores directed TF-target candidate pairs using gene features and a
training-derived regulatory graph.

## Installation

Use Python 3.10 on Linux. The requirements pin PyTorch 2.2.1 and CUDA 12.1
builds of the PyTorch Geometric extensions; a compatible NVIDIA driver is
required for GPU execution.

```bash
git clone https://github.com/sesh081421/TRACE.git
cd TRACE
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The install provides `trace-train`, `trace-sweep`, and `trace-summarize`.
The `scripts/` and `examples/` entry points also work directly from a checkout
once dependencies are installed. For CPU-only execution, install matching CPU
builds of PyTorch and the PyTorch Geometric extensions instead of CUDA builds.

## Data

Download one archive from the `dataset/` subfolder of the
[data download folder](https://drive.google.com/drive/folders/1WmWXiD0VyNyD_Bd8IkFWRRWi5N1LAsDp?usp=drive_link):

- `TRACE_example_data_BM.zip`: BM data for the quick example.
- `TRACE_preprocessed_data.zip`: all six datasets.

Extract the archive from the repository root, adjusting the download path:

```bash
unzip ../TRACE_preprocessed_data.zip -d .
```

The archive already contains `dataset/`. Do not extract it into `dataset/`
again. The prior file should be at `dataset/prior_network/tflink.h5`.

| Datasets | Default feature input |
| --- | --- |
| BM, UCB, hESC, hHep | RNA |
| PBMC, lymphoma | RNA+GAS |

GAS denotes gene-activity scores. See [dataset setup](dataset/README.md) for
file paths and [data formats](docs/DATA.md) for identifier and matrix requirements.

## Quick Check

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/run_quick_check.sh
python scripts/summarize.py --result_dir ./results_quick
```

This runs three epochs on BM and writes a checkpoint and metrics. It is an
execution check, not a full training run. Set `PYTHON_BIN` to use a different
Python interpreter.

## Train

For a PBMC run, prepare the full data archive and run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_trace.py \
  --data pbmc_granulocytes --data_root ./dataset \
  --result_dir ./results_pbmc_rna_gas \
  --epochs 100 --batch_size 128 --patience 10 --seed 8
```

Other dataset IDs are `GSE150774_RAW/BM`, `GSE150774_RAW/UCB`, `lymph`,
`hESC`, and `hHep`. The wrapper selects the corresponding input format.
Training uses one sampled background pair per positive by default; validation
uses a 1:1 ratio and test evaluation uses 1:10. Choose a GPU with
`CUDA_VISIBLE_DEVICES` and a separate output directory for each configuration.

To run all datasets with five seeds:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_sweep.py \
  --datasets all --seeds 8,18,28,38,48 \
  --data_root ./dataset --result_dir ./results_all --skip_existing
python scripts/summarize.py --result_dir ./results_all
```

## Outputs And Options

Runs save model checkpoints and metrics JSON under
`<result_dir>/<dataset>/seed_<seed>/`. The summarizer writes
`seed_sweep_raw.csv` and `seed_sweep_summary.csv`.

```bash
python scripts/run_trace.py --help
python scripts/run_sweep.py --help
python src/trace_tf_target/train.py --help
```

See [training options](docs/REPRODUCIBILITY.md), the
[model preset](configs/README.md), and [input and graph options](docs/INPUT_AND_GRAPH_CONTROLS.md).
This repository contains the TRACE model and its training tools; external
comparison-method runners and manuscript analysis pipelines are not included.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

Code is released under the [MIT License](LICENSE). Datasets and regulatory
resources retain their original terms.
