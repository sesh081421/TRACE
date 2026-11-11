# Training And Outputs

Run commands from the repository root after installing dependencies and
extracting the prepared data. The default data root is `./dataset`.

## Common Options

| Option | Purpose |
| --- | --- |
| `--data` | Dataset ID for a single run |
| `--datasets` | Comma-separated dataset IDs or `all` for the sweep |
| `--seed` / `--seeds` | Optimization seed or comma-separated sweep seeds |
| `--data_root` | Prepared data directory |
| `--result_dir` | Output directory |
| `--epochs` | Maximum training epochs |
| `--batch_size` | Positive pairs per training batch |
| `--patience` | Early-stopping patience |
| `--validation_neg_ratio` | Validation background-to-positive ratio |
| `--eval_neg_ratio` | Test background-to-positive ratio |

The wrappers use validation ratio 1 and test ratio 10 by default. They select
the checkpoint by validation average precision and exclude fixed evaluation
candidate pairs from training-negative sampling. Existing split and candidate
files are reused. Preserve those files when repeating runs.

The direct `src/trace_tf_target/train.py` CLI exposes additional options. With
that CLI, set `--validation_neg_ratio` explicitly when it differs from the test
ratio, and pass `--exclude_eval_candidates_from_train_negatives` to exclude
evaluation candidates during training.

## Results

Each run saves a metrics JSON and model checkpoint in
`<result_dir>/<short_dataset_name>/seed_<seed>/`. The metrics include configuration,
validation/test metrics, feature metadata, and graph information.

```bash
python scripts/summarize.py --result_dir ./results_all
```

This writes `seed_sweep_raw.csv` and `seed_sweep_summary.csv`. The CLI metric
name `AUPRC` denotes non-interpolated average precision.

The sweep's `--skip_existing` option skips runs with nonempty test metrics. It
does not compare all saved settings or checkpoint contents. Use a new result
directory when changing inputs, hyperparameters, or evaluation settings.

## Validation-Only Runs

The direct training CLI supports `--validation_only`. This mode writes
validation metrics but no checkpoint or test metrics. Summarize it separately:

```bash
python scripts/summarize.py --result_dir ./results_validation --metric_split validation
```

Outputs are `seed_sweep_validation_raw.csv` and
`seed_sweep_validation_summary.csv`. Validation and test results are not mixed.
