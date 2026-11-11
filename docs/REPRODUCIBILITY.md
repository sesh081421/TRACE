# Training and evaluation

## Running the training tools

Start with the [single-run validation example](../README.md#train-trace).
For a full training run, retain the metrics JSON, checkpoint, and generated
dataset artifacts together. In the direct training CLI, `--eval_neg_ratio` sets the test
background-to-positive ratio and also the validation ratio unless
`--validation_neg_ratio` is set separately. The `--neg_ratio` argument
controls dynamic training sampling. The training example builds neighborhoods from training positives
and excludes annotated positives when sampling training backgrounds. The direct
CLI also needs `--exclude_eval_candidates_from_train_negatives` for strict
candidate exclusion, as in the README command; the released wrapper and sweep
enable this rule automatically.
`--selection_metric AUPRC` selects the checkpoint by pooled validation AP.

Run `python scripts/run_sweep.py --help` from the repository root to configure
dataset and seed lists. The sweep uses the released TRACE preset, with separate
validation and test background ratios, and excludes fixed evaluation
candidates from training-negative sampling. Generate the dataset artifacts
before launching paired runs and preserve them across optimization seeds.
Use a separate data root when intentionally generating a fresh split or
candidate set. The [model preset](../configs/README.md) lists the model and
optimizer arguments used in the single-run example. For the full set of
training options, use `python src/trace_tf_target/train.py --help`.

`--validation_only` writes `test_metrics: null` and no checkpoint. Summarize
those runs with `python scripts/summarize.py --result_dir <path>
--metric_split validation`; this creates `seed_sweep_validation_raw.csv` and
`seed_sweep_validation_summary.csv`. The default summarizer selects test metrics
only and skips validation-only records. Validation and test outputs are never
pooled into one summary.

`--skip_existing` skips a run only when its metrics file contains nonempty test
metrics. It does not compare all saved hyperparameters or verify checkpoint
contents; use a new result directory when changing settings. A validation-only
file is not a completed test run and will be replaced by a full run unless you
choose a separate result directory.

## Evaluation designs in the manuscript

The descriptions below document the study designs. This model-only release
does not include every analysis runner or comparator implementation.

- **Primary fixed-candidate benchmark:** Partition annotated positive pairs
  into training, validation, and test sets (80:10:10). Use fixed and mutually
  disjoint validation and test candidate sets, with one unannotated candidate
  per positive for validation and ten for test. In the strict supervised
  protocol, exclude the union of validation and test candidate pairs from
  training-negative sampling. Reuse the same positive splits and candidate
  sets across methods and optimization seeds (Methods and Supplementary Notes
  1–2).
- **Complete-candidate analyses:** Rank directed non-self TF–target pairs
  after removing training and validation positives. The all-method comparison
  and the matched-checkpoint comparisons use distinct training protocols;
  record the checkpoint and candidate pool for each analysis. For the
  shared-unexposed analysis, additionally exclude validation candidate
  identities and the union of backgrounds sampled in training through the
  selected checkpoints (Methods and Supplementary Notes 4, 6, and 10).
- **Source-TF holdout:** Assign each source TF with annotated targets to a
  training, validation, or test role. Use only training sources for outgoing
  graph-prior edges, supervision, and sampled training negatives. Rank the
  complete non-self candidate set for the held-out source role with shared
  candidate identities and reference labels across methods. This evaluates
  transfer to a held-out regulator role within the same retained gene set
  (Methods and Supplementary Note 11).

For matched architecture and graph controls, use the condition definitions in
Supplementary Note 6. Pair comparisons on the same datasets and seeds, with
the same evaluation candidates and checkpoint rule unless the comparison
specifically varies that rule.

## Reporting

Identify the candidate pool, split, sampling rules, and checkpoint criterion
with each result. Report pooled AP across pairs separately from macro per-TF
AP and shortlist metrics. For the paper's primary benchmark, average the five
optimization runs within each dataset, then report the equal-weight mean and
sample standard deviation across datasets. Paired tests use dataset summaries;
the five-study sensitivity analysis combines BM and UCB before testing.
