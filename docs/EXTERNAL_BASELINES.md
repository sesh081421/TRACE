# Baseline setup

Use the wrapper for the method you want to run and inspect its command-line
options with `--help` after installing its dependencies.

| Method | Entry point | Setup |
| --- | --- | --- |
| GENELink | `trace_tf_target/run_genelink_baseline.py` | Point `--external_dir` to the GENELink checkout. |
| GRLGRN | `trace_tf_target/run_grlgrn_baseline.py` | Point `--external_dir` to the GRLGRN checkout. |
| GENIE3 | `trace_tf_target/run_genie3_baseline.py` | Install Arboreto. |
| GRNBoost2 and PANDA | `trace_tf_target/run_classic_baseline.py` | Install Arboreto and NetZooPy; select the method with `--model`. |
| Random ranking | `trace_tf_target/run_random_baseline.py` | Use the TRACE environment. |

The optional [baseline dependencies](../requirements-baselines.txt) also
support the additional wrappers. For methods that load an upstream checkout,
use the method's own installation instructions and pass its local path to the
wrapper.

For method comparisons, use shared positive splits, candidate identities,
reference labels, and metric definitions. Record any method-specific adapter,
environment, or checkpoint-selection rule with the run. Report sampled and
complete-candidate evaluations separately.
