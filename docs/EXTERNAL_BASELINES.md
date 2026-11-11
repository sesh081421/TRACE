# Baseline setup

Set up each comparison method using the implementation cited in the manuscript
and its installation instructions. Keep method-specific dependencies in a
separate environment when they conflict with the TRACE environment.

| Method | Setup |
| --- | --- |
| GENELink | Use the cited GENELink implementation and its required environment. |
| GRLGRN | Use the cited GRLGRN implementation and its required environment. |
| GENIE3 | Install Arboreto for its random-forest implementation. |
| GRNBoost2 | Install Arboreto for its gradient-boosting implementation. |
| PANDA | Install netZooPy and prepare the expression, regulatory-prior, and protein-interaction inputs. |
| Random ranking | Assign seeded scores to the same candidate pairs used for the model comparison. |

For method comparisons, use shared positive splits, candidate identities,
reference labels, and metric definitions. Record any method-specific adapter,
environment, or checkpoint-selection rule with the run. Report sampled and
complete-candidate evaluations separately. The manuscript and Supplementary
Information specify the settings used in the reported comparisons.
