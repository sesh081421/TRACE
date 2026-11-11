# Baseline setup

External comparison methods are not bundled with TRACE. Follow each method's
own installation instructions. Keep method-specific dependencies in a
separate environment when they conflict with the TRACE environment.

| Method | Setup |
| --- | --- |
| GENELink | Install GENELink and its required environment separately. |
| GRLGRN | Install GRLGRN and its required environment separately. |
| GENIE3 | Install Arboreto for its random-forest implementation. |
| GRNBoost2 | Install Arboreto for its gradient-boosting implementation. |
| PANDA | Install netZooPy and prepare the expression, regulatory-prior, and protein-interaction inputs. |
| Random ranking | Assign seeded scores to the same candidate pairs used for the model comparison. |

These packages are optional and are not required to install or run TRACE.
