# Verified analysis snapshot — CI run #136

This page records the output of the FlowOE analysis suite from GitHub Actions run #136. The run completed successfully on the project branch.

The experiment was synthetic-only. It is useful for regression and model-development diagnostics and must not be interpreted as the project's real-market performance.

## Run configuration

| Setting | Value |
|---|---:|
| Synthetic regimes | 4 |
| Rows per regime | 1,200 |
| CFM seed fits | 8 |
| Schedule/regime evaluations | 20 |
| Sampling-step settings | 6 |
| Quantity settings | 5 |
| L2 depth settings | 4 |
| Context-length settings | 3 |
| Side settings | 2 |
| Model comparison | CFM / PF-ODE |

## Regime diagnostics

| Regime | Median spread (bps) | p95 spread (bps) | One-step vol. (bps) | Median bid depth | Median ask depth |
|---|---:|---:|---:|---:|---:|
| Trend | 3.5621 | 4.2650 | 2.2015 | 7.9431 | 7.9567 |
| Volatile | 5.4577 | 7.2987 | 10.2694 | 7.9429 | 7.9111 |
| Mean-revert | 3.5795 | 4.2668 | 3.5662 | 7.9191 | 7.9044 |
| Thin-book | 3.5526 | 4.2352 | 3.1389 | 3.5476 | 3.5677 |

## Seed robustness

| Regime | Mean improvement (bps) | Minimum | Maximum | Positive-seed rate |
|---|---:|---:|---:|---:|
| Trend | -0.1932 | -0.2856 | -0.1009 | 0% |
| Volatile | -0.0196 | -0.2423 | 0.2030 | 50% |
| Mean-revert | -0.2707 | -0.2854 | -0.2560 | 0% |
| Thin-book | 0.4289 | 0.4099 | 0.4479 | 100% |

The trend-regime five-seed bootstrap interval for mean improvement was -0.2856 to -0.1009 bps.

## Sampling-step sensitivity

| Steps | FlowOE mean (bps) | TWAP mean (bps) | Difference (bps) |
|---:|---:|---:|---:|
| 4 | 1.6580 | 1.8613 | 0.2032 |
| 8 | 1.6495 | 1.8613 | 0.2118 |
| 12 | 1.8964 | 1.8613 | -0.0352 |
| 16 | 1.8022 | 1.8613 | 0.0591 |
| 24 | 1.4331 | 1.8613 | 0.4281 |
| 32 | 1.6574 | 1.8613 | 0.2038 |

## Order-size sensitivity

| Quantity | FlowOE mean (bps) | TWAP mean (bps) | Difference (bps) | Completion |
|---:|---:|---:|---:|---:|
| 0.25 | 1.3762 | 1.7045 | 0.3283 | 1.000 |
| 0.50 | 1.3762 | 1.7045 | 0.3283 | 1.000 |
| 1.00 | 1.3768 | 1.7045 | 0.3276 | 1.000 |
| 2.00 | 1.4112 | 1.7060 | 0.2948 | 1.000 |
| 4.00 | 1.5581 | 1.7833 | 0.2253 | 1.000 |

## L2 depth sensitivity

| Available levels | FlowOE mean (bps) | TWAP mean (bps) | Difference (bps) | Completion |
|---:|---:|---:|---:|---:|
| 1 | 1.8151 | 1.8223 | 0.0073 | 0.187 |
| 3 | 2.2895 | 2.2487 | -0.0407 | 0.473 |
| 5 | 2.6608 | 2.6618 | 0.0010 | 0.626 |
| 10 | 2.8347 | 2.9726 | 0.1379 | 0.827 |

## Context-length sensitivity

| Context | FlowOE mean (bps) | TWAP mean (bps) | Difference (bps) |
|---:|---:|---:|---:|
| 16 | 1.6774 | 1.7045 | 0.0271 |
| 32 | 1.2698 | 1.7045 | 0.4347 |
| 64 | 2.0214 | 1.7045 | -0.3169 |

## Buy / sell check

| Side | FlowOE mean (bps) | TWAP mean (bps) | Difference (bps) |
|---|---:|---:|---:|
| Buy | 1.2698 | 1.7045 | 0.4347 |
| Sell | 2.3368 | 1.9210 | -0.4158 |

This shows why the analysis suite reports sides separately rather than assuming symmetry.

## CFM vs probability-flow ODE

| Model | Mean slippage (bps) | Median slippage (bps) | p95 slippage (bps) | Difference vs TWAP (bps) |
|---|---:|---:|---:|---:|
| CFM | 2.7667 | 2.4259 | 8.6095 | -0.3920 |
| PF-ODE | 3.3757 | 0.6075 | 21.1691 | -1.0009 |

This is one synthetic volatile-regime comparison. It is a diagnostic rather than a model-selection claim.

## Figures

The repository includes readable vector figures derived from the same CI snapshot:

- [Seed robustness](../../assets/seed_robustness_ci136.svg)
- [Sampling-step sensitivity](../../assets/sampling_steps_ci136.svg)
- [Order-size sensitivity](../../assets/quantity_sensitivity_ci136.svg)
- [Context length](../../assets/context_sensitivity_ci136.svg)
- [Buy / sell symmetry](../../assets/side_symmetry_ci136.svg)
- [CFM vs PF-ODE](../../assets/cfm_pf_ci136.svg)

The CI job also produced the raw PNG analysis bundle as a workflow artifact.
