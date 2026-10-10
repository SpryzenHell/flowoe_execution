# FlowOE A100 Real-Execution Sweep #5

- Workflow: [FlowOE Real Execution Sweep #5](https://github.com/SpryzenHell/integrator/actions/runs/38078913959)
- Result artifact: `flowoe-real-execution-sweep-results` (artifact ID `11679842767`)
- Runner workflow commit: `10e8d22405317720a878fa4db38fd32d0e592e5f`
- Completed: 2026-10-10 19:34 UTC
- Outcome: workflow succeeded; all six train/evaluation runs and both latency sweeps completed.
- Source revision limitation: this run's manifest did not capture the exact FlowOE checkout SHA. Do not infer it from the current branch. Later runs now record both source revisions.

## A100 setup and tests

| Item | Recorded value |
|---|---|
| Host | `scn11-mn.mgmt.siddhi.param` |
| GPU | NVIDIA A100-SXM4-40GB, compute capability 8.0 |
| Driver / CUDA | 550.90.07 / 12.4.131 |
| Python / PyTorch | 3.11.17 / 2.5.1+cu124 |
| Ninja | `.venv/bin/ninja`; found by PyTorch extension loader |
| CUDA architecture | `TORCH_CUDA_ARCH_LIST=8.0` |
| FlowOE CUDA/PyTorch tests | 103 passed, 0 failures; 92% coverage |
| Data-prep/TensorRT compatibility unit tests | 35 passed |
| CUDA fused extension | Compiled and available |

This confirms the earlier “Ninja is required” setup failure is fixed; the CUDA extension tests were not removed or skipped to make the workflow green.

## Input data validation

### FI-2010

- Exact file: `Train_Dst_NoAuction_DecPre_CF_7.txt` (**DecPre**, not ZScore)
- Size: 607,324,298 bytes
- SHA-256: `11af4bbcf26f08a43a436c43619bdf772f552870d54a3b06de7b5344957279fc`
- 50,000 rows loaded for the experiment; 144 features, five label horizons; sample values finite and labels 1–3.

### BTCUSDT futures L2 and trades

- Public Binance USD-M futures, 2026-04-07 07:09:32.415 to 19:09:22.415 UTC.
- Target window: 12 hours; observed first-to-last snapshot span: 43,190 seconds (11 h 59 min 50 s).
- 8,639 reconstructed L2 snapshots at a 5-second target spacing. One target sample is missing; maximum observed gap was 5,442 ms.
- 895,669 downloaded trade rows; 895,566 inside the L2 evaluation window.
- 410,004 depth-update sequence links checked; the selected update-ID chain had no gaps; full-book depth payload detected.
- L2 SHA-256: `991c02e644fc84ff345100c1c9ce121543afa678c834e468ebcb40a2d68fa1a9`
- Trade SHA-256: `8eb72f1a5426aeb49e78ad5b6416d4c3d8269ec154713c42f544642558695dd9`

The independent reconstruction/coverage checks passed. The data is historical market data; execution fills and slippage below are simulator outputs, not live orders.

## Real-data execution evaluation

Configuration: buy and sell; order size 0.5 BTC; context length 32; horizon 8; training uses the first 60% of the time series and evaluation the final 40%; 20 epochs; seeds 7, 17 and 27; RK4 and fixed Euler with 2, 3, 4 or 6 steps. All six seed/side training jobs and evaluations passed. There were 144 non-overlapping evaluation windows per seed and side, reused across seed fits; these are 144 distinct time windows, not 432 independent market windows.

The table below reports signed slippage against trade VWAP. Positive improvement means the strategy has lower signed slippage than TWAP. The paired 95% intervals in the last column were recomputed from the saved per-episode results by averaging each shared timestamp's values across the three model seeds, then applying a paired circular moving-block bootstrap (5 consecutive episodes per block, 5,000 resamples, bootstrap seed 7). They describe uncertainty over the single held-out time window conditional on these trained seeds, not variation across market days.

| Side | Method | TWAP mean (bps) | Policy mean (bps) | Improvement vs TWAP (bps) | Pooled paired 95% CI (bps) | Mean completion |
|---|---|---:|---:|---:|---|---:|
| Buy | Fixed Euler 2 | -0.067921 | -0.073616 | +0.005695 | [-0.019984, +0.035846] | 99.8375% |
| Buy | Fixed Euler 3 | -0.067921 | -0.074933 | +0.007012 | [-0.003108, +0.018631] | 99.8792% |
| Buy | Fixed Euler 4 | -0.067921 | -0.074450 | +0.006530 | [-0.006819, +0.022189] | 99.8777% |
| Buy | Fixed Euler 6 | -0.067921 | -0.073986 | +0.006065 | [-0.010421, +0.025262] | 99.8757% |
| Buy | RK4 | -0.067921 | -0.073220 | +0.005299 | [-0.015657, +0.028869] | 99.8712% |
| Sell | Fixed Euler 2 | +0.083079 | +0.071308 | +0.011772 | [-0.012407, +0.038125] | 99.8050% |
| Sell | Fixed Euler 3 | +0.083079 | +0.078846 | +0.004234 | [-0.004974, +0.014925] | 99.8605% |
| Sell | Fixed Euler 4 | +0.083079 | +0.078747 | +0.004332 | [-0.009991, +0.020207] | 99.8538% |
| Sell | Fixed Euler 6 | +0.083079 | +0.078674 | +0.004406 | [-0.013714, +0.024130] | 99.8474% |
| Sell | RK4 | +0.083079 | +0.078506 | +0.004574 | [-0.018580, +0.029351] | 99.8387% |

**Research conclusion:** mean improvements were only about 0.004–0.012 bps, and every pooled interval includes zero. Improvements also vary by seed. This does not establish a repeatable execution benefit, and it is far below the 2.4 bps target. More non-overlapping market dates and out-of-sample windows are needed.

## Inference latency and schedule quality

The full model latency grid and the fixed-step quality sweep used a **synthetic smoke checkpoint and synthetic L2 input**, not a checkpoint trained on the real futures window. Timings are therefore synthetic-model inference measurements, not real-market performance evidence.

The full grid tested batch sizes 1, 8, 32 and 128 with RK4 counts 8, 16, 24 and 32; 50 warmups and 300 measured calls per configuration. On batch 1, RK4 end-to-end wall p99 ranged from 11.86 ms (8 steps) to 40.47 ms (32 steps). Four-step Euler end-to-end p99 was 1.12–1.23 ms across those grid configurations.

The separate fixed-step sweep compared Euler to a 64-step RK4 schedule reference with 50 warmups and 300 measurements:

| Batch | Euler steps | Wall p50 (ms) | Wall p95 (ms) | Wall p99 (ms) | Mean abs. schedule error vs RK4 | Mean KL vs RK4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.584 | 0.622 | 0.739 | 0.051663 | 0.141340 |
| 1 | 3 | 0.814 | 0.874 | 1.004 | 0.014701 | 0.010844 |
| 1 | 4 | 1.090 | 1.203 | 1.741 | 0.008943 | 0.003933 |
| 1 | 6 | 1.461 | 1.617 | 1.942 | 0.005020 | 0.001228 |
| 1 | 8 | 1.928 | 2.117 | 2.560 | 0.003487 | 0.000593 |
| 8 | 2 | 0.542 | 0.588 | 0.666 | 0.049002 | 0.121440 |
| 8 | 3 | 0.790 | 0.838 | 0.963 | 0.014728 | 0.010300 |
| 8 | 4 | 1.017 | 1.089 | 1.390 | 0.008844 | 0.003724 |
| 8 | 6 | 1.479 | 1.590 | 1.927 | 0.004946 | 0.001160 |
| 8 | 8 | 1.935 | 2.096 | 2.156 | 0.003437 | 0.000560 |

Two-step Euler is the fastest here but has materially larger schedule error. Increasing steps improves schedule similarity but increases latency. The fused CUDA extension only fuses the Euler update, not the vector-field network: maximum observed fused-vs-PyTorch output error was `2.98e-8`; mean wall-speedup ranged from 0.943x to 1.202x across tested configurations, so it is not an unconditional win. The latency and accuracy measurements come from different benchmark loops; p99 outliers were retained.

## TensorRT INT8 result

- The compatibility report confirms the INT32/INT64 shape-concatenation cast was applied (`shape_concat_dtype_changed=true`).
- TensorRT 10.5.0 built a loadable engine and the benchmark executed it, using 32 calibration samples from synthetic smoke L2 data and a synthetic smoke checkpoint.
- Timing used 50 warmups and 300 measured calls each at batches 1 and 4.
- The original benchmark compared matching input tensors, but only one batch-1 and one batch-4 input set. The maximum across those comparisons exceeded the already permissive error threshold of 0.05, so the accuracy gate failed.

| Batch | Runtime | Wall p50 (ms) | Wall p95 (ms) | Wall p99 (ms) | CUDA-event p50 (ms) | CUDA-event p95 (ms) | CUDA-event p99 (ms) | Max abs output error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | PyTorch fixed-step | 0.8580 | 0.8816 | 1.0035 | — | — | — | 0.000257 vs TRT |
| 1 | TensorRT INT8 | 0.3942 | 0.4079 | 0.4516 | 0.3686 | 0.3809 | 0.3874 | 0.000257 |
| 4 | PyTorch fixed-step | 0.8690 | 0.8987 | 1.0891 | — | — | — | 0.084224 vs TRT |
| 4 | TensorRT INT8 | 0.3960 | 0.4067 | 0.4255 | 0.3707 | 0.3809 | 0.4005 | 0.084224 |

Overall maximum absolute output error was 0.084224, above the allowed 0.05. The batch-4 TensorRT timing also had a 4.31 ms maximum outlier despite its reported p99 of 0.426 ms. Raw timing ratios are not accepted as an optimization because accuracy failed; the benchmark correctly recorded `latency_claim_allowed=false`. Neither the INT8 accuracy target nor the combined TensorRT/fused sub-2 ms target is established.

The follow-up A100 workflow now validates 32 distinct contexts/noise inputs at both batch sizes, reports mean/max absolute error and KL divergence, and uses a stricter 0.02 maximum-error threshold. It will test the checkpoint produced by the real-data workflow rather than accepting this smoke-only result.

## Artifacts and files

The downloadable artifact for the run contains `REAL_EXECUTION_SWEEP_SUMMARY.md`, all six training/evaluation JSON reports, the per-episode outputs, full latency grids, data hashes/reconstruction reports, test logs, the TensorRT patch report, engine build report and TensorRT timing JSON. Raw data, checkpoints and engine files remain in the workflow artifact rather than in Git.
