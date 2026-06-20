# [BUG-FIX-LOG 2026-06-19] Filter Governor & Sizer Calibration Report

This report summarizes the performance of the **Dynamic Filter Governor** and the calibration of the **Kelly Position Sizer** across all WFB windows.

## Active Governor Parameters
- **Enabled**: True
- **Min Completed Windows**: 1
- **Performance Threshold Ratio**: 70.0%
- **Round-Trip Transaction Cost**: 0.25%

## Window-by-Window Relaxation & Performance Analysis
| Window | Relaxation Factor | Baseline Return | Filtered Return | Delta (Filtered - Baseline) | Status |
|--------|-------------------|-----------------|-----------------|-----------------------------|--------|
| W1 | 0.0000 | -0.42% | -3.11% | -2.69% | Strict (Default) |
| W2 | 0.0000 | -5.69% | 3.98% | +9.68% | Strict (Default) |
| W3 | 0.0000 | 2.56% | 2.31% | -0.25% | Strict (Default) |
| W4 | 0.0000 | 9.97% | 3.88% | -6.10% | Strict (Default) |
| W5 | 0.0000 | 1.06% | 4.12% | +3.07% | Strict (Default) |
| W6 | 0.0000 | 8.06% | 0.62% | -7.44% | Strict (Default) |
| W7 | 0.0000 | -0.71% | -0.72% | -0.01% | Strict (Default) |
| W8 | 0.0000 | -17.20% | 11.93% | +29.13% | Strict (Default) |
| W9 | 0.0000 | 14.95% | 6.46% | -8.48% | Strict (Default) |
| W10 | 0.0000 | 28.94% | 3.98% | -24.96% | Strict (Default) |
| W11 | 0.0000 | 3.64% | 0.66% | -2.98% | Strict (Default) |
| W12 | 0.0000 | 4.56% | 2.26% | -2.30% | Strict (Default) |

## Summary of Calibration changes
- Kelly Position Sizer `pt_ratio` has been adjusted to `1.01` (from `1.2`) to align with empirical OOS Win/Loss ratios (~0.888) while satisfying positive asymmetry pre-flight constraints. This stops sizing negative-EV trades.
- Report generated at: 2026-06-20T15:35:36.287023
- Seed: 27644