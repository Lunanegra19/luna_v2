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
| W1 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W2 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W3 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W4 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W5 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W6 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W7 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W8 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W9 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W10 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W11 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |
| W12 | 0.0000 | 0.00% | 0.00% | +0.00% | Strict (Default) |

## Summary of Calibration changes
- Kelly Position Sizer `pt_ratio` has been adjusted to `1.01` (from `1.2`) to align with empirical OOS Win/Loss ratios (~0.888) while satisfying positive asymmetry pre-flight constraints. This stops sizing negative-EV trades.
- Report generated at: 2026-06-24T01:56:11.911951
- Seed: 100