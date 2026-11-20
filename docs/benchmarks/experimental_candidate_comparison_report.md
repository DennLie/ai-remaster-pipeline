# Experimental Candidate Comparison: Motion-Aware Scene-vs-Film Coordinate Detector (A-E)

- **Timestamp:** `2026-11-19T13:31:59Z`
- **Elapsed:** `982.35s`
- **Constraint Check:** Zero production detector, fusion, preset, restoration, or GUI files modified.

## 1. Unconfounded 2x2x2 Factorial Ablation (`M1_mc_evidence x M2_robust_cut x M3_centerline_geom`, Temporal-Only)

| Config | Mean Pos Recall | Mean Pos F1 | 000 Fast Pan (R / P) | 010 Tilted (R / P) | 010 Diag (R / P) | Static Diag 1f (R / P) | Total Neg Leak (px) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M1=0_M2=0_M3=0` | 0.0% | 0.000 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 684 |
| `M1=0_M2=0_M3=1` | 10.4% | 0.124 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 52.0% / 76.3% | 1148 |
| `M1=0_M2=1_M3=0` | 0.0% | 0.000 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 686 |
| `M1=0_M2=1_M3=1` | 10.4% | 0.124 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 52.0% / 76.3% | 1145 |
| `M1=1_M2=0_M3=0` | 5.2% | 0.081 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0 |
| `M1=1_M2=0_M3=1` | 54.0% | 0.601 | 0.0% / 0.0% | 99.2% / 96.9% | 80.6% / 89.2% | 64.2% / 96.2% | 0 |
| `M1=1_M2=1_M3=0` | 7.4% | 0.120 | 11.0% / 90.6% | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 0 |
| `M1=1_M2=1_M3=1` | 58.8% | 0.681 | 24.0% / 93.1% | 99.2% / 98.5% | 78.8% / 90.4% | 64.2% / 96.2% | 0 |

**Main Effects:**
- `M1_mc_evidence`: `d_recall=+0.2616`, `d_f1=+0.3090`, `d_neg_leak=-915.8 px`
- `M2_robust_cut`: `d_recall=+0.0174`, `d_f1=+0.0299`, `d_neg_leak=-0.2 px`
- `M3_centerline_geom`: `d_recall=+0.3025`, `d_f1=+0.3322`, `d_neg_leak=+230.8 px`

**Two-Way Interactions:**
- `M1 x M2`: `d_recall=+0.0174`, `d_f1=+0.0299`, `d_neg_leak=+0.2 px`
- `M1 x M3`: `d_recall=+0.1985`, `d_f1=+0.2084`, `d_neg_leak=-230.8 px`
- `M2 x M3`: `d_recall=+0.0062`, `d_f1=+0.0103`, `d_neg_leak=-1.2 px`

## 2. Scene-Locked vs. Film-Locked Coordinate Study (Candidate E2)

| Case | ROI px | HighConf (Repair) | Uncertain | Vetoed Intact | Median Flow | Median FB Err |
|---|---:|---:|---:|---:|---:|---:|
| `moving_transient_scratch_010` | 2070 | 2063 | 0 | 0 | 51.90 px | 0.18 px |
| `moving_persistent_film_locked_010` | 2082 | 2082 | 0 | 0 | 23.52 px | 0.45 px |
| `static_persistent_ambiguous_010f5` | 2070 | 0 | 2070 | 0 | 0.12 px | 0.04 px |
| `neighbors_f25_pipe_structure` | 2808 | 0 | 0 | 0 | 0.99 px | 0.17 px |
| `paleface_a_f8_actor_jacket` | 1930 | 0 | 305 | 13 | 0.49 px | 0.09 px |
| `paleface_b_f7_wall_crack` | 8000 | 0 | 26 | 123 | 1.55 px | 0.82 px |

## 3. Paired Clean Reference Collateral (Zero-Defect Sequences)

| Clean Ref | Candidate | Repair FP (px) | Severe FP Comp | Max FP H | Uncertain (px) | Intact Mod MC / Telea (px) |
|---|---|---:|---:|---:|---:|---:|
| `clean_ref_000` | `A1_baseline_temporal` | 0 | 0 | 0 | 0 | 0 / 0 |
| `clean_ref_000` | `A2_baseline_fused_conservative` | 5231 | 60 | 38 | 0 | 4983 / 5175 |
| `clean_ref_000` | `B_solution_1` | 54 | 1 | 18 | 0 | 53 / 51 |
| `clean_ref_000` | `C_solution_2` | 54 | 1 | 18 | 0 | 53 / 51 |
| `clean_ref_000` | `D_solution_3` | 1578 | 5 | 32 | 0 | 1558 / 1564 |
| `clean_ref_000` | `E1_experimental_temporal_only` | 0 | 0 | 0 | 1411 | 0 / 0 |
| `clean_ref_000` | `E2_experimental_fused_3state` | 4 | 0 | 4 | 1411 | 4 / 4 |
| `clean_ref_010` | `A1_baseline_temporal` | 0 | 0 | 0 | 0 | 0 / 0 |
| `clean_ref_010` | `A2_baseline_fused_conservative` | 1980 | 26 | 23 | 0 | 1859 / 1950 |
| `clean_ref_010` | `B_solution_1` | 727 | 10 | 53 | 0 | 673 / 725 |
| `clean_ref_010` | `C_solution_2` | 209 | 3 | 21 | 0 | 201 / 209 |
| `clean_ref_010` | `D_solution_3` | 907 | 3 | 21 | 0 | 883 / 902 |
| `clean_ref_010` | `E1_experimental_temporal_only` | 0 | 0 | 0 | 442 | 0 / 0 |
| `clean_ref_010` | `E2_experimental_fused_3state` | 0 | 0 | 0 | 442 | 0 / 0 |
| `clean_ref_020` | `A1_baseline_temporal` | 0 | 0 | 0 | 0 | 0 / 0 |
| `clean_ref_020` | `A2_baseline_fused_conservative` | 2735 | 17 | 16 | 0 | 2662 / 2714 |
| `clean_ref_020` | `B_solution_1` | 0 | 0 | 0 | 0 | 0 / 0 |
| `clean_ref_020` | `C_solution_2` | 0 | 0 | 0 | 0 | 0 / 0 |
| `clean_ref_020` | `D_solution_3` | 747 | 1 | 7 | 0 | 741 / 743 |
| `clean_ref_020` | `E1_experimental_temporal_only` | 0 | 0 | 0 | 911 | 0 / 0 |
| `clean_ref_020` | `E2_experimental_fused_3state` | 0 | 0 | 0 | 911 | 0 / 0 |

## 4. Track B Dev Regime Summary Across Candidates A-E

| Regime | Candidate | Repair R / P / F1 | All-Cand R / P | MC / Telea PSNR Gain | Intact Mod MC / Telea (Edge MC / Telea) | Abstain (Leg / E) |
|---|---|---:|---:|---:|---:|---:|
| `absdeg_000_fast_pan` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 100% / 0% |
| `absdeg_000_fast_pan` | `A2_baseline_fused_conservative` | 32.2% / 59.5% / 0.418 | 32.2% / 59.5% | +4.19 / +3.74 dB | 6741 / 7087 (2944 / 2966) | 100% / 0% |
| `absdeg_000_fast_pan` | `B_solution_1` | 41.4% / 68.2% / 0.515 | 41.4% / 68.2% | +2.98 / +2.51 dB | 5429 / 5888 (514 / 527) | 100% / 0% |
| `absdeg_000_fast_pan` | `C_solution_2` | 21.0% / 60.4% / 0.312 | 21.0% / 60.4% | +1.18 / +1.09 dB | 3854 / 4237 (412 / 422) | 100% / 0% |
| `absdeg_000_fast_pan` | `D_solution_3` | 37.5% / 65.5% / 0.477 | 37.5% / 65.5% | +3.32 / +2.87 dB | 5786 / 6168 (1539 / 1554) | 100% / 0% |
| `absdeg_000_fast_pan` | `E1_experimental_temporal_only` | 18.5% / 91.7% / 0.307 | 50.5% / 82.5% | +2.63 / +2.54 dB | 461 / 516 (46 / 46) | 100% / 0% |
| `absdeg_000_fast_pan` | `E2_experimental_fused_3state` | 24.8% / 92.1% / 0.390 | 54.6% / 83.0% | +3.76 / +3.59 dB | 611 / 668 (72 / 74) | 100% / 0% |
| `absdeg_010_moderate_pan` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 17% / 0% |
| `absdeg_010_moderate_pan` | `A2_baseline_fused_conservative` | 31.1% / 71.7% / 0.434 | 31.1% / 71.7% | +5.13 / +4.84 dB | 3615 / 3870 (1627 / 1680) | 17% / 0% |
| `absdeg_010_moderate_pan` | `B_solution_1` | 47.9% / 58.9% / 0.529 | 47.9% / 58.9% | +4.85 / +4.17 dB | 8597 / 9608 (2186 / 2288) | 17% / 0% |
| `absdeg_010_moderate_pan` | `C_solution_2` | 22.2% / 56.1% / 0.318 | 22.2% / 56.1% | +1.07 / +1.03 dB | 4227 / 4965 (994 / 1030) | 17% / 0% |
| `absdeg_010_moderate_pan` | `D_solution_3` | 39.7% / 64.3% / 0.491 | 39.7% / 64.3% | +4.95 / +4.59 dB | 5691 / 6423 (1746 / 1797) | 17% / 0% |
| `absdeg_010_moderate_pan` | `E1_experimental_temporal_only` | 35.5% / 89.0% / 0.507 | 60.7% / 85.8% | +5.78 / +5.11 dB | 1159 / 1349 (323 / 329) | 17% / 0% |
| `absdeg_010_moderate_pan` | `E2_experimental_fused_3state` | 39.1% / 89.2% / 0.544 | 63.4% / 85.7% | +6.92 / +6.22 dB | 1263 / 1457 (367 / 370) | 17% / 0% |
| `absdeg_020_dev_pan` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 0% / 0% |
| `absdeg_020_dev_pan` | `A2_baseline_fused_conservative` | 32.8% / 67.3% / 0.441 | 32.8% / 67.3% | +5.84 / +5.46 dB | 4840 / 5035 (2201 / 2229) | 0% / 0% |
| `absdeg_020_dev_pan` | `B_solution_1` | 21.5% / 57.3% / 0.312 | 21.5% / 57.3% | +2.63 / +2.22 dB | 4403 / 4578 (213 / 220) | 0% / 0% |
| `absdeg_020_dev_pan` | `C_solution_2` | 11.5% / 57.7% / 0.191 | 11.5% / 57.7% | +0.59 / +0.55 dB | 2352 / 2456 (159 / 165) | 0% / 0% |
| `absdeg_020_dev_pan` | `D_solution_3` | 26.6% / 68.0% / 0.383 | 26.6% / 68.0% | +3.32 / +3.02 dB | 3677 / 3811 (836 / 843) | 0% / 0% |
| `absdeg_020_dev_pan` | `E1_experimental_temporal_only` | 19.1% / 90.3% / 0.316 | 44.5% / 85.4% | +3.48 / +3.43 dB | 567 / 602 (28 / 28) | 0% / 0% |
| `absdeg_020_dev_pan` | `E2_experimental_fused_3state` | 23.2% / 90.3% / 0.369 | 47.8% / 85.5% | +4.45 / +4.34 dB | 707 / 747 (69 / 70) | 0% / 0% |
| `regime_persistent_bright_010` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 8% / 0% |
| `regime_persistent_bright_010` | `A2_baseline_fused_conservative` | 86.8% / 82.0% / 0.843 | 86.8% / 82.0% | +25.11 / +23.08 dB | 1860 / 1950 (1169 / 1199) | 8% / 0% |
| `regime_persistent_bright_010` | `B_solution_1` | 29.6% / 51.6% / 0.376 | 29.6% / 51.6% | +0.52 / +0.49 dB | 2163 / 2595 (650 / 690) | 8% / 0% |
| `regime_persistent_bright_010` | `C_solution_2` | 22.8% / 56.9% / 0.326 | 22.8% / 56.9% | +0.27 / +0.26 dB | 1277 / 1586 (254 / 263) | 8% / 0% |
| `regime_persistent_bright_010` | `D_solution_3` | 43.2% / 64.4% / 0.517 | 43.2% / 64.4% | +0.64 / +0.60 dB | 1906 / 2260 (749 / 767) | 8% / 0% |
| `regime_persistent_bright_010` | `E1_experimental_temporal_only` | 63.8% / 96.4% / 0.768 | 70.8% / 89.1% | +8.82 / +8.34 dB | 244 / 245 (205 / 205) | 8% / 0% |
| `regime_persistent_bright_010` | `E2_experimental_fused_3state` | 81.1% / 97.0% / 0.883 | 83.6% / 90.6% | +23.11 / +21.61 dB | 255 / 256 (208 / 208) | 8% / 0% |
| `regime_persistent_dark_tilted_020` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 0% / 0% |
| `regime_persistent_dark_tilted_020` | `A2_baseline_fused_conservative` | 99.9% / 78.6% / 0.880 | 99.9% / 78.6% | +27.09 / +21.06 dB | 2749 / 2814 (1907 / 1933) | 0% / 0% |
| `regime_persistent_dark_tilted_020` | `B_solution_1` | 33.0% / 66.2% / 0.440 | 33.0% / 66.2% | +3.85 / +3.21 dB | 1465 / 1618 (137 / 136) | 0% / 0% |
| `regime_persistent_dark_tilted_020` | `C_solution_2` | 15.2% / 66.6% / 0.248 | 15.2% / 66.6% | +0.75 / +0.67 dB | 692 / 770 (122 / 122) | 0% / 0% |
| `regime_persistent_dark_tilted_020` | `D_solution_3` | 42.9% / 74.3% / 0.544 | 42.9% / 74.3% | +5.56 / +4.96 dB | 1445 / 1506 (702 / 707) | 0% / 0% |
| `regime_persistent_dark_tilted_020` | `E1_experimental_temporal_only` | 85.3% / 95.4% / 0.901 | 85.5% / 87.4% | +10.91 / +10.32 dB | 426 / 425 (223 / 224) | 0% / 0% |
| `regime_persistent_dark_tilted_020` | `E2_experimental_fused_3state` | 99.7% / 95.8% / 0.977 | 99.7% / 88.8% | +24.02 / +19.82 dB | 455 / 452 (238 / 238) | 0% / 0% |
| `regime_subtle_transient_010` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 17% / 0% |
| `regime_subtle_transient_010` | `A2_baseline_fused_conservative` | 74.7% / 77.1% / 0.759 | 74.7% / 77.1% | +12.95 / +11.55 dB | 845 / 904 (512 / 526) | 17% / 0% |
| `regime_subtle_transient_010` | `B_solution_1` | 49.7% / 62.1% / 0.552 | 49.7% / 62.1% | +8.18 / +7.21 dB | 958 / 1083 (95 / 97) | 17% / 0% |
| `regime_subtle_transient_010` | `C_solution_2` | 29.5% / 62.1% / 0.400 | 29.5% / 62.1% | +2.63 / +2.45 dB | 537 / 638 (41 / 40) | 17% / 0% |
| `regime_subtle_transient_010` | `D_solution_3` | 71.0% / 74.2% / 0.726 | 71.0% / 74.2% | +11.57 / +9.89 dB | 801 / 903 (230 / 230) | 17% / 0% |
| `regime_subtle_transient_010` | `E1_experimental_temporal_only` | 41.8% / 97.7% / 0.586 | 42.6% / 92.9% | +7.08 / +6.47 dB | 40 / 41 (27 / 27) | 17% / 0% |
| `regime_subtle_transient_010` | `E2_experimental_fused_3state` | 70.0% / 97.9% / 0.816 | 70.0% / 94.9% | +12.41 / +11.37 dB | 62 / 63 (38 / 38) | 17% / 0% |
| `regime_diagonal_curved_010` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 17% / 0% |
| `regime_diagonal_curved_010` | `A2_baseline_fused_conservative` | 94.9% / 46.7% / 0.626 | 94.9% / 46.7% | +16.97 / +13.73 dB | 8236 / 8849 (2760 / 2858) | 17% / 0% |
| `regime_diagonal_curved_010` | `B_solution_1` | 35.8% / 61.5% / 0.453 | 35.8% / 61.5% | +1.58 / +1.53 dB | 1643 / 1776 (488 / 504) | 17% / 0% |
| `regime_diagonal_curved_010` | `C_solution_2` | 4.1% / 44.9% / 0.075 | 4.1% / 44.9% | +0.09 / +0.09 dB | 380 / 417 (180 / 183) | 17% / 0% |
| `regime_diagonal_curved_010` | `D_solution_3` | 80.5% / 64.8% / 0.718 | 80.5% / 64.8% | +7.29 / +5.76 dB | 3587 / 3761 (1558 / 1607) | 17% / 0% |
| `regime_diagonal_curved_010` | `E1_experimental_temporal_only` | 77.7% / 91.3% / 0.840 | 91.0% / 88.1% | +8.89 / +8.15 dB | 609 / 640 (315 / 323) | 17% / 0% |
| `regime_diagonal_curved_010` | `E2_experimental_fused_3state` | 80.8% / 90.4% / 0.853 | 98.6% / 50.1% | +9.39 / +8.52 dB | 705 / 739 (347 / 355) | 17% / 0% |
| `regime_horizontal_fissure_020` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 0% / 0% |
| `regime_horizontal_fissure_020` | `A2_baseline_fused_conservative` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 2087 / 2124 (1619 / 1631) | 0% / 0% |
| `regime_horizontal_fissure_020` | `B_solution_1` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 282 / 289 (68 / 68) | 0% / 0% |
| `regime_horizontal_fissure_020` | `C_solution_2` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 282 / 289 (68 / 68) | 0% / 0% |
| `regime_horizontal_fissure_020` | `D_solution_3` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 837 / 849 (518 / 522) | 0% / 0% |
| `regime_horizontal_fissure_020` | `E1_experimental_temporal_only` | 31.2% / 99.4% / 0.474 | 38.3% / 74.4% | +1.84 / +1.79 dB | 29 / 31 (0 / 0) | 0% / 0% |
| `regime_horizontal_fissure_020` | `E2_experimental_fused_3state` | 31.2% / 99.4% / 0.474 | 38.3% / 74.4% | +1.84 / +1.79 dB | 29 / 31 (0 / 0) | 0% / 0% |
| `regime_telecine_3_2_010` | `A1_baseline_temporal` | 0.0% / 0.0% / 0.000 | 0.0% / 0.0% | +0.00 / +0.00 dB | 0 / 0 (0 / 0) | 19% / 0% |
| `regime_telecine_3_2_010` | `A2_baseline_fused_conservative` | 100.0% / 84.9% / 0.918 | 100.0% / 84.9% | +28.42 / +24.31 dB | 1393 / 1456 (878 / 902) | 19% / 0% |
| `regime_telecine_3_2_010` | `B_solution_1` | 41.0% / 51.0% / 0.455 | 41.0% / 51.0% | +1.32 / +1.01 dB | 2665 / 2992 (977 / 1022) | 19% / 0% |
| `regime_telecine_3_2_010` | `C_solution_2` | 28.6% / 58.9% / 0.385 | 28.6% / 58.9% | +0.28 / +0.31 dB | 1219 / 1480 (279 / 288) | 19% / 0% |
| `regime_telecine_3_2_010` | `D_solution_3` | 49.8% / 65.6% / 0.566 | 49.8% / 65.6% | +0.94 / +0.96 dB | 1692 / 1933 (640 / 658) | 19% / 0% |
| `regime_telecine_3_2_010` | `E1_experimental_temporal_only` | 93.7% / 96.8% / 0.952 | 93.7% / 77.2% | +14.47 / +14.00 dB | 251 / 255 (126 / 124) | 19% / 0% |
| `regime_telecine_3_2_010` | `E2_experimental_fused_3state` | 100.0% / 97.0% / 0.984 | 100.0% / 78.4% | +28.04 / +24.13 dB | 246 / 255 (124 / 124) | 19% / 0% |

### 4b. Persistent Scratch Loss & Uncertainty Preservation Summary

| Candidate | Persistent GT (px) | Recovered Repair % | Surfaced Uncertain % | Erroneously Vetoed % | Silent Loss % |
|---|---:|---:|---:|---:|---:|
| `A1_baseline_temporal` | 53774 | 0.0% | 0.0% | 0.00% | 100.0% |
| `A2_baseline_fused_conservative` | 53774 | 67.3% | 0.0% | 0.00% | 32.7% |
| `B_solution_1` | 53774 | 24.4% | 0.0% | 0.00% | 75.7% |
| `C_solution_2` | 53774 | 12.4% | 0.0% | 0.00% | 87.5% |
| `D_solution_3` | 53774 | 37.6% | 0.0% | 0.00% | 62.4% |
| `E1_experimental_temporal_only` | 53774 | 65.3% | 5.7% | 0.00% | 29.0% |
| `E2_experimental_fused_3state` | 53774 | 72.9% | 5.5% | 0.78% | 21.6% |

### 4c. Dimensional Breakdown (Orientation, Polarity, Contrast, Persistence, Motion Regime — Repair R / P / F1)

| Dimension | Bucket | A2 Baseline | D Sol 3 | E1 Temporal | E2 Fused 3-State |
|---|---|---|---|---|---|
| `orientation` | `vertical` | 42.8%/71.3%/0.53 | 37.6%/66.0%/0.48 | 33.0%/92.6%/0.49 | 39.6%/93.0%/0.56 |
| `orientation` | `tilted_1.5deg` | 99.9%/78.6%/0.88 | 42.9%/74.3%/0.54 | 85.3%/95.4%/0.90 | 99.7%/95.8%/0.98 |
| `orientation` | `diagonal_38deg` | 94.9%/46.7%/0.63 | 80.5%/64.8%/0.72 | 77.7%/91.3%/0.84 | 80.8%/90.4%/0.85 |
| `orientation` | `horizontal_3deg` | 0.0%/0.0%/0.00 | 0.0%/0.0%/0.00 | 31.2%/99.4%/0.47 | 31.2%/99.4%/0.47 |
| `polarity` | `mixed` | 32.0%/65.7%/0.43 | 34.6%/65.7%/0.45 | 24.4%/90.0%/0.38 | 29.0%/90.3%/0.44 |
| `polarity` | `bright` | 90.9%/67.3%/0.77 | 59.0%/66.2%/0.62 | 72.6%/95.0%/0.82 | 84.5%/95.2%/0.90 |
| `polarity` | `dark` | 40.0%/67.8%/0.50 | 17.2%/65.1%/0.27 | 52.9%/96.8%/0.68 | 58.6%/96.9%/0.73 |
| `contrast` | `strong` | 44.4%/66.3%/0.53 | 35.7%/65.4%/0.46 | 38.8%/93.2%/0.55 | 44.5%/93.3%/0.60 |
| `contrast` | `low_0.38` | 74.7%/77.1%/0.76 | 71.0%/74.2%/0.73 | 41.8%/97.7%/0.59 | 70.0%/97.9%/0.82 |
| `persistence` | `mixed_orig` | 32.0%/65.7%/0.43 | 34.6%/65.7%/0.45 | 24.4%/90.0%/0.38 | 29.0%/90.3%/0.44 |
| `persistence` | `persistent_12f` | 93.5%/80.1%/0.86 | 43.1%/69.0%/0.53 | 74.7%/95.8%/0.84 | 90.5%/96.3%/0.93 |
| `persistence` | `transient_2f` | 74.7%/77.1%/0.76 | 71.0%/74.2%/0.73 | 41.8%/97.7%/0.59 | 70.0%/97.9%/0.82 |
| `persistence` | `medium_4f` | 34.1%/41.7%/0.38 | 28.9%/60.1%/0.39 | 47.9%/94.5%/0.64 | 49.0%/93.9%/0.64 |
| `persistence` | `medium_6f_telecine` | 100.0%/84.9%/0.92 | 49.8%/65.6%/0.57 | 93.7%/96.8%/0.95 | 100.0%/97.0%/0.98 |
| `motion_regime` | `fast_pan` | 32.2%/59.5%/0.42 | 37.5%/65.5%/0.48 | 18.5%/91.7%/0.31 | 24.8%/92.1%/0.39 |
| `motion_regime` | `moderate_pan` | 45.0%/66.0%/0.54 | 35.4%/65.9%/0.46 | 40.7%/93.0%/0.57 | 47.0%/93.2%/0.63 |
| `motion_regime` | `telecine_3_2` | 100.0%/84.9%/0.92 | 49.8%/65.6%/0.57 | 93.7%/96.8%/0.95 | 100.0%/97.0%/0.98 |

## 5. Archival Dev Negative Controls (Strict Zero-Leak Safety Gate & Full-Frame Collateral)

| Negative Control ROI | Candidate | Det Leak | Repair Leak | MC / Telea Mod | Full-Frame Rep / MC | Uncertain / Vetoed ROI | Gate |
|---|---|---:|---:|---:|---:|---:|:---:|
| `neighbors_frame25_pipe` | `A1_baseline_temporal` | 819 | 819 | 814 / 819 | 2878 / 2852 | 0 / 0 | **FAIL** |
| `neighbors_frame25_pipe` | `A2_baseline_fused_conservative` | 819 | 819 | 814 / 819 | 2623 / 2557 | 0 / 0 | **FAIL** |
| `neighbors_frame25_pipe` | `B_solution_1` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `neighbors_frame25_pipe` | `C_solution_2` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `neighbors_frame25_pipe` | `D_solution_3` | 0 | 0 | 0 / 0 | 12 / 9 | 0 / 0 | PASS |
| `neighbors_frame25_pipe` | `E1_experimental_temporal_only` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `neighbors_frame25_pipe` | `E2_experimental_fused_3state` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `paleface_seq_a_jacket` | `A1_baseline_temporal` | 641 | 641 | 625 / 636 | 8294 / 7858 | 0 / 0 | **FAIL** |
| `paleface_seq_a_jacket` | `A2_baseline_fused_conservative` | 641 | 641 | 626 / 636 | 10540 / 10051 | 0 / 0 | **FAIL** |
| `paleface_seq_a_jacket` | `B_solution_1` | 0 | 0 | 0 / 0 | 1557 / 1518 | 0 / 0 | PASS |
| `paleface_seq_a_jacket` | `C_solution_2` | 0 | 0 | 0 / 0 | 413 / 394 | 0 / 0 | PASS |
| `paleface_seq_a_jacket` | `D_solution_3` | 9 | 9 | 9 / 9 | 1544 / 1519 | 0 / 0 | **FAIL** |
| `paleface_seq_a_jacket` | `E1_experimental_temporal_only` | 305 | 0 | 0 / 0 | 101 / 101 | 305 / 0 | PASS |
| `paleface_seq_a_jacket` | `E2_experimental_fused_3state` | 305 | 0 | 0 / 0 | 101 / 101 | 305 / 13 | PASS |
| `paleface_seq_b_wall` | `A1_baseline_temporal` | 0 | 0 | 0 / 0 | 830 / 819 | 0 / 0 | PASS |
| `paleface_seq_b_wall` | `A2_baseline_fused_conservative` | 123 | 123 | 121 / 123 | 3733 / 3684 | 0 / 0 | **FAIL** |
| `paleface_seq_b_wall` | `B_solution_1` | 0 | 0 | 0 / 0 | 333 / 316 | 0 / 0 | PASS |
| `paleface_seq_b_wall` | `C_solution_2` | 0 | 0 | 0 / 0 | 288 / 270 | 0 / 0 | PASS |
| `paleface_seq_b_wall` | `D_solution_3` | 7 | 7 | 7 / 7 | 408 / 389 | 0 / 0 | **FAIL** |
| `paleface_seq_b_wall` | `E1_experimental_temporal_only` | 26 | 0 | 0 / 0 | 43 / 43 | 26 / 0 | PASS |
| `paleface_seq_b_wall` | `E2_experimental_fused_3state` | 26 | 0 | 0 / 0 | 46 / 46 | 26 / 123 | PASS |
| `college_seq_c_jamb` | `A1_baseline_temporal` | 0 | 0 | 0 / 0 | 245 / 225 | 0 / 0 | PASS |
| `college_seq_c_jamb` | `A2_baseline_fused_conservative` | 0 | 0 | 0 / 0 | 361 / 302 | 0 / 0 | PASS |
| `college_seq_c_jamb` | `B_solution_1` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `college_seq_c_jamb` | `C_solution_2` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `college_seq_c_jamb` | `D_solution_3` | 0 | 0 | 0 / 0 | 6 / 6 | 0 / 0 | PASS |
| `college_seq_c_jamb` | `E1_experimental_temporal_only` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `college_seq_c_jamb` | `E2_experimental_fused_3state` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `neighbors_seq_d_pole` | `A1_baseline_temporal` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `neighbors_seq_d_pole` | `A2_baseline_fused_conservative` | 0 | 0 | 0 / 0 | 57 / 38 | 0 / 0 | PASS |
| `neighbors_seq_d_pole` | `B_solution_1` | 96 | 96 | 96 / 96 | 350 / 350 | 0 / 0 | **FAIL** |
| `neighbors_seq_d_pole` | `C_solution_2` | 0 | 0 | 0 / 0 | 0 / 0 | 0 / 0 | PASS |
| `neighbors_seq_d_pole` | `D_solution_3` | 0 | 0 | 0 / 0 | 3 / 3 | 0 / 0 | PASS |
| `neighbors_seq_d_pole` | `E1_experimental_temporal_only` | 143 | 0 | 0 / 0 | 0 / 0 | 143 / 0 | PASS |
| `neighbors_seq_d_pole` | `E2_experimental_fused_3state` | 143 | 0 | 0 / 0 | 0 / 0 | 143 / 0 | PASS |

## 6. Runtime (Un-instrumented) & Peak Traced Memory Overhead

| Candidate | Mean ms/frame | Peak Traced Memory (MB) |
|---|---:|---:|
| `A1_baseline_temporal` | 138.07 | 201.43 |
| `A2_baseline_fused_conservative` | 235.98 | 218.29 |
| `B_solution_1` | 254.51 | 214.80 |
| `C_solution_2` | 509.01 | 214.80 |
| `D_solution_3` | 599.76 | 218.29 |
| `E1_experimental_temporal_only` | 1951.97 | 504.11 |
| `E2_experimental_fused_3state` | 2080.31 | 497.51 |

## 7. Artifacts Generated
- Composite Figure: `docs/benchmarks/visuals/experimental_candidates_a_to_e_comparison.png`
- Examples Directory: `docs/benchmarks/visuals/experimental_examples`
- Video 1: `docs/benchmarks/visuals/experimental_side_by_side_videos/absdeg_010_persistent_a_to_e.mp4`
- Video 2: `docs/benchmarks/visuals/experimental_side_by_side_videos/archival_negative_controls_a_to_e.mp4`
