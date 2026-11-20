# Diagnostic Audit: Motion-Gate Failure & Conservative Repair Leakage (Dev Only)

- **Timestamp:** `2026-11-18T15:39:08Z`
- **Constraint Check:** Zero detector thresholds, presets, fusion rules, or GUI files modified.

## 1. Motion-Gate Flow Distributions (Dev Clip `010` Frame 5)

| Region | Pixels | Raw Temporal Pass | Structure OK Pass | Motion OK (`flow<=1.2`) Pass | Mean Flow (px) | Diff Warped | Diff Unwarped |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gt_scratch` | 6885 | 6822 | 3974 | 0 | 20.25 | 64.72 | 51.28 |
| `intact_edges` | 124880 | 80185 | 21334 | 0 | 18.42 | 9.26 | 25.47 |
| `moving_regions` | 262038 | 111608 | 31435 | 0 | 20.21 | 5.98 | 26.22 |
| `ordinary_bg` | 551805 | 61825 | 6664 | 0 | 21.20 | 1.57 | 7.36 |

## 2. Stage-by-Stage Repair Leakage Trace (Archival Dev Negative Controls)

| Sequence ROI | Mode | Detector Mask (px) | Repair Mask (px) | Actual Modified MC (px) | Actual Modified Telea (px) | Zero-Leak Gate |
|---|---|---:|---:|---:|---:|:---:|
| `neighbors_frame25_pipe` | `spatial_only` | 0 | 0 | 0 | 0 | PASS |
| `neighbors_frame25_pipe` | `temporal_only` | 819 | 819 | 818 | 819 | **FAIL** |
| `neighbors_frame25_pipe` | `fused_conservative` | 819 | 819 | 818 | 819 | **FAIL** |
| `paleface_seq_a` | `spatial_only` | 21 | 13 | 13 | 13 | **FAIL** |
| `paleface_seq_a` | `temporal_only` | 641 | 641 | 622 | 636 | **FAIL** |
| `paleface_seq_a` | `fused_conservative` | 641 | 641 | 622 | 636 | **FAIL** |
| `paleface_seq_b` | `spatial_only` | 146 | 123 | 123 | 123 | **FAIL** |
| `paleface_seq_b` | `temporal_only` | 0 | 0 | 0 | 0 | PASS |
| `paleface_seq_b` | `fused_conservative` | 123 | 123 | 123 | 123 | **FAIL** |

## 3. Diagnostic Ablation Results (Dev Clips Only)

| Sequence | Variant | Det Recall | Det Prec | Det F1 | Neg ROI Leak (px) | MC Defect PSNR Gain | Intact Px Modified |
|---|---|---:|---:|---:|---:|---:|---:|
| `absdeg_000_dev` | `0_baseline_current_temporal` | 0.0% | 0.0% | 0.000 | N/A | +0.00 dB | 0 |
| `absdeg_000_dev` | `1_ablation_drop_flow_only` | 63.8% | 31.0% | 0.417 | N/A | +9.08 dB | 9368 |
| `absdeg_000_dev` | `2_solution_1_flank_novelty_thickness` | 37.8% | 69.4% | 0.490 | N/A | +3.54 dB | 1003 |
| `absdeg_000_dev` | `3_solution_2_fb_consistency` | 20.6% | 66.7% | 0.315 | N/A | +1.39 dB | 608 |
| `absdeg_000_dev` | `4_solution_3_fused_conservative_veto` | 33.5% | 66.9% | 0.447 | N/A | +3.37 dB | 1038 |
| `absdeg_010_dev` | `0_baseline_current_temporal` | 0.0% | 0.0% | 0.000 | N/A | +0.00 dB | 0 |
| `absdeg_010_dev` | `1_ablation_drop_flow_only` | 60.4% | 38.6% | 0.471 | N/A | +8.74 dB | 6119 |
| `absdeg_010_dev` | `2_solution_1_flank_novelty_thickness` | 42.3% | 54.0% | 0.475 | N/A | +5.59 dB | 2155 |
| `absdeg_010_dev` | `3_solution_2_fb_consistency` | 26.2% | 69.3% | 0.380 | N/A | +1.09 dB | 648 |
| `absdeg_010_dev` | `4_solution_3_fused_conservative_veto` | 40.4% | 71.5% | 0.516 | N/A | +6.66 dB | 958 |
| `absdeg_010_injected_drifting` | `0_baseline_current_temporal` | 0.0% | 0.0% | 0.000 | N/A | +0.00 dB | 0 |
| `absdeg_010_injected_drifting` | `1_ablation_drop_flow_only` | 15.1% | 6.7% | 0.093 | N/A | +0.36 dB | 4222 |
| `absdeg_010_injected_drifting` | `2_solution_1_flank_novelty_thickness` | 15.3% | 36.4% | 0.216 | N/A | +0.85 dB | 542 |
| `absdeg_010_injected_drifting` | `3_solution_2_fb_consistency` | 9.8% | 31.8% | 0.150 | N/A | +0.67 dB | 435 |
| `absdeg_010_injected_drifting` | `4_solution_3_fused_conservative_veto` | 19.8% | 38.6% | 0.262 | N/A | +1.22 dB | 656 |
| `neighbors_frame25_pipe` | `0_baseline_current_temporal` | N/A | N/A | N/A | 819 | N/A | N/A |
| `neighbors_frame25_pipe` | `1_ablation_drop_flow_only` | N/A | N/A | N/A | 819 | N/A | N/A |
| `neighbors_frame25_pipe` | `2_solution_1_flank_novelty_thickness` | N/A | N/A | N/A | 0 | N/A | N/A |
| `neighbors_frame25_pipe` | `3_solution_2_fb_consistency` | N/A | N/A | N/A | 0 | N/A | N/A |
| `neighbors_frame25_pipe` | `4_solution_3_fused_conservative_veto` | N/A | N/A | N/A | 0 | N/A | N/A |
| `paleface_seq_a` | `0_baseline_current_temporal` | N/A | N/A | N/A | 641 | N/A | N/A |
| `paleface_seq_a` | `1_ablation_drop_flow_only` | N/A | N/A | N/A | 641 | N/A | N/A |
| `paleface_seq_a` | `2_solution_1_flank_novelty_thickness` | N/A | N/A | N/A | 0 | N/A | N/A |
| `paleface_seq_a` | `3_solution_2_fb_consistency` | N/A | N/A | N/A | 0 | N/A | N/A |
| `paleface_seq_a` | `4_solution_3_fused_conservative_veto` | N/A | N/A | N/A | 9 | N/A | N/A |
| `paleface_seq_b` | `0_baseline_current_temporal` | N/A | N/A | N/A | 0 | N/A | N/A |
| `paleface_seq_b` | `1_ablation_drop_flow_only` | N/A | N/A | N/A | 76 | N/A | N/A |
| `paleface_seq_b` | `2_solution_1_flank_novelty_thickness` | N/A | N/A | N/A | 0 | N/A | N/A |
| `paleface_seq_b` | `3_solution_2_fb_consistency` | N/A | N/A | N/A | 0 | N/A | N/A |
| `paleface_seq_b` | `4_solution_3_fused_conservative_veto` | N/A | N/A | N/A | 7 | N/A | N/A |

Visual evidence saved to `docs/benchmarks/visuals/diagnostic_motion_and_leakage_audit.png`.
