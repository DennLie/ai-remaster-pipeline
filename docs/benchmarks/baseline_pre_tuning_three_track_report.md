# Validated Three-Track Film Defect & Restoration Safety Benchmark

- **Timestamp:** `2026-11-18T12:52:34Z`
- **Environment Isolation:** `True` (`I:\ai-remaster-pipeline\.venv\Scripts\python.exe`)
- **Detector Modification Check:** `0 thresholds, presets, or fusion rules modified (baseline frozen)`

> [!WARNING]
> Historical Frame 25 pipe benchmarks (`temporal_scratch_quality_gate.json`, `temporal_scratch_option2_benchmark.json`) and `absdiff` cross-film pseudo-labels (`cross_film_generalization_benchmark.json`) are **quarantined** and excluded from ground-truth status.

---

## Track A — Real Annotated Still Images (Spatial Evaluation Only)

| Dataset (`provenance`) | Detector | Precision | Recall | Pixel F1 | FP Pixels | Max FP Comp Area | Max FP Comp H | Severe Structural FPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `film_damage_simulator` (`manual_polygon_gt`) | `film_artefact_segmenter` | 0.8018 | 0.0461 | **0.0871** | 303 | 25 | 7 | 2 |
| `film_damage_simulator` (`manual_polygon_gt`) | `cnn_blotches` | 0.6781 | 0.0840 | **0.1494** | 1061 | 44 | 9 | 16 |
| `film_damage_simulator` (`manual_polygon_gt`) | `legacy` | 0.0096 | 0.0406 | **0.0155** | 111340 | 5007 | 1024 | 769 |
| `film_damage_simulator` (`manual_polygon_gt`) | `bopb` | 0.7190 | 0.4637 | **0.5638** | 4823 | 644 | 53 | 65 |
| `film_damage_simulator` (`manual_polygon_gt`) | `patchcore` | 0.5365 | 0.1872 | **0.2775** | 4304 | 359 | 39 | 50 |
| `film_damage_simulator` (`manual_polygon_gt`) | `padim` | 0.2638 | 0.1184 | **0.1635** | 8796 | 267 | 21 | 122 |
| `artefact` (`manual_segmentation_gt`) | `film_artefact_segmenter` | 0.0089 | 0.1301 | **0.0167** | 599519 | 490678 | 1012 | 194 |
| `artefact` (`manual_segmentation_gt`) | `cnn_blotches` | 0.0070 | 0.0976 | **0.0130** | 576694 | 459045 | 1010 | 169 |
| `artefact` (`manual_segmentation_gt`) | `legacy` | 0.0333 | 0.0716 | **0.0454** | 86459 | 11659 | 1024 | 649 |
| `artefact` (`manual_segmentation_gt`) | `bopb` | 0.2214 | 0.0773 | **0.1145** | 11281 | 1011 | 467 | 87 |
| `artefact` (`manual_segmentation_gt`) | `patchcore` | 0.0092 | 0.2226 | **0.0176** | 999792 | 850537 | 1024 | 111 |
| `artefact` (`manual_segmentation_gt`) | `padim` | 0.0066 | 0.1441 | **0.0126** | 906012 | 798599 | 1024 | 74 |

*All temporal/multi-frame detectors (`temporal_scratch`, `fused_fas_temporal`, `farneback_3f`, `dis_flow`, `ridge`, `consensus_5f`, `dart_harness`) were marked `INSUFFICIENT_TEMPORAL_CONTEXT` on Track A still images.*

---

## Track B — Exact Synthetic Defects on Real Clean Video (`AbsoluteDegradation` Clean References)

| Regime (`split`) | Mode | Det Precision | Det Recall | Det F1 | Max FP Height | Repair Cov % | Edge Repair FP px | MC Defect PSNR Gain (dB) | Intact Px Modified |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `absdeg_010_seed_20261106` (`dev_diagnostic`) | `spatial_only` | 0.6560 | 0.3619 | **0.4664** | 26 | 0.313% | 1407 | +5.37 dB | 4187 |
| `absdeg_010_seed_20261106` (`dev_diagnostic`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `absdeg_010_seed_20261106` (`dev_diagnostic`) | `fused_conservative` | 0.7159 | 0.3317 | **0.4533** | 25 | 0.313% | 1407 | +5.37 dB | 4187 |
| `absdeg_010_seed_20261106` (`dev_diagnostic`) | `fused_high_recall` | 0.7159 | 0.3317 | **0.4533** | 25 | 0.313% | 1407 | +5.37 dB | 4187 |
| `absdeg_020_seed_20261106` (`locked_holdout`) | `spatial_only` | 0.6098 | 0.3720 | **0.4621** | 43 | 0.349% | 1567 | +5.33 dB | 5726 |
| `absdeg_020_seed_20261106` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `absdeg_020_seed_20261106` (`locked_holdout`) | `fused_conservative` | 0.6712 | 0.3471 | **0.4576** | 27 | 0.349% | 1567 | +5.33 dB | 5726 |
| `absdeg_020_seed_20261106` (`locked_holdout`) | `fused_high_recall` | 0.6712 | 0.3471 | **0.4576** | 27 | 0.349% | 1567 | +5.33 dB | 5726 |
| `absdeg_020_seed_20261107` (`locked_holdout`) | `spatial_only` | 0.5720 | 0.3974 | **0.4690** | 28 | 0.253% | 1361 | +7.98 dB | 4530 |
| `absdeg_020_seed_20261107` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `absdeg_020_seed_20261107` (`locked_holdout`) | `fused_conservative` | 0.6413 | 0.3726 | **0.4714** | 25 | 0.253% | 1361 | +7.98 dB | 4530 |
| `absdeg_020_seed_20261107` (`locked_holdout`) | `fused_high_recall` | 0.6413 | 0.3726 | **0.4714** | 25 | 0.253% | 1361 | +7.98 dB | 4530 |
| `vertical_bright_drifting` (`locked_holdout`) | `spatial_only` | 0.7072 | 0.9999 | **0.8285** | 20 | 0.303% | 1211 | +7.63 dB | 3230 |
| `vertical_bright_drifting` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `vertical_bright_drifting` (`locked_holdout`) | `fused_conservative` | 0.7916 | 0.9993 | **0.8834** | 16 | 0.303% | 1211 | +7.63 dB | 3230 |
| `vertical_bright_drifting` (`locked_holdout`) | `fused_high_recall` | 0.7916 | 0.9993 | **0.8834** | 16 | 0.303% | 1211 | +7.63 dB | 3230 |
| `vertical_dark_persistent` (`locked_holdout`) | `spatial_only` | 0.7988 | 0.9978 | **0.8873** | 20 | 0.456% | 1257 | +3.39 dB | 3223 |
| `vertical_dark_persistent` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `vertical_dark_persistent` (`locked_holdout`) | `fused_conservative` | 0.8620 | 0.9974 | **0.9248** | 16 | 0.456% | 1257 | +3.39 dB | 3223 |
| `vertical_dark_persistent` (`locked_holdout`) | `fused_high_recall` | 0.8620 | 0.9974 | **0.9248** | 16 | 0.456% | 1257 | +3.39 dB | 3223 |
| `vertical_telecine_duplicates` (`locked_holdout`) | `spatial_only` | 0.6984 | 0.9806 | **0.8158** | 20 | 0.296% | 1272 | +4.17 dB | 3291 |
| `vertical_telecine_duplicates` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `vertical_telecine_duplicates` (`locked_holdout`) | `fused_conservative` | 0.7822 | 0.9745 | **0.8678** | 18 | 0.296% | 1272 | +4.17 dB | 3291 |
| `vertical_telecine_duplicates` (`locked_holdout`) | `fused_high_recall` | 0.7822 | 0.9745 | **0.8678** | 18 | 0.296% | 1272 | +4.17 dB | 3291 |
| `diagonal_bright_curved` (`locked_holdout`) | `spatial_only` | 0.4907 | 0.6850 | **0.5718** | 201 | 0.482% | 2730 | +3.39 dB | 10868 |
| `diagonal_bright_curved` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `diagonal_bright_curved` (`locked_holdout`) | `fused_conservative` | 0.5377 | 0.6581 | **0.5919** | 189 | 0.482% | 2730 | +3.39 dB | 10868 |
| `diagonal_bright_curved` (`locked_holdout`) | `fused_high_recall` | 0.5377 | 0.6581 | **0.5919** | 189 | 0.482% | 2730 | +3.39 dB | 10868 |
| `horizontal_fissure` (`locked_holdout`) | `spatial_only` | 0.5306 | 0.3347 | **0.4105** | 20 | 0.345% | 1360 | +0.69 dB | 7398 |
| `horizontal_fissure` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `horizontal_fissure` (`locked_holdout`) | `fused_conservative` | 0.5788 | 0.2757 | **0.3735** | 16 | 0.345% | 1360 | +0.69 dB | 7398 |
| `horizontal_fissure` (`locked_holdout`) | `fused_high_recall` | 0.5788 | 0.2757 | **0.3735** | 16 | 0.345% | 1360 | +0.69 dB | 7398 |
| `broad_abrasion_branching` (`locked_holdout`) | `spatial_only` | 0.8306 | 0.7738 | **0.8012** | 124 | 0.597% | 1011 | +1.64 dB | 3200 |
| `broad_abrasion_branching` (`locked_holdout`) | `temporal_only` | 0.0000 | 0.0000 | **0.0000** | 0 | 0.000% | 0 | +0.00 dB | 0 |
| `broad_abrasion_branching` (`locked_holdout`) | `fused_conservative` | 0.8707 | 0.6985 | **0.7751** | 119 | 0.597% | 1011 | +1.64 dB | 3200 |
| `broad_abrasion_branching` (`locked_holdout`) | `fused_high_recall` | 0.8707 | 0.6985 | **0.7751** | 119 | 0.597% | 1011 | +1.64 dB | 3200 |

---

## Track C — Real Archival Negative Controls (Strict Zero-Leakage Safety Gate)

| Sequence (`split`) | Critical ROI | Mode | Detector Leak (px) | Repair Leak (px) | Max Leak Height | Zero-Leak Gate |
|---|---|---|---:|---:|---:|:---:|
| `neighbors_frame25_pipe_audit` (`dev_diagnostic`) | `neighbors_frame25_pipe` | `spatial_only` | 0 | 0 | 0 | PASS |
| `neighbors_frame25_pipe_audit` (`dev_diagnostic`) | `neighbors_frame25_pipe` | `temporal_only` | 819 | 819 | 70 | **FAIL** |
| `neighbors_frame25_pipe_audit` (`dev_diagnostic`) | `neighbors_frame25_pipe` | `fused_conservative` | 819 | 819 | 70 | **FAIL** |
| `paleface_seq_a` (`dev_diagnostic`) | `paleface_actor_jacket_edge` | `spatial_only` | 21 | 13 | 5 | **FAIL** |
| `paleface_seq_a` (`dev_diagnostic`) | `paleface_actor_jacket_edge` | `temporal_only` | 641 | 641 | 193 | **FAIL** |
| `paleface_seq_a` (`dev_diagnostic`) | `paleface_actor_jacket_edge` | `fused_conservative` | 641 | 641 | 193 | **FAIL** |
| `paleface_seq_b` (`dev_diagnostic`) | `paleface_b_wall_curtain` | `spatial_only` | 146 | 123 | 21 | **FAIL** |
| `paleface_seq_b` (`dev_diagnostic`) | `paleface_b_wall_curtain` | `temporal_only` | 0 | 0 | 0 | PASS |
| `paleface_seq_b` (`dev_diagnostic`) | `paleface_b_wall_curtain` | `fused_conservative` | 123 | 123 | 21 | **FAIL** |
| `college_seq_c` (`dev_diagnostic`) | `college_door_jamb_student` | `spatial_only` | 1 | 0 | 0 | PASS |
| `college_seq_c` (`dev_diagnostic`) | `college_door_jamb_student` | `temporal_only` | 0 | 0 | 0 | PASS |
| `college_seq_c` (`dev_diagnostic`) | `college_door_jamb_student` | `fused_conservative` | 0 | 0 | 0 | PASS |
| `neighbors_seq_d` (`dev_diagnostic`) | `neighbors_d_telephone_pole` | `spatial_only` | 0 | 0 | 0 | PASS |
| `neighbors_seq_d` (`dev_diagnostic`) | `neighbors_d_telephone_pole` | `temporal_only` | 0 | 0 | 0 | PASS |
| `neighbors_seq_d` (`dev_diagnostic`) | `neighbors_d_telephone_pole` | `fused_conservative` | 0 | 0 | 0 | PASS |

Visual comparison & failure panel saved to `docs/benchmarks/visuals/three_track_validation_summary.png`.
