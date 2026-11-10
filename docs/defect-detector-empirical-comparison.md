# Empirical Multi-Candidate Defect Detector Comparison

## 1. Overview & Evaluation Protocol

This report documents the head-to-head empirical evaluation of five film defect detection candidates across identical preprocessed frame sequences, generated via [`scripts/candidate_detector_eval.py`](file:///I:/ai-remaster-pipeline/scripts/candidate_detector_eval.py).

### Evaluated Candidate Methods
1. **Baseline (`farneback_3f`)**: 3-frame Farneback Optical Flow Temporal Consensus ([`scripts/film_defect_masks.py`](file:///I:/ai-remaster-pipeline/scripts/film_defect_masks.py)).
2. **Candidate A (`dis_flow`)**: OpenCV Dense Inverse Search (`DISOpticalFlow_create(DISOPTICAL_FLOW_PRESET_FAST)`) 3-frame optical flow temporal consensus.
3. **Candidate B (`ridge`)**: Multi-scale steerable Ridge filter (`scales=(2, 4, 6)`) with directional vertical morphological tracking and temporal novelty bounds ([`scripts/film_scratch_ridge.py`](file:///I:/ai-remaster-pipeline/scripts/film_scratch_ridge.py)).
4. **Candidate C (`consensus_5f`)**: 5-frame temporal median and interquartile-spread consensus ($t-2 \dots t+2$) with horizontal flank preservation.
5. **Candidate D (`dart_harness`)**: Strictly isolated external DART MaskNet evaluation harness (evaluates exported soft masks or isolated multi-channel consensus without importing non-commercial DART/RTN weights into production).

### Standardized Preprocessing & Scoring Rules
- **Identical Frame Normalization**: All candidates receive identical `uint8` BGR and grayscale frame tensors validated by `preprocess_frames()` with zero silent resizing or interpolation.
- **Disjoint Defect Channels**: Every detector returns separate binary masks (`0` = clean, `255` = defect) for `scratch` and `dust` channels.
- **Micro-Aggregated Confusion Counts**: True Positives ($TP$), False Positives ($FP$), False Negatives ($FN$), and True Negatives ($TN$) are summed across frames prior to computing Precision, Recall, Pixel F1, IoU, and Unaffected Pixel False Positive Rate ($FPR = FP / (FP + TN)$). Ambiguous/ignore regions (`127`) are excluded from metric denominators.

---

## 2. Quantitative Spatial Metrics (Disjoint Scratch vs. Dust Channels)

Evaluated on paired synthetic sequences (`320x180`, 8 frames, sensitivity `0.65`, seed `42`) with parametric drifting vertical scratches (`width=2`, `opacity=0.85`, `duration=5`, `drift_px=3.0`) and transient multi-cluster dust (`clusters_per_frame=3`, `contrast=0.85`):

| Candidate | Scratch P | Scratch R | Scratch F1 | Scratch IoU | Scratch FPR | Dust P | Dust R | Dust F1 | Dust IoU | Dust FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `farneback_3f` (Baseline) | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 1.0000 | 0.3564 | 0.5255 | 0.3564 | 0.000000 |
| `dis_flow` (Candidate A) | 1.0000 | 0.0670 | 0.1257 | 0.0670 | 0.000000 | 1.0000 | 0.3653 | 0.5351 | 0.3653 | 0.000000 |
| `ridge` (Candidate B) | 0.6000 | 1.0000 | 0.7500 | 0.6000 | 0.003929 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000000 |
| `consensus_5f` (Candidate C) | 1.0000 | 0.9778 | 0.9888 | 0.9778 | 0.000000 | 1.0000 | 0.3695 | 0.5396 | 0.3695 | 0.000000 |
| `dart_harness` (Candidate D) | 0.6000 | 1.0000 | 0.7500 | 0.6000 | 0.003929 | 1.0000 | 0.3695 | 0.5396 | 0.3695 | 0.000000 |

### Archival & AbsoluteDegradation Cross-Reference
To ground the synthetic channel comparison against complex variable-width, curved, and partial-opacity scratches on real photographic textures, recall the micro-aggregated results from [`docs/absolute-scratch-channel-experiment.md`](file:///I:/ai-remaster-pipeline/docs/absolute-scratch-channel-experiment.md) (reserved test clips `010` and `020`, `1270x690`):

| Reference Method (AbsoluteDegradation Test Split) | Precision | Recall | Pixel F1 | Unaffected Pixel FPR |
|---|---:|---:|---:|---:|
| Temporal Dust Wrapper (`farneback_3f`) | 0.104569 | 0.136361 | 0.118367 | 0.4260% |
| Ridge Wrapper | 0.301839 | 0.058509 | 0.098019 | 0.0494% |
| Ridge Core | 0.279701 | 0.060666 | 0.099706 | 0.0570% |

---

## 3. Temporal Consistency Metrics

Measured across consecutive frame triplets ($t-1, t, t+1$) and active persistent scratch trajectories:

| Candidate | Scratch Flicker Rate | Scratch Continuity | Missed Persistent Rate | Dust Flicker Rate |
|---|---:|---:|---:|---:|
| `farneback_3f` (Baseline) | 0.000000 | 0.0000 | 1.0000 | 0.000000 |
| `dis_flow` (Candidate A) | 0.000000 | 0.4000 | 0.6000 | 0.000000 |
| `ridge` (Candidate B) | 0.000531 | 1.0000 | 0.0000 | 0.000000 |
| `consensus_5f` (Candidate C) | 0.000000 | 1.0000 | 0.0000 | 0.000000 |
| `dart_harness` (Candidate D) | 0.000531 | 1.0000 | 0.0000 | 0.000000 |

### Key Temporal Findings
- **Optical Flow Tracking Paradox on Persistent Scratches**: Both 3-frame optical flow methods (`farneback_3f` and `dis_flow`) warp neighbor frames $t-1$ and $t+1$ toward frame $t$. When a vertical scratch persists across consecutive frames and drifts slowly ($1\text{--}3\text{ px/frame}$), dense optical flow treats the high-contrast scratch as moving scene structure and aligns it across frames, cancelling out the temporal residual (`Scratch Continuity = 0.0000` for Farneback, `0.4000` for DIS).
- **5-Frame Rank Consensus & Multi-Scale Ridge**: `consensus_5f` and `ridge` achieve `1.0000` trajectory continuity (`0.0000` mid-trajectory dropout) on multi-frame drifting scratches because the 5-frame temporal median ($t-2 \dots t+2$) and sequence percentile bounds (`20th`/`80th` percentiles) reject features that occupy a given pixel column for only 1–2 frames.

---

## 4. Execution Latency & Memory Footprint Across Standard Resolutions

Profiled on CPU across `720p` (`1280x720`), `1080p` (`1920x1080`), and `4K` (`3840x2160`) 5-frame sequences using `time.perf_counter` and `tracemalloc`:

| Candidate | Resolution | Dimensions | Mean Latency (ms/frame) | Peak RAM Footprint (MB) |
|---|---|---|---:|---:|
| `farneback_3f` | 720p | 1280x720 | 10.87 | 26.37 |
| `farneback_3f` | 1080p | 1920x1080 | 26.38 | 59.33 |
| `farneback_3f` | 4K | 3840x2160 | 76.25 | 237.31 |
| `dis_flow` | 720p | 1280x720 | 11.07 | 26.37 |
| `dis_flow` | 1080p | 1920x1080 | 21.03 | 59.33 |
| `dis_flow` | 4K | 3840x2160 | 64.66 | 237.31 |
| `ridge` | 720p | 1280x720 | 177.42 | 83.50 |
| `ridge` | 1080p | 1920x1080 | 417.81 | 187.87 |
| `ridge` | 4K | 3840x2160 | 1856.24 | 751.47 |
| `consensus_5f` | 720p | 1280x720 | 22.24 | 43.95 |
| `consensus_5f` | 1080p | 1920x1080 | 48.10 | 98.88 |
| `consensus_5f` | 4K | 3840x2160 | 174.84 | 395.51 |
| `dart_harness` | 720p | 1280x720 | 193.49 | 109.87 |
| `dart_harness` | 1080p | 1920x1080 | 457.67 | 247.20 |
| `dart_harness` | 4K | 3840x2160 | 1841.73 | 988.78 |

### Performance Observations
1. **DIS Flow vs. Farneback**: At `1080p` and `4K`, `dis_flow` (`PRESET_FAST`) reduces per-frame latency by **20.3%** (`21.03 ms` vs. `26.38 ms` at 1080p) and **15.2%** (`64.66 ms` vs. `76.25 ms` at 4K) with identical peak memory (`59.33 MB` at 1080p, `237.31 MB` at 4K) and slightly higher dust recall (`0.3653` vs. `0.3564`).
2. **5-Frame Consensus (`consensus_5f`)**: Executes in `48.10 ms/frame` at 1080p (`98.88 MB` peak RAM), providing a lightweight multi-frame window without full sequence percentile sorting.
3. **Multi-Scale Ridge (`ridge`)**: Sequence-wide percentile computation (`np.percentile` across `T` frames) and multi-scale flank padding increase latency (`417.81 ms/frame` at 1080p) and RAM (`187.87 MB` at 1080p). Furthermore, `film_scratch_ridge.detect_sequence` enforces a hard geometry/memory guard (`h * w * len(frames) <= 96,000,000`), requiring windowed chunking on 4K sequences.

---

## 5. Multi-Panel Diagnostic Contact Sheets

`scripts/candidate_detector_eval.py` generates standardized 6-panel horizontal diagnostic contact sheets (`contact_sheet_<candidate>.png`) for visual inspection:
- **Panel A (`Source`)**: Input frame with candidate identifier.
- **Panel B (`Ground Truth`)**: Combined ground-truth mask (`255` = defect, `127` = ignore, `0` = clean).
- **Panel C (`Pred Scratch`)**: Candidate's predicted binary scratch channel mask.
- **Panel D (`Pred Dust`)**: Candidate's predicted binary dust/blotch channel mask.
- **Panel E (`Error Map`)**: Pixel-accurate confusion visualization in BGR (`Green (0, 255, 0)` = True Positive, `Red (0, 0, 255)` = False Positive, `Blue (255, 0, 0)` = False Negative, `Gray (64, 64, 64)` = Ignored).
- **Panel F (`Temporal Strip`)**: 5-frame temporal evolution strip ($t-2, t-1, t, t+1, t+2$) with the center frame $t$ highlighted to expose mask flicker and trajectory dropouts at a glance.

---

## 6. Licensing Audit & Architectural Recommendations

| Candidate | Core Dependencies | License Status | Production Eligibility |
|---|---|---|---|
| `farneback_3f` | OpenCV, NumPy | Apache-2.0 / BSD-3-Clause | **Cleared (Default Baseline)** |
| `dis_flow` | OpenCV, NumPy | Apache-2.0 / BSD-3-Clause | **Cleared** |
| `ridge` | OpenCV, NumPy | Apache-2.0 / BSD-3-Clause | **Cleared** |
| `consensus_5f` | OpenCV, NumPy | Apache-2.0 / BSD-3-Clause | **Cleared** |
| `dart_harness` | External DART / RTN checkpoints | Unresolved RTN provenance | **Research-Only (Isolated Harness)** |

### Recommendations for Phase 4 (`BaseDefectDetector` Registry)
1. **Pluggable Strategy Registry**: Register `farneback_3f` (default backward-compatible baseline), `dis_flow`, `ridge`, and `consensus_5f` as selectable commercial-safe backends in `scripts/film_defect_detectors.py`.
2. **Hybrid / Channel-Specific Routing**: Support pairing a fast motion-compensated dust detector (`dis_flow` or `farneback_3f`) with a dedicated persistent scratch channel (`consensus_5f` or windowed `ridge`), while keeping `dart_harness` strictly isolated in evaluation scripts.
