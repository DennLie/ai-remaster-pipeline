# Comprehensive Audit: Defect Class Assignment, DART Visual Artefacts & Benchmark Integrity

**Date:** 2026-11-11  
**Status:** Complete  
**Scope:** Investigation of visual defects, class confusion, channel-mapping leakage, threshold sensitivity, and negative controls in the automated Dust & Scratch benchmark suite.  
**Diagnostic Visual Assets:** [`docs/investigation/visual_crops/`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/)  
**Regression Test Suite:** [`tests/test_defect_class_assignment_audit.py`](file:///I:/ai-remaster-pipeline/tests/test_defect_class_assignment_audit.py)  

---

## 1. Executive Summary & Core Verdict

The investigation into the validated Dust & Scratch benchmark identified three interacting, compounding causes behind the reported defects and class inconsistencies:

1. **Native DART Model Limitation (Class-Agnostic Single-Channel Gate):**
   DART MaskNet is natively a **single-channel probability gate** predicting continuous degradation presence $p(\text{defect}) \in [0, 1]$ to gate recurrent temporal feature warping. It has **no internal concept of defect classes** (scratches vs. dust). At sequence boundaries (e.g., Frame 0), DART lacks bidirectional temporal optical flow context, resulting in elevated false alarms on intact scene structures. Furthermore, raw soft masks bloom significantly (dilating to $\approx 2\times$ authentic defect width at default threshold 160).
2. **Channel-Mapping Fallback Bug (`scripts/candidate_detector_eval.py`):**
   The benchmark harness contained an arbitrary fallback:
   ```python
   uncat = (binary > 0) & ~categorized
   if np.any(uncat):
       d_mask = np.where(uncat, np.uint8(255), d_mask)
   ```
   Any defect proposal that failed the rigid heuristic scratch filter was **unconditionally dumped into the dust channel**.
3. **Axis-Aligned Bounding Box & Safety Clamp Flaws in `split_defect_components`:**
   - **Tilted Scratch Bounding Box Collapse:** The filter required $cw \le \max(8, \text{int}(w \times 0.06))$. For a $128 \times 128$ frame, this required $cw \le 8$. However, even a 1-pixel hairline scratch tilted by just $4^\circ$ sweeps $\Delta x = 128 \times \tan(4^\circ) \approx 9\text{ px}$, producing an axis-aligned bounding box $cw = 10\text{ to }14 > 8$. Consequently, **100% of synthetic scratches were rejected by the scratch filter** and subsequently dumped into the dust channel by the fallback bug.
   - **Safety Clamp Wipeout:** The $2.5\%$ maximum mask area cutoff (`MAX_MASK_FRACTION = 0.025`) wiped valid scratch masks to zero on heavily damaged frames (e.g., `absdeg_010` frame 0 with 2.66% GT damage). Combined with the fallback bug, all 23,334 wiped scratch pixels were redirected into `dust_mask`, leaving `scratch_mask` completely empty.

### Resolution of the 5,192 Unmatched True Positives
In the baseline report (`.cache/benchmark_suite_report_validated_20261111_research/benchmark_report.json`), the synthetic dataset reported:
$$\text{Overall TP} = 14,352, \quad \text{Scratch TP} = 0, \quad \text{Dust TP} = 9,160, \quad \text{Difference} = 5,192$$
Our per-pixel audit accounts for this discrepancy down to the individual pixel:
- **5,120 pixels:** Authentic Ground-Truth scratch pixels detected by DART, but misclassified as dust due to the tilted bounding-box filter failure ($cw=14 > 8$) and the uncategorized-to-dust fallback dump.
- **72 pixels:** Authentic Ground-Truth dust cluster pixels detected by DART, but misclassified as scratches due to elongated transient cluster geometries.
$$\mathbf{5,120} \text{ (Scratch}\to\text{Dust)} + \mathbf{72} \text{ (Dust}\to\text{Scratch)} = \mathbf{5,192} \text{ Unmatched True Positives}$$

---

## 2. Per-Pixel Class Confusion Matrices (Task 1 & Task 7)

### A. Synthetic Benchmark (N=20 Frames, 327,680 Pixels)
Evaluated across `synthetic_scratch` (10 frames) and `synthetic_dust` (10 frames):

| Ground Truth Class | Pred Clean (Intact) | Pred Scratch Only | Pred Dust Only | Pred Both | Total GT Pixels | Class Recall |
|---|---:|---:|---:|---:|---:|---:|
| **GT Clean (Background)** | 302,772 | 69 | 10,419 | 0 | 313,260 | **96.65% TNR** |
| **GT Scratch** | 0 | **0** | **5,120** | 0 | 5,120 | **0.00%** *(100% leaked to dust)* |
| **GT Dust** | 68 | **72** | **9,160** | 0 | 9,300 | **98.49%** |

#### Dual-Evaluation Metric Decomposition:
- **Defect Localization (Defect vs. Clean Background):**
  - $\text{TP} = 5,120 + 72 + 9,160 = 14,352$
  - $\text{FP} = 69 + 10,419 = 10,488$
  - $\text{FN} = 68$
  - $\text{Localization Precision} = 14,352 / (14,352 + 10,488) = \mathbf{57.78\%}$
  - $\text{Localization Recall} = 14,352 / (14,352 + 68) = \mathbf{99.53\%}$
  - $\text{Localization F1} = \mathbf{0.7311}$
- **Defect Classification (Scratch vs. Dust on Detected Defects):**
  - Scratch Classification Recall: $0 / 5,120 = \mathbf{0.00\%}$
  - Dust Classification Recall: $9,160 / 9,232 = \mathbf{99.22\%}$
  - Scratch-to-Dust Leakage: $5,120 / 5,120 = \mathbf{100.00\%}$
  - Dust-to-Scratch Leakage: $72 / 9,232 = \mathbf{0.78\%}$

### B. AbsoluteDegradation Benchmark (N=12 Frames, 10,317,932 Valid Pixels)
Evaluated across clips `010` and `020` (excluding 197,668 ignore-band pixels where $0 < \text{opacity} < 0.10$):

| Ground Truth Class | Pred Clean (Intact) | Pred Scratch Only | Pred Dust Only | Pred Both | Total Valid GT Pixels |
|---|---:|---:|---:|---:|---:|
| **GT Clean (Background)** | 10,019,432 | 44,836 | 147,482 | 0 | 10,211,750 |
| **GT Scratch** | 9,256 | **61,201** | **35,725** | 0 | 106,182 |

#### Analysis:
- Authentic scratch detection: $\text{Overall TP} = 61,201 + 35,725 = 96,926$ ($91.28\%$ localization recall).
- **35,725 scratch pixels were leaked into the dust channel** (predominantly on Frame 0 where the fraction clamp wiped the mask, and on wide/curved scratches).
- **147,482 clean background pixels were misclassified as dust** due to scene edges and high-contrast vertical structures being dumped into dust.

---

## 3. Root Cause Attribution (Task 2)

| Component | Layer | Observed Failure Mechanism | Impact on Benchmark |
|---|---|---|---|
| **DART MaskNet** | Model Architecture | Single-channel recurrent probability gate. Predicts $p(\text{defect}) \in [0, 1]$. No scratch vs dust heads. | Requires artificial heuristic post-processing to separate classes. |
| **DART Temporal Warping** | Boundary Processing | Lack of past temporal frames at $t=0$ means optical flow cannot verify bilateral temporal persistence. | Elevated confidence on static edges on Frame 0 ($FPR = 3.61\%$ on `absdeg_010`). |
| **DART Confidence Profile** | Model Output | Mask blooming around defect boundaries at default threshold 160 (predicted stroke width $\approx 12.4\text{ px}$ vs. true $3.0\text{ px}$). | Inflated mask area, elevated FPR ($3.35\%$), component width $cw$ exceeds rigid thresholds. |
| **`split_defect_components`** | Channel Separation | Axis-aligned bounding box constraint: $cw \le \max(8, \text{int}(w \times 0.06))$. | Rejects tilted scratches ($cw \ge 10$) despite high length-to-width aspect ratios ($ch/cw > 6.5$). |
| **`split_defect_components`** | Guard Rails | Hard cutoff: `np.count_nonzero(mask) / mask.size > 0.025` wipes mask to 0. | Wipes frames with realistic heavy scratch damage ($2.66\%$ on `absdeg_010` frame 0). |
| **`candidate_detector_eval.py`** | Postprocessing Fallback | Arbitrary unassigned dump: `d_mask = np.where(uncat, 255, d_mask)`. | Scratches, wiped masks, and edge artifacts are unconditionally converted into dust. |
| **`run_defect_benchmark_suite.py`** | Evaluation Pipeline | Channel-specific metrics evaluate only matching damage records without recording cross-channel leakage. | Scratch channel reported $\text{TP}=0$ while overall reported $\text{TP}=14,352$, concealing the underlying leakage. |

---

## 4. Comprehensive Per-Frame Audit Across All Clips (Task 3)

### A. Synthetic Scratch Sequence (`synthetic_scratch`, 10 Frames)

| Frame | Filename | GT Scratch Px | Pred Scratch Px | Pred Dust Px | TP | FP | FN | Precision | Recall | F1 | Status & Diagnostic Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **00** | `000001.png` | 0 | 13 | 318 | 0 | 331 | 0 | 0.000 | 0.000 | 0.000 | **False-Positive Boundary:** GT is empty, but DART predicts 331 FPs due to boundary optical flow initialization. |
| **01** | `000002.png` | 0 | 0 | 161 | 0 | 161 | 0 | 0.000 | 0.000 | 0.000 | **False-Positive Boundary:** GT is empty; 161 FPs dumped into dust. |
| **02** | `000003.png` | 640 | 23 | 1,333 | 640 | 716 | 0 | 0.472 | 1.000 | 0.641 | **FP-Heavy / Scratch Onset:** Scratch first appears. DART detects scratch, but blooms to 1,356 px; 1,333 px dumped into dust. |
| **03** | `000004.png` | 640 | 0 | 1,145 | 640 | 505 | 0 | 0.559 | 1.000 | 0.717 | Scratch detected; 100% dumped into dust channel. |
| **04** | `000005.png` | 640 | 0 | 1,027 | 640 | 387 | 0 | 0.623 | 1.000 | 0.768 | Steady-state tracking; 100% dumped into dust. |
| **05** | `000006.png` | 640 | 0 | 969 | 640 | 329 | 0 | 0.660 | 1.000 | 0.796 | Steady-state tracking; 100% dumped into dust. |
| **06** | `000007.png` | 640 | 8 | 1,010 | 640 | 378 | 0 | 0.629 | 1.000 | 0.772 | Steady-state tracking; 99% dumped into dust. |
| **07** | `000008.png` | 640 | 0 | 953 | 640 | 313 | 0 | 0.672 | 1.000 | 0.804 | Steady-state tracking; 100% dumped into dust. |
| **08** | `000009.png` | 640 | 0 | 969 | 640 | 329 | 0 | 0.660 | 1.000 | 0.796 | Steady-state tracking; 100% dumped into dust. |
| **09** | `000010.png` | 640 | 0 | 886 | 640 | 246 | 0 | **0.722** | 1.000 | **0.839** | **Best Frame:** Lowest FP (246 px), highest precision (0.722) and F1 (0.839). |

### B. Synthetic Dust Sequence (`synthetic_dust`, 10 Frames)

| Frame | Filename | GT Dust Px | Pred Scratch Px | Pred Dust Px | TP | FP | FN | Precision | Recall | F1 | Status & Diagnostic Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **00** | `000001.png` | 1,036 | 46 | 1,841 | 1,032 | 855 | 4 | 0.547 | 0.996 | 0.706 | **FP-Heavy Boundary:** High false positives (855 px) from mask bloom; 46 px misclassified as scratch. |
| **01** | `000002.png` | 771 | 0 | 1,348 | 771 | 577 | 0 | 0.572 | 1.000 | 0.728 | Clean dust detection with 100% recall. |
| **02** | `000003.png` | 725 | 0 | 1,298 | 722 | 576 | 3 | 0.556 | 0.996 | 0.714 | Clean dust detection. |
| **03** | `000004.png` | 686 | 0 | 978 | 672 | 306 | 14 | **0.687** | 0.980 | **0.808** | **Best Frame:** Highest precision (0.687), highest F1 (0.808), lowest FP (306 px). |
| **04** | `000005.png` | 1,072 | 51 | 1,856 | 1,071 | 836 | 1 | 0.562 | 0.999 | 0.719 | FP-heavy (836 px); 51 px misclassified as scratch. |
| **05** | `000006.png` | 1,119 | 0 | 1,756 | 1,100 | 656 | 19 | 0.626 | 0.983 | 0.765 | Good defect coverage. |
| **06** | `000007.png` | 1,010 | 0 | 1,716 | 1,010 | 706 | 0 | 0.589 | 1.000 | 0.741 | 100% recall. |
| **07** | `000008.png` | 1,071 | 0 | 1,941 | 1,070 | 871 | 1 | 0.551 | 0.999 | 0.710 | **Highest FP Frame:** 871 false positive pixels. |
| **08** | `000009.png` | 998 | 0 | 1,758 | 993 | 765 | 5 | 0.565 | 0.995 | 0.721 | Stable tracking. |
| **09** | `000010.png` | 812 | 0 | 1,436 | 791 | 645 | 21 | 0.551 | 0.974 | **0.704** | **Worst Frame:** Lowest F1 (0.704) and highest FN (21 px). |

### C. AbsoluteDegradation Clips (`absdeg_010` and `absdeg_020`, 12 Frames)

| Clip | Frame | Filename | GT Scratch Px | Pred Scratch Px | Pred Dust Px | TP | FP | FN | F1 | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **010** | **00** | `000001.png` | 7,927 | **0** | **39,068** | 7,678 | 31,390 | 249 | **0.327** | **Worst / Most Broken Frame:** Safety clamp wiped scratch to 0; all 39,068 px dumped into dust. |
| **010** | **01** | `000002.png` | 7,882 | 15,199 | 11,061 | 6,659 | 19,601 | 1,223 | 0.390 | Fraction < 2.5%; scratches pass into scratch channel. |
| **010** | **02** | `000003.png` | 14,169 | 8,621 | 20,539 | 13,694 | 15,466 | 475 | **0.632** | **Best Frame:** Peak F1 in clip 010 (0.632), high recall (96.6%). |
| **010** | **03** | `000004.png` | 6,954 | 7,788 | 13,688 | 6,670 | 14,806 | 284 | 0.469 | Steady-state tracking. |
| **010** | **04** | `000005.png` | 9,274 | 7,917 | 15,338 | 7,880 | 15,375 | 1,394 | 0.484 | Steady-state tracking. |
| **010** | **05** | `000006.png` | 6,885 | 6,207 | 18,794 | 5,580 | 19,421 | 1,305 | 0.350 | Elevated false positives on background. |
| **020** | **00** | `000001.png` | 7,927 | 7,770 | 14,714 | 7,640 | 14,844 | 287 | 0.502 | Boundary frame; 96.4% recall. |
| **020** | **01** | `000002.png` | 7,882 | 8,754 | 13,268 | 7,060 | 14,962 | 822 | 0.472 | Lowest F1 in clip 020. |
| **020** | **02** | `000003.png` | 14,169 | 12,711 | 13,485 | 13,297 | 12,899 | 872 | **0.659** | **Best Frame:** Highest F1 in dataset (0.659), 93.8% recall. |
| **020** | **03** | `000004.png` | 6,954 | 10,021 | 9,899 | 6,743 | 13,177 | 211 | 0.502 | High recall (97.0%). |
| **020** | **04** | `000005.png` | 9,274 | 11,980 | 7,648 | 7,861 | 11,767 | 1,413 | 0.544 | Robust scratch detection. |
| **020** | **05** | `000006.png` | 6,885 | 9,069 | 5,705 | 6,164 | 8,610 | 721 | 0.569 | Lowest FPR in dataset (0.99%). |

---

## 5. Negative Controls Empirical Evaluation (Task 5)

To establish DART's susceptibility to non-defect scene features, 6-frame negative control sequences ($256 \times 256$, zero ground-truth defects, 393,216 total evaluated pixels per sequence) were evaluated using genuine DART GPU inference:

| Negative Control Regime | Simulated Visual Content | Total FP Pixels (>160) | False Positive Rate (FPR) | Raw Soft Mask Mean | Channel Destination | Primary Failure Mode |
|---|---|---:|---:|---:|---|---|
| **Clean Flat Gray** | Uniform neutral gray (128) | **0** | **0.000%** | 65.6 / 255 | N/A | **Pass.** Zero false alarms on flat fields. |
| **Film Grain** | Gaussian texture ($\sigma=15$) | **187** | **0.048%** | 57.3 / 255 | Dust (100%) | **Pass.** Excellent resistance to pure stationary/random noise. |
| **High-Contrast Edges** | 3 static sharp vertical bars ($Y=60\to 220$) | **46,906** | **11.929%** | 63.7 / 255 | Dust (100% via uncat dump) | **Critical Failure.** DART hallucinates defects along sharp vertical step edges. Uncat dump dumped 46k px into dust. |
| **Central Portrait / Face** | Oval face-tone portrait with smooth gradients | **27,835** | **7.079%** | 59.4 / 255 | Dust (100% via uncat dump) | **High Risk.** False alarms along facial contours and subject boundaries. |
| **Moving Objects** | Translating vertical bar ($dx = 20\text{ px/frame}$) | **18,004** | **4.579%** | 61.1 / 255 | Dust (100% via uncat dump) | **Flow Artifact.** RAFT optical flow warping errors trigger moving boundary false positives. |

---

## 6. Threshold Sensitivity & Precision-Recall Sweep (Task 6)

Evaluated on the synthetic dataset (20 frames, 327,680 total pixels, 14,420 ground truth defect pixels) without tuning on the held-out test sets:

| Soft Mask Gate Threshold | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Precision | Recall | Pixel F1 | IoU | False Positive Rate (FPR) | Mask Coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **32** | 14,419 | 224,695 | 1 | 88,565 | 0.0603 | 0.9999 | 0.1137 | 0.0603 | 71.731% | 72.972% |
| **64** | 14,417 | 104,177 | 3 | 209,083 | 0.1216 | 0.9998 | 0.2168 | 0.1216 | 33.256% | 36.192% |
| **96** | 14,414 | 44,806 | 6 | 268,454 | 0.2434 | 0.9996 | 0.3915 | 0.2434 | 14.303% | 18.073% |
| **128** | 14,406 | 20,576 | 14 | 292,684 | 0.4118 | 0.9990 | 0.5832 | 0.4116 | 6.568% | 10.676% |
| **160 (Baseline)** | 14,352 | 10,488 | 68 | 302,772 | 0.5778 | 0.9953 | 0.7311 | 0.5762 | 3.348% | 7.581% |
| **192** | 14,185 | 5,627 | 235 | 307,633 | 0.7160 | 0.9837 | 0.8288 | 0.7077 | 1.796% | 6.046% |
| **224 (Optimal Localization)** | **13,489** | **1,959** | **931** | **311,301** | **0.8732** | **0.9354** | **0.9032** | **0.8235** | **0.625%** | **4.714%** |
| **240** | 11,551 | 695 | 2,869 | 312,565 | 0.9432 | 0.8010 | 0.8663 | 0.7642 | 0.222% | 3.737% |

### Key Trade-Off Insights:
1. **Operating Point Defect:** At the default threshold of 160, DART produces 10,488 false positive pixels ($FPR = 3.35\%$), because continuous soft masks remain $\approx 160-200$ confidence across a halo surrounding each defect.
2. **Optimal Localization Frontier:** Increasing the threshold to **224** drops false positives by **81.3%** (from 10,488 down to 1,959) while preserving **93.54% Recall**, boosting Pixel F1 from 0.7311 to **0.9032** and IoU from 0.5762 to **0.8235**.
3. **Threshold vs. Classification Independence:** Tuning the threshold alters mask width, but **cannot resolve scratch vs. dust classification**, because DART's soft mask is inherently single-channel.

---

## 7. Visual Crop Diagnostics (Task 4)

Native-resolution, 7-panel diagnostic composites were generated and saved in [`docs/investigation/visual_crops/`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/):

1. **[`synthetic_scratch_frame_0_baseline_bug.png`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/synthetic_scratch_frame_0_baseline_bug.png):**  
   Demonstrates boundary false alarms on an empty frame (GT=0). Shows raw DART soft mask predicting 331 defect pixels on flat background, which are split into 13 scratch and 318 dust pixels.
2. **[`synthetic_scratch_frame_2_baseline_bug.png`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/synthetic_scratch_frame_2_baseline_bug.png) vs. [`synthetic_scratch_frame_2_corrected.png`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/synthetic_scratch_frame_2_corrected.png):**  
   Side-by-side comparison of the scratch onset frame.  
   - *Baseline Bug:* Raw DART detects the full vertical scratch, but because the $4^\circ$ tilt produces $cw=19 > 8$, Panel 5 (Scratch) is empty, and Panel 6 (Dust) receives all 1,333 scratch pixels.  
   - *Corrected:* Scratch channel correctly captures the tilted scratch ($TP=640$), leaving the dust channel clean ($d\_pred = 233$ small noise pixels).
3. **[`absdeg_010_frame_0_crop_baseline_bug.png`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/absdeg_010_frame_0_crop_baseline_bug.png) vs. [`absdeg_010_frame_0_crop_corrected.png`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/absdeg_010_frame_0_crop_corrected.png):**  
   High-resolution ($400 \times 400$) crop on AbsoluteDegradation clip 010.  
   - *Baseline Bug:* Vertical scratches visible in Source and GT are completely absent from Panel 5 (Scratch Channel is 0 px due to the 2.5% clamp wipeout), while Panel 6 (Dust Channel) contains 39,068 pixels of scratches and scene edges.  
   - *Corrected:* Scratches pass properly into the Scratch Channel ($TP=5,210$), and dust false positives drop by $>91\%$.
4. **[`synthetic_dust_frame_0_baseline_bug.png`](file:///I:/ai-remaster-pipeline/docs/investigation/visual_crops/synthetic_dust_frame_0_baseline_bug.png):**  
   Illustrates mask blooming on transient dust clusters and boundary optical flow effects.

---

## 8. Benchmark Metrics Before vs. After Correction (Task 10)

Comparative evaluation results between the original baseline (`.cache/benchmark_suite_report_validated_20261111_research/`) and the audited/corrected evaluation (`.cache/benchmark_suite_report_audit_20261111/`):

### A. Synthetic Dataset Benchmark

| Metric Component | Baseline Research Report (Before) | Audited / Corrected (After) | Delta & Significance |
|---|---:|---:|---|
| **Scratch TP** | **0** | **5,120** | **+5,120 TP** (100% scratch recall restored) |
| **Scratch FN** | **5,120** | **0** | **-5,120 FN** (Zero missed scratches) |
| **Scratch Pixel F1** | **0.0000** | **0.7914** | From total failure to robust detection |
| **Scratch Recall** | **0.0000** | **1.0000** | Full damage trajectory recovered |
| **Scratch Precision** | 0.0000 | 0.6548 | Elevated precision |
| **Dust TP** | 9,160 | **8,178** | **87.9% dust recall retained** without uncat dump |
| **Dust FN** | 140 | **1,122** | Only small fraction lost to tight non-cluster boundaries |
| **Dust Pixel F1** | 0.7262 | **0.6973** | High F1 without artificial scratch dump |
| **Dust FP** | 6,768 | **5,979** | Reduced FP from scratch/edge leakage elimination |
| **Dust FPR** | 4.379% | **3.869%** | Improved false-alarm rate |

### B. AbsoluteDegradation Benchmark

| Metric Component | Baseline Research Report (Before) | Audited / Corrected (After) | Delta & Significance |
|---|---:|---:|---|
| **Scratch TP** | 61,201 | **83,700** | **+22,499 authentic scratch pixels recovered** |
| **Scratch FN** | 44,981 | **22,482** | **-50.0% missed scratches** |
| **Scratch Pixel F1** | 0.5768 | **0.6319** | Substantial gain in scratch F1 |
| **Scratch Recall** | 0.5764 | **0.7883** | Significant increase in defect recall |
| **Overall False Positives** | 192,318 | **147,650** | **-44,668 false alarm pixels eliminated** |
| **Intact Pixel FPR** | 1.883% | **1.445%** | Lower false-alarm rate |

### C. Multi-Candidate Macro Benchmark Comparison (Audited)

Aggregated across all datasets in `.cache/benchmark_suite_report_audit_20261111/benchmark_report.md`:

| Candidate Detector | Backend Provenance | Evaluated Datasets | Scratch F1 | Dust F1 | Overall Pixel F1 | Macro F1 | Intact FPR | Frame Latency | Peak Memory |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **farneback_3f** | `in_tree_farneback_3f` | `synthetic, absdeg` | 0.2810 | 0.2024 | **0.2214** | 0.1884 | 0.605% | 612.6 ms | 931 MB |
| **dis_flow** | `in_tree_dis_flow` | `synthetic, absdeg` | 0.2913 | 0.2314 | **0.2202** | 0.1970 | 0.763% | 397.8 ms | 931 MB |
| **ridge** | `in_tree_ridge` | `synthetic, absdeg` | 0.0887 | N/A | **0.0907** | 0.0842 | **0.055%** | **45.8 ms** | 931 MB |
| **consensus_5f** | `in_tree_consensus_5f` | `synthetic, absdeg` | 0.1167 | 0.2076 | **0.0901** | 0.1303 | 1.332% | 371.7 ms | 931 MB |
| **legacy** | `in_tree_spatial_scratch` | `all 4 datasets` | 0.0258 | N/A | **0.0292** | 0.0487 | 5.530% | 98.5 ms | 931 MB |
| **dart_harness** | `genuine_dart_inference` | `synthetic, absdeg` | **0.6394** | **0.6973** | **0.5323** | **0.5694** | 1.445% | 1,917.0 ms | 932 MB |

---

## 9. Verification & Regression Test Suite (Task 9)

A dedicated, permanent regression test suite has been added in [`tests/test_defect_class_assignment_audit.py`](file:///I:/ai-remaster-pipeline/tests/test_defect_class_assignment_audit.py):
1. `test_tilted_scratch_classified_as_scratch_not_dust`: Proves $4^\circ$ tilted parametric scratches are classified as scratches and not leaked to dust.
2. `test_hairline_tilted_scratch_not_rejected_by_bounding_box`: Proves 1-pixel scratches with tilt are not rejected by axis-aligned bounding box limits.
3. `test_uncategorized_proposals_not_dumped_into_dust`: Proves irregular diagonal blobs are not dumped into the dust channel.
4. `test_dense_authentic_scratch_frame_not_wiped_by_fraction_cutoff`: Proves multi-scratch frames ($>2.5\%$ damage area) are preserved.
5. `test_class_confusion_and_unmatched_tp_identity`: Enforces the mathematical identity $5,120 + 72 = 5,192$.
6. `test_localization_vs_classification_metrics_distinction`: Validates decoupling of spatial defect detection from semantic defect categorization.
7. `test_bloomed_dust_cluster_classified_as_dust_not_rejected`: Proves compact bloomed dust clusters (e.g. 450px) are recognized as dust and not discarded.
8. Automated runner script [`scripts/run_negative_controls.py`](file:///I:/ai-remaster-pipeline/scripts/run_negative_controls.py) added for reproducible negative-controls evaluation.

All 81 defect benchmark tests pass cleanly in the isolated environment:
```powershell
.venv\Scripts\python.exe -m pytest tests/test_defect_class_assignment_audit.py tests/test_defect_mask_benchmark.py tests/test_benchmark_repairs_regression.py tests/test_absolute_scratch_benchmark.py tests/test_benchmark_infrastructure.py tests/test_candidate_detectors.py tests/test_dataset_adapters.py tests/test_defect_detector_registry.py tests/test_ridge_integration_comparison.py tests/test_ridge_pr24_replay.py tests/test_temporal_consistency.py
# 81 passed in 6.86s
```

---

## 10. Architectural Recommendations & Detector Roadmap (Task 10)

### A. Commercial ARP Production Recommendation
1. **Maintain Strict Prohibition on DART in Commercial Runtime:**
   - DART is derived from RTN code without commercial clearance (research-only).
   - Its latency ($\approx 1.9\text{ s/frame}$) and memory footprint make it unsuitable for high-throughput batch restoration.
   - Its high false-positive rate on high-contrast vertical structures ($FPR \approx 11.9\%$) would cause devastating over-repair on architectural scenes, doorframes, and window mullions.
2. **Path for In-Tree Classical Detectors:**
   - **`dis_flow` & `farneback_3f` remain the recommended production detectors** for commercial use. They achieve low FPR ($0.60\% - 0.76\%$), zero license contamination (100% BSD/Apache), and run at $400-600\text{ ms/frame}$.
   - **Do not adopt complex morphological heuristic splitters** for production unless they utilize minimum-area rotated bounding boxes and aspect ratio bounds that accommodate natural scratch tilt.

### B. Research Detector Roadmap (If Further Detector Work Is Justified)
Is further deep-learning detector development justified? **Yes, but NOT by patching DART postprocessing.**
- DART's fundamental flaw is being a **single-channel reconstruction gate**, not a multi-class semantic segmentation model.
- If a deep-learning detector is desired for future research or an enterprise tier:
  1. Train a native two-head or multi-class architecture (e.g., SegFormer or lightweight HRNet) with explicit disjoint heads: $p(\text{scratch})$, $p(\text{dust})$, and $p(\text{intact})$.
  2. Train on paired authentic historical film scans (such as ARTeFACT and FDS) rather than purely synthetic noise.
  3. Include explicit negative-control loss penalties on high-contrast static edges, moving camera pans, and human faces to suppress structural false alarms.
