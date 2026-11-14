---
title: "Scratch and Dust Mask Detection System Investigation & Benchmark Plan"
created: "2026-11-09T15:58:35+01:00"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "deep"
task_complexity: "medium"
---

# Scratch and Dust Mask Detection System Investigation & Benchmark Plan

## Problem Statement

### Context & Current State
The archive restoration pipeline (`ai-remaster-pipeline`) currently includes an experimental CPU-based temporal defect mask generator (`scripts/film_defect_masks.py`). It applies dense OpenCV Farneback optical flow across a 3-frame temporal window (`t-1, t, t+1`) to propose binary masks (0=clean, 255=defect) for transient dust and small blotches without modifying source pixels.

### Core Deficiencies
1. **Algorithmic & Speed Limitations**: Farneback optical flow is computationally demanding on high-resolution film scans (1080p/4K) and prone to motion compensation blur, leading to false positives on rapid camera pans or fine facial details.
2. **Defect Type Blindspots**: The 3-frame consensus mechanism is blind to multi-frame persistent defects, especially vertical emulsion scratches that span across 2 or more consecutive frames.
3. **Lack of Quantitative Benchmark Ground Truth**: No automated harness exists to measure pixel-level Precision, Recall, F1, and IoU against verified ground truth. Improvements cannot be empirically scored against regressions.
4. **Licensing Constraints**: Advanced AI models like DART possess unresolved copyright and commercial provenance issues (unlicensed RTN components). The core pipeline requires 100% unencumbered, commercial-safe algorithms.

### Scope of Current Phase
Perform repository analysis, design the pluggable detector architecture, specify a synthetic + archival benchmark harness, create GitHub issues for implementation tracking, and formulate a prioritized implementation plan with acceptance criteria—without modifying production code.

---

## Requirements

### Primary Objective: Detection Quality & Detail Preservation
The overarching goal is maximizing detection accuracy while minimizing damage to authentic film content:
- Maximize precision and recall across transient and persistent defects.
- Minimize false positives on high-frequency authentic details (fine textures, film grain, faces, moving edges, text/titles).
- Ensure temporal consistency across sequences without flicker or masking jitter.

### Functional Requirements
- **REQ-1 (Comprehensive Defect Taxonomy)**: Support detection of distinct film degradation classes:
  - Transient single-frame dust, specks, dirt, and chemical blotches.
  - Multi-frame persistent defects, especially vertical emulsion scratches extending across $\ge 2$ consecutive frames.
  - Challenging scene dynamics (rapid camera pans, high grain fields, motion blur, and exposure fluctuations).
- **REQ-2 (Empirical Evaluation of Candidate Detectors)**: Do not commit prematurely to any specific detector algorithm. Empirically benchmark candidate approaches on identical test data:
  - *Current baseline*: Farneback optical flow 3-frame consensus (`film_defect_masks.py`).
  - *Classical candidates*: Dense Inverse Search (DIS) optical flow, Multi-scale Ridge detection, Directional morphological temporal filters, 5-frame temporal median/rank filters.
  - *Neural / AI candidates*: DART MaskNet, lightweight U-Net / transformer segmenters (subject to REQ-6 licensing).
- **REQ-3 (Standardized Benchmark Harness & Separate Metrics Reporting)**:
  - Test on identical curated datasets combining controlled synthetic ground truth (e.g. AbsoluteDegradation / parametric damage generator with exact binary labels) and manually verified archival film clips (*Neighbors* 1920).
  - Report detection quality metrics (**Precision, Recall, Pixel F1, IoU, and Unaffected Pixel False Positive Rate**) **strictly separated** for (a) Scratches vs. (b) Dust/Blotches.
- **REQ-4 (GitHub Issue Tracking)**: Create granular, actionable GitHub issues in `DennLie/ai-remaster-pipeline` following repository triage standards (`ready-for-agent` / `ready-for-human`) to drive execution.

### Non-Functional Requirements & Hypotheses
- **REQ-5 (Performance as Hypothesis, Not Gate)**: Treat speedup targets as empirical evaluation hypotheses. First establish accurate computational baselines (CPU ms/frame, RAM, VRAM) on standard resolutions (720p, 1080p, 4K). Detection quality takes precedence over speed.
- **REQ-6 (Licensing & Commercial Integrity)**: Every candidate method must undergo a documented licensing audit. Candidate code/weights must be clearly classified as either commercial-safe (unencumbered MIT/BSD/Apache-2.0) or restricted/evaluation-only (e.g. DART RTN dependencies) before any production integration consideration.
- **REQ-7 (Zero Production Changes in Planning Phase)**: Strictly prohibit modifying production code during this research, benchmarking, and issue-creation phase.

---

## Approach

### Selected Approach: Empirical Multi-Candidate Benchmark & Evaluation Harness
Rather than pre-selecting a single detector algorithm, the selected approach establishes a rigorous, reproducible evaluation harness. This framework evaluates candidate methods side-by-side against identical test materials, reporting separate metrics for Scratches vs. Dust/Blotches to make data-driven architectural decisions.

### Candidate Methods under Evaluation
1. **Baseline**: 3-frame Farneback Optical Flow Temporal Consensus (`scripts/film_defect_masks.py`).
2. **Candidate A (DIS Optical Flow)**: Dense Inverse Search optical flow consensus (hypothesized for sharper motion boundary tracking and reduced computation time).
3. **Candidate B (Multi-scale Ridge + Directional Temporal)**: Steerable line filtering for continuous vertical scratches across consecutive frames (addressing the known ~6% recall gap identified in `docs/absolute-scratch-channel-experiment.md`).
4. **Candidate C (Multi-Frame Temporal Consensus)**: 5-frame temporal window ($t-2 \dots t+2$) with rank/median filtering to suppress 2-frame persistent blotches and jitter.
5. **Candidate D (Neural Segmentation / DART MaskNet)**: Supervised mask prediction evaluated in an isolated harness for complex organic tears and stains.

### Alternatives Considered & Rejected
- **Premature Algorithmic Pivot (e.g. immediate switch to DIS or Ridge)**: Rejected because the November 6 AbsoluteDegradation experiment showed Ridge had low recall (~5.9% on test content) while Temporal had elevated false positives (~0.42%). Algorithmic changes must be validated by empirical benchmark data before adoption.
- **Neural-Only Pipeline**: Rejected due to unresolved third-party RTN licensing provenance in DART weights and high GPU dependency requirements for standard CPU restoration nodes.

### Decision Matrix
| Evaluation Criterion | Weight | Approach: Multi-Candidate Benchmark (Selected) | Alternative 1: Premature Pivot | Alternative 2: Neural-Only |
| :--- | :---: | :--- | :--- | :--- |
| **Detection Quality & Preservation** | 35% | **5/5**: Measures exact P/R/F1 and authentic detail preservation | **2/5**: High risk of masking fine textures/grain without verification | **4/5**: Strong on severe damage, but risks hallucination/soft edges |
| **Defect Scope (Scratches + Dust)** | 25% | **5/5**: Evaluates specialized channels independently for dust vs. scratches | **2/5**: Single algorithm cannot cover both transient dust and vertical scratches | **3/5**: MaskNet mixes defect types without dedicated scratch persistence |
| **Licensing & Commercial Safety** | 20% | **5/5**: Evaluates open classical methods alongside isolated AI harnesses | **5/5**: Classical only, but lacks rigor | **1/5**: Unresolved RTN provenance creates commercial liability |
| **Operational Feasibility & Baselines** | 20% | **5/5**: Empirically profiles CPU/GPU baselines on standard footage | **3/5**: Untested performance on varied resolutions | **2/5**: Strict CUDA requirement excludes CPU workstations |
| **Weighted Total** | **100%** | **5.00 / 5.0** | **2.85 / 5.0** | **2.80 / 5.0** |

---

## Architecture & Evaluation Methodology

### Modular Benchmark Engine Architecture
```
                      ┌──────────────────────────────────────────────┐
                      │              Evaluation Datasets             │
                      │  - Synthetic GT (Exact Mathematical Labels)  │
                      │  - Archival GT (Annotated Real Damaged Film) │
                      │  - Expansion Set (Diverse Archival Footage)  │
                      └──────────────────────┬───────────────────────┘
                                             │ (Sequence-level Splits)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          Benchmark Scorer Engine             │
                      │  - Identical Preprocessing & Execution Rules │
                      │  - Separate Channels: Scratch vs. Dust       │
                      │  - Temporal Consistency Scorer               │
                      │  - Visual Inspection Artifact Generator      │
                      └──────────────────────┬───────────────────────┘
                                             │
      ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
      ▼                  ▼                   ▼                   ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Farneback   │   │   DIS Flow   │   │  Multi-scale │   │  5-Frame     │   │ DART MaskNet │
│  (Baseline)  │   │  Candidate   │   │ Ridge Filter │   │  Consensus   │   │ (Eval-only)  │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          Outputs & Visual Diagnostics        │
                      │  - Metrics: P / R / F1 / IoU / Unaffected FP │
                      │  - Temporal: Flicker, Continuity, Drift      │
                      │  - Visuals: Multi-panel Side-by-Side Diffs   │
                      └──────────────────────────────────────────────┘
```

### Dataset Design & Ground Truth Integrity
- **Dataset Splitting**: Strict sequence-level train / validation / test splits to prevent temporal autocorrelation and data leakage across adjacent frames.
- **Ground Truth Stratification**:
  - *Synthetic Ground Truth*: Injected onto certified pristine digital or reference frames, generating exact mathematical labels ($y_{true}$). Evaluates parametric scratch profiles (orientations, widths 1–8px, variable opacity, temporal durations 1–30 frames, lateral drift) vs. transient dust blotches.
  - *Archival Pilot Benchmark*: 5 representative 100-frame sequences from *Neighbors* (1920) featuring manually annotated ground truth (labels for authentic defects, authentic clean areas, and ignore/unlabeled regions). Existing authentic archival damage in background frames is explicitly annotated to prevent misclassifying real defects as clean background.
  - *Extensible Benchmark Architecture*: Designed to ingest additional diverse archival film sources (differing gauges, stocks, historical degradation patterns) beyond the initial pilot set.

### Temporal Consistency & Metrics Suite
- **Per-Channel Spatial Metrics**: Precision, Recall, Pixel F1, IoU, and Unaffected Pixel False Positive Rate reported **strictly separated** for:
  - Transient Defects (dust, dirt, specks, chemical blotches).
  - Persistent Defects (continuous running vertical/diagonal scratches).
- **Temporal Consistency Metrics**:
  - *Mask Flicker Rate*: High-frequency frame-to-frame mask alternation on static image regions.
  - *Detection Continuity*: Tracking completeness of persistent scratches across their active lifespan.
  - *Temporal False Positives*: Erroneous masks triggered by rapid camera motion, parallax, or lighting shifts.
  - *Missed Persistent Defects*: Detection dropouts mid-trajectory on continuous scratch lines.

### Visual Diagnostic Suite
- Automated generation of multi-panel comparison reels and contact sheets:
  - Panel A: Original source frame.
  - Panel B: Ground truth label mask (with color-coded defect types).
  - Panel C: Candidate predicted scratch mask.
  - Panel D: Candidate predicted dust mask.
  - Panel E: Color-coded Error Map (Green = True Positive, Red = False Positive, Blue = False Negative).
  - Panel F: Temporal difference strip showing mask evolution across $t-2 \dots t+2$.

### Decoupled Production Interface & Governance
- Retains abstract `BaseDefectDetector` strategy blueprint for future Sonnet 5.5 implementation.
- Strict firewall: AI/neural candidates (DART MaskNet) remain strictly sandboxed in the evaluation harness; only 100% unencumbered, commercial-safe algorithms are eligible for production integration.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **Annotation Ambiguity in Archival Footage** | MEDIUM | HIGH | Use 3-tier label masks (Defect, Clean, Ambiguous/Ignore) and calibrate with exact synthetic ground truth to avoid penalizing borderline grain. |
| **Overfitting to Single Film Stock (*Neighbors* 1920)** | MEDIUM | MEDIUM | Sequence-level train/val/test splits and an extensible architecture designed to ingest varied archival footage (different gauges, exposures, stocks). |
| **Temporal Mask Flicker & Inpainting Churn** | HIGH | MEDIUM | Explicit temporal metrics (Flicker Rate, Continuity, Mid-trajectory Dropout) treated as primary acceptance criteria before adoption. |
| **Licensing / Commercial Provenance Exposure** | HIGH | LOW | Strict architectural firewall keeping neural models (DART / RTN) isolated in evaluation-only test harnesses. Production path requires verified commercial-safe licenses. |
| **Quality vs Performance Compromise** | LOW | MEDIUM | Empirical profiling baselines established first. Detection quality, detail preservation, and temporal stability take strict precedence over arbitrary speed targets. |

---

## Success Criteria

1. Zero production code modified or committed in the repository during this planning phase.
2. Complete, approved Design Document stored under `docs/maestro/plans/2026-11-09-scratch-dust-investigation-design.md`.
3. Five prioritized, detailed GitHub issues created in `DennLie/ai-remaster-pipeline` adhering to triage conventions.
4. Finalized Implementation Plan containing phased work packages, dependency graphs, and testable criteria ready for immediate handoff to Claude Sonnet 5.5.
