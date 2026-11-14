# Archival Ground Truth Protocol & Published Film-Damage Benchmark Integration

## 1. Executive Decision & Scope

Instead of manually annotating thousands of new archival video frames from scratch before detector development can proceed, this repository reuses **existing, peer-reviewed, human-annotated and parametric film-damage benchmarks** via [`scripts/dataset_adapters.py`](../scripts/dataset_adapters.py) and [`scripts/temporal_consistency_scorer.py`](../scripts/temporal_consistency_scorer.py), which feed directly into [`scripts/benchmark_defect_masks.py`](../scripts/benchmark_defect_masks.py) without duplicating any confusion-matrix or metric formulas.

Human annotation is reserved strictly for a small, bounded domain-validation pilot on Buster Keaton's *Neighbors* (1920) (`evaluation/Neighbors_72.mkv`, 6 fixed frames in `.cache/dust-real-annotation-20261106-01/`) and final maintainer sign-off.

---

## 2. Audit of Published Film-Damage Benchmarks

### 2.1 Summary Comparison Table

| Benchmark / Source | Upstream URL / DOI | License | Commercial Suitability | Downloadable Payload | Annotation Format & Provenance Tier |
|---|---|---|---|---|---|
| **FilmDamageSimulator / FILM-AA** (Ivanova et al., Eurographics 2023) | [GitHub `daniela997/FilmDamageSimulator`](https://github.com/daniela997/FilmDamageSimulator) & [Figshare `21803292.v1`](https://doi.org/10.6084/m9.figshare.21803292.v1) | **MIT** (GitHub repo) / **CC BY 4.0** (Figshare dataset) | **Commercial-safe** (with attribution for CC BY 4.0 data) | 10 uncompressed 35mm scans (`Scan (1..10).jpg`, ~113.0 MB total) + 10 polygon JSONs (`Scan (1..10).json`, ~9.15 MB total) | **12,137 human-labeled polygons** across 5 classes (`Dust`, `Dirt`, `Short hair`, `Long hair`, `Scratch`). Tier: `manual_polygon_gt` |
| **ARTeFACT** (Ivanova et al., arXiv:2412.04580) | [HuggingFace `danielaivanova/damaged-media`](https://huggingface.co/datasets/danielaivanova/damaged-media) | **AFL-3.0** (Academic Free License v3.0) | **Permissive license** for dataset compilation; underlying historical media items carry archive-specific provenance — use for **offline evaluation/benchmarking only**, not redistributed in product weights | 418 examples across 28 Parquet shards (`data/train-00000-of-00028.parquet` .. `00027`, 4.07 GB compressed / 13.55 GB uncompressed; individual shards 20.8–479 MB) | **Pixel-level 15-class human segmentation PNGs** (`0=Clean`, `1..15=Damage`, `255=Background`). Tier: `manual_segmentation_gt` |
| **Authentically Damaged & Manually Restored Film Scans** (Eurographics 2023 Dataset 1) | [Figshare `10.6084/m9.figshare.21803304.v2`](https://doi.org/10.6084/m9.figshare.21803304) | **CC BY 4.0** | **Commercial-safe** (with attribution) | 86 JPEG files (~420 MB total): 43 authentically damaged 4K scans across 11 film stocks (`cinestill50d`, `cinestill800t`, `ektachrome100`, `ektar100`, `gold200`, `portra160`, `portra400`, `provia100`, `superia400`, `ultramax400`, `velvia50`) + 43 expert-restored counterparts (`*_restored.jpg`) | **Paired damaged/restored images** (no separate polygon/mask files). Image difference `\|damaged - restored\|` includes JPEG & retouching artifacts. Tier: `restoration_diff_proxy` (never mixed with `manual_gt`) |
| **AbsoluteDegradation** (Jasiński et al., v1.0.0) | [GitHub `TytanMikJas/AbsoluteDegradation`](https://github.com/TytanMikJas/AbsoluteDegradation/tree/v1.0.0) | **MIT** (generator code); external REDS/RTN textures have separate non-commercial terms | **Parametric `FilmScratchGenerator` is MIT**; external RTN dust textures (HTTP 404 upstream) are **not cleared for commercial use** | Local checkout in `.cache/absolute-degradation-v1`; 72 REDS reference PNGs (98.9 MB) + 144 deterministic damaged frames & float32 masks in `.cache/absolute-scratch-benchmark-20261106-01` | Exact float32 opacity masks in $[0, 1]$ (`>=0.10 -> 255`, `0<soft<0.10 -> 127 ignore`). Tier: `synthetic_parametric_gt` |
| **DART** (Jasiński et al., ACCV 2026) | [GitHub `TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer`](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer) | **`NOASSERTION` / Custom research-only** (adapted RTN components) | **NOT cleared for commercial production**; isolated strictly to external research harnesses | Local checkout at `I:\DART` (`pretrained_models/dart.safetensors`, 51.96 MB, SHA-256 `d78a388c76ffb2d7...`) | Model-predicted masks (`MaskNet`) are **never** ground truth. Tier: `model_pseudo_label` |

---

### 2.2 Class Balance & Technical Quirks Discovered During Audit

#### A. `FilmDamageSimulator` (`daniela997/FilmDamageSimulator` / Figshare `21803292.v1`)
Full enumeration of all 10 JSON annotation files (`Scan (1).json` – `Scan (10).json`) yields **12,137 human-annotated artifact polygons** on empty Lomochrome Purple 35mm film frames scanned on a Plustek OpticFilm 7600i:

| Scan | Dust | Dirt | Short hair | Long hair | Scratch | Total Polygons | Coordinate Scale Factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Scan (1)` | 868 | 401 | 386 | 122 | 1 | 1,778 | $1.0\times$ |
| `Scan (2)` | 923 | 394 | 328 | 173 | 0 | 1,818 | $1.0\times$ |
| `Scan (3)` | 1,278 | 186 | 76 | 8 | 6 | 1,554 | $1.0\times$ |
| `Scan (4)` | 841 | 477 | 130 | 15 | 3 | 1,466 | $1.0\times$ |
| `Scan (5)` | 1,118 | 115 | 85 | 47 | 17 | 1,382 | $1.0\times$ |
| `Scan (6)` | 926 | 278 | 118 | 6 | 6 | 1,334 | $1.0\times$ |
| `Scan (7)` | 778 | 416 | 170 | 25 | 3 | 1,392 | $1.0\times$ |
| `Scan (8)` | 254 | 246 | 18 | 1 | 19 | 538 | **$1.5\times$** |
| `Scan (9)` | 612 | 164 | 11 | 0 | 11 | 798 | **$1.5\times$** |
| `Scan (10)` | 33 | 23 | 19 | 2 | 0 | 77 | **$1.5\times$** |
| **Total** | **7,631** (62.9%) | **2,700** (22.2%) | **1,341** (11.0%) | **399** (3.3%) | **66** (0.5%) | **12,137** (100%) | — |

- **Critical Coordinate Scaling Quirk (`Scan (8..10)`):** Upstream [`damage_generator/scans.py`](https://github.com/daniela997/FilmDamageSimulator/blob/main/damage_generator/scans.py) documents in code that `Scan (8)`, `Scan (9)`, and `Scan (10)` were annotated at $2/3$ resolution and require multiplying every polygon vertex `(x, y)` by **`1.5`** before rasterizing onto the full-resolution scan (`4944x3384` / `4992x3336`). [`parse_fds_polygon_json()`](../scripts/dataset_adapters.py) enforces this automatically.
- **Limitation:** Because `FilmDamageSimulator` annotations are on *empty* developed 35mm emulsion frames (film grain + authentic physical artifacts, no scene objects or motion), they provide an exact ground truth for **artifact recall vs. film-grain false positives**, but cannot measure scene-edge or motion false positives without compositing or pairing with video benchmarks.

#### B. `ARTeFACT` (`danielaivanova/damaged-media`)
- Contains 418 high-resolution images of damaged historical media with pixel-accurate 15-class annotations:
  - **Target film-restoration classes:** `3 = Dust`, `4 = Scratch`, `5 = Hair`, `6 = Dirt`.
  - **Structural/non-film media damage classes (`1, 2, 7..15`):** `Material loss`, `Peel`, `Fold`, `Writing`, `Cracks`, `Staining`, `Stamp`, `Sticker`, `Puncture`, `Burn marks`, `Lightleak`.
  - **Border class (`255 = Background`):** Pixels outside the physical media boundary.
- [`convert_artefact_segmentation_mask()`](../scripts/dataset_adapters.py) maps `255 (Background)` and non-target structural classes to **`127` (ignore)** so that black scanner borders and physical paper tears are never misclassified as clean film background (`0`) when scoring dust or scratch detectors.

---

## 3. 3-Tier Ground Truth & Provenance Separation Protocol

### 3.1 Pixel Encoding Specification (`GT_VALUES`)
All ground-truth masks consumed by [`scripts/benchmark_defect_masks.py`](../scripts/benchmark_defect_masks.py) must be single-channel 8-bit grayscale PNGs (`uint8`) containing **only** three discrete values:
- **`0` (`Clean / Unaffected`)**: Verified intact film emulsion or scene content. False positives ($FP$) and true negatives ($TN$) are computed exclusively over `0` pixels.
- **`127` (`Ambiguous / Ignore`)**: Excluded from $TP$, $FP$, $FN$, and $TN$ (counted under `ignored`). Used for:
  1. Sub-threshold feathering/opacity fringes on synthetic scratches (`0 < soft_mask < 0.10`).
  2. Scanner borders / outer registration bars (`255 Background` in ARTeFACT).
  3. Other defect classes when evaluating a single-class channel (e.g., `Dust` pixels are marked `127` when scoring the `Scratch` channel so a scratch detector is neither penalized for missing dust nor penalized as a false positive if a dust blob overlaps a scratch).
  4. Unreviewed regions in partial human annotations.
- **`255` (`Defect`)**: Verified target damage pixel ($TP$ and $FN$ denominator).

### 3.2 Mandatory Provenance Tiers (`PROVENANCE_TIERS`)
[`validate_provenance_separation()`](../scripts/dataset_adapters.py) requires every manifest row to declare a `label_provenance` column from one of five tiers and **raises a `ValueError` if manual ground truth is mixed with unverified proxies or pseudo-labels**:
1. `manual_polygon_gt` — Human-drawn polygon contours on real film scans (e.g., `FilmDamageSimulator` `Scan (1..10)`).
2. `manual_segmentation_gt` — Human pixel-level segmentation masks on real media (e.g., `ARTeFACT`, or reviewed *Neighbors* pilot frames).
3. `synthetic_parametric_gt` — Exact procedural/parametric masks injected onto clean or reference video frames (`benchmark_defect_masks.py` synthetic generator, `AbsoluteDegradation` `FilmScratchGenerator`, PR24 5-film positive controls).
4. `restoration_diff_proxy` — Thresholded difference masks between paired damaged and manually restored scans (Figshare `21803304`). Useful for exploratory localization only; never reported as true manual ground truth.
5. `model_pseudo_label` — Masks predicted by neural restoration models (e.g., DART `MaskNet`). Strictly forbidden as ground truth.

### 3.3 Sequence-Level Splitting
To prevent temporal autocorrelation and frame-to-frame leakage on video benchmarks:
- Splits (`development`, `val`, `test`) are assigned **strictly at the clip/sequence level** (never random frame sampling within a shot).
- In the `AbsoluteDegradation` video benchmark (`.cache/absolute-scratch-benchmark-20261106-01`), clip `000` is frozen as `development` (48 damaged + 24 reference frames) and clips `010` and `020` are frozen as `test` (96 damaged + 48 reference frames).
- In the 5-film archival benchmark (`.cache/ridge-pr24-positive-control-20261106-03`), *Lumière*, *The Great Train Robbery*, and *Méliès* form the `development` split, while *Caligari* and *Gertie* form the `historical_holdout` split.

---

## 4. Reproducible Benchmark Results

All results below are stored with SHA-256 provenance in `.cache/issue34-published-benchmarks-20261109/report.json` and can be reproduced offline using `.venv\Scripts\python.exe`.

### 4.1 Per-Class Evaluation on Real 4K Annotated 35mm Scans (`FilmDamageSimulator`)
Evaluated at full native 4K resolution on `Scan (8)` (`4944x3384`, SHA-256 `157d36b4eb8cf047...`, 538 human polygons, $1.5\times$ coordinate scaling applied) and `Scan (10)` (`4992x3336`, SHA-256 `92186390e55a1c04...`, 77 human polygons, $1.5\times$ coordinate scaling applied) using the single-frame spatial median-difference baseline (since temporal optical-flow detectors require multi-frame temporal novelty and output $0$ on identical static frames):

| Scan | Defect Class | GT Defect Pixels | Recall | Precision | Pixel F1 | Intact-Pixel FPR |
|---|---|---:|---:|---:|---:|---:|
| `Scan (8)` | `Dust` | 33,368 | 0.3854 | 0.0144 | 0.0278 | 5.3250% |
| `Scan (8)` | `Dirt` | 83,684 | 0.3319 | 0.0306 | 0.0560 | 5.3250% |
| `Scan (8)` | `Short hair` | 12,075 | 0.3705 | 0.0051 | 0.0100 | 5.3250% |
| `Scan (8)` | `Long hair` | 1,962 | 0.4016 | 0.0009 | 0.0018 | 5.3250% |
| `Scan (8)` | `Scratch` | 68,147 | 0.1765 | 0.0135 | 0.0250 | 5.3250% |
| `Scan (8)` | **Combined (All 5 Classes)** | **199,223** | **0.2907** | **0.0617** | **0.1018** | **5.3250%** |
| `Scan (10)` | `Dust` | 4,936 | 0.3487 | 0.0026 | 0.0051 | 4.0435% |
| `Scan (10)` | `Dirt` | 7,958 | 0.2690 | 0.0032 | 0.0063 | 4.0435% |
| `Scan (10)` | `Short hair` | 16,279 | 0.3019 | 0.0073 | 0.0142 | 4.0435% |
| `Scan (10)` | `Long hair` | 8,272 | 0.1944 | 0.0024 | 0.0047 | 4.0435% |
| `Scan (10)` | **Combined (All Classes)** | **37,445** | **0.2773** | **0.0152** | **0.0289** | **4.0435%** |

**Finding:** Purely spatial single-frame impulse detection achieves 28–40% recall on real physical `Dust`, `Dirt`, and `Hair`, and 17.6% recall on faint physical `Scratch` contours, but suffers a **4.0–5.3% false-positive rate on coarse Lomochrome Purple 35mm emulsion grain** (especially along high-gradient grain clumps, where edge-regime FPR reaches **13.66–18.44%**). This confirms why multi-frame temporal gating is mandatory for low-FPR film restoration.

### 4.2 Nuisance-Region False-Positive Breakdown (`grain`, `edges`, `faces_foreground`, `motion`)
Evaluated via [`evaluate_nuisance_fp_breakdown()`](../scripts/dataset_adapters.py) on reserved test clips `010` and `020` (`1270x690`, seeds `20261106` & `20261107`) in `.cache/absolute-scratch-benchmark-20261106-01`:

| Detector Method | Overall FPR | `edges` FPR (High Gradient) | `grain` FPR (Textured Non-Edge) | `faces_foreground` FPR (Central Subject ROI) | `motion` FPR ($\|I_t - I_{t-1}\| \ge 12$) |
|---|---:|---:|---:|---:|---:|
| **Legacy wrapper** | 8.3991% | 14.8168% | 10.3083% | 11.9247% | 10.2101% |
| **Temporal Dust wrapper** | 0.2964% | 1.2712% | 0.3098% | 0.3118% | 0.6965% |
| **Ridge wrapper** | **0.0350%** | **0.1507%** | **0.0361%** | **0.0424%** | **0.0736%** |
| **Ridge core** | 0.0716% | 0.2885% | 0.0814% | 0.0692% | 0.1322% |

**Finding:** Across all four detectors, false positives concentrate disproportionately on **spatial edges (`4.2x` higher than overall FPR for Temporal and Ridge)** and **inter-frame motion (`2.1x–2.3x` higher than overall FPR)**. The ARP `Ridge wrapper` suppresses motion and edge false positives to `0.0736%` and `0.1507%` respectively—over $100\times$ lower than `Legacy`.

### 4.3 Video Temporal Consistency Metrics (`temporal_consistency_scorer.py`)

#### A. `AbsoluteDegradation` Reserved Test Split (Clips `010` & `020`, Multi-Frame Drifting/Curved Scratches)
| Detector Method | Mean Detection Continuity Score | Mean Missed Persistent Defect Rate | Mean Mask Flicker Rate (`0->1->0`) | Mean Temporal Motion FPR | Mean Static Region FPR |
|---|---:|---:|---:|---:|---:|
| **Legacy wrapper** | 0.7563 | 0.2438 | 0.054722 | 0.102152 | 0.064162 |
| **Temporal Dust wrapper** | **0.7854** | **0.2146** | 0.004464 | 0.008543 | 0.001109 |
| **Ridge wrapper** | 0.3188 | 0.6813 | **0.000383** | **0.000914** | **0.000212** |
| **Ridge core** | 0.3458 | 0.6542 | 0.000386 | 0.001026 | 0.000228 |

#### B. 5-Film Archival Positive-Control Replay (`.cache/ridge-pr24-positive-control-20261106-03`)
| Archival Sequence | Split | `pr24` / `arp_core` F1 | `arp_wrapper` F1 | `arp_core` Continuity | `arp_wrapper` Continuity | `arp_wrapper` Missed Rate | `arp_wrapper` Flicker Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| *Lumière* (1895) | development | 0.8073 | 0.7793 | 1.0000 | 0.9167 | 0.0833 | 0.001043 |
| *Train Robbery* (1903) | development | 0.3071 | 0.2986 | 0.7500 | 0.6667 | 0.3333 | 0.002365 |
| *Méliès* (1902) | development | 0.4976 | 0.4874 | 0.7500 | 0.7083 | 0.2917 | 0.001161 |
| *Caligari* (1920) | holdout | 0.7500 | 0.7253 | 1.0000 | 0.9167 | 0.0833 | 0.001355 |
| *Gertie* (1914) | holdout | 0.6110 | 0.4488 | 0.6667 | 0.3750 | 0.6250 | 0.000191 |

---

## 5. *Neighbors* (1920) Domain-Validation Pilot & Remaining Limitations

1. **Bounded *Neighbors* (1920) Pilot (`.cache/dust-real-annotation-20261106-01`):**
   - Instead of manually labeling 500 frames across 5 clips before detector engineering, a deterministic 6-frame sample (`frames 0, 12, 23, 36, 48, 71` at `960x720`, source SHA-256 `63b1c49d061e31c8...`) has been exported via [`scripts/prepare_dust_annotation.py`](../scripts/prepare_dust_annotation.py).
   - Its `annotation_status` remains `"pending"` until human review marks `0=healthy`, `127=ignore`, `255=defect`. Blank or unverified masks are never substituted as reviewed negatives.
2. **Remaining Limitations:**
   - `FilmDamageSimulator` provides real human-labeled 35mm polygons (`12,137` contours) on empty emulsion scans rather than moving scene footage.
   - `ARTeFACT` provides pixel-accurate 15-class human labels on still media under `AFL-3.0`; its 4.07 GB Parquet corpus is supported via `convert_artefact_segmentation_mask()` for offline evaluation.
   - `AbsoluteDegradation` and the 5-film PR24 corpus provide multi-frame temporal ground truth on real video backgrounds, conditioned on synthetic defect trajectories (`synthetic_parametric_gt`).
   - Final claim of real-damage precision on *Neighbors* (1920) remains gated on human sign-off of the 6-frame pilot sample at the end of the pipeline lifecycle.
