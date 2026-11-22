---
title: "Color Bible (Phase 2)"
created: "2026-11-22T14:49:00Z"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "standard"
task_complexity: "complex"
---

# Color Bible (Phase 2) Design Document

## Problem Statement

In DennLie/ai-remaster-pipeline (ARP), long-form black-and-white feature films (60–120 minutes, containing 480–1,600 shots) suffer from long-range color drift across hard scene cuts. While adjacent shots can share references via `reference_images_for()`, costumes and characters appearing at Minute 5 are not bound to the same color palette when they reappear at Minute 80. Furthermore, characters alter wardrobe, and lighting changes across interior/exterior sets. 

To resolve this without non-commercial models or transient GPU-memory dependencies, ARP requires a persistent, film-wide **Color Bible** module (Phase 2 of the Commercial Remaster Audit) that separates permanent character/object identity from appearance variants, binds them to shot ranges, and manages approved color references with mathematical CIE-LAB distributions and RGB target palettes.

## Requirements

### Functional Requirements

- **REQ-1 (Schema & Persistence)**: Versioned, idempotent SQLite migrations with a clean repository/service abstraction separating storage from business logic (enabling future PostgreSQL adoption). Structure tables: `films`, `sequences`, `shots`; Entity tables: `characters`, `appearance_variants`, `objects`, `locations`, `shot_entity_assignments`; Color Reference table: `color_references`.
- **REQ-2 (Zero Biometrics / GDPR Art. 4(14) & Art. 9)**: Strictly zero automated facial recognition, face-clustering, or biometric embeddings in Phase 2 (safe by default for living persons). Keep `shot_entity_assignments` (`confidence`, `approval_status`, `provenance`) cleanly extensible so an optional future opt-in auto-suggestion plugin (e.g. for historical films with deceased actors) can propose assignments for human approval later.
- **REQ-3 (Manifest Sync & Non-Contiguous Shot Ranges)**: Idempotently import and synchronize ARP's existing shot manifests (`manifests/references/*.csv` or `manifests/colorize/*.csv`). Support assigning character + costume variants to multiple disjoint/non-contiguous shot ranges (e.g., Shots 12–18 AND Shots 410–435) with nullable `shot_id` in range assignments to avoid row fan-out.
- **REQ-4 (CIE-LAB Statistics & RGB Palette Extraction)**: Pure vectorized CPU calculation of CIE-LAB percentiles (`p10, p25, p50, p75, p90` for $L^*, a^*, b^*$) and deterministic dominant RGB palette swatches from keyframes with optional binary mask support.
- **REQ-5 (Deterministic Snapshot Hash & Compliance Wiring)**: Compute a canonical, deterministic SHA-256 snapshot hash across approved references (`approval_status = 'approved'`) and assignments for a film, and pass it directly to `compliance.append_compliance_record(..., color_bible_version=...)`.
- **REQ-6 (GUI Workbench & API Routes)**: Add a dedicated, responsive `Color Bible` workbench tab to `ai_remaster_gui` and secure HTTP endpoints (`/api/color-bible/...`) for managing characters, costume variants, shot range assignments, reference palettes, and approvals, strictly validating paths via `resolve_served`.

### Non-Functional Requirements

- **NFR-1 (Performance & Testability)**: 100% CPU-testable on Windows; zero external GPU or unpinned C++ library requirements for tests.
- **NFR-2 (Robustness & Integrity)**: Idempotent schema migrations and manifest sync; transactional consistency on range updates and approvals; `PRAGMA foreign_keys = ON` on all SQLite connections.
- **NFR-3 (HTTP Security)**: Strict local-origin and host validation on all new `/api/color-bible/...` routes, consistent with ARP's existing security model.

### Constraints

- **CON-1**: Do not implement Phase 3 (reference retrieval in `scripts/qwen_colorize_references.py`) or Phase 4 (SAM 2 video tracking) yet.
- **CON-2**: Zero placeholders, zero `TODO` comments, zero stubbed UI callbacks.

## Approach

### Selected Approach: Layered Repository & Color-Science Service (SQLite + Vectorized NumPy + Split Workbench GUI)

The selected approach structures the Color Bible into 4 distinct layers:
1. **Persistence & Migration Layer (`ai_remaster_gui/color_bible_db.py`)**: SQLite schema management with versioned migrations (`schema_migrations`), foreign key cascades, and a repository interface (`ColorBibleRepository`) isolating SQL queries.
2. **Domain Service & Color Engine Layer (`ai_remaster_gui/color_bible_service.py` & `ai_remaster_gui/color_science.py`)**:
   - Manifest synchronization mapping ARP CSV rows to `films`, `sequences`, and `shots`.
   - Shot range resolution logic mapping non-contiguous intervals (e.g. `[(12, 18), (410, 435)]`) to assignments while preserving `approval_status`, `confidence`, and `provenance`.
   - Pure NumPy vectorized color extraction: sRGB to linear RGB $\to$ CIE XYZ (D65) $\to$ CIE-LAB ($L^*, a^*, b^*$). Masked pixel extraction, percentile distributions (`p10, p25, p50, p75, p90`), and deterministic quantized RGB dominant palette clustering.
   - Deterministic canonical snapshot hash calculation (`sha256:...`) across all approved references and assignments.
3. **CLI Interface (`scripts/color_bible.py`)**: Subcommands for `init-db`, `sync-manifest`, `add-character`, `add-variant`, `assign-range`, `extract-palette`, `approve-reference`, `reject-reference`, and `snapshot-hash`.
4. **GUI & API Layer (`ai_remaster_gui/http_handler.py`, `server.py`, `render-color-bible.js`, `styles.css`)**:
   - `/api/color-bible/*` routes protected by local-origin, host security, and `resolve_served` path confinement.
   - Split workbench UI: entity & shot-range management on the left, reference stills, RGB palettes, and approval actions on the right.

### Alternatives Considered

#### Alternative A: In-Memory JSON / YAML File Manifests
- **Description**: Storing color bible definitions inside JSON sidecars alongside `manifests/references/*.csv`.
- **Pros**: Zero SQL setup; easily inspectable plain text.
- **Cons**: Concurrency hazards, lack of referential integrity (e.g. deleting a character orphans costume variants), slow range queries across 1,600 shots, no clean migration path to PostgreSQL.
- **Rejected Because**: Lacks transactional safety and relational integrity needed for 1,600-shot commercial productions.

#### Alternative B: Heavy PyTorch/CUDA Color Extraction & Monolithic CLI
- **Description**: Utilizing PyTorch / torchvision GPU kernels for color extraction.
- **Pros**: Fast if GPU is already active.
- **Cons**: Fails on CPU-only test environments, adds heavy runtime startup overhead for simple color science, conflicts with lightweight headless testing.
- **Rejected Because**: Unacceptable for fast local unit testing and CPU-only CI environments.

### Decision Matrix

| Criterion | Weight | Selected Approach (SQLite + NumPy Service) | Alternative A (JSON/YAML Files) | Alternative B (PyTorch/GPU Engine) |
| :--- | :---: | :--- | :--- | :--- |
| **Referential Integrity & Range Queries** | 30% | **5/5**: Relational foreign keys and indexed range searches | **2/5**: Manual parsing and reconciliation across multiple files | **4/5**: Relational or DB needed anyway |
| **Testability & Portability (CPU on Windows)** | 25% | **5/5**: Pure Python + NumPy, 100% testable in milliseconds | **5/5**: Plain file parsing | **2/5**: Requires Torch/CUDA runtime |
| **PostgreSQL Extensibility** | 20% | **5/5**: Repository pattern isolates SQLite specifics | **1/5**: Complete rewrite required for enterprise DB | **4/5**: Compatible if decoupled |
| **Colorist UX & Ergonomics** | 15% | **5/5**: Dedicated split workbench with immediate palette feedback | **2/5**: Manual file edits | **4/5**: UI possible but heavier backend |
| **Compliance & Snapshot Hash Stability** | 10% | **5/5**: Deterministic canonical sorting over indexed approved states | **3/5**: File order differences can create unstable hashes | **4/5**: Hash calculation feasible |
| **Weighted Total** | 100% | **4.95 / 5.0** | **2.45 / 5.0** | **3.40 / 5.0** |

## Architecture

### Component Diagram

```
+-----------------------------------------------------------------------------------+
|                            ai_remaster_gui Frontend                              |
|   +--------------------------------------+  +---------------------------------+   |
|   | Left Panel: Entities & Shot Ranges   |  | Right Panel: Stills & Palettes  |   |
|   | - Character / Costume Variant Lists  |  | - Color Reference Thumbnails    |   |
|   | - Range Editor (12-18, 410-435)      |  | - RGB Swatches & LAB Curves     |   |
|   | - Props & Sets / Locations           |  | - One-Click Approve / Reject    |   |
|   +--------------------------------------+  +---------------------------------+   |
+------------------------------------------+----------------------------------------+
                                           | HTTP JSON API (/api/color-bible/*)
                                           v
+-----------------------------------------------------------------------------------+
|                        Server Layer (http_handler.py)                             |
|   - Localhost & Local-Origin Security Verification                                |
|   - Path confinement via resolve_served                                           |
|   - Request validation & JSON serialization                                       |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                 Service & Color Science Layer (color_bible_service.py)            |
|   +---------------------------------------------------------------------------+   |
|   | ColorBibleService:                                                        |   |
|   | - sync_manifest(film_id, manifest_path)                                   |   |
|   | - assign_shot_ranges(film_id, entity_type, entity_id, variant_id, ranges) |   |
|   | - register_reference(film_id, entity_type, entity_id, variant_id, image)  |   |
|   | - approve_reference(ref_id, approved_by)                                  |   |
|   | - export_snapshot_hash(film_id) -> sha256:abcd...                         |   |
|   +---------------------------------------------------------------------------+   |
|   | color_science.py:                                                         |   |
|   | - sRGB -> Linear RGB -> CIE-XYZ (D65) -> CIE-LAB (Vectorized NumPy)       |   |
|   | - p10, p25, p50, p75, p90 Percentile Distributions                        |   |
|   | - Deterministic Dominant RGB Target Palette Extraction                    |   |
|   +---------------------------------------------------------------------------+   |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                  Repository & Persistence (color_bible_db.py)                     |
|   - ColorBibleRepository Protocol & SQLite Implementation                         |
|   - Versioned Idempotent Schema Migrations (schema_migrations table)              |
|   - Foreign Key Integrity (PRAGMA foreign_keys = ON) & Range Indexes              |
|   - Database Path Resolution: CLI --db -> ARP_COLOR_BIBLE_DB -> manifests/*.db    |
+-----------------------------------------------------------------------------------+
                                           |
                                           +--------> compliance.py (Version Hash)
```

### Data Flow

1. **Manifest Ingestion**: Colorist or CLI imports a shot manifest (`manifests/references/*.csv`). Service parses shot entries, generates or looks up the `film_id`, and syncs shot rows idempotently into `films` and `shots`.
2. **Entity & Appearance Modeling**: Colorist defines Characters (e.g., "Elena") and Appearance Variants (e.g., "Costume 1 - Wool Trench", "Costume 2 - Evening Gown").
3. **Range Assignment**: Colorist assigns Elena's Costume 1 to non-contiguous shot ranges (e.g. `[(1, 15), (105, 120)]`). The service writes `shot_entity_assignments` with start and end boundaries and default `approved` status.
4. **Reference Registration & Analysis**: An approved keyframe (and optional binary mask) is linked to a variant. `color_science.py` computes $L^*, a^*, b^*$ percentiles and dominant RGB swatches via vectorized NumPy, storing JSON distributions in `color_references`.
5. **Approval & Snapshot Generation**: Colorist reviews swatches in the workbench and marks reference `approved`. The service calculates the canonical SHA-256 snapshot hash across all approved references and assignments for the film.
6. **Compliance Injection**: Pipeline runs retrieve the snapshot hash and pass it to `compliance.append_compliance_record(..., color_bible_version=...)`.

### Key Database Schema

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS films (
    film_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_video TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequences (
    sequence_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    sequence_index INTEGER NOT NULL,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shots (
    shot_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    sequence_id TEXT REFERENCES sequences(sequence_id) ON DELETE SET NULL,
    shot_index INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    source_reference TEXT,
    color_reference TEXT,
    qc_status TEXT DEFAULT 'pending',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shots_film_index ON shots(film_id, shot_index);

CREATE TABLE IF NOT EXISTS characters (
    character_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appearance_variants (
    variant_id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(character_id) ON DELETE CASCADE,
    costume_id TEXT NOT NULL,
    hair_profile TEXT,
    skin_reference TEXT,
    lighting_context TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    kind TEXT CHECK(kind IN ('prop', 'vehicle', 'other')) NOT NULL,
    name TEXT NOT NULL,
    palette_json TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS locations (
    location_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    set_name TEXT NOT NULL,
    lighting_context TEXT,
    palette_json TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shot_entity_assignments (
    assignment_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    shot_id TEXT REFERENCES shots(shot_id) ON DELETE SET NULL,
    entity_type TEXT CHECK(entity_type IN ('character', 'object', 'location')) NOT NULL,
    entity_id TEXT NOT NULL,
    variant_id TEXT REFERENCES appearance_variants(variant_id) ON DELETE SET NULL,
    valid_shot_range_start INTEGER NOT NULL,
    valid_shot_range_end INTEGER NOT NULL,
    confidence REAL DEFAULT 1.0,
    approval_status TEXT CHECK(approval_status IN ('draft', 'pending_review', 'approved', 'rejected')) NOT NULL DEFAULT 'approved',
    provenance TEXT DEFAULT 'manual',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assignments_shot ON shot_entity_assignments(film_id, shot_id);
CREATE INDEX IF NOT EXISTS idx_assignments_range ON shot_entity_assignments(film_id, valid_shot_range_start, valid_shot_range_end);

CREATE TABLE IF NOT EXISTS color_references (
    ref_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    entity_type TEXT CHECK(entity_type IN ('character', 'object', 'location')) NOT NULL,
    entity_id TEXT NOT NULL,
    variant_id TEXT REFERENCES appearance_variants(variant_id) ON DELETE SET NULL,
    source_shot_id TEXT REFERENCES shots(shot_id) ON DELETE SET NULL,
    source_keyframe_path TEXT NOT NULL,
    mask_uri TEXT,
    lab_stats_json TEXT NOT NULL,
    rgb_palette_json TEXT NOT NULL,
    provenance TEXT CHECK(provenance IN ('historical_research', 'ddcolor', 'qwen2511', 'manual')) NOT NULL,
    approval_status TEXT CHECK(approval_status IN ('draft', 'pending_review', 'approved', 'rejected')) NOT NULL DEFAULT 'draft',
    approved_by TEXT,
    approved_at TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refs_entity ON color_references(film_id, entity_type, entity_id, variant_id);
```

## Agent Team

| Phase | Agent | Parallel | Deliverables |
| :---: | :---: | :---: | :--- |
| **Phase 1** | `coder` | No | `ai_remaster_gui/color_bible_db.py`: Versioned migrations, repository layer, path resolution |
| **Phase 2** | `coder` | No | `ai_remaster_gui/color_science.py`, `ai_remaster_gui/color_bible_service.py`: Vectorized color algorithms, service layer |
| **Phase 3** | `coder` | No | `scripts/color_bible.py`, `compliance.py` wiring: CLI tools and compliance record integration |
| **Phase 4** | `coder` | No | `ai_remaster_gui/http_handler.py`, `render-color-bible.js`: Color Bible API endpoints and workbench tab |
| **Phase 5** | `tester` | No | `tests/test_color_bible_*.py`: Comprehensive unit and integration test suite |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
| :--- | :---: | :---: | :--- |
| **Path Traversal / Arbitrary File Access** | HIGH | LOW | Enforce `resolve_served()` validation and relative path checks on all keyframe and mask URIs. |
| **Color Science Drift** | MEDIUM | LOW | Standard D65 observer matrices and vectorized inverse companding tested against ground-truth color swatches. |
| **Snapshot Hash Instability** | MEDIUM | LOW | Canonical JSON serialization with sorted keys and normalized precision before SHA-256 computation. |
| **Windows File Lock Contention** | MEDIUM | MEDIUM | Enable WAL mode (`PRAGMA journal_mode = WAL`) and context-managed connection closures. |
| **UI Regressions in Existing Tabs** | HIGH | LOW | Non-destructive insertion of `color_bible` tab into `availableTabs()` and clean route isolation in `http_handler.py`. |

## Success Criteria

1. **Idempotent Schema & Migrations**: Running database initialization multiple times succeeds without error or data loss; tables enforce foreign key cascades with `PRAGMA foreign_keys = ON`.
2. **Disjoint Shot Range Resolution**: A character wearing Costume A can be assigned to non-contiguous ranges (e.g. shots 12–18 and shots 410–435) while preserving distinct appearance variants for shots 500–550 (Costume B).
3. **Accurate & Masked Color Extraction**: Vectorized CIE-LAB percentiles (`p10, p25, p50, p75, p90`) and dominant RGB palettes match known synthetic target values, respecting binary masks when provided.
4. **Deterministic Snapshot Hashing & Compliance Integration**: `get_snapshot_hash()` outputs a stable `sha256:...` digest across runs, and passes cleanly into `compliance.append_compliance_record(..., color_bible_version=...)`.
5. **Full GUI & API Operability**: The Color Bible tab renders in the browser workbench, allows creating entities/variants, assigning ranges, inspecting palette swatches, and approving/rejecting references with strict HTTP security.
6. **Zero Codebase Gaps**: Zero `TODO`s, zero placeholders, strictly typed Python, 100% CPU testable on Windows.
7. **Passing Acceptance Test Gate**:
   ```bash
   pytest tests/test_color_bible*.py tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_ddcolor_reference.py tests/test_http_routes.py tests/test_http_security.py tests/test_gui_smoke.py tests/test_pipeline_order.py
   ```
   All tests in these suites pass 100% on CPU.
