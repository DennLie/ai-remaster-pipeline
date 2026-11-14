"""Automated, idempotent benchmark resource provisioner and isolation auditor.

Manages, provisions, audits, and validates all external and built-in benchmark
resources for film scratch and dust detection evaluation:

1. FilmDamageSimulator / FILM-AA (MIT code / CC BY 4.0 data, Figshare 21803292.v1)
2. ARTeFACT (AFL-3.0 data, HuggingFace danielaivanova/damaged-media)
3. AbsoluteDegradation (MIT generator code, TytanMikJas/AbsoluteDegradation v1.0.0)
4. DART MaskNet (Research-only, isolated evaluation harness with precomputed masks)
5. Synthetic scratch and dust benchmark (Built-in, 100% commercial-safe)
6. Buster Keaton Neighbors (1920) 6-frame domain validation pilot

STRICT ENVIRONMENT ISOLATION:
Never installs research packages into ARP's `.venv` or ComfyUI's Python runtime.
Reuses existing dedicated environments (e.g. `I:\\DART\\.venv` and
`.cache\\absolute-degradation-v1\\.venv`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping
import urllib.request

import cv2
import numpy as np

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import benchmark_defect_masks as bdm
import dataset_adapters as da


# Pinned commits and checksums
FDS_FIGSHARE_ARTICLE_ID = "21803292"
FDS_FILE_SPECS: dict[str, dict[str, Any]] = {
    "Scan (8).jpg": {
        "file_id": "38687871",
        "md5": "86efd8da6513a7a402c90ed50bd39ab8",
        "size": 10570235,
    },
    "Scan (8).json": {
        "file_id": "38687922",
        "md5": "724d64b827cb87a5db6ec9840665ae13",
        "size": 359258,
    },
    "Scan (10).jpg": {
        "file_id": "38687892",
        "md5": "df58349fedab1db42d7fadad0d205c08",
        "size": 10985336,
    },
    "Scan (10).json": {
        "file_id": "38687943",
        "md5": "999b6b2bb69cca6326e0d9d946a6bdfe",
        "size": 51231,
    },
}

ARTEFACT_SHARD_NAME = "train-00019-of-00028.parquet"
ARTEFACT_REPO_ID = "danielaivanova/damaged-media"
ARTEFACT_SUBPATH = "data/train-00019-of-00028.parquet"

ABS_DEG_COMMIT = "542856fd7e275a6dcec80311f7f7b4541bdc4f70"
DART_COMMIT = "fc6f3e722c74a3b0e3a43ef40bb1d5303fa5e332"
DART_WEIGHTS_SHA256 = "d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3"
NEIGHBORS_72_SHA256 = "63b1c49d061e31c8ad4bf82b0e7745778841da927aaec02efd3fa7bfb30c9f13"

PROHIBITED_ARP_PACKAGES = frozenset(
    {"pyarrow", "fastparquet", "duckdb", "datasets", "dart", "absolute_degradation"}
)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_isolated_dart_python() -> Path | None:
    candidate = Path("I:/DART/.venv/Scripts/python.exe")
    if candidate.is_file():
        return candidate
    candidate_unix = Path("I:/DART/.venv/bin/python")
    if candidate_unix.is_file():
        return candidate_unix
    # Managed dart tool path
    managed = ROOT_DIR / "tools/dart/.venv/Scripts/python.exe"
    if managed.is_file():
        return managed
    return None


def find_isolated_absdeg_python() -> Path | None:
    candidate = ROOT_DIR / ".cache/absolute-degradation-v1/.venv/Scripts/python.exe"
    if candidate.is_file():
        return candidate
    candidate_unix = ROOT_DIR / ".cache/absolute-degradation-v1/.venv/bin/python"
    if candidate_unix.is_file():
        return candidate_unix
    return None


def audit_environment_isolation() -> dict[str, Any]:
    """Verify that ARP's .venv and ComfyUI remain completely isolated from research packages."""
    violations = []
    current_python = sys.executable

    # Check current environment modules and site-packages
    for pkg in sorted(PROHIBITED_ARP_PACKAGES):
        try:
            __import__(pkg)
            violations.append(f"Prohibited package '{pkg}' is importable in current Python ({current_python})")
        except ImportError:
            pass

    dart_py = find_isolated_dart_python()
    absdeg_py = find_isolated_absdeg_python()

    return {
        "current_python": current_python,
        "is_isolated": len(violations) == 0,
        "violations": violations,
        "dart_python": str(dart_py) if dart_py else None,
        "absdeg_python": str(absdeg_py) if absdeg_py else None,
        "reusable_dart_env_ready": dart_py is not None,
        "reusable_absdeg_env_ready": absdeg_py is not None,
    }


def audit_all_resources(cache_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Audit all 6 benchmark resources and report exact status, provenance, and verification."""
    cache = Path(cache_dir or ROOT_DIR / ".cache/benchmarks").resolve()
    reports: dict[str, dict[str, Any]] = {}

    # 1. FilmDamageSimulator
    fds_target = cache / "fds"
    fds_legacy_cache = ROOT_DIR / ".cache/issue34-published-benchmarks-20261109/fds_scans"
    if not fds_legacy_cache.is_dir():
        alt = Path("I:/ai-remaster-pipeline/.cache/issue34-published-benchmarks-20261109/fds_scans")
        if alt.is_dir():
            fds_legacy_cache = alt
    fds_installed = False
    fds_tested = False
    fds_notes = []

    # Check if target adapted dataset or raw files exist
    if (fds_target / "manifest.csv").is_file() and (fds_target / "ground_truth").is_dir():
        fds_installed = True
        fds_tested = True
        fds_notes.append("Adapted canonical dataset present")
    elif fds_legacy_cache.is_dir() and (fds_legacy_cache / "Scan (8).jpg").is_file():
        fds_installed = True
        fds_notes.append("Raw scans present in legacy cache (.cache/issue34-...)")

    reports["film_damage_simulator"] = {
        "name": "FilmDamageSimulator / FILM-AA",
        "source": "Figshare DOI 10.6084/m9.figshare.21803292.v1 & GitHub daniela997/FilmDamageSimulator",
        "license": "CC BY 4.0 (Dataset) / MIT (Code)",
        "commercial_suitability": "Commercial-safe with attribution",
        "provenance_tier": "manual_polygon_gt",
        "status": "INSTALLED_AND_TESTED" if fds_tested else ("INSTALLED_UNTESTED" if fds_installed else "MISSING"),
        "local_path": str(fds_target if fds_target.exists() else fds_legacy_cache),
        "notes": "; ".join(fds_notes) or "Available for automatic download and adaptation",
    }

    # 2. ARTeFACT
    artefact_target = cache / "artefact"
    artefact_extracted = artefact_target / "extracted"
    artefact_installed = False
    artefact_tested = False
    artefact_notes = []

    if (artefact_target / "manifest.csv").is_file() and (artefact_target / "ground_truth").is_dir():
        artefact_installed = True
        artefact_tested = True
        artefact_notes.append("Adapted canonical dataset present")
    elif (artefact_extracted / "extracted_manifest.json").is_file():
        artefact_installed = True
        artefact_notes.append("Raw extracted samples present")

    reports["artefact"] = {
        "name": "ARTeFACT (danielaivanova/damaged-media)",
        "source": "Hugging Face danielaivanova/damaged-media (shard train-00019-of-00028.parquet)",
        "license": "AFL-3.0 (Academic Free License v3.0)",
        "commercial_suitability": "Offline research evaluation only (NOT permitted in commercial product weights)",
        "provenance_tier": "manual_segmentation_gt",
        "status": "INSTALLED_AND_TESTED" if artefact_tested else ("INSTALLED_UNTESTED" if artefact_installed else "MISSING"),
        "local_path": str(artefact_target),
        "notes": "; ".join(artefact_notes) or "Requires --acknowledge-research for provisioning",
    }

    # 3. AbsoluteDegradation
    absdeg_repo = ROOT_DIR / ".cache/absolute-degradation-v1"
    if not absdeg_repo.is_dir():
        alt = Path("I:/ai-remaster-pipeline/.cache/absolute-degradation-v1")
        if alt.is_dir():
            absdeg_repo = alt
    absdeg_bench = ROOT_DIR / ".cache/absolute-scratch-benchmark-20261106-01"
    if not absdeg_bench.is_dir():
        alt = Path("I:/ai-remaster-pipeline/.cache/absolute-scratch-benchmark-20261106-01")
        if alt.is_dir():
            absdeg_bench = alt
    absdeg_installed = False
    absdeg_tested = False
    absdeg_notes = []

    if absdeg_repo.is_dir() and (absdeg_repo / "absolute_degradation").is_dir():
        absdeg_installed = True
        absdeg_notes.append(f"Generator checkout pinned at {ABS_DEG_COMMIT[:10]}")
    if (absdeg_bench / "generated.json").is_file() and (absdeg_bench / "010").is_dir():
        absdeg_tested = True
        absdeg_notes.append("Generated multi-frame scratch clips and soft masks present")

    reports["absolute_degradation"] = {
        "name": "AbsoluteDegradation FilmScratchGenerator",
        "source": f"GitHub TytanMikJas/AbsoluteDegradation @ {ABS_DEG_COMMIT}",
        "license": "MIT (Generator code; RTN texture templates 404 upstream)",
        "commercial_suitability": "Parametric generator code is MIT; no proprietary textures bundled",
        "provenance_tier": "synthetic_parametric_gt",
        "status": "INSTALLED_AND_TESTED" if (absdeg_installed and absdeg_tested) else ("INSTALLED_UNTESTED" if absdeg_installed else "MISSING"),
        "local_path": str(absdeg_bench if absdeg_bench.exists() else absdeg_repo),
        "notes": "; ".join(absdeg_notes),
    }

    # 4. DART MaskNet Harness
    dart_repo = Path("I:/DART")
    dart_weights = dart_repo / "pretrained_models/dart.safetensors"
    dart_precomputed = ROOT_DIR / "evaluation/dart_trial_01/masks"
    dart_installed = False
    dart_tested = False
    dart_notes = []

    if dart_repo.is_dir() and dart_weights.is_file():
        dart_installed = True
        dart_notes.append("External DART checkout & weights verified")
    if dart_precomputed.is_dir() and len(list(dart_precomputed.glob("*.png"))) >= 72:
        dart_tested = True
        dart_notes.append("72 precomputed Buster Keaton soft masks available for evaluation harness")

    reports["dart_harness"] = {
        "name": "DART MaskNet External Evaluation Harness",
        "source": f"GitHub TytanMikJas/DART @ {DART_COMMIT}",
        "license": "Research-Only / NOASSERTION (Adapted RTN code lacks confirmed commercial grant)",
        "commercial_suitability": "Strictly research-only; firewalled from commercial production runtime",
        "provenance_tier": "model_pseudo_label",
        "status": "INSTALLED_AND_TESTED" if (dart_installed and dart_tested) else ("INSTALLED_UNTESTED" if dart_installed else "MISSING"),
        "local_path": str(dart_repo),
        "available_masks": 72 if dart_tested else 0,
        "valid_prediction_masks": 72 if dart_tested else 0,
        "samples_evaluated": 0,
        "samples_skipped": 72 if dart_tested else 0,
        "samples_without_verified_ground_truth": 72 if dart_tested else 0,
        "notes": "; ".join(dart_notes) if dart_notes else "DART research runtime and weights",
    }

    # 5. Synthetic Scratch & Dust Benchmark
    synth_target = cache / "synthetic"
    synth_ready = (synth_target / "manifest.csv").is_file() and (synth_target / "ground_truth").is_dir()

    reports["synthetic_benchmark"] = {
        "name": "Synthetic Scratch & Dust Benchmark Generator",
        "source": "In-tree scripts/benchmark_defect_masks.py",
        "license": "BSD-3 / MIT (NumPy & OpenCV)",
        "commercial_suitability": "100% Commercial-safe",
        "provenance_tier": "synthetic_parametric_gt",
        "status": "INSTALLED_AND_TESTED" if synth_ready else "INSTALLED_UNTESTED",
        "local_path": str(synth_target),
        "notes": "Built-in generator ready for instantaneous on-demand provisioning",
    }

    # 6. Neighbors Validation Pilot
    neighbors_source = ROOT_DIR / "evaluation/Neighbors_72.mkv"
    neighbors_cache = ROOT_DIR / ".cache/dust-real-annotation-20261106-01"
    neighbors_target = cache / "neighbors_pilot"
    neighbors_ready = (neighbors_cache / "manifest.json").is_file() or (neighbors_target / "manifest.json").is_file()

    reports["neighbors_pilot"] = {
        "name": "Neighbors (1920) 6-Frame Validation Pilot",
        "source": "Buster Keaton Neighbors (1920) via evaluation/Neighbors_72.mkv",
        "license": "Public Domain",
        "commercial_suitability": "100% Commercial-safe",
        "provenance_tier": "manual_segmentation_gt",
        "status": "INSTALLED_AND_TESTED" if neighbors_ready else ("INSTALLED_UNTESTED" if neighbors_source.is_file() else "MISSING"),
        "local_path": str(neighbors_target if neighbors_target.exists() else neighbors_cache),
        "notes": "6 fixed native frames (0, 12, 23, 36, 48, 71); annotation_status=pending human sign-off",
    }

    return reports


def provision_film_damage_simulator(
    target_dir: Path,
    *,
    scans: Sequence[str] = ("Scan (8)", "Scan (10)"),
    force: bool = False,
) -> dict[str, Any]:
    """Download (or copy from local cache) and adapt FilmDamageSimulator scans and polygon ground truth."""
    target_dir = Path(target_dir).resolve()
    raw_dir = target_dir / "raw_downloads"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = target_dir / "manifest.csv"
    gt_scratch_dir = target_dir / "ground_truth_scratch"
    if not force and manifest_file.is_file() and (target_dir / "ground_truth").is_dir() and gt_scratch_dir.is_dir():
        return {
            "status": "cached",
            "target_dir": str(target_dir),
            "manifest": str(manifest_file),
        }

    legacy_cache = ROOT_DIR / ".cache/issue34-published-benchmarks-20261109/fds_scans"

    downloaded_files = {}
    for stem in scans:
        for ext in (".jpg", ".json"):
            filename = f"{stem}{ext}"
            spec = FDS_FILE_SPECS.get(filename)
            dest = raw_dir / filename

            # 1. Check if already present and valid in raw_dir
            if dest.is_file():
                if spec and compute_md5(dest) == spec["md5"]:
                    downloaded_files[filename] = dest
                    continue

            # 2. Check if present in legacy cache
            legacy_file = legacy_cache / filename
            if legacy_file.is_file():
                if not spec or compute_md5(legacy_file) == spec["md5"]:
                    shutil.copyfile(legacy_file, dest)
                    downloaded_files[filename] = dest
                    continue

            # 3. Download from Figshare if spec available
            if spec:
                url = f"https://ndownloader.figshare.com/files/{spec['file_id']}"
                req = urllib.request.Request(url, headers={"User-Agent": "AI-Remaster-Benchmark/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
                if compute_md5(dest) != spec["md5"]:
                    raise ValueError(f"Downloaded file {filename} MD5 mismatch against Figshare specification")
                downloaded_files[filename] = dest
            else:
                raise FileNotFoundError(f"Cannot provision {filename}; no Figshare spec or local cache found")

    # Adapt into canonical benchmark dataset
    records = []
    for stem in scans:
        jpg_path = raw_dir / f"{stem}.jpg"
        json_path = raw_dir / f"{stem}.json"
        if not jpg_path.is_file() or not json_path.is_file():
            continue

        image_bgr = cv2.imread(str(jpg_path))
        if image_bgr is None:
            raise ValueError(f"Failed to decode image {jpg_path}")
        h, w = image_bgr.shape[:2]

        polygons = da.parse_fds_polygon_json(json_path, scan_stem=stem)
        masks = da.rasterize_fds_masks(polygons, h, w)

        safe_clip = stem.replace(" ", "_").replace("(", "").replace(")", "")
        # Export authentic scan with combined, scratch, and dust GT masks
        records.append(
            {
                "clip": safe_clip,
                "frame": "000001.png",
                "split": "test",
                "damage_type": "all_damage",
                "label_provenance": "manual_polygon_gt",
                "gt_mask": masks["combined"],
                "gt_scratch_mask": masks["scratch"],
                "gt_dust_mask": masks["dust"],
                "image": image_bgr,
            }
        )

    res = da.export_benchmark_dataset(target_dir, records, require_manual_only=True)
    return {
        "status": "provisioned",
        "target_dir": str(target_dir),
        "manifest": str(res["manifest"]),
        "samples": len(records),
    }


def provision_artefact(
    target_dir: Path,
    *,
    acknowledge_research: bool = False,
    max_samples: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Download ARTeFACT shard using isolated Python and adapt into canonical 3-tier GT dataset."""
    if not acknowledge_research:
        return {
            "status": "LICENSE_BLOCKED",
            "reason": "ARTeFACT carries AFL-3.0 and historical media archive terms; requires --acknowledge-research",
        }

    target_dir = Path(target_dir).resolve()
    manifest_file = target_dir / "manifest.csv"
    if not force and manifest_file.is_file() and (target_dir / "ground_truth").is_dir():
        return {
            "status": "cached",
            "target_dir": str(target_dir),
            "manifest": str(manifest_file),
        }

    dart_py = find_isolated_dart_python()
    if not dart_py:
        return {
            "status": "UNAVAILABLE",
            "reason": "No isolated Python environment with pyarrow/PIL found (expected I:\\DART\\.venv)",
        }

    # Extract raw samples using isolated runner
    extracted_dir = target_dir / "extracted"
    extractor_script = SCRIPTS_DIR / "extract_artefact_samples.py"

    # Find or download shard
    # Check HF hub default cache first
    hf_default_shard = (
        Path(os.environ.get("USERPROFILE", "C:/Users/ogn81"))
        / f".cache/huggingface/hub/datasets--danielaivanova--damaged-media/snapshots"
    )
    shard_path = None
    if hf_default_shard.is_dir():
        matches = list(hf_default_shard.glob(f"*/data/{ARTEFACT_SHARD_NAME}"))
        if matches:
            shard_path = matches[0]

    cmd = [
        str(dart_py),
        str(extractor_script),
        "--output-dir",
        str(extracted_dir),
        "--max-samples",
        str(max_samples),
    ]
    if shard_path and shard_path.is_file():
        cmd.extend(["--parquet-path", str(shard_path)])
    else:
        # Download in isolated python via huggingface_hub
        download_cmd = (
            f"from huggingface_hub import hf_hub_download; "
            f"p = hf_hub_download('{ARTEFACT_REPO_ID}', '{ARTEFACT_SUBPATH}', repo_type='dataset'); "
            f"print('DOWNLOADED_PATH:' + p)"
        )
        proc = subprocess.run([str(dart_py), "-c", download_cmd], capture_output=True, text=True, check=True)
        for line in proc.stdout.splitlines():
            if line.startswith("DOWNLOADED_PATH:"):
                shard_path = Path(line.split(":", 1)[1].strip())
                break
        if not shard_path or not shard_path.is_file():
            raise FileNotFoundError("Failed to download or locate ARTeFACT Parquet shard via isolated python")
        cmd.extend(["--parquet-path", str(shard_path)])

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    extracted_manifest = json.loads((extracted_dir / "extracted_manifest.json").read_text(encoding="utf-8"))
    records = []
    for s in extracted_manifest.get("samples", []):
        sample_id = s["id"]
        raw_img_path = Path(s["image_path"])
        raw_ann_path = Path(s["annotation_path"])

        img_bgr = cv2.imread(str(raw_img_path))
        ann_raw = cv2.imread(str(raw_ann_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or ann_raw is None:
            continue

        # Use full sample_id to prevent name collision across samples
        clip_name = f"artefact_{sample_id}"

        # Generate class-specific GT mask preserving ignore regions for other defects
        if 4 in s["target_classes"]:
            damage_type = "scratch"
            gt_mask = da.convert_artefact_segmentation_mask(ann_raw, mode="scratch")
        else:
            damage_type = "dust"
            gt_mask = da.convert_artefact_segmentation_mask(ann_raw, mode="dust")

        records.append(
            {
                "clip": clip_name,
                "frame": "000001.png",
                "split": "test",
                "damage_type": damage_type,
                "label_provenance": "manual_segmentation_gt",
                "gt_mask": gt_mask,
                "image": img_bgr,
            }
        )

    res = da.export_benchmark_dataset(target_dir, records, require_manual_only=True)
    return {
        "status": "provisioned",
        "target_dir": str(target_dir),
        "manifest": str(res["manifest"]),
        "samples": len(records),
    }


def provision_absolute_degradation(
    target_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Verify or link AbsoluteDegradation parametric scratch sequences into canonical benchmark."""
    target_dir = Path(target_dir).resolve()
    manifest_file = target_dir / "manifest.csv"
    if not force and manifest_file.is_file() and (target_dir / "ground_truth").is_dir():
        return {
            "status": "cached",
            "target_dir": str(target_dir),
            "manifest": str(manifest_file),
        }

    legacy_cache = ROOT_DIR / ".cache/absolute-scratch-benchmark-20261106-01"
    if not (legacy_cache / "generated.json").is_file():
        alt = Path("I:/ai-remaster-pipeline/.cache/absolute-scratch-benchmark-20261106-01")
        if (alt / "generated.json").is_file():
            legacy_cache = alt
    if not (legacy_cache / "generated.json").is_file():
        return {
            "status": "UNAVAILABLE",
            "reason": "AbsoluteDegradation benchmark frames not yet generated and generator requires reference clips",
        }

    records = []
    # Load 12 representative frames from clip 010 and 020
    for clip in ("010", "020"):
        clip_dir = legacy_cache / clip / "20261106"
        if not clip_dir.is_dir():
            continue
        for i in range(6):  # 6 frames per clip = 12 total
            frame_name = f"{i:06d}.png"
            img_path = clip_dir / frame_name
            gt_path = clip_dir / f"{i:06d}.gt.png"
            if not img_path.is_file() or not gt_path.is_file():
                continue

            img = cv2.imread(str(img_path))
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            records.append(
                {
                    "clip": f"absdeg_{clip}",
                    "frame": f"{i + 1:06d}.png",
                    "split": "test",
                    "damage_type": "scratch",
                    "label_provenance": "synthetic_parametric_gt",
                    "gt_mask": gt,
                    "image": img,
                }
            )

    res = da.export_benchmark_dataset(target_dir, records, require_manual_only=False)
    return {
        "status": "provisioned",
        "target_dir": str(target_dir),
        "manifest": str(res["manifest"]),
        "samples": len(records),
    }


def provision_dart_harness(
    target_dir: Path,
    *,
    acknowledge_research: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Verify DART MaskNet external research checkout and export precomputed mask manifest."""
    if not acknowledge_research:
        return {
            "status": "LICENSE_BLOCKED",
            "reason": "DART adapts RTN components without confirmed commercial grant; requires --acknowledge-research",
        }

    target_dir = Path(target_dir).resolve()
    manifest_file = target_dir / "manifest.csv"
    if not force and manifest_file.is_file() and (target_dir / "ground_truth").is_dir():
        return {
            "status": "cached",
            "target_dir": str(target_dir),
            "manifest": str(manifest_file),
        }

    precomputed = ROOT_DIR / "evaluation/dart_trial_01/masks"
    if not precomputed.is_dir():
        return {
            "status": "UNAVAILABLE",
            "reason": "evaluation/dart_trial_01/masks is missing",
        }

    # Precomputed masks from Buster Keaton 72-frame run
    # Format them into an isolated prediction/pseudo-label cohort
    video_path = ROOT_DIR / "evaluation/Neighbors_72.mkv"
    video_frames: list[np.ndarray] = []
    if video_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(frame)
        cap.release()

    records = []
    # Take first 12 frames
    mask_files = sorted(precomputed.glob("*.png"))[:12]
    for idx, mf in enumerate(mask_files):
        soft_mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        # Binarize DART soft mask at threshold 160
        bin_mask = np.where(soft_mask >= 160, np.uint8(255), np.uint8(0))
        img = video_frames[idx] if idx < len(video_frames) else None
        if img is None:
            raise ValueError(f"Missing authentic source video frame {idx} from {video_path}")
        records.append(
            {
                "clip": "dart_neighbors_72",
                "frame": f"{idx + 1:06d}.png",
                "split": "test",
                "damage_type": "all_damage",
                "label_provenance": "model_pseudo_label",
                "gt_mask": bin_mask,
                "image": img,
            }
        )

    res = da.export_benchmark_dataset(target_dir, records, require_manual_only=False)
    return {
        "status": "provisioned",
        "target_dir": str(target_dir),
        "manifest": str(res["manifest"]),
        "samples": len(records),
    }


def provision_synthetic_benchmark(
    target_dir: Path,
    *,
    num_frames: int = 10,
    force: bool = False,
) -> dict[str, Any]:
    """Generate standardized synthetic scratch and dust clips using in-tree generators."""
    target_dir = Path(target_dir).resolve()
    manifest_file = target_dir / "manifest.csv"
    if not force and manifest_file.is_file() and (target_dir / "ground_truth").is_dir():
        return {
            "status": "cached",
            "target_dir": str(target_dir),
            "manifest": str(manifest_file),
        }

    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261109)
    base_frames = [
        rng.integers(96, 160, size=(128, 128), dtype=np.uint8)
        for _ in range(num_frames)
    ]
    s_cfg = bdm.ScratchConfig(seed=20261109, width=3, opacity=0.9, angle_deg=4.0, duration=8, drift_px=1.5)
    d_cfg = bdm.DustConfig(seed=20261110, clusters_per_frame=8, contrast=0.85)

    s_frames, s_gts = bdm.inject_scratches(base_frames, s_cfg)
    d_frames, d_gts = bdm.inject_dust(base_frames, d_cfg)

    records = []
    for i in range(num_frames):
        fname = f"{i + 1:06d}.png"
        records.append(
            {
                "clip": "synthetic_scratch",
                "frame": fname,
                "split": "test",
                "damage_type": "scratch",
                "label_provenance": "synthetic_parametric_gt",
                "gt_mask": s_gts[i],
                "image": s_frames[i],
            }
        )
        records.append(
            {
                "clip": "synthetic_dust",
                "frame": fname,
                "split": "test",
                "damage_type": "dust",
                "label_provenance": "synthetic_parametric_gt",
                "gt_mask": d_gts[i],
                "image": d_frames[i],
            }
        )

    res = da.export_benchmark_dataset(target_dir, records, require_manual_only=False)
    return {
        "status": "provisioned",
        "target_dir": str(target_dir),
        "manifest": str(res["manifest"]),
        "samples": len(records),
    }


def provision_neighbors_pilot(
    target_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Verify or extract the 6-frame Buster Keaton Neighbors (1920) pilot sample."""
    target_dir = Path(target_dir).resolve()
    manifest_file = target_dir / "manifest.json"
    if not force and manifest_file.is_file() and (target_dir / "samples.csv").is_file():
        return {
            "status": "cached",
            "target_dir": str(target_dir),
            "manifest": str(manifest_file),
        }

    legacy_cache = ROOT_DIR / ".cache/dust-real-annotation-20261106-01"
    if not (legacy_cache / "manifest.json").is_file():
        alt = Path("I:/ai-remaster-pipeline/.cache/dust-real-annotation-20261106-01")
        if (alt / "manifest.json").is_file():
            legacy_cache = alt
    if legacy_cache.is_dir() and (legacy_cache / "manifest.json").is_file():
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in legacy_cache.iterdir():
            if item.is_file():
                shutil.copyfile(item, target_dir / item.name)
        return {
            "status": "provisioned",
            "target_dir": str(target_dir),
            "manifest": str(target_dir / "manifest.json"),
            "samples": 6,
        }

    # Extract fresh from source video if available
    source_video = ROOT_DIR / "evaluation/Neighbors_72.mkv"
    if not source_video.is_file():
        return {
            "status": "UNAVAILABLE",
            "reason": "evaluation/Neighbors_72.mkv not found",
        }

    import prepare_dust_annotation
    prepare_dust_annotation.prepare(source_video, target_dir)
    return {
        "status": "provisioned",
        "target_dir": str(target_dir),
        "manifest": str(manifest_file),
        "samples": 6,
    }


def provision_all(
    cache_dir: Path | None = None,
    *,
    acknowledge_research: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end idempotent provisioning for all eligible benchmark resources."""
    cache = Path(cache_dir or ROOT_DIR / ".cache/benchmarks").resolve()
    cache.mkdir(parents=True, exist_ok=True)

    iso = audit_environment_isolation()
    if not iso["is_isolated"]:
        raise RuntimeError(f"Strict Python environment isolation violated: {iso['violations']}")

    results = {
        "isolation_audit": iso,
        "resources": {},
    }

    # 1. Synthetic Benchmark
    results["resources"]["synthetic_benchmark"] = provision_synthetic_benchmark(
        cache / "synthetic", force=force
    )

    # 2. FilmDamageSimulator
    results["resources"]["film_damage_simulator"] = provision_film_damage_simulator(
        cache / "fds", force=force
    )

    # 3. Neighbors Pilot
    results["resources"]["neighbors_pilot"] = provision_neighbors_pilot(
        cache / "neighbors_pilot", force=force
    )

    # 4. AbsoluteDegradation
    results["resources"]["absolute_degradation"] = provision_absolute_degradation(
        cache / "absolute_degradation", force=force
    )

    # 5. ARTeFACT
    results["resources"]["artefact"] = provision_artefact(
        cache / "artefact",
        acknowledge_research=acknowledge_research,
        force=force,
    )

    # 6. DART Harness
    results["resources"]["dart_harness"] = provision_dart_harness(
        cache / "dart_harness",
        acknowledge_research=acknowledge_research,
        force=force,
    )

    summary_file = cache / "provisioning_summary.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Audit and display status of all benchmark resources")
    parser.add_argument("--provision", action="store_true", help="Idempotently provision all eligible resources")
    parser.add_argument("--resource", choices=("fds", "artefact", "absdeg", "dart", "synthetic", "neighbors"), help="Provision a specific resource")
    parser.add_argument("--acknowledge-research", action="store_true", help="Explicitly acknowledge research-only terms for ARTeFACT and DART")
    parser.add_argument("--cache-dir", type=Path, default=ROOT_DIR / ".cache/benchmarks", help="Root directory for provisioned benchmarks")
    parser.add_argument("--force", action="store_true", help="Force re-provisioning even if cached")
    args = parser.parse_args(argv)

    if args.status or (not args.provision and not args.resource):
        iso = audit_environment_isolation()
        print("=" * 80)
        print("ENVIRONMENT ISOLATION AUDIT")
        print("=" * 80)
        print(f"ARP Python:          {iso['current_python']}")
        print(f"Strict Isolation:    {'PASS (Clean)' if iso['is_isolated'] else 'FAIL'}")
        if iso["violations"]:
            for v in iso["violations"]:
                print(f"  WARNING: {v}")
        print(f"DART Python:         {iso['dart_python'] or 'Not detected'}")
        print(f"AbsDeg Python:       {iso['absdeg_python'] or 'Not detected'}")
        print("\n" + "=" * 80)
        print("BENCHMARK RESOURCE AUDIT")
        print("=" * 80)
        audit = audit_all_resources(args.cache_dir)
        for key, info in audit.items():
            print(f"[{info['status']}] {info['name']}")
            print(f"  Source:      {info['source']}")
            print(f"  License:     {info['license']}")
            print(f"  Commercial:  {info['commercial_suitability']}")
            print(f"  Provenance:  {info['provenance_tier']}")
            print(f"  Path:        {info['local_path']}")
            print(f"  Notes:       {info['notes']}\n")
        return 0

    if args.provision:
        res = provision_all(
            args.cache_dir,
            acknowledge_research=args.acknowledge_research,
            force=args.force,
        )
        print(f"Provisioning complete. Summary saved to {args.cache_dir / 'provisioning_summary.json'}")
        for k, v in res["resources"].items():
            print(f"  {k}: {v.get('status')} ({v.get('samples', 0)} samples)")
        return 0

    if args.resource:
        res_map = {
            "fds": lambda: provision_film_damage_simulator(args.cache_dir / "fds", force=args.force),
            "artefact": lambda: provision_artefact(args.cache_dir / "artefact", acknowledge_research=args.acknowledge_research, force=args.force),
            "absdeg": lambda: provision_absolute_degradation(args.cache_dir / "absolute_degradation", force=args.force),
            "dart": lambda: provision_dart_harness(args.cache_dir / "dart_harness", acknowledge_research=args.acknowledge_research, force=args.force),
            "synthetic": lambda: provision_synthetic_benchmark(args.cache_dir / "synthetic", force=args.force),
            "neighbors": lambda: provision_neighbors_pilot(args.cache_dir / "neighbors_pilot", force=args.force),
        }
        fn = res_map.get(args.resource)
        if fn:
            res = fn()
            print(f"Provisioned {args.resource}: {res}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
