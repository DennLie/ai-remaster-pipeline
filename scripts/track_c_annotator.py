#!/usr/bin/env python3
"""Offline local Windows human-annotation interface for Track C archival keyframes.

Strictly raw-only and model-independent:
- Shows ONLY original archival frames I_t and raw neighboring frames I_{t-2..t+2}.
- NEVER imports or displays detector masks, E1/E2 predictions, optical flow,
  repaired images, or temporal/spatial residual maps.
- Reads and writes strictly 3-tier 2D uint8 PNG masks {0, 127, 255} across separate
  scratch, dust, and negative-control channels without modifying raw frames.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse
import urllib.request
import webbrowser

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_track_c_annotation_pack import (  # noqa: E402
    ALLOWED_REVIEW_STATUSES,
    MANIFEST_OUT,
    STATUS_REPORT_JSON,
    STATUS_REPORT_MD,
    VALID_TIER_VALUES,
    frame_review_key,
    generate_status_report,
    load_annotation_reviews,
    resolve_manifest_path,
    resolve_pack_dir,
    verify_human_labels,
)

# Known raw spatial regions for quick viewport navigation only (zero auto-labeling).
RAW_ROI_BOOKMARKS: dict[tuple[str, int], list[dict[str, Any]]] = {
    ("neighbors_72", 25): [
        {
            "label": "Vertical Pipe Structure",
            "hint": "Inspect pipe column x:180..192, y:48..282 across neighbors for intact scene structure",
            "x": 140,
            "y": 24,
            "w": 96,
            "h": 280,
        }
    ],
    ("neighbors_72", 61): [
        {
            "label": "Full Frame Screening",
            "hint": "Screen entire frame carefully; compare t-2..t+2 for persistent branching scratch vs film structure",
            "x": 0,
            "y": 0,
            "w": 960,
            "h": 720,
        }
    ],
    ("neighbors_72", 62): [
        {
            "label": "Full Frame Screening",
            "hint": "Screen entire frame carefully; track defect continuity from frame 61 across t-2..t+2",
            "x": 0,
            "y": 0,
            "w": 960,
            "h": 720,
        }
    ],
    ("paleface_seq_a", 8): [
        {
            "label": "Actor Jacket Silhouette",
            "hint": "Inspect jacket edge x:258..268, y:188..381 with neighbor flicker to avoid false defect marks on moving actor",
            "x": 210,
            "y": 150,
            "w": 110,
            "h": 260,
        }
    ],
    ("paleface_seq_b", 7): [
        {
            "label": "Intact Wall & Curtain",
            "hint": "Inspect wooden wall/curtain texture x:200..280, y:80..180 (dev negative-control reference region)",
            "x": 170,
            "y": 55,
            "w": 145,
            "h": 155,
        }
    ],
    ("college_seq_c", 8): [
        {
            "label": "Door Jamb / Student",
            "hint": "Inspect door jamb/student x:280..340, y:150..280 carefully before classifying vertical edges",
            "x": 245,
            "y": 120,
            "w": 130,
            "h": 190,
        }
    ],
    ("neighbors_seq_d", 2): [
        {
            "label": "Utility Pole Structure",
            "hint": "Inspect pole x:150..220, y:200..400 across t-2..t+2 to separate intact vertical wood from defects",
            "x": 120,
            "y": 175,
            "w": 135,
            "h": 250,
        }
    ],
    ("neighbors_seq_d", 30): [
        {
            "label": "Utility Pole Structure",
            "hint": "Inspect pole x:150..220, y:200..400 across t-2..t+2 to separate intact vertical wood from defects",
            "x": 120,
            "y": 175,
            "w": 135,
            "h": 250,
        }
    ],
}

STRIP_HEADER_PX = 28


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _atomic_write_png(target: Path, arr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError(f"Failed to PNG-encode mask for {target}")
    _atomic_write_bytes(target, encoded.tobytes())


class TrackCAnnotatorBackend:
    """Thread-safe raw-only Track C mask storage and validation backend."""

    def __init__(
        self,
        *,
        manifest_path: Path = MANIFEST_OUT,
        pack_dir: Path | None = None,
        project_root: Path = PROJECT_ROOT,
        report_md: Path | None = None,
        report_json: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.manifest_data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.pack_dir = resolve_pack_dir(self.manifest_data, pack_dir=pack_dir, project_root=self.project_root)
        self.pilot_pack_rel = str(self.manifest_data.get("pilot_pack_dir") or ".cache/benchmarks/track_c_pilot_pack")
        self.report_md = report_md
        self.report_json = report_json
        self._lock = threading.Lock()
        self._items_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for item in self.manifest_data.get("pilot_keyframes", []):
            self._items_by_key[(str(item["clip_id"]), int(item["frame_index"]))] = item

    def _resolve(self, rel_or_abs: str) -> Path:
        return resolve_manifest_path(
            rel_or_abs,
            pack_dir=self.pack_dir,
            pilot_pack_rel=self.pilot_pack_rel,
            project_root=self.project_root,
        )

    def _get_item(self, clip_id: str, frame_index: int) -> dict[str, Any]:
        item = self._items_by_key.get((clip_id, int(frame_index)))
        if item is None:
            raise KeyError(f"Unknown pilot keyframe: {clip_id}#{frame_index}")
        return item

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            report = verify_human_labels(
                manifest_path=self.manifest_path,
                pack_dir=self.pack_dir,
                project_root=self.project_root,
            )
            for pf in report.get("per_frame", []):
                clip_id = str(pf["clip_id"])
                frame_idx = int(pf["frame_index"])
                item = self._items_by_key.get((clip_id, frame_idx), {})
                pf["roi_bookmarks"] = RAW_ROI_BOOKMARKS.get((clip_id, frame_idx), [])
                pf["manifest_notes"] = str(item.get("notes") or "")
            return report

    def read_context_png(self, clip_id: str, frame_index: int, offset: int) -> tuple[bytes, bool]:
        """Return lossless PNG bytes of raw frame I_{t+offset} and whether it is in-bounds."""
        if offset not in (-2, -1, 0, 1, 2):
            raise ValueError(f"Offset must be in -2..2, got {offset}")
        item = self._get_item(clip_id, frame_index)
        raw_path = self._resolve(str(item["raw_frame_path"]))
        if offset == 0:
            return raw_path.read_bytes(), True

        strip_path = self._resolve(str(item["neighbor_strip_path"]))
        raw_img = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        strip_img = cv2.imread(str(strip_path), cv2.IMREAD_COLOR)
        if raw_img is None or strip_img is None:
            raise FileNotFoundError(f"Missing raw/strip file for {clip_id}#{frame_index}")
        h, w = int(raw_img.shape[0]), int(raw_img.shape[1])
        slot = offset + 2
        x0 = slot * w
        x1 = x0 + w
        y0 = STRIP_HEADER_PX if strip_img.shape[0] >= h + STRIP_HEADER_PX else 0
        patch = strip_img[y0 : y0 + h, x0:x1]
        if patch.shape[0] != h or patch.shape[1] != w:
            raise ValueError(f"Unexpected neighbor strip geometry {strip_img.shape} for {h}x{w}")
        in_bounds = bool(np.any(patch != 0))
        ok, encoded = cv2.imencode(".png", patch)
        if not ok:
            raise RuntimeError("Failed to encode context PNG")
        return encoded.tobytes(), in_bounds

    def get_frame_masks(self, clip_id: str, frame_index: int) -> dict[str, Any]:
        with self._lock:
            item = self._get_item(clip_id, frame_index)
            scr_p = self._resolve(str(item["scratch_template_path"]))
            dst_p = self._resolve(str(item["dust_template_path"]))
            neg_p = self._resolve(str(item["neg_control_template_path"]))
            scr = cv2.imread(str(scr_p), cv2.IMREAD_UNCHANGED)
            dst = cv2.imread(str(dst_p), cv2.IMREAD_UNCHANGED)
            neg = cv2.imread(str(neg_p), cv2.IMREAD_UNCHANGED)
            for name, arr in (("scratch", scr), ("dust", dst), ("neg_control", neg)):
                if arr is None or arr.ndim != 2 or arr.dtype != np.uint8:
                    raise ValueError(f"Invalid {name} mask for {clip_id}#{frame_index}")
                unexpected = np.setdiff1d(np.unique(arr), VALID_TIER_VALUES)
                if unexpected.size > 0:
                    raise ValueError(f"{name} mask contains invalid tier values: {unexpected.tolist()}")
            if scr.shape != dst.shape or scr.shape != neg.shape:
                raise ValueError("Template mask dimensions disagree")

            h, w = int(scr.shape[0]), int(scr.shape[1])
            reviews = load_annotation_reviews(self.pack_dir)
            rev = reviews.get(frame_review_key(clip_id, frame_index), {})

            return {
                "clip_id": clip_id,
                "frame_index": int(frame_index),
                "width": w,
                "height": h,
                "dataset_role": str(item.get("split", "dev_diagnostic")),
                "scratch_b64": base64.b64encode(np.ascontiguousarray(scr).tobytes()).decode("ascii"),
                "dust_b64": base64.b64encode(np.ascontiguousarray(dst).tobytes()).decode("ascii"),
                "neg_control_b64": base64.b64encode(np.ascontiguousarray(neg).tobytes()).decode("ascii"),
                "review": {
                    "annotator": str(rev.get("annotator") or ""),
                    "second_pass_reviewer": str(rev.get("second_pass_reviewer") or ""),
                    "review_status": str(rev.get("review_status") or "pending"),
                    "notes": str(rev.get("notes") or item.get("notes") or ""),
                    "updated_at": str(rev.get("updated_at") or ""),
                },
                "roi_bookmarks": RAW_ROI_BOOKMARKS.get((clip_id, int(frame_index)), []),
                "manifest_notes": str(item.get("notes") or ""),
            }

    @staticmethod
    def _decode_tier_plane(b64_data: str, height: int, width: int, name: str) -> np.ndarray:
        raw = base64.b64decode(b64_data.encode("ascii"), validate=True)
        expected = height * width
        if len(raw) != expected:
            raise ValueError(f"{name} plane length {len(raw)} != expected {expected}")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width)).copy()
        unexpected = np.setdiff1d(np.unique(arr), VALID_TIER_VALUES)
        if unexpected.size > 0:
            raise ValueError(f"{name} plane contains disallowed values {unexpected.tolist()}; allowed {{0, 127, 255}}")
        return arr

    def save_frame_masks(self, clip_id: str, frame_index: int, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            item = self._get_item(clip_id, frame_index)
            scr_p = self._resolve(str(item["scratch_template_path"]))
            dst_p = self._resolve(str(item["dust_template_path"]))
            neg_p = self._resolve(str(item["neg_control_template_path"]))

            ref = cv2.imread(str(scr_p), cv2.IMREAD_UNCHANGED)
            if ref is None or ref.ndim != 2:
                raise ValueError(f"Existing scratch template unreadable for {clip_id}#{frame_index}")
            h, w = int(ref.shape[0]), int(ref.shape[1])

            scr = self._decode_tier_plane(str(payload["scratch_b64"]), h, w, "scratch")
            dst = self._decode_tier_plane(str(payload["dust_b64"]), h, w, "dust")
            neg = self._decode_tier_plane(str(payload["neg_control_b64"]), h, w, "neg_control")

            _atomic_write_png(scr_p, scr)
            _atomic_write_png(dst_p, dst)
            _atomic_write_png(neg_p, neg)

            review_in = payload.get("review", {}) if isinstance(payload.get("review"), dict) else {}
            reviewed_any = bool(np.any(scr != 127) or np.any(dst != 127) or np.any(neg != 127))
            status = str(review_in.get("review_status") or ("first_pass_in_progress" if reviewed_any else "pending"))
            if status not in ALLOWED_REVIEW_STATUSES:
                status = "first_pass_in_progress" if reviewed_any else "pending"

            review_path = self.pack_dir / "annotation_reviews.json"
            existing: dict[str, Any] = {"protocol_version": self.manifest_data.get("protocol_version", "2.0.0"), "frames": {}}
            if review_path.exists():
                try:
                    loaded = json.loads(review_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict) and isinstance(loaded.get("frames"), dict):
                        existing = loaded
                except Exception:
                    pass

            now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
            fkey = frame_review_key(clip_id, frame_index)
            prev_entry = existing["frames"].get(fkey, {}) if isinstance(existing.get("frames"), dict) else {}
            history = list(prev_entry.get("history", []))[-9:]
            history.append(
                {
                    "timestamp": now_iso,
                    "review_status": status,
                    "annotator": str(review_in.get("annotator") or "").strip(),
                    "second_pass_reviewer": str(review_in.get("second_pass_reviewer") or "").strip(),
                }
            )
            existing["frames"][fkey] = {
                "clip_id": clip_id,
                "frame_index": int(frame_index),
                "annotator": str(review_in.get("annotator") or "").strip(),
                "second_pass_reviewer": str(review_in.get("second_pass_reviewer") or "").strip(),
                "review_status": status,
                "notes": str(review_in.get("notes") or "").strip(),
                "updated_at": now_iso,
                "history": history,
            }
            _atomic_write_bytes(review_path, (json.dumps(existing, indent=2) + "\n").encode("utf-8"))

            report = verify_human_labels(
                manifest_path=self.manifest_path,
                pack_dir=self.pack_dir,
                project_root=self.project_root,
            )
            frame_audit = next(
                (f for f in report["per_frame"] if f["clip_id"] == clip_id and int(f["frame_index"]) == int(frame_index)),
                None,
            )
            return {"ok": True, "frame_audit": frame_audit, "summary": report}

    def export_status_report(self) -> dict[str, Any]:
        with self._lock:
            md_path = self.report_md if self.report_md is not None else (self.pack_dir / "track_c_pilot_status_report.md")
            json_path = self.report_json if self.report_json is not None else (self.pack_dir / "track_c_pilot_status_report.json")
            report = generate_status_report(
                manifest_path=self.manifest_path,
                pack_dir=self.pack_dir,
                project_root=self.project_root,
                md_out=md_path,
                json_out=json_path,
            )
            return {"ok": True, "md_path": str(md_path), "json_path": str(json_path), "summary": report}


def build_annotator_html() -> str:
    """Return the self-contained offline HTML5/JS Track C annotator single-page app."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Track C Independent Archival Annotator (Raw Frames Only)</title>
<style>
  :root {
    --bg: #0f1318;
    --panel: #171d24;
    --panel-2: #1f2730;
    --border: #2e3846;
    --text: #e6edf3;
    --muted: #93a1b2;
    --accent: #38bdf8;
    --warn: #f59e0b;
    --danger: #ef4444;
    --ok: #22c55e;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; width: 100%; height: 100%;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px; overflow: hidden;
  }
  .app {
    display: grid;
    grid-template-rows: 46px 1fr 142px;
    grid-template-columns: 292px 1fr 310px;
    height: 100vh; width: 100vw;
  }
  header {
    grid-column: 1 / 4;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 14px; background: var(--panel); border-bottom: 1px solid var(--border);
    gap: 10px;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 600; }
  .pill {
    padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600;
    background: #102a43; color: #7dd3fc; border: 1px solid #0284c7;
  }
  .pill.warn { background: #3b2407; color: #fcd34d; border-color: #d97706; }
  .pill.ok { background: #0f2f1e; color: #86efac; border-color: #16a34a; }
  .pill.danger { background: #3b1116; color: #fca5a5; border-color: #dc2626; }
  .top-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  button, select, input, textarea {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; font-family: inherit; font-size: 12px;
  }
  button {
    padding: 5px 10px; cursor: pointer; font-weight: 500;
    transition: border-color 0.12s, background 0.12s;
  }
  button:hover:not(:disabled) { border-color: var(--accent); background: #26313d; }
  button.active { background: #0c4a6e; border-color: #38bdf8; color: #fff; }
  button.primary { background: #0369a1; border-color: #0ea5e9; color: #fff; }
  button.primary:hover { background: #0284c7; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  aside {
    background: var(--panel); overflow-y: auto; padding: 10px;
    display: flex; flex-direction: column; gap: 10px;
  }
  .left-rail { border-right: 1px solid var(--border); }
  .right-rail { border-left: 1px solid var(--border); }
  .card {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; display: flex; flex-direction: column; gap: 8px;
  }
  .card h3 {
    margin: 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); display: flex; align-items: center; justify-content: space-between;
  }
  .frame-list { display: flex; flex-direction: column; gap: 5px; max-height: 290px; overflow-y: auto; }
  .frame-item {
    display: flex; flex-direction: column; gap: 3px; text-align: left;
    padding: 7px 8px; border-radius: 6px; border: 1px solid var(--border);
    background: #141a21; cursor: pointer;
  }
  .frame-item.selected { border-color: var(--accent); background: #102536; }
  .frame-item-top { display: flex; justify-content: space-between; font-weight: 600; font-size: 12px; }
  .frame-item-sub { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; }
  .palette-grid { display: grid; grid-template-columns: 1fr; gap: 5px; }
  .swatch-btn {
    display: flex; align-items: center; justify-content: space-between; padding: 6px 8px;
    text-align: left;
  }
  .swatch-left { display: flex; align-items: center; gap: 8px; }
  .dot { width: 12px; height: 12px; border-radius: 3px; border: 1px solid rgba(255,255,255,0.4); }
  .tool-row { display: flex; gap: 6px; flex-wrap: wrap; }
  .tool-row button { flex: 1; }
  main.stage-wrap {
    position: relative; background: #090c10; overflow: hidden;
    display: flex; flex-direction: column;
  }
  .stage-hud {
    position: absolute; top: 8px; left: 8px; right: 8px; pointer-events: none;
    display: flex; justify-content: space-between; align-items: flex-start; z-index: 5;
  }
  .hud-box {
    pointer-events: auto;
    background: rgba(15, 19, 24, 0.88); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 9px; font-size: 12px;
    display: flex; align-items: center; gap: 8px; backdrop-filter: blur(4px);
  }
  canvas#mainCanvas { width: 100%; height: 100%; display: block; cursor: crosshair; }
  footer.strip-dock {
    grid-column: 1 / 4;
    background: var(--panel); border-top: 1px solid var(--border);
    padding: 6px 12px; display: grid;
    grid-template-columns: 230px 1fr; gap: 12px; align-items: center;
  }
  .strip-controls { display: flex; flex-direction: column; gap: 5px; }
  .offset-btns { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; }
  .strip-thumbs {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; height: 100%;
    align-items: center;
  }
  .thumb-card {
    display: flex; flex-direction: column; align-items: center; gap: 3px;
    background: #10151b; border: 1px solid var(--border); border-radius: 6px;
    padding: 4px; cursor: pointer; height: 124px; justify-content: center;
  }
  .thumb-card.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .thumb-card canvas { width: 100%; height: 96px; object-fit: contain; image-rendering: pixelated; }
  .thumb-cap { font-size: 11px; color: var(--muted); }
  .stat-grid { display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; font-size: 12px; }
  .stat-val { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right; }
  .roi-btn { width: 100%; text-align: left; }
  label { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: var(--muted); }
  input, select, textarea { padding: 5px 7px; width: 100%; }
  textarea { resize: vertical; min-height: 54px; }
  .kbd {
    font-family: ui-monospace, Consolas, monospace; font-size: 10px;
    padding: 1px 4px; background: #10151b; border: 1px solid var(--border); border-radius: 3px;
    color: #cbd5e1;
  }
  .banner-raw {
    font-size: 11px; color: #7dd3fc; background: #0c2233;
    border: 1px solid #0369a1; border-radius: 6px; padding: 6px 8px; line-height: 1.35;
  }
  .banner-warn {
    font-size: 11px; color: #fcd34d; background: #332006;
    border: 1px solid #b45309; border-radius: 6px; padding: 6px 8px; line-height: 1.35;
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="brand">
      <span>Track C Independent Archival Annotator</span>
      <span class="pill">RAW FRAMES ONLY (NO MODEL OUTPUT)</span>
      <span class="pill warn" id="rolePill">DEV DIAGNOSTIC PILOT</span>
      <span id="unsavedPill" class="pill ok">Saved</span>
    </div>
    <div class="top-actions">
      <button id="btnUndo" title="Ctrl+Z">Undo</button>
      <button id="btnRedo" title="Ctrl+Y / Ctrl+Shift+Z">Redo</button>
      <button id="btnFit">Fit Screen</button>
      <button id="btnZoom1">100%</button>
      <button id="btnZoom4">400%</button>
      <button id="btnZoom10">1000%</button>
      <button id="btnExportReport">Write Status Report</button>
      <button id="btnSave" class="primary" title="Ctrl+S">Save Frame (Ctrl+S)</button>
    </div>
  </header>

  <aside class="left-rail">
    <div class="banner-raw">
      <strong>Strict Independence Rule:</strong> Only original archival frame <code>I_t</code> and raw neighbors <code>t-2..t+2</code> are displayed. Unreviewed pixels start at <code>127</code> and are never initialized to <code>0</code>.
    </div>

    <div class="card">
      <h3>Pilot Keyframes <span id="packProgress">0/14</span></h3>
      <div class="frame-list" id="frameList"></div>
    </div>

    <div class="card">
      <h3>Label Class (3-Tier {0, 127, 255})</h3>
      <div class="palette-grid">
        <button class="swatch-btn active" data-class="scratch" id="cls-scratch">
          <span class="swatch-left"><span class="dot" style="background:#00e5ff"></span><span>Scratch Defect (255)</span></span>
          <span class="kbd">1</span>
        </button>
        <button class="swatch-btn" data-class="dust" id="cls-dust">
          <span class="swatch-left"><span class="dot" style="background:#ff9800"></span><span>Dust / Dirt (255)</span></span>
          <span class="kbd">2</span>
        </button>
        <button class="swatch-btn" data-class="intact" id="cls-intact">
          <span class="swatch-left"><span class="dot" style="background:#22c55e"></span><span>Verified Intact (0)</span></span>
          <span class="kbd">3</span>
        </button>
        <button class="swatch-btn" data-class="neg_control" id="cls-neg_control">
          <span class="swatch-left"><span class="dot" style="background:#e879f9"></span><span>Neg Control ROI (255+0)</span></span>
          <span class="kbd">4</span>
        </button>
        <button class="swatch-btn" data-class="uncertain" id="cls-uncertain">
          <span class="swatch-left"><span class="dot" style="background:#64748b"></span><span>Uncertain / Reset (127)</span></span>
          <span class="kbd">5</span>
        </button>
      </div>
    </div>

    <div class="card">
      <h3>Drawing Tool &amp; View</h3>
      <div class="tool-row">
        <button id="tool-brush" class="active" data-tool="brush">Brush <span class="kbd">B</span></button>
        <button id="tool-line" data-tool="line">Line <span class="kbd">L</span></button>
        <button id="tool-rect" data-tool="rect">Rect <span class="kbd">R</span></button>
      </div>
      <label>
        Brush Radius (px): <strong id="brushSizeVal">2 px</strong> (<span class="kbd">[</span> / <span class="kbd">]</span>)
        <input type="range" id="brushSize" min="1" max="32" value="2" />
      </label>
      <div class="tool-row">
        <button id="btnToggleOverlay">Mask Overlay: <span id="overlayVal">50%</span> <span class="kbd">H</span></button>
      </div>
      <div class="tool-row">
        <button id="btnToggleUnreviewed">Highlight 127 Unreviewed: <span id="unrevVal">OFF</span> <span class="kbd">U</span></button>
      </div>
    </div>
  </aside>

  <main class="stage-wrap" id="stageWrap">
    <div class="stage-hud">
      <div class="hud-box">
        <strong id="hudFrameTitle">-</strong>
        <span>Showing Raw: <strong id="hudOffsetLabel" style="color:#38bdf8">t (I_t)</strong></span>
        <span>Zoom: <strong id="hudZoom">100%</strong></span>
      </div>
      <div class="hud-box" id="hudPixelInfo">
        <span>(x, y): -</span>
      </div>
    </div>
    <canvas id="mainCanvas"></canvas>
  </main>

  <aside class="right-rail">
    <div class="card">
      <h3>Raw ROI Quick Zoom</h3>
      <div id="roiList" style="display:flex; flex-direction:column; gap:5px;"></div>
      <div id="roiHint" style="font-size:11px; color:var(--muted); line-height:1.35;"></div>
    </div>

    <div class="card">
      <h3>Current Frame Coverage &amp; Audit</h3>
      <div class="stat-grid">
        <span>Scratch (`255`) px:</span><span class="stat-val" id="stScratch">0</span>
        <span>Dust (`255`) px:</span><span class="stat-val" id="stDust">0</span>
        <span>Neg Control ROI px:</span><span class="stat-val" id="stNegCtrl">0</span>
        <span>Verified Intact (`0`) px:</span><span class="stat-val" id="stIntact">0</span>
        <span>Uncertain / Unreviewed (`127`):</span><span class="stat-val" id="stUncertain">0</span>
        <span>Reviewed Coverage:</span><span class="stat-val" id="stCoverage">0.00%</span>
        <span>Channel Conflicts:</span><span class="stat-val" id="stConflicts">0</span>
      </div>
      <div id="conflictWarn" class="banner-warn" style="display:none;"></div>
    </div>

    <div class="card">
      <h3>Provenance &amp; 2nd-Pass Review</h3>
      <label>Annotator (1st pass)
        <input id="inpAnnotator" type="text" placeholder="e.g. DL" />
      </label>
      <label>2nd-Pass Reviewer
        <input id="inpReviewer" type="text" placeholder="independent verifier initials" />
      </label>
      <label>Review Status
        <select id="selStatus">
          <option value="pending">pending</option>
          <option value="first_pass_in_progress">first_pass_in_progress</option>
          <option value="first_pass_complete">first_pass_complete</option>
          <option value="second_pass_verified">second_pass_verified</option>
          <option value="needs_revision">needs_revision</option>
        </select>
      </label>
      <label>Frame Notes / Ambiguities
        <textarea id="txtNotes" placeholder="Note motion blur, weave, ambiguous cracks left at 127..."></textarea>
      </label>
      <div style="font-size:11px; color:var(--muted);">
        Last saved: <span id="lblUpdatedAt">-</span>
      </div>
    </div>

    <div class="card">
      <h3>Shortcuts</h3>
      <div style="font-size:11px; color:var(--muted); line-height:1.45;">
        <div><span class="kbd">Left-Drag</span> Paint active class</div>
        <div><span class="kbd">Wheel</span> Zoom to cursor &bull; <span class="kbd">Right/Middle-Drag</span> Pan</div>
        <div><span class="kbd">Q</span>/<span class="kbd">W</span> Neighbor frame &bull; <span class="kbd">T</span> Blink <span class="kbd">G</span> Raw <code>t</code></div>
        <div><span class="kbd">B</span> Brush &bull; <span class="kbd">L</span> Line &bull; <span class="kbd">R</span> Rect &bull; <span class="kbd">[</span>/<span class="kbd">]</span> Size</div>
      </div>
    </div>
  </aside>

  <footer class="strip-dock">
    <div class="strip-controls">
      <div style="font-weight:600;">Raw Neighbor Flip (<span class="kbd">Q</span>/<span class="kbd">W</span>)</div>
      <div class="offset-btns">
        <button data-offset="-2" id="off--2">t-2</button>
        <button data-offset="-1" id="off--1">t-1</button>
        <button data-offset="0" id="off-0" class="active">t</button>
        <button data-offset="1" id="off-1">t+1</button>
        <button data-offset="2" id="off-2">t+2</button>
      </div>
      <div style="display:flex; gap:6px;">
        <button id="btnBlink" style="flex:1;">Blink t-1..t+1 (T)</button>
        <button id="btnCenterT">t (G)</button>
      </div>
    </div>
    <div class="strip-thumbs" id="stripThumbs">
      <div class="thumb-card" data-offset="-2"><canvas id="thumb--2"></canvas><span class="thumb-cap">t-2</span></div>
      <div class="thumb-card" data-offset="-1"><canvas id="thumb--1"></canvas><span class="thumb-cap">t-1</span></div>
      <div class="thumb-card active" data-offset="0"><canvas id="thumb-0"></canvas><span class="thumb-cap">t (target)</span></div>
      <div class="thumb-card" data-offset="1"><canvas id="thumb-1"></canvas><span class="thumb-cap">t+1</span></div>
      <div class="thumb-card" data-offset="2"><canvas id="thumb-2"></canvas><span class="thumb-cap">t+2</span></div>
    </div>
  </footer>
</div>

<script>
"use strict";
(() => {
  const OFFSETS = [-2, -1, 0, 1, 2];
  const state = {
    status: null,
    currentClip: null,
    currentFrame: null,
    width: 0,
    height: 0,
    scr: null,
    dst: null,
    neg: null,
    ctxImages: {},
    ctxInBounds: {},
    ctxCanvases: {},
    rawSampleCtx: null,
    overlayCanvas: document.createElement("canvas"),
    overlayCtx: null,
    overlayImgData: null,
    activeOffset: 0,
    blinkTimer: null,
    paintClass: "scratch",
    tool: "brush",
    brushRadius: 2,
    overlayMode: 1, // 0=0%, 1=50%, 2=80%
    highlightUnreviewed: false,
    zoom: 1,
    panX: 20,
    panY: 20,
    dirty: false,
    undoStack: [],
    redoStack: [],
    isDrawing: false,
    isPanning: false,
    panStart: null,
    lastPt: null,
    shapeStart: null,
    shapeEnd: null,
    hoverPt: null,
    activeStrokePatches: null,
  };
  state.overlayCtx = state.overlayCanvas.getContext("2d");

  const canvas = document.getElementById("mainCanvas");
  const ctx = canvas.getContext("2d");
  const stageWrap = document.getElementById("stageWrap");

  function decodeB64Bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function encodeBytesB64(arr) {
    let bin = "";
    const chunk = 0x8000;
    for (let i = 0; i < arr.length; i += chunk) {
      bin += String.fromCharCode.apply(null, arr.subarray(i, i + chunk));
    }
    return btoa(bin);
  }

  function setDirty(flag) {
    state.dirty = flag;
    const pill = document.getElementById("unsavedPill");
    if (flag) {
      pill.textContent = "Unsaved Changes";
      pill.className = "pill warn";
    } else {
      pill.textContent = "Saved";
      pill.className = "pill ok";
    }
  }

  function updatePaletteUI() {
    document.querySelectorAll("[data-class]").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.class === state.paintClass);
    });
    document.querySelectorAll("[data-tool]").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tool === state.tool);
    });
    document.getElementById("brushSizeVal").textContent = `${state.brushRadius} px`;
    document.getElementById("brushSize").value = String(state.brushRadius);
    const modes = ["0%", "50%", "80%"];
    document.getElementById("overlayVal").textContent = modes[state.overlayMode];
    document.getElementById("unrevVal").textContent = state.highlightUnreviewed ? "ON" : "OFF";
    document.getElementById("btnUndo").disabled = state.undoStack.length === 0;
    document.getElementById("btnRedo").disabled = state.redoStack.length === 0;
  }

  function computeOverlayPixel(i, buf32) {
    const s = state.scr[i];
    const d = state.dst[i];
    const n = state.neg[i];
    let r = 0, g = 0, b = 0, a = 0;
    if (s === 255 && d === 255) {
      r = 255; g = 40; b = 80; a = 240; // conflict highlight
    } else if (n === 255 && (s === 255 || d === 255)) {
      r = 255; g = 0; b = 60; a = 245; // conflict highlight
    } else if (s === 255) {
      r = 0; g = 229; b = 255; a = 220;
    } else if (d === 255) {
      r = 255; g = 152; b = 0; a = 220;
    } else if (n === 255) {
      r = 232; g = 121; b = 249; a = 185;
    } else if (s === 0 && d === 0) {
      r = 34; g = 197; b = 94; a = 155;
    } else if (state.highlightUnreviewed) {
      const x = i % state.width;
      const y = (i / state.width) | 0;
      const checker = ((x >> 2) + (y >> 2)) & 1;
      r = checker ? 148 : 90; g = checker ? 163 : 102; b = checker ? 184 : 118; a = 115;
    }
    buf32[i] = (a << 24) | (b << 16) | (g << 8) | r;
  }

  function refreshOverlayRect(x0, y0, x1, y1) {
    if (!state.overlayImgData) return;
    const w = state.width, h = state.height;
    const cx0 = Math.max(0, x0 | 0), cy0 = Math.max(0, y0 | 0);
    const cx1 = Math.min(w - 1, x1 | 0), cy1 = Math.min(h - 1, y1 | 0);
    const buf32 = new Uint32Array(state.overlayImgData.data.buffer);
    for (let y = cy0; y <= cy1; y++) {
      let idx = y * w + cx0;
      for (let x = cx0; x <= cx1; x++, idx++) {
        computeOverlayPixel(idx, buf32);
      }
    }
    state.overlayCtx.putImageData(state.overlayImgData, 0, 0, cx0, cy0, cx1 - cx0 + 1, cy1 - cy0 + 1);
  }

  function refreshStatsUI() {
    if (!state.scr) return;
    const n = state.width * state.height;
    let sc = 0, ds = 0, ng = 0, intact = 0, bothRev = 0, conflicts = 0;
    for (let i = 0; i < n; i++) {
      const s = state.scr[i], d = state.dst[i], g = state.neg[i];
      if (s === 255) sc++;
      if (d === 255) ds++;
      if (g === 255) ng++;
      if (s === 0 && d === 0) intact++;
      if (s !== 127 && d !== 127) bothRev++;
      if ((s === 255 && d === 255) || (g === 255 && (s === 255 || d === 255))) conflicts++;
    }
    const cov = (100.0 * bothRev) / Math.max(n, 1);
    document.getElementById("stScratch").textContent = sc.toLocaleString();
    document.getElementById("stDust").textContent = ds.toLocaleString();
    document.getElementById("stNegCtrl").textContent = ng.toLocaleString();
    document.getElementById("stIntact").textContent = intact.toLocaleString();
    document.getElementById("stUncertain").textContent = (n - bothRev).toLocaleString();
    document.getElementById("stCoverage").textContent = `${cov.toFixed(2)}%`;
    document.getElementById("stConflicts").textContent = conflicts.toLocaleString();
    const warn = document.getElementById("conflictWarn");
    if (conflicts > 0) {
      warn.style.display = "block";
      warn.textContent = `${conflicts} pixel conflict(s) (scratch+dust overlap or defect inside negative-control ROI). Resolve before approving.`;
    } else {
      warn.style.display = "none";
    }
  }

  function paintPixel(idx) {
    const cls = state.paintClass;
    if (cls === "scratch") {
      state.scr[idx] = 255; state.dst[idx] = 0; if (state.neg[idx] === 255) state.neg[idx] = 127;
    } else if (cls === "dust") {
      state.scr[idx] = 0; state.dst[idx] = 255; if (state.neg[idx] === 255) state.neg[idx] = 127;
    } else if (cls === "intact") {
      state.scr[idx] = 0; state.dst[idx] = 0;
    } else if (cls === "neg_control") {
      state.neg[idx] = 255; state.scr[idx] = 0; state.dst[idx] = 0;
    } else if (cls === "uncertain") {
      state.scr[idx] = 127; state.dst[idx] = 127; state.neg[idx] = 127;
    }
  }

  function collectDiskPoints(cx, cy, r) {
    const pts = [];
    const rr = r <= 1 ? 0 : r - 0.5;
    const r2 = rr * rr;
    const rad = Math.ceil(rr);
    for (let dy = -rad; dy <= rad; dy++) {
      for (let dx = -rad; dx <= rad; dx++) {
        if (dx * dx + dy * dy <= r2 + 0.01) {
          const x = cx + dx, y = cy + dy;
          if (x >= 0 && y >= 0 && x < state.width && y < state.height) pts.push([x, y]);
        }
      }
    }
    return pts;
  }

  function applyMaskRegion(coords) {
    if (!coords.length || !state.scr) return;
    let minX = state.width, minY = state.height, maxX = -1, maxY = -1;
    const visited = new Set();
    for (const [x, y] of coords) {
      if (x < 0 || y < 0 || x >= state.width || y >= state.height) continue;
      const idx = y * state.width + x;
      if (!visited.has(idx)) {
        visited.add(idx);
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
    if (visited.size === 0) return;

    const rw = maxX - minX + 1, rh = maxY - minY + 1;
    const oldScr = new Uint8Array(rw * rh);
    const oldDst = new Uint8Array(rw * rh);
    const oldNeg = new Uint8Array(rw * rh);
    const newScr = new Uint8Array(rw * rh);
    const newDst = new Uint8Array(rw * rh);
    const newNeg = new Uint8Array(rw * rh);

    for (let ry = 0; ry < rh; ry++) {
      for (let rx = 0; rx < rw; rx++) {
        const gi = (minY + ry) * state.width + (minX + rx);
        const li = ry * rw + rx;
        oldScr[li] = state.scr[gi];
        oldDst[li] = state.dst[gi];
        oldNeg[li] = state.neg[gi];
        if (visited.has(gi)) paintPixel(gi);
        newScr[li] = state.scr[gi];
        newDst[li] = state.dst[gi];
        newNeg[li] = state.neg[gi];
      }
    }
    state.undoStack.push({ x: minX, y: minY, w: rw, h: rh, oldScr, oldDst, oldNeg, newScr, newDst, newNeg });
    if (state.undoStack.length > 50) state.undoStack.shift();
    state.redoStack = [];
    refreshOverlayRect(minX, minY, maxX, maxY);
    setDirty(true);
    refreshStatsUI();
    updatePaletteUI();
    render();
  }

  function applyLiveBrushSegment(x0, y0, x1, y1) {
    const points = [];
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let s = 0; s <= steps; s++) {
      const px = Math.round(x0 + ((x1 - x0) * s) / steps);
      const py = Math.round(y0 + ((y1 - y0) * s) / steps);
      for (const p of collectDiskPoints(px, py, state.brushRadius)) points.push(p);
    }
    let minX = state.width, minY = state.height, maxX = -1, maxY = -1;
    for (const [px, py] of points) {
      const gi = py * state.width + px;
      if (!state.activeStrokePatches.has(gi)) {
        state.activeStrokePatches.set(gi, [state.scr[gi], state.dst[gi], state.neg[gi]]);
      }
      paintPixel(gi);
      if (px < minX) minX = px;
      if (py < minY) minY = py;
      if (px > maxX) maxX = px;
      if (py > maxY) maxY = py;
    }
    if (maxX >= minX) {
      refreshOverlayRect(minX, minY, maxX, maxY);
      setDirty(true);
      refreshStatsUI();
      render();
    }
  }

  function finishLiveStroke() {
    if (!state.activeStrokePatches || state.activeStrokePatches.size === 0) {
      state.activeStrokePatches = null;
      return;
    }
    let minX = state.width, minY = state.height, maxX = -1, maxY = -1;
    for (const gi of state.activeStrokePatches.keys()) {
      const x = gi % state.width, y = (gi / state.width) | 0;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    const rw = maxX - minX + 1, rh = maxY - minY + 1;
    const oldScr = new Uint8Array(rw * rh);
    const oldDst = new Uint8Array(rw * rh);
    const oldNeg = new Uint8Array(rw * rh);
    const newScr = new Uint8Array(rw * rh);
    const newDst = new Uint8Array(rw * rh);
    const newNeg = new Uint8Array(rw * rh);
    for (let ry = 0; ry < rh; ry++) {
      for (let rx = 0; rx < rw; rx++) {
        const gi = (minY + ry) * state.width + (minX + rx);
        const li = ry * rw + rx;
        const prev = state.activeStrokePatches.get(gi);
        oldScr[li] = prev ? prev[0] : state.scr[gi];
        oldDst[li] = prev ? prev[1] : state.dst[gi];
        oldNeg[li] = prev ? prev[2] : state.neg[gi];
        newScr[li] = state.scr[gi];
        newDst[li] = state.dst[gi];
        newNeg[li] = state.neg[gi];
      }
    }
    state.undoStack.push({ x: minX, y: minY, w: rw, h: rh, oldScr, oldDst, oldNeg, newScr, newDst, newNeg });
    if (state.undoStack.length > 50) state.undoStack.shift();
    state.redoStack = [];
    state.activeStrokePatches = null;
    updatePaletteUI();
  }

  function applyPatch(patch, forward) {
    const sSrc = forward ? patch.newScr : patch.oldScr;
    const dSrc = forward ? patch.newDst : patch.oldDst;
    const nSrc = forward ? patch.newNeg : patch.oldNeg;
    for (let ry = 0; ry < patch.h; ry++) {
      for (let rx = 0; rx < patch.w; rx++) {
        const gi = (patch.y + ry) * state.width + (patch.x + rx);
        const li = ry * patch.w + rx;
        state.scr[gi] = sSrc[li];
        state.dst[gi] = dSrc[li];
        state.neg[gi] = nSrc[li];
      }
    }
    refreshOverlayRect(patch.x, patch.y, patch.x + patch.w - 1, patch.y + patch.h - 1);
    setDirty(true);
    refreshStatsUI();
    updatePaletteUI();
    render();
  }

  function undo() {
    if (!state.undoStack.length) return;
    const p = state.undoStack.pop();
    state.redoStack.push(p);
    applyPatch(p, false);
  }

  function redo() {
    if (!state.redoStack.length) return;
    const p = state.redoStack.pop();
    state.undoStack.push(p);
    applyPatch(p, true);
  }

  function screenToImage(sx, sy) {
    return [
      Math.floor((sx - state.panX) / state.zoom),
      Math.floor((sy - state.panY) / state.zoom),
    ];
  }

  function fitScreen() {
    if (!state.width || !state.height) return;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    const z = Math.min((cw - 24) / state.width, (ch - 24) / state.height);
    state.zoom = Math.max(0.25, z);
    state.panX = (cw - state.width * state.zoom) / 2;
    state.panY = (ch - state.height * state.zoom) / 2;
    render();
  }

  function zoomToRect(rx, ry, rw, rh) {
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    const z = Math.min((cw - 40) / rw, (ch - 40) / rh);
    state.zoom = Math.max(0.5, Math.min(24, z));
    state.panX = cw / 2 - (rx + rw / 2) * state.zoom;
    state.panY = ch / 2 - (ry + rh / 2) * state.zoom;
    render();
  }

  function setZoomCenter(newZoom, cx, cy) {
    const clamped = Math.max(0.25, Math.min(24, newZoom));
    const ix = (cx - state.panX) / state.zoom;
    const iy = (cy - state.panY) / state.zoom;
    state.zoom = clamped;
    state.panX = cx - ix * state.zoom;
    state.panY = cy - iy * state.zoom;
    render();
  }

  function updateStripThumbnails() {
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    const centerX = state.hoverPt ? state.hoverPt[0] : (cw / 2 - state.panX) / state.zoom;
    const centerY = state.hoverPt ? state.hoverPt[1] : (ch / 2 - state.panY) / state.zoom;
    const patchSize = Math.max(32, Math.min(180, Math.round(140 / Math.max(state.zoom, 0.7))));
    const sx = Math.max(0, Math.min(state.width - patchSize, Math.round(centerX - patchSize / 2)));
    const sy = Math.max(0, Math.min(state.height - patchSize, Math.round(centerY - patchSize / 2)));

    for (const off of OFFSETS) {
      const tc = document.getElementById(`thumb-${off}`);
      const tctx = tc.getContext("2d");
      tc.width = 160; tc.height = 96;
      tctx.imageSmoothingEnabled = false;
      tctx.fillStyle = "#090c10";
      tctx.fillRect(0, 0, tc.width, tc.height);
      const src = state.ctxImages[off];
      if (src) {
        tctx.drawImage(src, sx, sy, patchSize, patchSize, 16, 0, 128, 96);
        tctx.strokeStyle = "rgba(56, 189, 248, 0.7)";
        tctx.lineWidth = 1;
        tctx.beginPath();
        tctx.moveTo(80, 38); tctx.lineTo(80, 58);
        tctx.moveTo(70, 48); tctx.lineTo(90, 48);
        tctx.stroke();
      }
    }
  }

  function render() {
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw; canvas.height = ch;
    }
    ctx.fillStyle = "#090c10";
    ctx.fillRect(0, 0, cw, ch);
    if (!state.width) return;

    const img = state.ctxImages[state.activeOffset] || state.ctxImages[0];
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    if (img) {
      ctx.drawImage(img, state.panX, state.panY, state.width * state.zoom, state.height * state.zoom);
    }
    const alphas = [0.0, 0.5, 0.8];
    const alpha = alphas[state.overlayMode];
    if (alpha > 0 && state.overlayImgData) {
      ctx.globalAlpha = alpha;
      ctx.drawImage(state.overlayCanvas, state.panX, state.panY, state.width * state.zoom, state.height * state.zoom);
      ctx.globalAlpha = 1.0;
    }

    if (state.zoom >= 6) {
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = 1;
      const xStart = Math.max(0, Math.floor(-state.panX / state.zoom));
      const xEnd = Math.min(state.width, Math.ceil((cw - state.panX) / state.zoom));
      const yStart = Math.max(0, Math.floor(-state.panY / state.zoom));
      const yEnd = Math.min(state.height, Math.ceil((ch - state.panY) / state.zoom));
      ctx.beginPath();
      for (let x = xStart; x <= xEnd; x++) {
        const sx = Math.round(state.panX + x * state.zoom) + 0.5;
        ctx.moveTo(sx, state.panY + yStart * state.zoom);
        ctx.lineTo(sx, state.panY + yEnd * state.zoom);
      }
      for (let y = yStart; y <= yEnd; y++) {
        const sy = Math.round(state.panY + y * state.zoom) + 0.5;
        ctx.moveTo(state.panX + xStart * state.zoom, sy);
        ctx.lineTo(state.panX + xEnd * state.zoom, sy);
      }
      ctx.stroke();
    }

    if (state.isDrawing && state.shapeStart && state.shapeEnd) {
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      if (state.tool === "line") {
        ctx.beginPath();
        ctx.moveTo(state.panX + (state.shapeStart[0] + 0.5) * state.zoom, state.panY + (state.shapeStart[1] + 0.5) * state.zoom);
        ctx.lineTo(state.panX + (state.shapeEnd[0] + 0.5) * state.zoom, state.panY + (state.shapeEnd[1] + 0.5) * state.zoom);
        ctx.stroke();
      } else if (state.tool === "rect") {
        const rx0 = Math.min(state.shapeStart[0], state.shapeEnd[0]);
        const ry0 = Math.min(state.shapeStart[1], state.shapeEnd[1]);
        const rw = Math.abs(state.shapeEnd[0] - state.shapeStart[0]) + 1;
        const rh = Math.abs(state.shapeEnd[1] - state.shapeStart[1]) + 1;
        ctx.strokeRect(state.panX + rx0 * state.zoom, state.panY + ry0 * state.zoom, rw * state.zoom, rh * state.zoom);
      }
    }

    if (state.hoverPt && state.tool === "brush") {
      const [hx, hy] = state.hoverPt;
      if (hx >= 0 && hy >= 0 && hx < state.width && hy < state.height) {
        ctx.strokeStyle = "rgba(255,255,255,0.85)";
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.arc(
          state.panX + (hx + 0.5) * state.zoom,
          state.panY + (hy + 0.5) * state.zoom,
          Math.max(0.5, state.brushRadius - 0.3) * state.zoom,
          0, Math.PI * 2
        );
        ctx.stroke();
      }
    }
    ctx.restore();

    document.getElementById("hudZoom").textContent = `${Math.round(state.zoom * 100)}%`;
    const lbl = state.activeOffset === 0 ? "t (I_t)" : (state.activeOffset > 0 ? `t+${state.activeOffset}` : `t${state.activeOffset}`);
    const outOfRange = state.ctxInBounds[state.activeOffset] === false ? " [OUT OF RANGE / BORDER]" : "";
    document.getElementById("hudOffsetLabel").textContent = lbl + outOfRange;
    updateStripThumbnails();
  }

  function setOffset(off) {
    state.activeOffset = off;
    document.querySelectorAll("[data-offset]").forEach(el => {
      el.classList.toggle("active", Number(el.dataset.offset) === off);
    });
    render();
  }

  function renderFrameList() {
    const list = document.getElementById("frameList");
    list.innerHTML = "";
    const report = state.status;
    document.getElementById("packProgress").textContent = `${report.completed_keyframes}/${report.total_keyframes} (2nd: ${report.second_pass_verified_keyframes})`;
    for (const pf of report.per_frame) {
      const btn = document.createElement("button");
      const active = pf.clip_id === state.currentClip && Number(pf.frame_index) === Number(state.currentFrame);
      btn.className = "frame-item" + (active ? " selected" : "");
      const statusColor = pf.review_status === "second_pass_verified" ? "#86efac" : (pf.hard_conflict_px > 0 ? "#fca5a5" : (pf.any_channel_reviewed_px > 0 ? "#7dd3fc" : "#93a1b2"));
      btn.innerHTML = `
        <div class="frame-item-top">
          <span>${pf.clip_id} #${pf.frame_index}</span>
          <span style="color:${statusColor}">${pf.review_status}</span>
        </div>
        <div class="frame-item-sub">
          <span>cov: ${pf.reviewed_coverage_pct.toFixed(1)}% | scr:${pf.scratch_px} dst:${pf.dust_px}</span>
          <span>${pf.hard_conflict_px > 0 ? "CONFLICT:" + pf.hard_conflict_px : pf.dataset_role}</span>
        </div>
      `;
      btn.onclick = () => selectKeyframe(pf.clip_id, pf.frame_index);
      list.appendChild(btn);
    }
  }

  function loadImagePromise(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url + "?t=" + Date.now();
    });
  }

  async function selectKeyframe(clipId, frameIdx) {
    if (state.dirty && state.currentClip !== null) {
      const ok = confirm("You have unsaved mask edits on the current frame. Save before switching?");
      if (ok) await saveCurrentFrame();
    }
    if (state.blinkTimer) { clearInterval(state.blinkTimer); state.blinkTimer = null; }

    const res = await fetch(`/api/frame/${encodeURIComponent(clipId)}/${frameIdx}/masks`);
    const data = await res.json();
    state.currentClip = clipId;
    state.currentFrame = Number(frameIdx);
    state.width = data.width;
    state.height = data.height;
    state.scr = decodeB64Bytes(data.scratch_b64);
    state.dst = decodeB64Bytes(data.dust_b64);
    state.neg = decodeB64Bytes(data.neg_control_b64);
    state.undoStack = [];
    state.redoStack = [];
    setDirty(false);

    state.overlayCanvas.width = state.width;
    state.overlayCanvas.height = state.height;
    state.overlayImgData = state.overlayCtx.createImageData(state.width, state.height);
    refreshOverlayRect(0, 0, state.width - 1, state.height - 1);

    document.getElementById("inpAnnotator").value = data.review.annotator || "";
    document.getElementById("inpReviewer").value = data.review.second_pass_reviewer || "";
    document.getElementById("selStatus").value = data.review.review_status || "pending";
    document.getElementById("txtNotes").value = data.review.notes || "";
    document.getElementById("lblUpdatedAt").textContent = data.review.updated_at || "not yet saved";
    document.getElementById("hudFrameTitle").textContent = `${clipId} frame ${frameIdx} (${state.width}x${state.height})`;

    const roiList = document.getElementById("roiList");
    const roiHint = document.getElementById("roiHint");
    roiList.innerHTML = "";
    roiHint.textContent = data.manifest_notes || "";
    const fitBtn = document.createElement("button");
    fitBtn.className = "roi-btn";
    fitBtn.textContent = "Full Frame View";
    fitBtn.onclick = () => fitScreen();
    roiList.appendChild(fitBtn);
    for (const r of (data.roi_bookmarks || [])) {
      const b = document.createElement("button");
      b.className = "roi-btn";
      b.textContent = `Zoom: ${r.label} (x:${r.x} y:${r.y})`;
      b.onclick = () => {
        zoomToRect(r.x, r.y, r.w, r.h);
        roiHint.textContent = r.hint;
      };
      roiList.appendChild(b);
    }

    state.ctxImages = {};
    state.ctxInBounds = {};
    await Promise.all(OFFSETS.map(async (off) => {
      const imgResp = await fetch(`/api/frame/${encodeURIComponent(clipId)}/${frameIdx}/context/${off}`);
      state.ctxInBounds[off] = imgResp.headers.get("X-In-Bounds") !== "false";
      const blob = await imgResp.blob();
      const url = URL.createObjectURL(blob);
      state.ctxImages[off] = await new Promise((resolve) => {
        const im = new Image();
        im.onload = () => { URL.revokeObjectURL(url); resolve(im); };
        im.src = url;
      });
    }));

    const sampleCan = document.createElement("canvas");
    sampleCan.width = state.width; sampleCan.height = state.height;
    state.rawSampleCtx = sampleCan.getContext("2d", { willReadFrequently: true });
    state.rawSampleCtx.drawImage(state.ctxImages[0], 0, 0);

    setOffset(0);
    fitScreen();
    refreshStatsUI();
    updatePaletteUI();
    renderFrameList();
  }

  async function saveCurrentFrame() {
    if (!state.currentClip) return;
    const payload = {
      scratch_b64: encodeBytesB64(state.scr),
      dust_b64: encodeBytesB64(state.dst),
      neg_control_b64: encodeBytesB64(state.neg),
      review: {
        annotator: document.getElementById("inpAnnotator").value,
        second_pass_reviewer: document.getElementById("inpReviewer").value,
        review_status: document.getElementById("selStatus").value,
        notes: document.getElementById("txtNotes").value,
      },
    };
    const res = await fetch(`/api/frame/${encodeURIComponent(state.currentClip)}/${state.currentFrame}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const out = await res.json();
    if (out.ok) {
      state.status = out.summary;
      setDirty(false);
      document.getElementById("lblUpdatedAt").textContent = out.frame_audit?.updated_at || "just now";
      renderFrameList();
    } else {
      alert("Save error: " + (out.error || "unknown error"));
    }
  }

  async function boot() {
    const res = await fetch("/api/status");
    state.status = await res.json();
    renderFrameList();
    if (state.status.per_frame.length > 0) {
      const f = state.status.per_frame[0];
      await selectKeyframe(f.clip_id, f.frame_index);
    }
  }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    if (e.button === 1 || e.button === 2 || e.altKey) {
      state.isPanning = true;
      state.panStart = [e.clientX - state.panX, e.clientY - state.panY];
      return;
    }
    if (e.button !== 0 || !state.scr) return;
    const [ix, iy] = screenToImage(sx, sy);
    state.isDrawing = true;
    state.lastPt = [ix, iy];
    if (state.tool === "brush") {
      state.activeStrokePatches = new Map();
      applyLiveBrushSegment(ix, iy, ix, iy);
    } else {
      state.shapeStart = [ix, iy];
      state.shapeEnd = [ix, iy];
      render();
    }
  });

  window.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const [ix, iy] = screenToImage(sx, sy);
    state.hoverPt = [ix, iy];

    if (state.isPanning && state.panStart) {
      state.panX = e.clientX - state.panStart[0];
      state.panY = e.clientY - state.panStart[1];
      render();
      return;
    }

    if (state.scr && ix >= 0 && iy >= 0 && ix < state.width && iy < state.height) {
      const idx = iy * state.width + ix;
      let rgbTxt = "";
      if (state.rawSampleCtx) {
        const px = state.rawSampleCtx.getImageData(ix, iy, 1, 1).data;
        rgbTxt = `RGB(${px[0]},${px[1]},${px[2]})`;
      }
      document.getElementById("hudPixelInfo").textContent =
        `(${ix},${iy}) ${rgbTxt} | S:${state.scr[idx]} D:${state.dst[idx]} N:${state.neg[idx]}`;
    }

    if (state.isDrawing) {
      if (state.tool === "brush" && state.lastPt) {
        applyLiveBrushSegment(state.lastPt[0], state.lastPt[1], ix, iy);
        state.lastPt = [ix, iy];
      } else {
        state.shapeEnd = [ix, iy];
        render();
      }
    } else {
      render();
    }
  });

  window.addEventListener("mouseup", () => {
    if (state.isPanning) { state.isPanning = false; state.panStart = null; }
    if (!state.isDrawing) return;
    state.isDrawing = false;
    if (state.tool === "brush") {
      finishLiveStroke();
    } else if (state.shapeStart && state.shapeEnd) {
      const coords = [];
      const [x0, y0] = state.shapeStart, [x1, y1] = state.shapeEnd;
      if (state.tool === "line") {
        const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
        for (let s = 0; s <= steps; s++) {
          const px = Math.round(x0 + ((x1 - x0) * s) / steps);
          const py = Math.round(y0 + ((y1 - y0) * s) / steps);
          for (const p of collectDiskPoints(px, py, state.brushRadius)) coords.push(p);
        }
      } else if (state.tool === "rect") {
        const ax0 = Math.max(0, Math.min(x0, x1)), ax1 = Math.min(state.width - 1, Math.max(x0, x1));
        const ay0 = Math.max(0, Math.min(y0, y1)), ay1 = Math.min(state.height - 1, Math.max(y0, y1));
        for (let y = ay0; y <= ay1; y++) for (let x = ax0; x <= ax1; x++) coords.push([x, y]);
      }
      state.shapeStart = null; state.shapeEnd = null;
      applyMaskRegion(coords);
    }
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    setZoomCenter(state.zoom * factor, e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });

  document.querySelectorAll("[data-class]").forEach(btn => {
    btn.onclick = () => { state.paintClass = btn.dataset.class; updatePaletteUI(); };
  });
  document.querySelectorAll("[data-tool]").forEach(btn => {
    btn.onclick = () => { state.tool = btn.dataset.tool; updatePaletteUI(); };
  });
  document.querySelectorAll("[data-offset]").forEach(el => {
    el.onclick = () => {
      if (state.blinkTimer) { clearInterval(state.blinkTimer); state.blinkTimer = null; }
      setOffset(Number(el.dataset.offset));
    };
  });
  document.getElementById("brushSize").oninput = (e) => {
    state.brushRadius = Number(e.target.value); updatePaletteUI(); render();
  };
  document.getElementById("btnToggleOverlay").onclick = () => {
    state.overlayMode = (state.overlayMode + 1) % 3; updatePaletteUI(); render();
  };
  document.getElementById("btnToggleUnreviewed").onclick = () => {
    state.highlightUnreviewed = !state.highlightUnreviewed;
    refreshOverlayRect(0, 0, state.width - 1, state.height - 1);
    updatePaletteUI(); render();
  };
  document.getElementById("btnUndo").onclick = undo;
  document.getElementById("btnRedo").onclick = redo;
  document.getElementById("btnFit").onclick = fitScreen;
  document.getElementById("btnZoom1").onclick = () => setZoomCenter(1, canvas.clientWidth / 2, canvas.clientHeight / 2);
  document.getElementById("btnZoom4").onclick = () => setZoomCenter(4, canvas.clientWidth / 2, canvas.clientHeight / 2);
  document.getElementById("btnZoom10").onclick = () => setZoomCenter(10, canvas.clientWidth / 2, canvas.clientHeight / 2);
  document.getElementById("btnCenterT").onclick = () => {
    if (state.blinkTimer) { clearInterval(state.blinkTimer); state.blinkTimer = null; }
    setOffset(0);
  };
  document.getElementById("btnBlink").onclick = () => {
    if (state.blinkTimer) {
      clearInterval(state.blinkTimer); state.blinkTimer = null; setOffset(0);
    } else {
      const seq = [-1, 0, 1];
      let idx = 1;
      state.blinkTimer = setInterval(() => {
        idx = (idx + 1) % seq.length;
        setOffset(seq[idx]);
      }, 280);
    }
  };
  document.getElementById("btnSave").onclick = saveCurrentFrame;
  document.getElementById("btnExportReport").onclick = async () => {
    const res = await fetch("/api/export_report", { method: "POST" });
    const out = await res.json();
    if (out.ok) alert(`Status report exported to:\n${out.md_path}\n${out.json_path}`);
  };

  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault(); saveCurrentFrame(); return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault(); e.shiftKey ? redo() : undo(); return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
      e.preventDefault(); redo(); return;
    }
    const k = e.key.toLowerCase();
    if (k === "1") { state.paintClass = "scratch"; updatePaletteUI(); }
    else if (k === "2") { state.paintClass = "dust"; updatePaletteUI(); }
    else if (k === "3") { state.paintClass = "intact"; updatePaletteUI(); }
    else if (k === "4") { state.paintClass = "neg_control"; updatePaletteUI(); }
    else if (k === "5" || k === "e") { state.paintClass = "uncertain"; updatePaletteUI(); }
    else if (k === "b") { state.tool = "brush"; updatePaletteUI(); }
    else if (k === "l") { state.tool = "line"; updatePaletteUI(); }
    else if (k === "r") { state.tool = "rect"; updatePaletteUI(); }
    else if (k === "[") { state.brushRadius = Math.max(1, state.brushRadius - 1); updatePaletteUI(); render(); }
    else if (k === "]") { state.brushRadius = Math.min(32, state.brushRadius + 1); updatePaletteUI(); render(); }
    else if (k === "q") { setOffset(Math.max(-2, state.activeOffset - 1)); }
    else if (k === "w") { setOffset(Math.min(2, state.activeOffset + 1)); }
    else if (k === "g") { document.getElementById("btnCenterT").click(); }
    else if (k === "t") { document.getElementById("btnBlink").click(); }
    else if (k === "h") { document.getElementById("btnToggleOverlay").click(); }
    else if (k === "u") { document.getElementById("btnToggleUnreviewed").click(); }
  });

  new ResizeObserver(() => render()).observe(stageWrap);
  boot();
})();
</script>
</body>
</html>
"""


def make_handler(backend: TrackCAnnotatorBackend) -> type[BaseHTTPRequestHandler]:
    html_bytes = build_annotator_html().encode("utf-8")

    class AnnotatorHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, payload: Any) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            try:
                if parsed.path in ("/", "/index.html"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_bytes)))
                    self.end_headers()
                    self.wfile.write(html_bytes)
                    return
                if parsed.path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if parsed.path == "/api/status":
                    self._send_json(HTTPStatus.OK, backend.get_status())
                    return
                if len(parts) == 5 and parts[0] == "api" and parts[1] == "frame" and parts[4] == "masks":
                    clip_id, frame_idx = parts[2], int(parts[3])
                    self._send_json(HTTPStatus.OK, backend.get_frame_masks(clip_id, frame_idx))
                    return
                if len(parts) == 6 and parts[0] == "api" and parts[1] == "frame" and parts[4] == "context":
                    clip_id, frame_idx, offset = parts[2], int(parts[3]), int(parts[5])
                    png_bytes, in_bounds = backend.read_context_png(clip_id, frame_idx, offset)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(png_bytes)))
                    self.send_header("X-In-Bounds", "true" if in_bounds else "false")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(png_bytes)
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path: {parsed.path}"})
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            parts = [p for p in parsed.path.split("/") if p]
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(body.decode("utf-8")) if body else {}
                if parsed.path == "/api/export_report":
                    self._send_json(HTTPStatus.OK, backend.export_status_report())
                    return
                if len(parts) == 5 and parts[0] == "api" and parts[1] == "frame" and parts[4] == "save":
                    clip_id, frame_idx = parts[2], int(parts[3])
                    self._send_json(HTTPStatus.OK, backend.save_frame_masks(clip_id, frame_idx, payload))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path: {parsed.path}"})
            except KeyError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

    return AnnotatorHandler


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    """Windows-safe HTTP server that rejects SO_REUSEADDR port sharing/hijacking."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _is_ipv6_loopback_busy(port: int) -> bool:
    if port <= 0 or not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.settimeout(0.15)
            return s.connect_ex(("::1", port)) == 0
    except OSError:
        return False


def start_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    strict_port: bool = False,
    manifest_path: Path = MANIFEST_OUT,
    pack_dir: Path | None = None,
    report_md: Path | None = STATUS_REPORT_MD,
    report_json: Path | None = STATUS_REPORT_JSON,
    open_browser: bool = True,
) -> ThreadingHTTPServer:
    backend = TrackCAnnotatorBackend(
        manifest_path=manifest_path,
        pack_dir=pack_dir,
        report_md=report_md,
        report_json=report_json,
    )
    handler_cls = make_handler(backend)

    if port == 0:
        candidates = [0]
    elif strict_port:
        candidates = [port]
    else:
        candidates = list(dict.fromkeys([port, 8787, 8788, 8789, 8790, 8791, 8792, 0]))

    server: _ExclusiveHTTPServer | None = None
    last_err: OSError | None = None
    for cand in candidates:
        if not strict_port and host == "127.0.0.1" and cand != 0 and _is_ipv6_loopback_busy(cand):
            last_err = OSError(f"Loopback [::1]:{cand} is already occupied (likely an SSH tunnel)")
            continue
        try:
            server = _ExclusiveHTTPServer((host, cand), handler_cls)
            if cand != port and port != 0:
                print(
                    f"Notice: Port {port} is busy; automatically selected free exclusive port {server.server_address[1]}.",
                    flush=True,
                )
            break
        except OSError as exc:
            last_err = exc
            if strict_port:
                raise

    if server is None:
        raise last_err if last_err is not None else RuntimeError("Could not bind any local HTTP port")

    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"Track C Independent Archival Annotator serving on {url}", flush=True)
    print(f"Pack directory: {backend.pack_dir}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    return server


def _verify_live_loopback(url: str) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + "api/status", timeout=3.0) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Self-check returned HTTP {resp.status} for {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--strict-port", action="store_true", help="Fail instead of falling back if --port is occupied.")
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_OUT)
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=STATUS_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=STATUS_REPORT_JSON)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    srv = start_server(
        host=args.host,
        port=args.port,
        strict_port=args.strict_port,
        manifest_path=args.manifest_path,
        pack_dir=args.pack_dir,
        report_md=args.report_md,
        report_json=args.report_json,
        open_browser=False,
    )
    actual_port = srv.server_address[1]
    url = f"http://{args.host}:{actual_port}/"
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        _verify_live_loopback(url)
        print(f"Self-check OK: HTTP 200 confirmed on {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        while srv_thread.is_alive():
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nShutting down Track C annotator.")
        srv.shutdown()
        srv.server_close()
