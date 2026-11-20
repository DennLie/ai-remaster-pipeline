from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_track_c_human_gt import (  # noqa: E402
    evaluate_frame_masks_against_human_gt,
    run_track_c_evaluation,
)
from prepare_track_c_annotation_pack import (  # noqa: E402
    MANIFEST_OUT,
    generate_status_report,
    verify_human_labels,
)
from track_c_annotator import (  # noqa: E402
    TrackCAnnotatorBackend,
    build_annotator_html,
    start_server,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_temp_pilot_pack(tmp_path: Path, *, with_video: bool = False) -> tuple[Path, Path]:
    pack_dir = tmp_path / "pack"
    clip_dir = pack_dir / "clip_a"
    (clip_dir / "raw_frames").mkdir(parents=True, exist_ok=True)
    (clip_dir / "neighbor_strips").mkdir(parents=True, exist_ok=True)
    (clip_dir / "templates").mkdir(parents=True, exist_ok=True)

    w, h = 96, 64
    rng = np.random.default_rng(1234)
    raw_frames: list[np.ndarray] = []
    for t in range(5):
        img = np.full((h, w, 3), 110 + t * 4, dtype=np.uint8)
        cv2.line(img, (20, 8), (20, 56), (210, 210, 210), 2)
        noise = rng.integers(-3, 4, size=img.shape, dtype=np.int16)
        frame = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if t == 2:
            # Bright vertical scratch in target frame
            cv2.line(frame, (55, 6), (55, 58), (245, 245, 245), 1)
        raw_frames.append(frame)

    raw_p = clip_dir / "raw_frames" / "frame_000002.png"
    cv2.imwrite(str(raw_p), raw_frames[2])

    strip = np.zeros((h + 28, w * 5, 3), dtype=np.uint8)
    for slot in range(5):
        strip[28 : 28 + h, slot * w : (slot + 1) * w] = raw_frames[slot]
    strip_p = clip_dir / "neighbor_strips" / "strip_000002.png"
    cv2.imwrite(str(strip_p), strip)

    tmpl_init = np.full((h, w), 127, dtype=np.uint8)
    scr_p = clip_dir / "templates" / "frame_000002_scratch_gt.png"
    dst_p = clip_dir / "templates" / "frame_000002_dust_gt.png"
    neg_p = clip_dir / "templates" / "frame_000002_neg_control_gt.png"
    for p in (scr_p, dst_p, neg_p):
        cv2.imwrite(str(p), tmpl_init)

    video_rel = "clip_a/source.avi"
    if with_video:
        vid_p = pack_dir / video_rel
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vw = cv2.VideoWriter(str(vid_p), fourcc, 24.0, (w, h))
        assert vw.isOpened()
        for f in raw_frames:
            vw.write(f)
        vw.release()

    manifest = {
        "protocol_version": "2.0.0",
        "pilot_pack_dir": str(pack_dir),
        "pilot_keyframes": [
            {
                "clip_id": "clip_a",
                "source_video": video_rel,
                "split": "dev_diagnostic",
                "frame_index": 2,
                "width": w,
                "height": h,
                "raw_frame_path": "clip_a/raw_frames/frame_000002.png",
                "raw_sha256": _sha256_bytes(raw_p.read_bytes()),
                "neighbor_strip_path": "clip_a/neighbor_strips/strip_000002.png",
                "scratch_template_path": "clip_a/templates/frame_000002_scratch_gt.png",
                "dust_template_path": "clip_a/templates/frame_000002_dust_gt.png",
                "neg_control_template_path": "clip_a/templates/frame_000002_neg_control_gt.png",
                "notes": "Synthetic offline test fixture",
            }
        ],
    }
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pack_dir, manifest_p


def test_annotator_strict_raw_independence_imports() -> None:
    forbidden = {
        "temporal_scratch_analyzer",
        "film_defect_detectors",
        "film_defect_masks",
        "film_scratch_ridge",
        "experimental_scene_film_scratch_prototype",
        "run_experimental_candidate_comparison",
        "run_three_track_validation",
        "ai_remaster_gui",
    }
    for script_name in ("track_c_annotator.py", "generate_track_c_illustrated_guide.py"):
        tree = ast.parse((SCRIPTS_DIR / script_name).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not (imported & forbidden), f"{script_name} must not import model modules: {imported & forbidden}"


def test_annotator_inline_js_syntax(tmp_path: Path) -> None:
    html = build_annotator_html()
    m = re.search(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    assert m is not None
    js_path = tmp_path / "annotator.js"
    js_path.write_text(m.group(1), encoding="utf-8")
    node_exe = shutil.which("node")
    if node_exe is not None:
        proc = subprocess.run([node_exe, "--check", str(js_path)], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"JS syntax error: {proc.stderr}"


def test_three_tier_validation_conflicts_review_and_http_roundtrip(tmp_path: Path) -> None:
    pack_dir, manifest_p = _create_temp_pilot_pack(tmp_path)
    raw_before = (pack_dir / "clip_a" / "raw_frames" / "frame_000002.png").read_bytes()

    initial = verify_human_labels(manifest_path=manifest_p, pack_dir=pack_dir, project_root=tmp_path)
    assert initial["total_keyframes"] == 1
    assert initial["completed_keyframes"] == 0
    assert initial["pending_keyframes"] == 1
    assert initial["mean_reviewed_coverage_pct"] == 0.0
    assert initial["ready_for_real_film_claim"] is False

    srv = start_server(
        host="127.0.0.1",
        port=0,
        manifest_path=manifest_p,
        pack_dir=pack_dir,
        report_md=tmp_path / "status.md",
        report_json=tmp_path / "status.json",
        open_browser=False,
    )
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/") as resp:
            assert resp.status == 200
            assert b"RAW FRAMES ONLY" in resp.read()

        with urllib.request.urlopen(f"{base}/api/frame/clip_a/2/context/-1") as resp:
            assert resp.status == 200
            assert resp.headers.get("X-In-Bounds") == "true"
            ctx_arr = cv2.imdecode(np.frombuffer(resp.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            assert ctx_arr is not None and ctx_arr.shape == (64, 96, 3)

        with urllib.request.urlopen(f"{base}/api/frame/clip_a/2/masks") as resp:
            mask_payload = json.loads(resp.read().decode("utf-8"))
        h, w = mask_payload["height"], mask_payload["width"]

        # Reject invalid pixel value (e.g., 42)
        bad_scr = np.full((h, w), 127, dtype=np.uint8)
        bad_scr[5, 5] = 42
        req_bad = urllib.request.Request(
            f"{base}/api/frame/clip_a/2/save",
            data=json.dumps(
                {
                    "scratch_b64": base64.b64encode(bad_scr.tobytes()).decode("ascii"),
                    "dust_b64": mask_payload["dust_b64"],
                    "neg_control_b64": mask_payload["neg_control_b64"],
                    "review": {"annotator": "DL"},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad)
        assert exc_info.value.code == 400

        # Deliberately save a scratch+dust conflict and verify detection blocks readiness
        scr = np.full((h, w), 127, dtype=np.uint8)
        dst = np.full((h, w), 127, dtype=np.uint8)
        neg = np.full((h, w), 127, dtype=np.uint8)
        scr[10:40, 55] = 255
        dst[10:15, 55] = 255  # 5 overlap pixels
        scr[5:55, 18:23] = 0
        dst[5:55, 18:23] = 0
        neg[5:55, 18:23] = 255

        req_conf = urllib.request.Request(
            f"{base}/api/frame/clip_a/2/save",
            data=json.dumps(
                {
                    "scratch_b64": base64.b64encode(scr.tobytes()).decode("ascii"),
                    "dust_b64": base64.b64encode(dst.tobytes()).decode("ascii"),
                    "neg_control_b64": base64.b64encode(neg.tobytes()).decode("ascii"),
                    "review": {"annotator": "DL", "review_status": "first_pass_complete"},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_conf) as resp:
            conf_out = json.loads(resp.read().decode("utf-8"))
        assert conf_out["frame_audit"]["scratch_dust_overlap_px"] == 5
        assert conf_out["summary"]["ready_for_real_film_claim"] is False

        # Resolve conflict, add clean verified dust + intact, mark second_pass_verified
        dst[10:15, 55] = 0
        dst[20:23, 70:73] = 255
        scr[20:23, 70:73] = 0
        req_ok = urllib.request.Request(
            f"{base}/api/frame/clip_a/2/save",
            data=json.dumps(
                {
                    "scratch_b64": base64.b64encode(scr.tobytes()).decode("ascii"),
                    "dust_b64": base64.b64encode(dst.tobytes()).decode("ascii"),
                    "neg_control_b64": base64.b64encode(neg.tobytes()).decode("ascii"),
                    "review": {
                        "annotator": "DL",
                        "second_pass_reviewer": "REV2",
                        "review_status": "second_pass_verified",
                        "notes": "Confirmed vertical scratch and dust speck; border left 127",
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok) as resp:
            ok_out = json.loads(resp.read().decode("utf-8"))
        assert ok_out["ok"] is True
        assert ok_out["frame_audit"]["hard_conflict_px"] == 0
        assert ok_out["frame_audit"]["second_pass_verified"] is True
        assert ok_out["summary"]["ready_for_real_film_claim"] is True
        assert ok_out["summary"]["ready_with_second_pass"] is True

        with urllib.request.urlopen(urllib.request.Request(f"{base}/api/export_report", method="POST", data=b"{}")) as resp:
            rep_out = json.loads(resp.read().decode("utf-8"))
            assert rep_out["ok"] is True
            assert (tmp_path / "status.md").exists()
            assert (tmp_path / "status.json").exists()
    finally:
        srv.shutdown()
        srv.server_close()

    # Verify raw frame was never modified & masks resumed cleanly on fresh backend
    assert (pack_dir / "clip_a" / "raw_frames" / "frame_000002.png").read_bytes() == raw_before
    resumed = TrackCAnnotatorBackend(manifest_path=manifest_p, pack_dir=pack_dir, project_root=tmp_path).get_frame_masks("clip_a", 2)
    assert resumed["review"]["second_pass_reviewer"] == "REV2"
    assert resumed["review"]["review_status"] == "second_pass_verified"


def test_evaluator_excludes_127_and_blocks_empty_pack(tmp_path: Path) -> None:
    pack_dir, manifest_p = _create_temp_pilot_pack(tmp_path, with_video=True)
    with pytest.raises(RuntimeError, match="0 completed human annotations"):
        run_track_c_evaluation(manifest_path=manifest_p, pack_dir=pack_dir, project_root=tmp_path)

    # Synthetic 3-tier metric exclusion verification
    scr_gt = np.full((30, 30), 127, dtype=np.uint8)
    dst_gt = np.full((30, 30), 127, dtype=np.uint8)
    neg_gt = np.full((30, 30), 127, dtype=np.uint8)

    # Verified scratch (5 px), verified intact (25 px), neg ROI (10 px subset of intact), rest remains 127!
    scr_gt[2:7, 10] = 255
    dst_gt[2:7, 10] = 0
    scr_gt[10:15, 20:25] = 0
    dst_gt[10:15, 20:25] = 0
    neg_gt[10:12, 20:25] = 255

    # Put a giant false prediction ONLY on unreviewed 127 region (row 20..29) + 1 px leak into neg_gt + 4 TP on scratch
    det = np.zeros((30, 30), dtype=np.uint8)
    rep = np.zeros((30, 30), dtype=np.uint8)
    unc = np.zeros((30, 30), dtype=np.uint8)
    det[20:29, 0:10] = 255
    rep[20:29, 0:10] = 255  # Must be excluded because GT is 127
    det[2:6, 10] = 255
    rep[2:6, 10] = 255      # 4 TP
    unc[6, 10] = 255        # 1 scratch px routed to Uncertain
    det[10, 21] = 255
    rep[10, 21] = 255       # 1 FP inside neg_gt ROI

    m = evaluate_frame_masks_against_human_gt(
        scr_gt=scr_gt,
        dst_gt=dst_gt,
        neg_gt=neg_gt,
        det_mask=det,
        repair_mask=rep,
        uncertain_mask=unc,
        actual_modified_mask=(rep > 0),
    )
    assert m["reviewed_px"] == 30
    assert m["repair_tp_scratch_px"] == 4
    assert m["repair_fp_intact_px"] == 1
    assert m["repair_roi_leak_px"] == 1
    assert m["scratch_routed_to_uncertain_px"] == 1
    assert m["severe_localized_failure"] is True

    # Also test end-to-end evaluator execution on temp video pack
    scr_p = pack_dir / "clip_a" / "templates" / "frame_000002_scratch_gt.png"
    dst_p = pack_dir / "clip_a" / "templates" / "frame_000002_dust_gt.png"
    neg_p = pack_dir / "clip_a" / "templates" / "frame_000002_neg_control_gt.png"
    s_arr = np.full((64, 96), 0, dtype=np.uint8)
    d_arr = np.full((64, 96), 0, dtype=np.uint8)
    n_arr = np.full((64, 96), 127, dtype=np.uint8)
    s_arr[6:58, 55] = 255
    n_arr[8:56, 19:22] = 255
    cv2.imwrite(str(scr_p), s_arr)
    cv2.imwrite(str(dst_p), d_arr)
    cv2.imwrite(str(neg_p), n_arr)

    eval_out = run_track_c_evaluation(
        manifest_path=manifest_p,
        pack_dir=pack_dir,
        project_root=pack_dir,
        pre_safety_report_path=None,
        md_out=tmp_path / "eval.md",
        json_out=tmp_path / "eval.json",
    )
    assert eval_out["evaluated_keyframes"] == 1
    assert eval_out["annotation_classification"] == "dev_diagnostic_pilot_only"
    assert "Corrected_E2_fused_3state" in eval_out["variants"]
    assert eval_out["variants"]["Previous_E2_pre_safety_fix"]["status"] == "not_archived_per_pixel"
    assert (tmp_path / "eval.md").exists()


def test_real_track_c_pilot_pack_templates_valid() -> None:
    if not MANIFEST_OUT.exists():
        pytest.skip("Track C manifest not present")
    report = verify_human_labels(manifest_path=MANIFEST_OUT)
    assert report["total_keyframes"] == 14
    assert report["invalid_files"] == []
    assert report["raw_integrity_failures"] == []


def test_exclusive_port_fallback_avoids_windows_collision(tmp_path: Path) -> None:
    import socket

    pack_dir, manifest_p = _create_temp_pilot_pack(tmp_path)
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    busy_port = blocker.getsockname()[1]
    try:
        with pytest.raises(OSError):
            start_server(
                host="127.0.0.1",
                port=busy_port,
                strict_port=True,
                manifest_path=manifest_p,
                pack_dir=pack_dir,
                open_browser=False,
            )

        srv = start_server(
            host="127.0.0.1",
            port=busy_port,
            strict_port=False,
            manifest_path=manifest_p,
            pack_dir=pack_dir,
            open_browser=False,
        )
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            assert srv.server_address[1] != busy_port
            with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/status") as resp:
                assert resp.status == 200
        finally:
            srv.shutdown()
            srv.server_close()
    finally:
        blocker.close()

