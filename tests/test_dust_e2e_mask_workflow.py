"""Isolated end-to-end tests for editable dust masks, project persistence, repair backends, and real film frames."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from unittest import mock

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
for p in (str(ROOT_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ai_remaster_gui import (
    app as app_mod,
    cache as cache_mod,
    config as config_mod,
    dust as dust_stage,
    dust_mask_edits,
    frame_inspector,
    http_handler,
    media as media_mod,
    paths as paths_mod,
    project_io,
    runtime_settings,
    server,
    stage_stamps,
    state,
)
from ai_remaster_gui.dust_mask_edits import (
    BIT_DUST_ADD,
    BIT_DUST_REJECT,
    BIT_PROTECT_INTACT,
    BIT_REVIEWED,
    BIT_SCRATCH_ADD,
    BIT_SCRATCH_REJECT,
    BIT_UNCERTAIN_REJECT,
    GROUND_TRUTH_STATUS,
    apply_stroke_transition,
    decode_bitmask_b64,
    encode_bitmask_b64,
)
from ai_remaster_gui.runtime_settings import base_settings
from scripts.dust_native_composite import compose as native_compose, plane_masks, plane_shapes


class _Handler(http_handler.Handler):
    def __init__(self, path: str, body: dict | None = None) -> None:
        self.path = path
        self.headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        self._body = body or {}
        self.responses: dict[str, object] = {}

    def read_json(self) -> dict:
        return self._body

    def send_json(self, payload) -> None:  # noqa: D401
        self.responses["json"] = payload

    def send_error(self, code, message=None, explain=None) -> None:  # noqa: D401
        self.responses["error"] = code


@contextmanager
def _isolated_workspace(root: Path, settings: dict | None = None):
    settings_path = root / ".ai_remaster_gui.json"
    app = server.PipelineApp.__new__(server.PipelineApp)
    app.settings = settings if settings is not None else base_settings()
    app.project_path = None
    app.log = []
    app.process = None
    app.quitting = False
    app.queue_running = False
    app.running_stage = ""
    app.running_stage_key = ""
    app.running_reference_manifest = ""
    app.running_reference_index = None
    app.denoise_grid_request = {}
    app.run_started_at = 0.0
    app.recomp_total_frames = 0
    app.recomp_progress_run = 0.0
    app.lock = threading.Lock()
    app.source_analysis_lock = threading.Lock()
    app.source_analysis_status = {}
    app.source_analysis_results = {}
    app.source_analysis_threads = set()
    app.source_tone_default_keys = set()

    patches = (
        mock.patch.object(config_mod, "ROOT", root),
        mock.patch.object(config_mod, "SETTINGS_FILE", settings_path),
        mock.patch.object(paths_mod, "ROOT", root),
        mock.patch.object(cache_mod, "ROOT", root),
        mock.patch.object(media_mod, "ROOT", root),
        mock.patch.object(media_mod, "PREVIEW_DIR", root / ".cache" / "previews"),
        mock.patch.object(media_mod, "FILE_PREVIEW_DIR", root / ".cache" / "file_previews"),
        mock.patch.object(media_mod, "ASPECT_PREVIEW_DIR", root / ".cache" / "aspect_previews"),
        mock.patch.object(media_mod, "MEDIA_CLIP_DIR", root / ".cache" / "media_clips"),
        mock.patch.object(media_mod, "SOURCE_PLAYBACK_DIR", root / ".cache" / "source_playback"),
        mock.patch.object(dust_mask_edits, "ROOT", root),
        mock.patch.object(dust_stage, "ROOT", root),
        mock.patch.object(project_io, "ROOT", root),
        mock.patch.object(frame_inspector, "ROOT", root),
        mock.patch.object(frame_inspector, "CACHE", root / ".cache" / "frame_inspector"),
        mock.patch.object(stage_stamps, "STAMP_DIR", root / ".cache" / "stage_stamps"),
        mock.patch.object(runtime_settings, "SETTINGS_FILE", settings_path),
        mock.patch.object(app_mod, "SETTINGS_FILE", settings_path),
        mock.patch.object(server, "ROOT", root),
        mock.patch.object(server, "SETTINGS_FILE", settings_path),
        mock.patch.object(state, "APP", app),
    )
    with ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        yield app


def _snapshot_folder(folder: Path) -> dict[str, tuple[str, int]]:
    return {
        p.name: (dust_stage.digest(p), p.stat().st_mtime_ns)
        for p in sorted(folder.iterdir())
        if p.is_file()
    }


def _make_synthetic_damaged_clip(path: Path, *, width: int = 128, height: int = 128, num_frames: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    canvas = rng.normal(115.0, 14.0, size=(height + 24, width + 64)).astype(np.float32)
    canvas = cv2.GaussianBlur(canvas, (7, 7), 1.5)
    # Intact panning scene structure bar
    cv2.rectangle(canvas, (50, 12), (72, height + 10), 65.0, 2)
    canvas_u8 = np.clip(canvas, 20, 220).astype(np.uint8)

    writer = dust_stage.Writer(path, {"frames": num_frames, "width": width, "height": height, "fps": "24"})
    for idx in range(num_frames):
        crop = canvas_u8[10 : 10 + height, 4 + idx * 6 : 4 + idx * 6 + width]
        frame = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if idx == 2:
            # Strong transient vertical scratch (auto-detected by farneback_3f)
            frame[16:56, 28:31] = 248
            # Strong transient dust blotch (auto-detected as false positive to reject)
            frame[88:94, 92:98] = 246
            # Subtler vertical line (to be manually added via BIT_SCRATCH_ADD)
            frame[18:48, 60:62] = np.clip(frame[18:48, 60:62].astype(np.int16) + 60, 0, 255).astype(np.uint8)
            # Subtler dark dust spot (to be manually added via BIT_DUST_ADD)
            frame[96:102, 24:30] = np.clip(frame[96:102, 24:30].astype(np.int16) - 65, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.close()


def test_full_synthetic_editable_mask_e2e_workflow() -> None:
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        ws1 = Path(td1)
        ws2 = Path(td2)
        source_rel = "input/clip.mkv"
        source1 = ws1 / source_rel
        _make_synthetic_damaged_clip(source1)
        source_sha_initial = dust_stage.digest(source1)

        settings1 = base_settings()
        settings1["global"].update(source=source_rel, dust="true", expand_outpaint="false", colorize="false")
        settings1["dust"].update(detector="farneback_3f", preset="balanced", dilate="1", repair="telea")

        with _isolated_workspace(ws1, settings1) as app1:
            # 1. Run production detection (farneback_3f / balanced) and snapshot immutable <base_key>/
            base_vals = {"detector": "farneback_3f", "preset": "balanced", "repair": "telea"}
            base_report = dust_stage.run(source1, base_vals, "mask")
            assert base_report["metadata"]["frames"] == 5
            base_folder = dust_stage.folder_for(source1, base_vals)
            base_snapshot = _snapshot_folder(base_folder)

            base_eff_frames = [m[:, :, 0] > 127 for m in dust_stage.frames(base_folder / "effective_mask.mkv")]
            assert np.any(base_eff_frames[2]), "farneback_3f must detect synthetic defects on frame 2"

            # 2. Edit via HTTP handlers / PipelineApp on frame 2
            get_h = _Handler("/api/dust-mask-frame?frame=2&analysis_source=stage")
            get_h.do_GET()
            assert get_h.responses["json"]["ok"] is True
            assert get_h.responses["json"]["edit_revision"] == 0

            edit_f2 = np.zeros((128, 128), dtype=np.uint8)
            sel_scratch_add = np.zeros((128, 128), dtype=bool)
            sel_scratch_add[18:48, 60:62] = True
            sel_dust_add = np.zeros((128, 128), dtype=bool)
            sel_dust_add[96:102, 24:30] = True
            # Reject auto-detected blotch region (covering dilation radius around 88:94, 92:98)
            sel_fp_reject = np.zeros((128, 128), dtype=bool)
            sel_fp_reject[84:98, 88:102] = True
            # PROTECT_INTACT overlapping BOTH auto proposal (16:26, 26:34) and added scratch (18:26, 58:64)
            sel_protect = np.zeros((128, 128), dtype=bool)
            sel_protect[16:26, 26:34] = True
            sel_protect[18:26, 58:64] = True

            edit_f2 = apply_stroke_transition(edit_f2, sel_scratch_add, "scratch_add")
            edit_f2 = apply_stroke_transition(edit_f2, sel_dust_add, "dust_add")
            edit_f2 = apply_stroke_transition(edit_f2, sel_fp_reject, "reject_all")
            edit_f2 = apply_stroke_transition(edit_f2, sel_protect, "protect")
            edit_f2 = apply_stroke_transition(edit_f2, np.ones((128, 128), dtype=bool), "mark_reviewed")

            save_h = _Handler(
                "/api/dust-mask-frame",
                {
                    "frame": 2,
                    "edit_mask_b64": encode_bitmask_b64(edit_f2),
                    "analysis_source": "stage",
                    "expected_revision": 0,
                },
            )
            save_h.do_POST()
            save_res = save_h.responses["json"]
            assert save_res["ok"] is True
            assert save_res["edit_revision"] == 1
            assert save_res["reviewed"] is True
            assert save_res["edited_frames"] == [2]
            sig1 = save_res["edit_signature"]
            tid1 = save_res["timeline_id"]
            assert sig1.endswith(":rev1")

            # 3. Export .arpp bundle via save_project, close workspace, open fresh temp workspace via load_project
            arpp_path = ws1 / "projects" / "e2e_clip.arpp"
            with mock.patch.object(server, "browse_path", return_value=str(arpp_path)):
                saved_info = app1.save_project(save_as=True)
            assert Path(saved_info["path"]) == arpp_path
            assert arpp_path.is_file()

        # Open completely fresh workspace ws2 and load .arpp project
        source2 = ws2 / source_rel
        source2.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source1, source2)

        with _isolated_workspace(ws2) as app2:
            with mock.patch.object(server, "browse_path", return_value=str(arpp_path)):
                loaded_info = app2.load_project()
            assert Path(loaded_info["path"]) == arpp_path

            edit_state2 = app2.dust_edit_state()["stage"]
            assert edit_state2["compatible"] is True
            assert edit_state2["edit_revision"] == 1
            assert edit_state2["edit_signature"] == sig1
            assert edit_state2["timeline_id"] == tid1
            assert edit_state2["edited_frames"] == [2]

            reloaded_f2 = app2.load_dust_mask_frame(2, analysis_source="stage")
            assert reloaded_f2["reviewed"] is True
            np.testing.assert_array_equal(decode_bitmask_b64(reloaded_f2["edit_mask_b64"], 128, 128), edit_f2)

        # Resume in ws1 to verify immutable base caches & run detector switch + all 4 repair paths
        with _isolated_workspace(ws1, settings1) as app1:
            # 4. Switch detector (farneback_3f -> opt-in experimental_e2) and rerun detection
            with pytest.raises(ValueError, match="experimental"):
                dust_stage.run(source1, {"detector": "experimental_e2", "research_confirmed": "false"}, "mask")

            e2_vals = {
                "detector": "experimental_e2",
                "preset": "experimental_e2_research",
                "research_confirmed": "true",
                "repair": "telea",
            }
            e2_base_report = dust_stage.run(source1, e2_vals, "mask")
            assert "e2_metadata" in e2_base_report
            e2_base_folder = dust_stage.folder_for(source1, e2_vals)
            e2_base_snapshot = _snapshot_folder(e2_base_folder)
            assert (e2_base_folder / "vetoed_mask.mkv").is_file()

            # Compose E2 with active edits
            e2_comp_vals = {**e2_vals, "edit_signature": sig1, "timeline_id": tid1}
            e2_comp_report = dust_stage.run(source1, e2_comp_vals, "mask")
            e2_comp_folder = dust_stage.folder_for(source1, e2_comp_vals)
            assert e2_comp_report["edit_signature"] == sig1
            e2_eff_f2 = list(dust_stage.frames(e2_comp_folder / "effective_mask.mkv"))[2][:, :, 0] > 127
            assert np.all(e2_eff_f2[28:48, 60:62]), "Human scratch_add survives E2 detector rerun"
            assert np.all(e2_eff_f2[96:102, 24:30]), "Human dust_add survives E2 detector rerun"
            assert not np.any(e2_eff_f2[sel_protect]), "PROTECT_INTACT survives E2 detector rerun"

            # Verify immutable base caches stayed bit-for-bit untouched
            assert _snapshot_folder(base_folder) == base_snapshot
            assert _snapshot_folder(e2_base_folder) == e2_base_snapshot

            # Add another edit on frame 1 -> new edit_signature updates composed folder hash
            edit_f1 = np.zeros((128, 128), dtype=np.uint8)
            edit_f1[40:44, 40:44] = BIT_DUST_ADD
            save_rev2 = app1.save_dust_mask_frame(
                1, encode_bitmask_b64(edit_f1), analysis_source="stage", expected_revision=1
            )
            sig2 = save_rev2["edit_signature"]
            assert save_rev2["edit_revision"] == 2
            assert sig2 != sig1

            comp_folder_rev1 = dust_stage.folder_for(source1, {**base_vals, "edit_signature": sig1, "timeline_id": tid1})
            comp_folder_rev2 = dust_stage.folder_for(source1, {**base_vals, "edit_signature": sig2, "timeline_id": tid1})
            assert comp_folder_rev2 != comp_folder_rev1

            # 5A. Repair with Telea (real execution)
            telea_vals = {**base_vals, "edit_signature": sig2, "timeline_id": tid1}
            telea_report = dust_stage.run(source1, telea_vals, "repair")
            assert telea_report["repair_complete"] is True
            telea_folder = dust_stage.folder_for(source1, telea_vals)

            telea_inputs = list(dust_stage.frames(telea_folder / "input.mkv"))
            telea_masks = [m[:, :, 0] > 127 for m in dust_stage.frames(telea_folder / "effective_mask.mkv")]
            telea_outputs = list(dust_stage.frames(telea_folder / "repaired.mkv"))
            for idx, (inp, msk, out) in enumerate(zip(telea_inputs, telea_masks, telea_outputs, strict=True)):
                changed = np.any(inp != out, axis=2)
                assert int(np.count_nonzero(changed & ~msk)) == 0
                if idx == 2:
                    assert int(np.count_nonzero(changed & sel_protect)) == 0
                    assert int(np.count_nonzero(changed & sel_fp_reject)) == 0
                    assert int(np.count_nonzero(changed[28:48, 60:62])) > 0
                    assert int(np.count_nonzero(changed[96:102, 24:30])) > 0

            # 5B. Repair with ProPainter (deterministic test double verifying composed mask folder is supplied)
            pp_vals = {
                **base_vals,
                "repair": "propainter",
                "license_confirmed": "true",
                "edit_signature": sig2,
                "timeline_id": tid1,
            }
            pp_expected_folder = dust_stage.folder_for(source1, pp_vals)
            seen_pp_target: list[Path] = []

            def _fake_propainter(target, tmp, meta, url, output, report, cancel_file=None):
                seen_pp_target.append(Path(target))
                assert (Path(target) / "effective_mask.mkv").is_file()
                for _ in range(meta["frames"]):
                    yield np.full((meta["height"], meta["width"], 3), 170, dtype=np.uint8)

            with (
                mock.patch.object(dust_stage, "propainter_frames", side_effect=_fake_propainter),
                mock.patch(
                    "dust_model_preflight.inspect_propainter",
                    return_value={"ready": True, "backend": "propainter", "test_double": True},
                ),
            ):
                pp_report = dust_stage.run(source1, pp_vals, "repair")
            assert pp_report["repair_complete"] is True
            assert seen_pp_target == [pp_expected_folder]
            pp_inputs = list(dust_stage.frames(pp_expected_folder / "input.mkv"))
            pp_masks = [m[:, :, 0] > 127 for m in dust_stage.frames(pp_expected_folder / "effective_mask.mkv")]
            pp_outputs = list(dust_stage.frames(pp_expected_folder / "repaired.mkv"))
            for idx, (inp, msk, out) in enumerate(zip(pp_inputs, pp_masks, pp_outputs, strict=True)):
                changed = np.any(inp != out, axis=2)
                assert int(np.count_nonzero(changed & ~msk)) == 0
                if idx == 2:
                    assert int(np.count_nonzero(changed & sel_protect)) == 0
                    assert int(np.count_nonzero(changed[28:48, 60:62])) > 0

            # 5C. Repair with DART (deterministic test double verifying composed mask enforcement)
            dart_vals = {
                "detector": "dart",
                "repair": "dart",
                "preset": "dart_optimal_f1",
                "research_confirmed": "true",
                "license_confirmed": "true",
                "edit_signature": sig2,
                "timeline_id": tid1,
            }
            dart_comp_folder = dust_stage.folder_for(source1, dart_vals)
            real_subproc_run = subprocess.run

            def _fake_dart_subproc(cmd, *args, **kwargs):
                if isinstance(cmd, (list, tuple)) and any("inference.py" in str(part) for part in cmd):
                    out_dir = Path(cmd[cmd.index("--output") + 1])
                    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
                    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
                    for f_idx in range(5):
                        restored_img = np.full((128, 128, 3), 185, dtype=np.uint8)
                        auto_mask_img = np.zeros((128, 128), dtype=np.uint8)
                        if f_idx == 2:
                            auto_mask_img[16:50, 28:31] = 255
                        cv2.imwrite(str(out_dir / "frames" / f"frame_{f_idx:06d}.png"), restored_img)
                        cv2.imwrite(str(out_dir / "masks" / f"frame_{f_idx:06d}.png"), auto_mask_img)
                    return subprocess.CompletedProcess(cmd, 0)
                return real_subproc_run(cmd, *args, **kwargs)

            with (
                mock.patch("ai_remaster_gui.dart_runtime.require_installation", return_value={"ready": True}),
                mock.patch(
                    "dust_model_preflight.inspect_dart",
                    return_value={"ready": True, "backend": "dart", "test_double": True},
                ),
                mock.patch.object(dust_stage.subprocess, "run", side_effect=_fake_dart_subproc),
            ):
                dart_mask_report = dust_stage.run(source1, dart_vals, "mask")
                assert dart_mask_report["edit_signature"] == sig2
                dart_repair_report = dust_stage.run(source1, dart_vals, "repair")
            assert dart_repair_report["repair_complete"] is True
            dart_inputs = list(dust_stage.frames(dart_comp_folder / "input.mkv"))
            dart_masks = [m[:, :, 0] > 127 for m in dust_stage.frames(dart_comp_folder / "effective_mask.mkv")]
            dart_outputs = list(dust_stage.frames(dart_comp_folder / "repaired.mkv"))
            for idx, (inp, msk, out) in enumerate(zip(dart_inputs, dart_masks, dart_outputs, strict=True)):
                changed = np.any(inp != out, axis=2)
                assert int(np.count_nonzero(changed & ~msk)) == 0
                if idx == 2:
                    assert int(np.count_nonzero(changed & sel_protect)) == 0
                    assert int(np.count_nonzero(changed[28:48, 60:62])) > 0

            # 5D. Direct 10-bit dust_native_composite.compose() with composed effective_mask.mkv
            native_dir = ws1 / "native_10bit"
            native_dir.mkdir(parents=True, exist_ok=True)
            native_meta = {"width": 128, "height": 128, "frames": 5, "fps": "24", "pixel_format": "yuv420p10le"}
            shapes = plane_shapes(native_meta)
            planes = [
                (np.indices((h, w), dtype=np.uint16).sum(axis=0) * 7 % 700 + 120).astype("<u2")
                for h, w in shapes
            ]
            raw_10bit = b"".join(p.tobytes() for p in planes)
            native_src = native_dir / "source_10bit.mkv"
            subprocess.run(
                [
                    dust_stage.find_ffmpeg(),
                    "-v", "error",
                    "-f", "rawvideo", "-pixel_format", "yuv420p10le",
                    "-s", "128x128", "-r", "24", "-i", "pipe:0",
                    "-c:v", "ffv1", str(native_src),
                ],
                input=raw_10bit * 5,
                check=True,
            )
            patch_mkv = native_dir / "patch.mkv"
            patch_writer = dust_stage.Writer(patch_mkv, {"width": 128, "height": 128, "frames": 5, "fps": "24"})
            for _ in range(5):
                patch_writer.write(np.full((128, 128, 3), 235, dtype=np.uint8))
            patch_writer.close()

            native_out = native_dir / "composed_10bit.mkv"
            native_counts = native_compose(
                dust_stage.find_ffmpeg(),
                native_src,
                patch_mkv,
                telea_folder / "effective_mask.mkv",
                native_out,
                native_meta,
            )
            assert all(int(v) == 0 for row in native_counts for k, v in row.items() if k.endswith("outside_samples"))
            decoded_raw = subprocess.check_output(
                [dust_stage.find_ffmpeg(), "-v", "error", "-i", str(native_out), "-pix_fmt", "yuv420p10le", "-f", "rawvideo", "pipe:1"]
            )
            frame_bytes = len(raw_10bit)
            f2_bytes = decoded_raw[2 * frame_bytes : 3 * frame_bytes]
            f2_mask = telea_masks[2]
            y_out = np.frombuffer(f2_bytes[: 128 * 128 * 2], "<u2").reshape((128, 128))
            np.testing.assert_array_equal(y_out[~f2_mask], planes[0][~f2_mask])
            np.testing.assert_array_equal(y_out[sel_protect], planes[0][sel_protect])
            assert np.any(y_out[28:48, 60:62] != planes[0][28:48, 60:62])

            # Confirm source video and base detector caches stayed untouched
            assert dust_stage.digest(source1) == source_sha_initial
            assert _snapshot_folder(base_folder) == base_snapshot
            assert _snapshot_folder(e2_base_folder) == e2_base_snapshot

            # 6. Optional provenance export via HTTP handler
            exp_h = _Handler(
                "/api/dust-mask-export",
                {"analysis_source": "stage", "values": base_vals, "include_auto_proposals": True},
            )
            exp_h.do_POST()
            exp_res = exp_h.responses["json"]
            assert exp_res["ok"] is True
            prov_json = json.loads(Path(exp_res["provenance_path"]).read_text(encoding="utf-8"))
            assert prov_json["ground_truth_status"] == GROUND_TRUTH_STATUS
            assert prov_json["ground_truth_status"] == "operational_edits_only_not_independent_gt"
            assert prov_json["protect_or_reject_auto_promoted_to_gt"] is False
            assert prov_json["untouched_auto_as_gt"] is False
            exp_dir = Path(exp_res["export_dir"])
            assert (exp_dir / "human_edits" / "f000002.png").is_file()
            assert (exp_dir / "protect_intact" / "f000002.png").is_file()
            assert (exp_dir / "effective_repair" / "f000002.png").is_file()
            assert (exp_dir / "auto_proposals" / "scratch" / "f000002.png").is_file()


def test_real_neighbors_film_frame_e2e_workflow() -> None:
    neighbors_path = ROOT_DIR / "evaluation" / "Neighbors_72.mkv"
    assert neighbors_path.is_file(), f"Required evaluation fixture missing: {neighbors_path}"
    neighbors_sha_before = dust_stage.digest(neighbors_path)
    neighbors_mtime_before = neighbors_path.stat().st_mtime_ns

    # Extract 3 cropped frames (128x96) from Neighbors_72.mkv into temp source.mkv
    cap = cv2.VideoCapture(str(neighbors_path))
    assert cap.isOpened()
    raw_frames: list[np.ndarray] = []
    try:
        for idx in range(5):
            ok, img = cap.read()
            assert ok and img is not None
            if idx >= 2:
                raw_frames.append(img[150:246, 400:528].copy())
    finally:
        cap.release()
    assert len(raw_frames) == 3

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        source_rel = "input/source.mkv"
        source = ws / source_rel
        source.parent.mkdir(parents=True, exist_ok=True)
        writer = dust_stage.Writer(source, {"frames": 3, "width": 128, "height": 96, "fps": "30"})
        for fr in raw_frames:
            writer.write(fr)
        writer.close()
        source_sha_before = dust_stage.digest(source)

        settings = base_settings()
        settings["global"].update(source=source_rel, dust="true", expand_outpaint="false", colorize="false")
        settings["dust"].update(detector="farneback_3f", preset="balanced", repair="telea")

        with _isolated_workspace(ws, settings) as app:
            # 1. Production farneback_3f detection on real cropped frames
            fb_vals = {"detector": "farneback_3f", "preset": "balanced", "repair": "telea"}
            dust_stage.run(source, fb_vals, "mask")
            fb_base_folder = dust_stage.folder_for(source, fb_vals)
            fb_base_snapshot = _snapshot_folder(fb_base_folder)

            fb_eff_f1 = list(dust_stage.frames(fb_base_folder / "effective_mask.mkv"))[1][:, :, 0] > 127
            auto_ys, auto_xs = np.where(fb_eff_f1)
            assert len(auto_ys) > 0, "Expected real auto proposals on Neighbors_72 middle frame"

            # 2. Apply scratch_add + false-positive reject on auto proposal + PROTECT_INTACT
            edit_f1 = np.zeros((96, 128), dtype=np.uint8)
            sel_fp_reject = np.zeros((96, 128), dtype=bool)
            cy, cx = int(auto_ys[0]), int(auto_xs[0])
            sel_fp_reject[max(0, cy - 1) : min(96, cy + 2), max(0, cx - 1) : min(128, cx + 2)] = True

            # Choose a scratch_add stripe over textured real film pixels away from the protected region
            sel_scratch_add = np.zeros((96, 128), dtype=bool)
            sel_scratch_add[10:34, 90:92] = True

            sel_protect = np.zeros((96, 128), dtype=bool)
            sel_protect[40:56, 20:36] = True

            edit_f1 = apply_stroke_transition(edit_f1, sel_fp_reject, "reject_all")
            edit_f1 = apply_stroke_transition(edit_f1, sel_scratch_add, "scratch_add")
            edit_f1 = apply_stroke_transition(edit_f1, sel_protect, "protect")
            edit_f1 = apply_stroke_transition(edit_f1, np.ones((96, 128), dtype=bool), "mark_reviewed")

            b64_saved = encode_bitmask_b64(edit_f1)
            save_res = app.save_dust_mask_frame(1, b64_saved, analysis_source="stage", expected_revision=0)
            sig = save_res["edit_signature"]
            tid = save_res["timeline_id"]
            assert save_res["edit_revision"] == 1

            reloaded = app.load_dust_mask_frame(1, analysis_source="stage")
            np.testing.assert_array_equal(decode_bitmask_b64(reloaded["edit_mask_b64"], 128, 96), edit_f1)

            # 3. Run Telea composed repair
            comp_vals = {**fb_vals, "edit_signature": sig, "timeline_id": tid}
            repair_report = dust_stage.run(source, comp_vals, "repair")
            assert repair_report["repair_complete"] is True
            comp_folder = dust_stage.folder_for(source, comp_vals)

            in_frames = list(dust_stage.frames(comp_folder / "input.mkv"))
            eff_masks = [m[:, :, 0] > 127 for m in dust_stage.frames(comp_folder / "effective_mask.mkv")]
            out_frames = list(dust_stage.frames(comp_folder / "repaired.mkv"))
            for idx, (inp, msk, out) in enumerate(zip(in_frames, eff_masks, out_frames, strict=True)):
                changed = np.any(inp != out, axis=2)
                assert int(np.count_nonzero(changed & ~msk)) == 0
                if idx == 1:
                    assert int(np.count_nonzero(changed & sel_protect)) == 0
                    assert not msk[cy, cx], "Rejected false-positive pixel must be excluded from effective mask"
                    assert int(np.count_nonzero(changed[10:34, 90:92])) > 0

            # Verify repair_measurements.csv matches zero outside & protected changes
            with (comp_folder / "repair_measurements.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            assert all(int(r["changed_outside_pixels"]) == 0 for r in rows)

            # Verify original source.mkv SHA256 and immutable <base_key>/ preserved
            assert dust_stage.digest(source) == source_sha_before
            assert _snapshot_folder(fb_base_folder) == fb_base_snapshot

            # 4. Run opt-in experimental_e2 on the real 3-frame clip and verify separate 3-state outputs
            e2_vals = {
                "detector": "experimental_e2",
                "preset": "experimental_e2_research",
                "research_confirmed": "true",
                "repair": "telea",
            }
            e2_report = dust_stage.run(source, e2_vals, "mask")
            e2_folder = dust_stage.folder_for(source, e2_vals)
            assert e2_report["e2_metadata"]["non_flattened_uncertainty"] is True
            for name in ("scratch_mask.mkv", "dust_mask.mkv", "uncertain_mask.mkv", "vetoed_mask.mkv"):
                assert (e2_folder / name).is_file()
            # Dust channel remains unsupported (all zeros)
            for d_frame in dust_stage.frames(e2_folder / "dust_mask.mkv"):
                assert int(np.count_nonzero(d_frame)) == 0
            # Verify inspect_dust surfaces vetoed_url cleanly
            with mock.patch("ai_remaster_gui.frame_inspector._decode", return_value="decoded.png"):
                insp = app.dust_inspect(e2_vals, frame=1)
            assert insp["vetoed_url"]
            assert "vetoed_mask.mkv" in insp["auto_proposal_urls"]

    assert dust_stage.digest(neighbors_path) == neighbors_sha_before
    assert neighbors_path.stat().st_mtime_ns == neighbors_mtime_before
