"""Routing/behaviour tests for the GUI request handler.

These pin down the set of routes the handler answers and the response shapes produced by the two
shared helpers (_send_result and _send_action), so the consolidation of the per-endpoint
boilerplate cannot silently drop a route or change a payload.
"""

from __future__ import annotations

import inspect
import re
import threading
import unittest
from types import MethodType
from unittest import mock

from ai_remaster_gui import http_handler, state
from ai_remaster_gui import server  # noqa: F401  (imported for its bind_context side effect)

EXPECTED_GET_ROUTES = {
    "/",
    "/site.webmanifest",
    "/api/state",
    "/api/command",
    "/api/existing-outputs", "/api/existing-shot-list",
    "/api/media-status",
    "/api/comfy",
    "/api/logfile",
    "/api/openai-models",
    "/api/shot-preview",
    "/api/shot-sheets",
    "/api/upscale-shot-comparison",
    "/api/aspect-preview",
    "/api/outpaint-auto-crop",
    "/api/outpaint-custom-mask",
    "/api/outpaint-frame-mask-frames",
    "/api/outpaint-frame-mask",
    "/api/outpaint-chunk-preview",
    "/api/outpaint-guide-preview",
    "/api/denoise-inspector",
    "/api/stabilization-footprint",
    "/api/denoise-frame-differences",
    "/api/denoise-grid-download",
    "/api/denoise-diff-video",
    "/api/denoise-playback-video",
    "/api/cleanup-frame-differences",
    "/api/frame-inspector",
    "/api/dust-mask-frame",
    "/api/color-bible/state",
    "/api/color-bible/shot-assignments",
    "/api/color-bible/snapshot-hash",
    "/media",
}

EXPECTED_POST_ROUTES = {
    "/api/settings", "/api/run", "/api/upscale-preview", "/api/stop", "/api/quit",
    "/api/shot-scrub", "/api/shot-prompt", "/api/shot-enabled", "/api/shot-merge",
    "/api/shot-split", "/api/shot-boundary", "/api/shot-fade", "/api/shot-upscale-strength", "/api/shot-ltx-strength", "/api/reference-ltx-strength", "/api/shot-cq-luma-only", "/api/shot-cq-skip", "/api/shot-cq-guides", "/api/shot-cq-anchor", "/api/reference-regenerate",
    "/api/reference-add", "/api/reference-remove", "/api/reference-delete", "/api/reference-custom", "/api/reference-mask-sam",
    "/api/guide-frame-mask-sam", "/api/reference-edit-preview", "/api/reference-edit-accept",
    "/api/reference-edit-revert", "/api/reference-paint-save", "/api/export-media",
    "/api/outpaint-custom-mask-save", "/api/outpaint-custom-mask-clear",
    "/api/outpaint-frame-mask-save", "/api/outpaint-frame-mask-clear",
    "/api/outpaint-chunk", "/api/outpaint-chunk-regenerate", "/api/outpaint-anchor",
    "/api/outpaint-anchor-clear", "/api/outpaint-anchor-generate", "/api/outpaint-end-anchor",
    "/api/outpaint-end-anchor-clear", "/api/outpaint-end-anchor-generate", "/api/guide-frame-add",
    "/api/guide-frame-remove", "/api/guide-frame-save", "/api/guide-frame-upload",
    "/api/guide-frame-clear", "/api/guide-frame-generate", "/api/guide-frame-edit-preview",
    "/api/guide-frame-edit-accept", "/api/guide-frame-edit-revert", "/api/guide-paint-save",
    "/api/browse", "/api/browse-global-source", "/api/source-sequence-fps", "/api/overview-clear", "/api/project-save",
    "/api/project-load", "/api/cache-delete",
    "/api/denoise-batch", "/api/denoise-grid", "/api/denoise-diff",
    "/api/denoise-playback",
    "/api/deflicker-batch",
    "/api/enhancement-batch",
    "/api/stabilize-batch", "/api/stabilize-adopt",
    "/api/dust-status", "/api/dust-inspect", "/api/dust-run", "/api/dust-adopt",
    "/api/dust-mask-frame", "/api/dust-mask-reset", "/api/dust-mask-export",
    "/api/color-bible/sync-manifest",
    "/api/color-bible/film",
    "/api/color-bible/character",
    "/api/color-bible/variant",
    "/api/color-bible/object",
    "/api/color-bible/location",
    "/api/color-bible/assign-ranges",
    "/api/color-bible/reference",
    "/api/color-bible/reference-approval",
    "/api/sam2-track-shot",
}


def _routes_in(method_name: str) -> set[str]:
    source = inspect.getsource(getattr(http_handler.Handler, method_name))
    return set(re.findall(r'parsed\.path == "([^"]+)"', source))


class RouteCoverageTests(unittest.TestCase):
    def test_get_routes_unchanged(self) -> None:
        self.assertEqual(_routes_in("do_GET"), EXPECTED_GET_ROUTES)

    def test_post_routes_unchanged(self) -> None:
        self.assertEqual(_routes_in("do_POST"), EXPECTED_POST_ROUTES)


class FakeApp:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.settings: dict = {"global": {}}

    def state(self, view: str = "") -> dict:
        return {"_view": view}


class _Handler(http_handler.Handler):
    def __init__(self, path: str, body: dict | None = None) -> None:
        self.path = path
        self.headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        self._body = body or {}
        self.responses: dict[str, object] = {}

    def read_json(self) -> dict:
        return self._body

    def send_json(self, payload):  # noqa: D102
        self.responses["json"] = payload

    def send_error(self, code, message=None, explain=None):  # noqa: D102
        self.responses["error"] = code


class HelperBehaviourTests(unittest.TestCase):
    def test_stabilization_routes_pass_explicit_selection_and_candidate(self):
        rows = [{'method': 'zoom0', 'preset': 'settings'}]
        state.APP.run_stabilization_batch = mock.Mock(return_value=(True, 'started'))
        handler = _Handler('/api/stabilize-batch', {'variants': rows})
        handler.do_POST()
        state.APP.run_stabilization_batch.assert_called_once_with(rows)
        self.assertTrue(handler.responses['json']['ok'])
        state.APP.adopt_stabilization = mock.Mock(side_effect=ValueError('stale result'))
        handler = _Handler('/api/stabilize-adopt', {'method': 'zoom0', 'variant': 'old.mkv'})
        handler.do_POST()
        state.APP.adopt_stabilization.assert_called_once_with('zoom0', 'old.mkv')
        self.assertFalse(handler.responses['json']['ok'])
        self.assertIn('stale result', handler.responses['json']['error'])

    def test_settings_validation_returns_json_error(self) -> None:
        state.APP.update_settings = mock.Mock(side_effect=ValueError('Stop processing before changing the pipeline order.'))
        handler = _Handler('/api/settings', {'stage': 'global', 'values': {'preprocessing_order': 'deflicker,denoise,enhancement'}})
        handler.do_POST()
        self.assertFalse(handler.responses['json']['ok'])
        self.assertIn('Stop processing', handler.responses['json']['error'])

    def setUp(self) -> None:
        self._real_app = state.APP
        state.APP = FakeApp()

    def tearDown(self) -> None:
        state.APP = self._real_app

    def test_denoise_batch_passes_selection_and_preserves_legacy_all_request(self) -> None:
        state.APP.run_denoise_batch = mock.Mock(return_value=(True, "started"))
        variants = [{"method": "bm3d", "preset": "strong"}]
        handler = _Handler("/api/denoise-batch", {"variants": variants})
        handler.do_POST()
        state.APP.run_denoise_batch.assert_called_once_with(variants)
        self.assertTrue(handler.responses["json"]["ok"])
        state.APP.run_denoise_batch.reset_mock()
        _Handler("/api/denoise-batch").do_POST()
        state.APP.run_denoise_batch.assert_called_once_with(None)

    def test_denoise_playback_passes_frame_index_without_coercing_fractional_input(self) -> None:
        from ai_remaster_gui import denoise_playback
        handler = _Handler('/api/denoise-playback', {'variant': 'ready.mkv', 'frame': 11400, 'mode': 'rgb', 'contrast': 64})
        with mock.patch.object(denoise_playback, 'playback_segment', return_value={'url': 'cached.mp4'}) as render:
            handler.do_POST()
        render.assert_called_once_with(state.APP, 'ready.mkv', 11400, 'rgb', 64, 'denoise', 'input')
        self.assertEqual(handler.responses['json'], {'ok': True, 'url': 'cached.mp4'})

    def test_denoise_playback_video_rejects_unknown_tokens(self) -> None:
        handler = _Handler('/api/denoise-playback-video?token=../../unrelated.mp4')
        handler.do_GET()
        self.assertEqual(handler.responses['error'], 404)

    def test_dust_mask_playback_forwards_saved_options_without_a_repair(self):
        from ai_remaster_gui import denoise_playback
        values={'detector':'temporal','repair':'telea'}
        handler=_Handler('/api/denoise-playback',{'stage':'dust','frame':7,'mask_values':values})
        with mock.patch.object(denoise_playback,'playback_segment',return_value={}) as call:
            handler.do_POST()
        call.assert_called_once_with(state.APP,'',7,'direction',32,'dust','input',mask_values=values)

    def test_shared_inspector_chart_and_playback_forward_deflicker_stage(self) -> None:
        from ai_remaster_gui import denoise_playback
        for route, symbol, args in (("/api/denoise-inspector?variant=ready.mkv&frame=7&stage=deflicker", "inspect_denoise", (state.APP, "ready.mkv", 7, "deflicker", 'input')),
                                    ("/api/denoise-frame-differences?variant=ready.mkv&stage=deflicker", "denoise_frame_differences", (state.APP, "ready.mkv", "deflicker"))):
            with self.subTest(route=route), mock.patch.object(http_handler, symbol, return_value={}) as call:
                _Handler(route).do_GET()
                call.assert_called_once_with(*args)
        with mock.patch.object(denoise_playback, "playback_segment", return_value={}) as call:
            _Handler('/api/denoise-playback', {'variant':'ready.mkv','frame':7,'stage':'deflicker'}).do_POST()
            call.assert_called_once_with(state.APP, 'ready.mkv', 7, 'direction', 32, 'deflicker', 'input')
        state.APP.run_deflicker_batch = mock.Mock(return_value=(True, "started"))
        rows = [{"method":"ffmpeg_median", "preset":"settings"}]
        handler = _Handler('/api/deflicker-batch', {'variants':rows})
        handler.do_POST()
        state.APP.run_deflicker_batch.assert_called_once_with(rows)
        self.assertEqual(handler.responses['json']['state'], {'_view':'deflicker'})

    def test_enhancement_subset_and_shared_media_keep_their_stage(self) -> None:
        from ai_remaster_gui import denoise_playback
        rows = [{'method': 'nafnet', 'preset': 'settings'}]
        state.APP.run_enhancement_batch = mock.Mock(return_value=(True, 'started'))
        handler = _Handler('/api/enhancement-batch', {'variants': rows})
        handler.do_POST()
        state.APP.run_enhancement_batch.assert_called_once_with(rows)
        self.assertEqual(handler.responses['json']['state'], {'_view': 'enhancement'})
        with mock.patch.object(denoise_playback, 'playback_segment', return_value={}) as call:
            _Handler('/api/denoise-playback', {'variant': 'ready.mkv', 'frame': 7, 'stage': 'enhancement'}).do_POST()
            call.assert_called_once_with(state.APP, 'ready.mkv', 7, 'direction', 32, 'enhancement', 'input')
        with mock.patch.object(http_handler, 'inspect_denoise', return_value={}) as call:
            _Handler('/api/denoise-inspector?variant=ready.mkv&frame=7&stage=enhancement').do_GET()
            call.assert_called_once_with(state.APP, 'ready.mkv', 7, 'enhancement', 'input')

    def test_send_result_success_merges_dict_and_state(self) -> None:
        with mock.patch.object(http_handler, "merge_manifest_shots", return_value={"merged": 2}):
            handler = _Handler("/api/shot-merge", {"manifest": "m.csv", "index": 0})
            handler.do_POST()
        self.assertEqual(handler.responses["json"], {"ok": True, "merged": 2, "state": {"_view": "shots"}})

    def test_send_result_failure_logs_and_reports(self) -> None:
        boom = RuntimeError("kaboom")
        with mock.patch.object(http_handler, "merge_manifest_shots", side_effect=boom):
            handler = _Handler("/api/shot-merge", {"manifest": "m.csv", "index": 0})
            handler.do_POST()
        self.assertEqual(handler.responses["json"], {"ok": False, "error": "kaboom"})
        self.assertTrue(any("Shot merge failed: kaboom" in line for line in state.APP.log))

    def test_send_result_side_effect_only_returns_ok(self) -> None:
        with mock.patch.object(http_handler, "update_reference_prompt", return_value={}) as save:
            handler = _Handler("/api/shot-prompt", {"manifest": "m.csv", "index": 1, "prompt": "hi"})
            handler.do_POST()
        self.assertEqual(handler.responses["json"], {"ok": True})
        save.assert_called_once_with("m.csv", 1, 0, "hi")  # no reference_index: reference 1

    def test_shot_boundary_returns_row_patch_instead_of_full_state(self) -> None:
        with (
            mock.patch.object(http_handler, "update_shot_boundary", return_value={"frame": "12"}) as boundary,
            mock.patch.object(http_handler, "shot_rows_for_indices", return_value=[{"index": 0}, {"index": 1}]) as rows,
        ):
            handler = _Handler("/api/shot-boundary", {"manifest": "m.csv", "index": 0, "edge": "end", "frame": 12})
            handler.do_POST()

        payload = handler.responses["json"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["rows"], [{"index": 0}, {"index": 1}])
        self.assertNotIn("state", payload)
        boundary.assert_called_once_with("m.csv", 0, "end", 0.0, 12)
        rows.assert_called_once_with("m.csv", (-2, -1, 0, 1))

    def test_state_route_passes_active_view(self) -> None:
        state.APP.state = mock.Mock(return_value={"ok": True})
        handler = _Handler("/api/state?active=shots")
        handler.do_GET()

        self.assertEqual(handler.responses["json"], {"ok": True})
        state.APP.state.assert_called_once_with("shots", False)

    def test_state_route_passes_section_retry(self) -> None:
        state.APP.state = mock.Mock(return_value={"ok": True})
        handler = _Handler("/api/state?active=outpaint&retry_section=1")
        handler.do_GET()

        state.APP.state.assert_called_once_with("outpaint", True)

    def test_upscale_shot_comparison_returns_cached_frame_pair(self) -> None:
        state.APP.upscale_shot_comparison_frames = mock.Mock(
            return_value={"before": "before.jpg", "after": "after.jpg", "stale": "false"}
        )
        handler = _Handler("/api/upscale-shot-comparison?index=3&time=4.25")
        handler.do_GET()

        self.assertEqual(
            handler.responses["json"],
            {"ok": True, "before": "before.jpg", "after": "after.jpg", "stale": "false"},
        )
        state.APP.upscale_shot_comparison_frames.assert_called_once_with(3, 4.25)

    def test_send_action_returns_ok_message_state(self) -> None:
        state.APP.run_reference_regeneration = lambda *a: (True, "done")
        handler = _Handler("/api/reference-regenerate", {"manifest": "m.csv", "index": 0})
        handler.do_POST()
        self.assertEqual(
            handler.responses["json"],
            {"ok": True, "message": "done", "state": {"_view": ""}, "error": ""},
        )

    def test_send_action_failure_has_null_state(self) -> None:
        state.APP.run_reference_regeneration = lambda *a: (False, "nope")
        handler = _Handler("/api/reference-regenerate", {"manifest": "m.csv", "index": 0})
        handler.do_POST()
        payload = handler.responses["json"]
        self.assertEqual(payload["ok"], False)
        self.assertIsNone(payload["state"])
        self.assertEqual(payload["error"], "nope")

    def test_send_action_extra_key_carries_preview(self) -> None:
        state.APP.run_guide_edit_preview = lambda *a: (True, "edited", "preview.png")
        handler = _Handler("/api/guide-frame-edit-preview", {"chunk_index": 0, "guide_index": 0})
        handler.do_POST()
        self.assertEqual(handler.responses["json"]["preview"], "preview.png")

    def test_send_action_exception_reports_its_type(self) -> None:
        # The guide editor pops up NoEditRegionError rather than burying it in the status line.
        from ai_remaster_gui.outpaint_guides import NoEditRegionError

        def refuse(*_args):
            raise NoEditRegionError("nothing to edit")

        state.APP.run_guide_edit_preview = refuse
        handler = _Handler("/api/guide-frame-edit-preview", {"chunk_index": 0, "guide_index": 0})
        handler.do_POST()
        payload = handler.responses["json"]
        self.assertEqual((payload["ok"], payload["error"], payload["error_type"]), (False, "nothing to edit", "NoEditRegionError"))


class WorkerOutputTests(unittest.TestCase):
    def test_worker_output_survives_characters_outside_the_windows_code_page(self) -> None:
        # A curly closing quote is E2 80 9D in UTF-8; 0x9D is undefined in Windows-1252, so a
        # reader on the code page died there and the worker hung on its next print.
        import subprocess
        import sys

        app = server.PipelineApp.__new__(server.PipelineApp)
        script = "print('A cart labeled “BAE HAY”'); print('done')"
        # Workers write UTF-8 when the environment says so (as ARP's did) whatever the code page.
        with mock.patch.dict("os.environ", {"PYTHONIOENCODING": "utf-8"}), mock.patch("locale.getpreferredencoding", return_value="cp1252"):
            kwargs = app.child_process_kwargs()
            process = subprocess.Popen([sys.executable, "-c", script], **kwargs)
            lines = [line.rstrip() for line in process.stdout]
        process.wait()
        self.assertEqual(lines, ["A cart labeled “BAE HAY”", "done"])


class StateLockTests(unittest.TestCase):
    def test_shot_view_builds_after_app_lock_is_released(self) -> None:
        app = server.PipelineApp.__new__(server.PipelineApp)
        app.settings = {"global": {"colorize": "true"}}
        app.project_path = None
        app.log = []
        app.process = None
        app.running_stage = ""
        app.running_stage_key = ""
        app.running_reference_manifest = ""
        app.running_reference_index = None
        app.run_started_at = 0.0
        app.lock = threading.Lock()

        app.source_media_state = MethodType(lambda self, source: {
            "previews": [],
            "info": {},
            "monochrome": True,
            "analysis": {},
            "aspect_preview": "",
        }, app)
        app.files_for = MethodType(lambda self, stage: [], app)
        app.active_stages = MethodType(lambda self: (), app)
        app.progress = MethodType(lambda self: [], app)
        app.phase_progress = MethodType(lambda self, progress_rows=None: {"global": {"percent": 0, "label": "Waiting"}, "stages": []}, app)
        app.expected_outputs = MethodType(lambda self, key: [], app)
        app.existing_outputs = MethodType(lambda self, key: [], app)
        app.upscale_preview_state = MethodType(lambda self: {}, app)
        app.output_selection_state = MethodType(lambda self: {}, app)

        def fake_shot_views(settings, view):
            self.assertTrue(app.lock.acquire(blocking=False))
            app.lock.release()
            return {"shots": [{"index": 0}], "shots_manifest": "m.csv"}

        with (
            mock.patch.object(server, "shot_views", side_effect=fake_shot_views),
            mock.patch.object(server, "source_section_state", return_value={}),
            mock.patch.object(server, "system_status", return_value={}),
        ):
            payload = app.state("shots")

        self.assertEqual(payload["shot_views"]["shots"], [{"index": 0}])


class DustMaskEditRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_app = state.APP
        state.APP = FakeApp()

    def tearDown(self) -> None:
        state.APP = self._real_app

    def test_dust_mask_get_and_post_routes_delegate_and_include_state_and_error_type(self) -> None:
        from ai_remaster_gui.dust_mask_edits import EditRevisionConflictError, EditStoreIntegrityError

        state.APP.load_dust_mask_frame = mock.Mock(
            return_value={
                "frame": 2,
                "width": 32,
                "height": 24,
                "frame_count": 5,
                "analysis_source": "stage",
                "timeline_id": "a" * 24,
                "edit_revision": 0,
                "zip_sha256": "",
                "edit_signature": "",
                "compatible": True,
                "compatibility_reason": "",
                "edited_frames": [],
                "edit_mask_b64": "",
                "reviewed": False,
                "frame_stats": {},
                "summary": {},
            }
        )
        get_handler = _Handler("/api/dust-mask-frame?frame=2&analysis_source=stage&timeline_id=" + ("a" * 24))
        get_handler.do_GET()
        state.APP.load_dust_mask_frame.assert_called_once_with(
            2,
            analysis_source="stage",
            timeline_id="a" * 24,
            project_expected=None,
        )
        self.assertTrue(get_handler.responses["json"]["ok"])
        self.assertEqual(get_handler.responses["json"]["frame"], 2)

        state.APP.load_dust_mask_frame = mock.Mock(side_effect=EditStoreIntegrityError("corrupt edit zip"))
        err_handler = _Handler("/api/dust-mask-frame?frame=0&analysis_source=stage")
        err_handler.do_GET()
        self.assertEqual(err_handler.responses["json"]["ok"], False)
        self.assertEqual(err_handler.responses["json"]["error_type"], "EditStoreIntegrityError")

        bad_frame = _Handler("/api/dust-mask-frame?frame=abc")
        bad_frame.do_GET()
        self.assertEqual(bad_frame.responses["json"]["ok"], False)
        self.assertEqual(bad_frame.responses["json"]["error_type"], "ValueError")

        state.APP.save_dust_mask_frame = mock.Mock(
            return_value={"frame": 1, "edit_revision": 1, "edit_signature": ("b" * 64) + ":rev1"}
        )
        save_handler = _Handler(
            "/api/dust-mask-frame",
            {"frame": 1, "edit_mask_b64": "AAAA", "analysis_source": "stage", "expected_revision": 0},
        )
        save_handler.do_POST()
        state.APP.save_dust_mask_frame.assert_called_once()
        self.assertTrue(save_handler.responses["json"]["ok"])
        self.assertEqual(save_handler.responses["json"]["state"], {"_view": "dust"})

        state.APP.save_dust_mask_frame = mock.Mock(side_effect=EditRevisionConflictError("stale rev"))
        conflict_handler = _Handler("/api/dust-mask-frame", {"frame": 1, "edit_mask_b64": "AAAA"})
        conflict_handler.do_POST()
        self.assertEqual(conflict_handler.responses["json"]["ok"], False)
        self.assertEqual(conflict_handler.responses["json"]["error_type"], "EditRevisionConflictError")

        state.APP.reset_dust_masks = mock.Mock(
            return_value={"scope": "frame", "edit_revision": 2, "edited_frames": []}
        )
        reset_handler = _Handler("/api/dust-mask-reset", {"scope": "frame", "frame": 1, "analysis_source": "stage"})
        reset_handler.do_POST()
        self.assertTrue(reset_handler.responses["json"]["ok"])
        self.assertEqual(reset_handler.responses["json"]["state"], {"_view": "dust"})

        state.APP.export_dust_provenance = mock.Mock(
            return_value={
                "export_dir": "exports/dust_mask_provenance/clip",
                "metadata_path": "exports/dust_mask_provenance/clip/metadata.json",
                "provenance_path": "exports/dust_mask_provenance/clip/provenance.json",
                "exported_frames": [1],
                "summary": {"ground_truth_status": "operational_edits_only_not_independent_gt"},
            }
        )
        export_handler = _Handler("/api/dust-mask-export", {"analysis_source": "stage", "include_auto_proposals": True})
        export_handler.do_POST()
        self.assertTrue(export_handler.responses["json"]["ok"])
        self.assertEqual(export_handler.responses["json"]["exported_frames"], [1])

    def test_dust_mask_routes_reject_client_paths_and_cross_origin_requests(self) -> None:
        bad_save = _Handler("/api/dust-mask-frame", {"frame": 0, "edit_mask_b64": "", "path": "C:/evil.zip"})
        bad_save.do_POST()
        self.assertFalse(bad_save.responses["json"]["ok"])
        self.assertEqual(bad_save.responses["json"]["error_type"], "ValueError")

        bad_export = _Handler("/api/dust-mask-export", {"export_dir": "C:/evil"})
        bad_export.do_POST()
        self.assertFalse(bad_export.responses["json"]["ok"])
        self.assertEqual(bad_export.responses["json"]["error_type"], "ValueError")

        cross_origin = _Handler("/api/dust-mask-frame", {"frame": 0, "edit_mask_b64": ""})
        cross_origin.headers = {"Host": "127.0.0.1:8765", "Origin": "https://attacker.example"}
        cross_origin.do_POST()
        self.assertEqual(cross_origin.responses["error"], 403)

        remote_host = _Handler("/api/dust-mask-frame?frame=0")
        remote_host.headers = {"Host": "attacker.example"}
        remote_host.do_GET()
        self.assertEqual(remote_host.responses["error"], 403)

    def test_existing_dust_routes_delegate_to_app_methods(self) -> None:
        state.APP.dust_status = mock.Mock(return_value={"frames": 3, "dust_edits": {"has_edits": True}})
        status_h = _Handler("/api/dust-status", {"values": {"detector": "farneback_3f"}, "source": "s.mkv", "project": ""})
        status_h.do_POST()
        state.APP.dust_status.assert_called_once_with({"detector": "farneback_3f"}, "s.mkv", "")
        self.assertTrue(status_h.responses["json"]["ok"])

        state.APP.dust_inspect = mock.Mock(return_value={"frame": 0, "urls": {}, "vetoed_url": ""})
        insp_h = _Handler("/api/dust-inspect", {"values": {"detector": "farneback_3f"}, "frame": 0, "source": "s.mkv", "project": ""})
        insp_h.do_POST()
        state.APP.dust_inspect.assert_called_once_with({"detector": "farneback_3f"}, 0, "s.mkv", "")
        self.assertTrue(insp_h.responses["json"]["ok"])


class DustMaskEditEndToEndRouteTests(unittest.TestCase):
    def test_real_pipeline_app_edit_roundtrip_compatibility_adoption_export_and_arpp(self) -> None:
        import tempfile
        from pathlib import Path
        import numpy as np
        from ai_remaster_gui import dust as dust_stage, project_io
        from ai_remaster_gui.dust_mask_edits import (
            BIT_PROTECT_INTACT,
            BIT_REVIEWED,
            BIT_SCRATCH_ADD,
            decode_bitmask_b64,
            encode_bitmask_b64,
        )
        from ai_remaster_gui.runtime_settings import base_settings

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "input" / "clip.mkv"
            source.parent.mkdir(parents=True, exist_ok=True)
            writer = dust_stage.Writer(source, {"frames": 3, "width": 64, "height": 48, "fps": "12"})
            for _ in range(3):
                writer.write(np.full((48, 64, 3), 120, np.uint8))
            writer.close()

            settings = base_settings()
            settings["global"].update(source=str(source), dust="true", expand_outpaint="false", colorize="false")
            settings["dust"].update(detector="legacy", repair="telea")

            app = server.PipelineApp.__new__(server.PipelineApp)
            app.settings = settings
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
            app.save = mock.Mock()
            app.ensure_pipeline_source = mock.Mock()
            app.remember_project_path = mock.Mock()

            with (
                mock.patch.object(server, "ROOT", root),
                mock.patch.object(dust_stage, "ROOT", root),
                mock.patch.object(project_io, "ROOT", root),
                mock.patch.object(state, "APP", app),
            ):
                # 1. Unedited frame load returns exact dimensions & clean summary
                loaded0 = app.load_dust_mask_frame(0, analysis_source="stage")
                self.assertEqual((loaded0["width"], loaded0["height"], loaded0["frame_count"]), (64, 48, 3))
                self.assertEqual(loaded0["edit_mask_b64"], "")
                self.assertEqual(loaded0["edit_revision"], 0)
                self.assertTrue(loaded0["compatible"])

                # 2. Run base detection & repair and adopt it before editing
                dust_stage.run(source, {"detector": "legacy", "repair": "telea"}, "mask")
                dust_stage.run(source, {"detector": "legacy", "repair": "telea"}, "repair")
                app.adopt_dust({"detector": "legacy", "repair": "telea"})
                app.validate_dust_adoption()

                # 3. Paint lossless bitmask edit on frame 1 via POST route contract
                bitmask = np.zeros((48, 64), dtype=np.uint8)
                bitmask[10:20, 12:14] = BIT_SCRATCH_ADD | BIT_REVIEWED
                bitmask[25:35, 30:40] = BIT_PROTECT_INTACT | BIT_REVIEWED
                b64 = encode_bitmask_b64(bitmask)

                save_h = _Handler(
                    "/api/dust-mask-frame",
                    {"frame": 1, "edit_mask_b64": b64, "analysis_source": "stage", "expected_revision": 0},
                )
                save_h.do_POST()
                save_res = save_h.responses["json"]
                self.assertTrue(save_res["ok"])
                self.assertEqual(save_res["edit_revision"], 1)
                self.assertTrue(save_res["edit_signature"].endswith(":rev1"))
                self.assertEqual(save_res["edited_frames"], [1])
                self.assertTrue(save_res["reviewed"])
                self.assertIn("dust_edits", save_res["state"])
                self.assertEqual(save_res["state"]["dust_edits"]["stage"]["edit_signature"], save_res["edit_signature"])

                # 4. Exact byte-for-byte GET round-trip
                get_h = _Handler("/api/dust-mask-frame?frame=1&analysis_source=stage")
                get_h.do_GET()
                get_res = get_h.responses["json"]
                self.assertTrue(get_res["ok"])
                np.testing.assert_array_equal(decode_bitmask_b64(get_res["edit_mask_b64"], 64, 48), bitmask)

                # 5. Stale expected_revision fails closed
                stale_h = _Handler(
                    "/api/dust-mask-frame",
                    {"frame": 1, "edit_mask_b64": b64, "expected_revision": 0},
                )
                stale_h.do_POST()
                self.assertFalse(stale_h.responses["json"]["ok"])
                self.assertEqual(stale_h.responses["json"]["error_type"], "EditRevisionConflictError")

                # 6. Stale adoption now fails closed until composed repair is run & adopted
                with self.assertRaises((FileNotFoundError, ValueError)):
                    app.validate_dust_adoption()

                # 7. Clicking Repair right after painting allows saved_base_mask and forwards --edit-signature
                with mock.patch.object(app, "_start_denoise_matrix", return_value=(True, "launched")) as launch:
                    ok, _msg = app.run_dust({"detector": "legacy", "repair": "telea"}, "repair")
                    self.assertTrue(ok)
                    cmd = launch.call_args.args[1]
                    self.assertIn("--edit-signature", cmd)
                    self.assertIn(save_res["edit_signature"], cmd)
                    self.assertIn("--timeline-id", cmd)
                    self.assertIn(save_res["timeline_id"], cmd)

                # 8. Inspecting before composed repair is materialized falls back to base + active edit overlay
                with mock.patch("ai_remaster_gui.frame_inspector._decode", return_value="frame.png"):
                    insp = app.dust_inspect({"detector": "legacy", "repair": "telea"}, 1)
                    self.assertTrue(insp["composed_pending"])
                    self.assertTrue(insp["base_mask_ready"])
                    self.assertEqual(insp["edit_mask_b64"], b64)
                    self.assertEqual(insp["edit_signature"], save_res["edit_signature"])

                # 9. Materialize composed repair, adopt, and verify validate_dust_adoption passes
                dust_stage.run(
                    source,
                    {
                        "detector": "legacy",
                        "repair": "telea",
                        "edit_signature": save_res["edit_signature"],
                        "timeline_id": save_res["timeline_id"],
                    },
                    "repair",
                )
                app.adopt_dust({"detector": "legacy", "repair": "telea"})
                app.validate_dust_adoption()

                # 10. Export provenance bundle strictly under ROOT/exports/dust_mask_provenance
                exp_h = _Handler("/api/dust-mask-export", {"analysis_source": "stage"})
                exp_h.do_POST()
                exp_res = exp_h.responses["json"]
                self.assertTrue(exp_res["ok"])
                self.assertTrue(Path(exp_res["export_dir"]).is_relative_to(root / "exports" / "dust_mask_provenance"))
                self.assertTrue(Path(exp_res["provenance_path"]).is_file())
                self.assertEqual(
                    exp_res["summary"]["ground_truth_status"],
                    "operational_edits_only_not_independent_gt",
                )

                # 11. .arpp bundle round-trip preserves manifests/dust_masks/<tid>.zip + .meta.json
                arpp_path = root / "projects" / "clip.arpp"
                project_io.write_project_file(arpp_path, app.settings)
                zip_member = f"manifests/dust_masks/{save_res['timeline_id']}.zip"
                meta_member = f"manifests/dust_masks/{save_res['timeline_id']}.meta.json"
                import zipfile
                with zipfile.ZipFile(arpp_path, "r") as zf:
                    names = set(zf.namelist())
                self.assertIn(zip_member, names)
                self.assertIn(meta_member, names)

                # 12. Tampered edit ZIP fails closed on GET and marks state incompatible
                zip_disk = root / zip_member
                zip_bytes = zip_disk.read_bytes()
                zip_disk.write_bytes(b"tampered-bytes")
                tampered_h = _Handler("/api/dust-mask-frame?frame=1&analysis_source=stage")
                tampered_h.do_GET()
                self.assertFalse(tampered_h.responses["json"]["ok"])
                self.assertEqual(tampered_h.responses["json"]["error_type"], "EditStoreIntegrityError")
                self.assertFalse(app.dust_edit_state()["stage"]["compatible"])
                zip_disk.write_bytes(zip_bytes)

                # 13. Replaced clip with mismatched dimensions reports compatible=False on GET
                source_diff = root / "input" / "clip.mkv"
                source_diff.unlink()
                writer2 = dust_stage.Writer(source_diff, {"frames": 3, "width": 80, "height": 48, "fps": "12"})
                for _ in range(3):
                    writer2.write(np.full((48, 80, 3), 120, np.uint8))
                writer2.close()
                mismatch_h = _Handler("/api/dust-mask-frame?frame=0&analysis_source=stage")
                mismatch_h.do_GET()
                mismatch_res = mismatch_h.responses["json"]
                self.assertTrue(mismatch_res["ok"])
                self.assertFalse(mismatch_res["compatible"])
                self.assertTrue(mismatch_res["compatibility_reason"])
                self.assertEqual(mismatch_res["edited_frames"], [])

                # 14. Reset all clears conflict cleanly
                reset_h = _Handler("/api/dust-mask-reset", {"scope": "all", "analysis_source": "stage"})
                reset_h.do_POST()
                self.assertTrue(reset_h.responses["json"]["ok"])
                self.assertTrue(reset_h.responses["json"]["state"]["dust_edits"]["stage"]["compatible"])


class SAM2TrackShotRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_app = state.APP
        state.APP = FakeApp()

    def tearDown(self) -> None:
        state.APP = self._real_app

    def test_sam2_track_shot_success(self) -> None:
        from pathlib import Path
        with (
            mock.patch("ai_remaster_gui.sam2_tracker.verify_sam2_installation") as mock_verify,
            mock.patch("ai_remaster_gui.commercial_policy.evaluate_component") as mock_eval,
            mock.patch("ai_remaster_gui.sam2_tracker.run_shot_tracking_pipeline") as mock_run,
        ):
            mock_eval.return_value = mock.MagicMock(allowed=True, reason="")
            mock_run.return_value = Path("intermediate/shot_masks/shot_001.masks.npz")

            handler = _Handler("/api/sam2-track-shot", {
                "shot_id": "shot_001",
                "frames": "test_frames",
                "prompts": {"0": {"points": [[10, 10]], "labels": [1]}},
            })
            handler.do_POST()

            mock_verify.assert_called_once()
            mock_eval.assert_called_once()
            mock_run.assert_called_once()
            self.assertEqual(handler.responses["json"], {
                "ok": True,
                "output_npz_path": str(Path("intermediate/shot_masks/shot_001.masks.npz")),
            })

    def test_sam2_track_shot_commercial_guard_blocks(self) -> None:
        with (
            mock.patch("ai_remaster_gui.sam2_tracker.verify_sam2_installation"),
            mock.patch("ai_remaster_gui.commercial_policy.evaluate_component") as mock_eval,
        ):
            mock_eval.return_value = mock.MagicMock(allowed=False, reason="Non-commercial model")

            handler = _Handler("/api/sam2-track-shot", {
                "shot_id": "shot_001",
                "frames": "test_frames",
                "prompts": {},
            })
            handler.do_POST()

            self.assertFalse(handler.responses["json"]["ok"])
            self.assertIn("Commercial Model Guard blocked sam2", handler.responses["json"]["error"])

    def test_sam2_track_shot_path_traversal_rejected(self) -> None:
        with (
            mock.patch("ai_remaster_gui.sam2_tracker.verify_sam2_installation"),
            mock.patch("ai_remaster_gui.commercial_policy.evaluate_component") as mock_eval,
        ):
            mock_eval.return_value = mock.MagicMock(allowed=True, reason="")

            handler = _Handler("/api/sam2-track-shot", {
                "shot_id": "shot_001",
                "frames": "test_frames",
                "prompts": {},
                "output_npz_path": "../../outside/evil.masks.npz",
            })
            handler.do_POST()

            self.assertFalse(handler.responses["json"]["ok"])
            self.assertIn("path traversal", handler.responses["json"]["error"].lower())


if __name__ == "__main__":
    unittest.main()
