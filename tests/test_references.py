from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai_remaster_gui import references, state
from ai_remaster_gui.manifests import read_manifest, write_manifest_details


class ReferenceScrubTests(unittest.TestCase):
    def test_shot_view_falls_back_to_loaded_reference_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
            folder = Path(tmp_text)
            loaded_manifest = folder / "loaded-project-shots.csv"
            loaded_manifest.write_text("enabled,start_frame,end_frame,end\ntrue,0,24,00:00:01.000\n", encoding="utf-8")
            missing_derived = folder / "new-derived-name.csv"
            loaded_text = references.rel(loaded_manifest)

            def fake_rows(manifest, **_kwargs):
                return [{"manifest": manifest}] if manifest else []

            with (
                mock.patch.object(references, "manifest_for_outpainted", return_value=references.rel(missing_derived)),
                mock.patch.object(references, "shot_rows", side_effect=fake_rows),
            ):
                views = references.shot_views({
                    "shots": {"outpainted_video": "intermediate/outpainted/renamed.mp4"},
                    "references": {"manifest": loaded_text},
                    "colour": {"manifest": loaded_text},
                }, "shots")

        self.assertEqual(views["shots_manifest"], loaded_text)
        self.assertEqual(views["shots"], [{"manifest": loaded_text}])

    def test_extract_reference_frame_persists_selected_frame(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start", "end", "source_reference", "color_reference"],
                    [{
                        "start": "00:00:00.000",
                        "end": "00:00:02.000",
                        "source_reference": "",
                        "color_reference": "",
                    }],
                )

                fake_result = SimpleNamespace(returncode=0, stderr="", stdout="")
                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references, "ffprobe_info", return_value={"frame_rate": "25.000 fps"}),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result),
                ):
                    result = references.extract_reference_frame(references.rel(manifest), 0, 1.24)

                row = read_manifest(manifest)[0]
                self.assertEqual(row["selected_frame"], "31")
                self.assertEqual(result["selected_frame"], "31")
                self.assertEqual(row["source_reference"], result["source_reference"])
        finally:
            state.APP = previous_app

    def test_extract_reference_frame_uses_exact_frame_when_provided(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start", "end", "source_reference", "color_reference"],
                    [{
                        "start": "00:00:00.000",
                        "end": "00:00:02.000",
                        "source_reference": "",
                        "color_reference": "",
                    }],
                )

                fake_result = SimpleNamespace(returncode=0, stderr="", stdout="")
                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result) as run,
                ):
                    result = references.extract_reference_frame(references.rel(manifest), 0, 1.251, frame=30)

                command = run.call_args.args[0]
                self.assertEqual(command[command.index("-vf") + 1], "trim=start_frame=30:end_frame=31,setpts=PTS-STARTPTS")
                self.assertEqual(read_manifest(manifest)[0]["selected_frame"], "30")
                self.assertEqual(result["selected_frame"], "30")
        finally:
            state.APP = previous_app

    def test_additional_reference_round_trips_inside_one_shot(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                primary = folder / "primary.png"
                primary.write_bytes(b"primary")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start_frame", "end_frame", "selected_frame", "source_reference", "color_reference"],
                    [{
                        "start_frame": "0",
                        "end_frame": "100",
                        "selected_frame": "20",
                        "source_reference": references.rel(primary),
                        "color_reference": references.rel(primary),
                    }],
                )

                fake_result = SimpleNamespace(returncode=0, stderr="", stdout="")
                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references, "manifest_fps", return_value=25.0),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result),
                ):
                    added = references.add_reference_frame(references.rel(manifest), 0, 2.4, frame=60)

                row = read_manifest(manifest)[0]
                items = references.reference_items(row)
                self.assertEqual(added["reference_index"], "1")
                self.assertEqual([item["selected_frame"] for item in items], ["20", "60"])
                view = references.shot_rows(references.rel(manifest))[0]
                self.assertEqual(view["reference_count"], 2)
                self.assertEqual(view["reference_items"][1]["selected_frame"], 60)

                with (
                    mock.patch.object(references, "local_tool", return_value="ffmpeg"),
                    mock.patch.object(references, "manifest_fps", return_value=25.0),
                    mock.patch.object(references.subprocess, "run", return_value=fake_result),
                ):
                    references.extract_reference_frame(
                        references.rel(manifest), 0, 3.2, frame=80, reference_index=1
                    )
                updated = references.reference_items(read_manifest(manifest)[0])
                self.assertEqual([item["selected_frame"] for item in updated], ["20", "80"])

                references.remove_additional_reference(references.rel(manifest), 0, 1)
                self.assertEqual(len(references.reference_items(read_manifest(manifest)[0])), 1)
        finally:
            state.APP = previous_app

    def test_merging_shots_preserves_the_following_shot_as_a_keyframe(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(log=[], settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                manifest = Path(tmp_text) / "shots.csv"
                write_manifest_details(
                    manifest,
                    "source.mp4",
                    ["start_frame", "end_frame", "selected_frame", "end", "source_reference", "color_reference"],
                    [
                        {"start_frame": "0", "end_frame": "50", "selected_frame": "20", "end": "00:00:02.000", "source_reference": "a.png", "color_reference": "a_color.png"},
                        {"start_frame": "50", "end_frame": "100", "selected_frame": "70", "end": "00:00:04.000", "source_reference": "b.png", "color_reference": "b_color.png"},
                    ],
                )
                with mock.patch.object(references, "manifest_fps", return_value=25.0):
                    references.merge_manifest_shots(references.rel(manifest), 0)

                rows = read_manifest(manifest)
                self.assertEqual(len(rows), 1)
                items = references.reference_items(rows[0])
                self.assertEqual([item["selected_frame"] for item in items], ["20", "70"])
                self.assertEqual([item["color_reference"] for item in items], ["a_color.png", "b_color.png"])
        finally:
            state.APP = previous_app

    def test_preview_reference_frame_can_use_exact_frame(self) -> None:
        with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
            folder = Path(tmp_text)
            source = folder / "source.mp4"
            source.write_bytes(b"video placeholder")
            manifest = folder / "shots.csv"
            write_manifest_details(
                manifest,
                references.rel(source),
                ["start", "end", "source_reference", "color_reference"],
                [{"start": "00:00:00.000", "end": "00:00:02.000", "source_reference": "", "color_reference": ""}],
            )

            with mock.patch.object(references, "extract_video_frame_at_frame", return_value="preview.jpg") as exact:
                path = references.preview_reference_frame(references.rel(manifest), 0, 1.251, frame=30)

        self.assertEqual(path, "preview.jpg")
        self.assertEqual(exact.call_args.args[3], 30)

    def test_shot_rows_can_limit_preview_generation_to_selected_indices(self) -> None:
        previous_app = state.APP
        state.APP = SimpleNamespace(settings={"references": {}})
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                source = folder / "source.mp4"
                source.write_bytes(b"video placeholder")
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    references.rel(source),
                    ["start_frame", "end_frame", "end", "source_reference", "color_reference"],
                    [
                        {"start_frame": "0", "end_frame": "24", "end": "00:00:01.000", "source_reference": "", "color_reference": ""},
                        {"start_frame": "24", "end_frame": "48", "end": "00:00:02.000", "source_reference": "", "color_reference": ""},
                        {"start_frame": "48", "end_frame": "72", "end": "00:00:03.000", "source_reference": "", "color_reference": ""},
                    ],
                )

                calls = []

                def fake_preview(_manifest, index, _seconds, frame=None):
                    calls.append((index, frame))
                    return f"preview-{index}-{frame}.jpg"

                with (
                    mock.patch.object(references, "manifest_fps", return_value=24.0),
                    mock.patch.object(references, "preview_reference_frame", side_effect=fake_preview),
                ):
                    rows = references.shot_rows(references.rel(manifest), include_previews=True, preview_indices={1})

            self.assertNotIn("start_preview", rows[0])
            self.assertEqual(rows[1]["start_preview"], "preview-1-24.jpg")
            self.assertNotIn("start_preview", rows[2])
            self.assertEqual([index for index, _frame in calls], [1, 1, 1])
        finally:
            state.APP = previous_app


class ColorBibleIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prev_app = state.APP
        state.APP = SimpleNamespace(
            log=[],
            settings={
                "global": {"color_bible_film_id": "film_test"},
                "references": {
                    "manifest": "",
                    "color_bible_film_id": "film_test",
                    "reference_runner": "auto",
                    "confidence_threshold": "0.85",
                },
            },
        )

    def tearDown(self) -> None:
        state.APP = self.prev_app

    def test_shot_rows_exposes_qc_status_and_color_bible_fields(self) -> None:
        with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "shots.csv"
            write_manifest_details(
                manifest,
                "source.mp4",
                ["start_frame", "end_frame", "end", "source_reference", "color_reference", "qc_status"],
                [
                    {
                        "start_frame": "0",
                        "end_frame": "24",
                        "end": "00:00:01.000",
                        "source_reference": "a.png",
                        "color_reference": "a_color.png",
                        "qc_status": "needs_review",
                    },
                    {
                        "start_frame": "24",
                        "end_frame": "48",
                        "end": "00:00:02.000",
                        "source_reference": "b.png",
                        "color_reference": "b_color.png",
                        "qc_status": "approved",
                    },
                ],
            )

            mock_service = mock.MagicMock()
            mock_service.get_shot_palette_assignments.return_value = [
                {
                    "assignment": {"approval_status": "approved", "confidence": 0.9},
                    "entity_type": "character",
                    "entity_id": "char_1",
                    "variant_id": "var_1",
                    "entity": {"name": "Alice"},
                    "color_references": [],
                }
            ]

            with (
                mock.patch("ai_remaster_gui.color_bible_service.ColorBibleService", return_value=mock_service),
                mock.patch.object(references, "manifest_fps", return_value=24.0),
            ):
                rows = references.shot_rows(references.rel(manifest))

            self.assertEqual(len(rows), 2)
            # Shot 0 has qc_status='needs_review'
            self.assertEqual(rows[0]["qc_status"], "needs_review")
            self.assertTrue(rows[0]["needs_review"])
            self.assertIn("Shot QC status is 'needs_review'", rows[0]["review_reasons"][0])
            self.assertEqual(len(rows[0]["color_bible_entities"]), 1)

            # Shot 1 has qc_status='approved'
            self.assertEqual(rows[1]["qc_status"], "approved")
            self.assertFalse(rows[1]["needs_review"])
            self.assertEqual(rows[1]["review_reasons"], [])

    def test_regeneration_commands_pass_color_bible_args(self) -> None:
        with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
            folder = Path(tmp_text)
            manifest = folder / "shots.csv"
            write_manifest_details(
                manifest,
                "source.mp4",
                ["start_frame", "end_frame", "end", "source_reference", "color_reference"],
                [
                    {
                        "start_frame": "0",
                        "end_frame": "24",
                        "end": "00:00:01.000",
                        "source_reference": "src.png",
                        "color_reference": "dst.png",
                    }
                ],
            )

            # 1. ddcolor command
            cmd_dd, out_dd = references.ddcolor_reference_regeneration_command(references.rel(manifest), 0)
            self.assertIn("--film-id", cmd_dd)
            self.assertEqual(cmd_dd[cmd_dd.index("--film-id") + 1], "film_test")
            self.assertIn("--reference-runner", cmd_dd)
            self.assertEqual(cmd_dd[cmd_dd.index("--reference-runner") + 1], "auto")
            self.assertIn("--confidence-threshold", cmd_dd)
            self.assertEqual(cmd_dd[cmd_dd.index("--confidence-threshold") + 1], "0.85")

            # 2. qwen21 command
            cmd_q21, out_q21 = references.qwen21_reference_regeneration_command(references.rel(manifest), 0)
            self.assertIn("--film-id", cmd_q21)
            self.assertEqual(cmd_q21[cmd_q21.index("--film-id") + 1], "film_test")
            self.assertIn("--reference-runner", cmd_q21)
            self.assertEqual(cmd_q21[cmd_q21.index("--reference-runner") + 1], "auto")
            self.assertIn("--confidence-threshold", cmd_q21)
            self.assertEqual(cmd_q21[cmd_q21.index("--confidence-threshold") + 1], "0.85")

            # 3. single reference command
            with mock.patch("ai_remaster_gui.references.qwen_workflow_for", return_value="dummy_workflow.json"):
                cmd_ref, out_ref = references.reference_regeneration_command(references.rel(manifest), 0)
                self.assertIn("--film-id", cmd_ref)
                self.assertEqual(cmd_ref[cmd_ref.index("--film-id") + 1], "film_test")
                self.assertIn("--reference-runner", cmd_ref)
                self.assertEqual(cmd_ref[cmd_ref.index("--reference-runner") + 1], "auto")
                self.assertIn("--confidence-threshold", cmd_ref)
                self.assertEqual(cmd_ref[cmd_ref.index("--confidence-threshold") + 1], "0.85")

    def test_shot_rows_safe_when_app_is_none(self) -> None:
        """Verify shot_rows executes without AttributeError when state.APP is None."""
        prev_app = state.APP
        state.APP = None
        try:
            with tempfile.TemporaryDirectory(dir=references.ROOT) as tmp_text:
                folder = Path(tmp_text)
                manifest = folder / "shots.csv"
                write_manifest_details(
                    manifest,
                    "video.mp4",
                    ["start", "end", "source_reference", "color_reference"],
                    [{"start": "00:00:00.000", "end": "00:00:01.000", "source_reference": "", "color_reference": ""}],
                )
                rows = references.shot_rows(references.rel(manifest))
                self.assertEqual(len(rows), 1)
                self.assertFalse(rows[0]["masked_edit_available"])
        finally:
            state.APP = prev_app


if __name__ == "__main__":
    unittest.main()
