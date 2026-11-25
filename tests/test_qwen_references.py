from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_colorize_references as qwen  # noqa: E402

WORKFLOW = ROOT / "workflows" / "qwen_image_edit" / "Image Edit (Qwen 2511).json"


class Qwen2511ReferenceGraphTests(unittest.TestCase):
    def build(self, reference_names: list[str]) -> dict:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "grey.png"
            source.write_bytes(b"png")
            args = qwen.build_parser().parse_args(["--manifest", "x.csv", "--workflow", str(WORKFLOW), "--comfy-dir", temp])
            args.active_seed = 5
            args.active_reference_names = reference_names
            workflow = qwen.load_workflow(WORKFLOW)
            qwen.patch_qwen_model_backend(args, workflow)
            return qwen.patch_workflow(args, workflow, source, Path(temp) / "out.png", "colour it")

    def test_without_references_the_graph_is_unchanged(self) -> None:
        prompt = self.build([])
        encoders = [node for node in prompt.values() if node["class_type"] == "TextEncodeQwenImageEditPlus"]
        self.assertTrue(all("vae" in node["inputs"] and "image2" not in node["inputs"] for node in encoders))
        self.assertFalse(any(node["class_type"] == "ReferenceLatent" for node in prompt.values()))

    def test_references_are_seen_but_only_the_grey_frame_is_a_reference_picture(self) -> None:
        prompt = self.build(["arp_qwen_ref_images/a.png", "arp_qwen_ref_images/b.png"])
        encoders = {key: node for key, node in prompt.items() if node["class_type"] == "TextEncodeQwenImageEditPlus"}
        references = {key: node for key, node in prompt.items() if node["class_type"] == "ReferenceLatent"}
        loads = {key: node["inputs"]["image"] for key, node in prompt.items() if node["class_type"] == "LoadImage"}
        encode = next(key for key, node in prompt.items() if node["class_type"] == "VAEEncode")
        for node in encoders.values():
            self.assertNotIn("vae", node["inputs"])  # every image is vision tokens only
            self.assertEqual(loads[node["inputs"]["image2"][0]], "arp_qwen_ref_images/a.png")
            self.assertEqual(loads[node["inputs"]["image3"][0]], "arp_qwen_ref_images/b.png")
        self.assertEqual(len(references), len(encoders))
        for node in references.values():
            self.assertIn(node["inputs"]["conditioning"][0], encoders)
            self.assertEqual(node["inputs"]["latent"][0], encode)  # the grey frame's own latent
        # The sampler's conditioning now passes through the ReferenceLatent nodes.
        consumers = [node for node in prompt.values() if node["class_type"] == "FluxKontextMultiReferenceLatentMethod"]
        self.assertTrue(all(node["inputs"]["conditioning"][0] in references for node in consumers))


class QwenSemanticMatchingIntegrationTests(unittest.TestCase):
    def test_build_parser_options(self) -> None:
        parser = qwen.build_parser()
        args = parser.parse_args([
            "--manifest", "shots.csv",
            "--workflow", "workflow.json",
            "--reference-runner", "ddcolor_only",
            "--guided-backend", "qwen2511",
            "--film-id", "film_noir_1947",
            "--color-bible-db", "color_bible.db",
            "--confidence-threshold", "0.85",
        ])
        self.assertEqual(args.reference_runner, "ddcolor_only")
        self.assertEqual(args.guided_backend, "qwen2511")
        self.assertEqual(args.film_id, "film_noir_1947")
        self.assertEqual(args.color_bible_db, Path("color_bible.db"))
        self.assertEqual(args.confidence_threshold, 0.85)

        # Defaults
        default_args = parser.parse_args(["--manifest", "shots.csv", "--workflow", "workflow.json"])
        self.assertEqual(default_args.reference_runner, "guided_only")
        self.assertEqual(default_args.guided_backend, "qwen2511")
        self.assertIsNone(default_args.film_id)
        self.assertIsNone(default_args.color_bible_db)
        self.assertEqual(default_args.confidence_threshold, 0.8)

    def test_reference_images_for_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ref_path = Path(temp) / "ref_001.png"
            ref_path.write_bytes(b"image")
            rows = [
                {"source_reference": "shot0.png", "color_reference": str(ref_path), "_shot_index": "0"},
                {"source_reference": "shot1.png", "color_reference": "", "_shot_index": "1"},
            ]
            refs = qwen.reference_images_for(rows, index=1, count=1)
            self.assertEqual(refs, [ref_path])

    def test_reference_images_for_delegates_to_service(self) -> None:
        mock_service = unittest.mock.MagicMock()
        with tempfile.TemporaryDirectory() as temp:
            ref_path = Path(temp) / "bible_ref.png"
            ref_path.write_bytes(b"bible image")
            mock_service.get_shot_palette_assignments.return_value = [
                {
                    "assignment": {"approval_status": "approved", "confidence": 0.95},
                    "entity_type": "character",
                    "entity_id": "mickey",
                    "color_references": [
                        {
                            "ref_id": "ref_mic",
                            "approval_status": "approved",
                            "source_keyframe_path": str(ref_path),
                            "version": 1,
                        }
                    ],
                }
            ]
            mock_service.get_snapshot_hash.return_value = "sha256:abc"
            rows = [
                {"source_reference": "shot0.png", "color_reference": "c0.png", "_shot_index": "0"}
            ]
            refs = qwen.reference_images_for(
                rows,
                index=0,
                count=2,
                film_id="film_mickey",
                color_bible_service=mock_service,
                confidence_threshold=0.8,
            )
            self.assertEqual(refs, [ref_path])

    def test_resolve_shot_references_helper(self) -> None:
        from ai_remaster_gui.semantic_matching import SemanticMatchResult
        mock_service = unittest.mock.MagicMock()
        mock_service.get_shot_palette_assignments.return_value = []
        mock_service.get_snapshot_hash.return_value = "v1"
        rows = [{"source_reference": "shot0.png", "color_reference": "", "_shot_index": "0"}]
        result = qwen.resolve_shot_references(
            rows, index=0, count=2, film_id="film_test", color_bible_service=mock_service
        )
        self.assertIsInstance(result, SemanticMatchResult)
        self.assertEqual(result.color_bible_version, "v1")

    def test_signature_incorporates_provenance_and_scoped_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / "shot.png"
            src.write_bytes(b"frame")
            wf = Path(temp) / "wf.json"
            wf.write_text("{}", encoding="utf-8")
            parser = qwen.build_parser()
            args = parser.parse_args(["--manifest", "x.csv", "--workflow", str(wf)])
            sig = qwen.signature(
                args,
                workflow_path=wf,
                source_path=src,
                prompt="Colorize Mickey",
                shot_index=4,
                retrieved_ref_hashes={"ref_1": "v1_hash"},
                retrieved_ref_ids=["ref_1"],
                color_bible_version="sha256:cb_version",
            )
            self.assertIn("scoped_cache_key", sig)
            self.assertIn("retrieved_ref_ids", sig)
            self.assertEqual(sig["retrieved_ref_ids"], ["ref_1"])
            self.assertEqual(sig["color_bible_version"], "sha256:cb_version")
            self.assertEqual(sig["shot_index"], 4)

    def test_main_with_args_review_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            t = Path(temp)
            src = t / "shot00.png"
            src.write_bytes(b"shot00 bytes")
            dst = t / "shot00_color.png"
            manifest = t / "manifest.csv"
            manifest.write_text(
                "enabled,source_reference,color_reference\ntrue,"
                + str(src)
                + ","
                + str(dst)
                + "\n",
                encoding="utf-8",
            )
            wf = t / "workflow.json"
            wf.write_text("{}", encoding="utf-8")

            parser = qwen.build_parser()
            args = parser.parse_args([
                "--manifest", str(manifest),
                "--workflow", str(wf),
                "--film-id", "film_review_test",
                "--confidence-threshold", "0.85",
                "--dry-run",
            ])

            mock_service = unittest.mock.MagicMock()
            # Low confidence assignment -> triggers review gate
            mock_service.get_shot_palette_assignments.return_value = [
                {
                    "assignment": {"approval_status": "approved", "confidence": 0.60},
                    "entity_type": "character",
                    "entity_id": "donald",
                    "entity": {"name": "Donald"},
                    "color_references": [],
                }
            ]
            mock_service.get_snapshot_hash.return_value = "v1"

            with (
                unittest.mock.patch("ai_remaster_gui.color_bible_service.ColorBibleService", return_value=mock_service),
                unittest.mock.patch.object(qwen, "colorize_reference_shot") as mock_colorize,
                unittest.mock.patch.object(qwen, "render") as mock_render,
            ):
                ret = qwen.main_with_args(args)
                self.assertEqual(ret, 0)
                # Inference was skipped cleanly
                mock_colorize.assert_not_called()
                mock_render.assert_not_called()

            # Manifest CSV row was updated with qc_status='needs_review'
            from ai_remaster_gui.manifests import read_manifest
            rows = read_manifest(manifest)
            self.assertEqual(rows[0].get("qc_status"), "needs_review")

    def test_main_with_args_ddcolor_routing(self) -> None:
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            t = Path(temp)
            src = t / "shot01.png"
            Image.new("RGB", (8, 8), color=(128, 128, 128)).save(src)
            dst = t / "shot01_color.png"
            manifest = t / "manifest.csv"
            manifest.write_text(
                "enabled,source_reference,color_reference\ntrue,"
                + str(src)
                + ","
                + str(dst)
                + "\n",
                encoding="utf-8",
            )
            wf = t / "workflow.json"
            wf.write_text("{}", encoding="utf-8")

            parser = qwen.build_parser()
            args = parser.parse_args([
                "--manifest", str(manifest),
                "--workflow", str(wf),
                "--film-id", "film_ddcolor_test",
                "--reference-runner", "ddcolor_only",
                "--dry-run",
            ])

            mock_service = unittest.mock.MagicMock()
            mock_service.get_shot_palette_assignments.return_value = []
            mock_service.get_snapshot_hash.return_value = "v1"

            with (
                unittest.mock.patch("ai_remaster_gui.color_bible_service.ColorBibleService", return_value=mock_service),
                unittest.mock.patch.object(qwen, "colorize_reference_shot") as mock_colorize,
                unittest.mock.patch.object(qwen, "render") as mock_render,
            ):
                ret = qwen.main_with_args(args)
                self.assertEqual(ret, 0)
                mock_colorize.assert_called_once()
                mock_render.assert_not_called()


if __name__ == "__main__":
    unittest.main()
