"""Independent Light Deflicker, legacy migration and signed shared comparison."""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_remaster_gui import deflicker, denoise_review, denoise_playback, frame_inspector, project_io, runtime_settings, server
from ai_remaster_gui.commercial_policy import stage_issue
from ai_remaster_gui.runtime_settings import base_settings, migrate_deflicker_settings


class DeflickerMigrationTests(unittest.TestCase):
    def test_legacy_only_project_moves_to_an_independent_enabled_stage(self):
        old = {"global": {"cleanup": "true"}, "cleanup": {"light_deflicker": "true", "dearchive": "false",
               "deflicker_method": "ffmpeg_median", "deflicker_window": "7", "deflicker_strength": "0.4"}}
        untouched = copy.deepcopy(old)
        new = migrate_deflicker_settings(old)
        self.assertEqual(old, untouched)
        self.assertEqual(new["global"], {"cleanup": "false", "deflicker": "true"})
        self.assertEqual(new["deflicker"], {"method": "ffmpeg_median", "window": "7", "strength": "0.4"})
        self.assertNotIn("light_deflicker", new["cleanup"])
        self.assertEqual(migrate_deflicker_settings(new), new)

    def test_other_cleanup_and_explicit_new_stage_choices_are_preserved(self):
        old = {"global": {"cleanup": "true", "deflicker": "false"},
               "deflicker": {"method": "arp", "strength": "0.2"},
               "cleanup": {"light_deflicker": "true", "deflicker_method": "localized_stream",
                           "deflicker_strength": "0.8", "localized_confirm_film_experiment": "true", "devignette": "true"}}
        new = migrate_deflicker_settings(old)
        self.assertEqual(new["global"]["cleanup"], "true")
        self.assertEqual(new["global"]["deflicker"], "false")
        self.assertEqual(new["deflicker"]["method"], "arp")
        self.assertEqual(new["deflicker"]["strength"], "0.2")
        self.assertEqual(new["deflicker"]["localized_confirm_film_experiment"], "true")

    def test_disabled_legacy_cleanup_does_not_silently_enable_processing(self):
        old = {"global": {"cleanup": "false"}, "cleanup": {"light_deflicker": "true"}}
        self.assertEqual(migrate_deflicker_settings(old)["global"]["deflicker"], "false")

    def test_project_and_machine_settings_migrate_before_defaults_are_merged(self):
        old = {"global": {"source": "input/film.mkv", "cleanup": "true"},
               "cleanup": {"light_deflicker": "true", "dearchive": "false", "deflicker_method": "localized_stream",
                           "localized_confirm_film_experiment": "true", "deflicker_strength": "0.4"}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            from ai_remaster_gui import app
            with patch.object(app, "SETTINGS_FILE", path), patch.object(runtime_settings, "newest", return_value=None), patch.object(app, "newest", return_value=None):
                machine = runtime_settings.load_settings()
                path.write_text(json.dumps({"schema_version": 2, "settings": old}), encoding="utf-8")
                with patch.object(project_io, "load_settings", return_value=base_settings()):
                    project = project_io.read_project_file(path)
        for loaded in (machine, project):
            self.assertEqual(loaded["global"]["deflicker"], "true")
            self.assertEqual(loaded["global"]["cleanup"], "false")
            self.assertEqual(loaded["deflicker"]["method"], "localized_stream")
            self.assertEqual(loaded["deflicker"]["strength"], "0.4")
            self.assertEqual(loaded["deflicker"]["localized_confirm_film_experiment"], "true")
            self.assertNotIn("light_deflicker", loaded["cleanup"])

    def test_batch_validates_confirmation_and_never_accepts_browser_output_paths(self):
        settings = base_settings()
        settings["global"].update(source="input/film.mkv", deflicker="true")
        with patch.object(server.APP, "settings", settings), patch.object(server.APP, "ensure_pipeline_source"), \
             patch.object(server.APP, "process", None), patch.object(server.APP, "_start_denoise_matrix", return_value=(True, "started")) as start, \
             patch.object(Path, "is_file", return_value=True), \
             patch.object(frame_inspector, "_metadata", return_value={"width":64, "height":48, "frames":12, "fps":12}):
            ok, message = server.APP.run_deflicker_batch([{"method":"localized_research", "preset":"settings"}])
            self.assertFalse(ok)
            self.assertIn("confirmation", message)
            start.assert_not_called()
            settings["deflicker"]["localized_confirm_single_shot"] = "true"
            rows = [{"method":"localized_research", "preset":"settings", "output":"input/do-not-overwrite.mkv"}]
            ok, _ = server.APP.run_deflicker_batch(rows)
            self.assertTrue(ok)
            command = start.call_args.args[1]
            self.assertNotIn("--output", command)
            self.assertEqual(start.call_args.args[2], "deflicker")
            before = copy.deepcopy(settings)
            with patch.object(deflicker.backend, "run") as run, patch("sys.argv", command[3:]):
                deflicker.main()
            self.assertNotIn("output", run.call_args.kwargs)
            self.assertEqual(settings, before)

    def test_stage_order_routing_and_cleanup_never_apply_a_second_correction(self):
        settings = base_settings()
        settings["global"].update(source="input/source.mkv", denoise="true", deflicker="true", cleanup="true",
                                  stabilize="true", detail="true", expand_outpaint="false", colorize="false")
        settings["deflicker"].update(method="ffmpeg_am", window="7")
        settings["cleanup"].update(light_deflicker="true", deflicker_method="arp", dearchive="false", devignette="true")
        with patch.object(server.APP, "settings", settings):
            app = server.APP
            self.assertEqual([item.key for item in app.active_stages()], ["denoise", "deflicker", "cleanup", "stabilize", "detail"])
            flicker, cleanup = app.command_for("deflicker"), app.command_for("cleanup")
            self.assertEqual(flicker[flicker.index("--source")+1], app.denoise_output())
            self.assertEqual(cleanup[cleanup.index("--source")+1], app.deflicker_output())
            self.assertNotIn("--light-deflicker", cleanup)
            self.assertNotIn("--deflicker-method", cleanup)
            self.assertEqual(app.cleanup_comparison_state()["source"], app.deflicker_output())
            before = app.cleanup_output()
            settings["deflicker"]["window"] = "9"
            self.assertNotEqual(app.cleanup_output(), before)
            settings["global"]["cleanup"] = "false"
            self.assertEqual(app.cleaned_source_for_downstream(), app.deflicker_output())
            settings["global"]["deflicker"] = "false"
            self.assertEqual(app.cleaned_source_for_downstream(), app.denoise_output())

    def test_model_free_stage_is_permitted_by_the_commercial_model_guard(self):
        settings = base_settings()
        settings["global"]["commercial_model_guard"] = "true"
        self.assertEqual(stage_issue(settings, "deflicker"), "")

    def test_variant_and_parameter_validation_is_fail_closed(self):
        for rows in ([], None, [{"method":"bad"}], [{"method":"arp"}]*2, [{"method":[]}], [{"method":"arp","preset":"strong"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                deflicker.validate_variants(rows)
        for values in ({"method":"wrong"}, {"method":"arp","strength":"nan"}, {"method":"arp","max_correction":1},
                       {"method":"ffmpeg_am","window":"5.1"}, {"method":"ffmpeg_am","window":"4"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                deflicker.backend.options(values)


@unittest.skipUnless(server.local_tool("ffmpeg"), "FFmpeg required")
class DeflickerMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="arp_deflicker_stage_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.settings = base_settings()
        self.settings["global"].update(source=str(self.source), deflicker="true", expand_outpaint="false", colorize="false")
        self.settings["cloud"]["intermediate_format"] = "lossless"
        self.settings["deflicker"].update(localized_confirm_single_shot="true", localized_confirm_film_experiment="true")
        subprocess.run([server.local_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12:duration=1",
                        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=1", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(self.source)], check=True)
        self.app = SimpleNamespace(settings=self.settings, denoised_source_for_downstream=lambda:str(self.source))
        for change in (patch.object(deflicker.backend, "ROOT", self.root), patch.object(denoise_review, "ROOT", self.root),
                       patch.object(frame_inspector, "ROOT", self.root), patch.object(frame_inspector, "CACHE", self.root/"stills"),
                       patch.object(denoise_playback, "_CACHE", self.root/"reels")):
            change.start();self.addCleanup(change.stop)

    def render(self, method="arp"):
        values = {**self.settings["deflicker"], "method":method}
        return deflicker.backend.run(self.source, values, "lossless")

    def test_unsigned_denoise_is_neither_a_pipeline_input_nor_a_finished_output(self):
        output = self.root / 'intermediate' / 'denoised' / 'selected.mkv'
        output.parent.mkdir(parents=True)
        output.write_bytes(self.source.read_bytes())
        self.settings['global'].update(denoise='true', deflicker='false', cleanup='false',
            stabilize='false', detail='false', upscale='false', add_soundtrack='false')
        self.settings['denoise'].update(method='hqdn3d', preset='gentle')
        with patch.object(server.APP, 'settings', self.settings), patch.object(server.APP, 'denoise_output', return_value=str(output)):
            self.assertEqual(server.APP.existing_outputs('denoise'), [])
            self.assertNotEqual(server.APP.output_selection_state().get('kind'), 'denoised')
            with patch.object(server.APP, 'save'):
                server.APP.hydrate_stage_inputs()
            self.assertEqual(self.settings['stabilize']['input_video'], '')
            # Enable the downstream stage only after checking output selection.
            self.settings['global']['deflicker'] = 'true'
            ok, message = server.APP.run_stage('deflicker')
            self.assertFalse(ok)
            self.assertIn('Run Remove Noise first', message)
            with patch.object(server.APP, 'ensure_pipeline_source'):
                ok, message = server.APP.run_deflicker_batch([{'method': 'arp', 'preset': 'settings'}])
            self.assertFalse(ok)
            self.assertIn('Run Remove Noise first', message)

    def test_all_five_existing_algorithms_keep_frames_geometry_audio_and_reuse(self):
        original = self.source.read_bytes()
        for method in deflicker.METHODS:
            with self.subTest(method=method):
                output = self.render(method)
                self.assertTrue(output.is_file())
                source, checked = denoise_review.review_pair(self.app, str(output), "deflicker")
                self.assertEqual(source, self.source)
                self.assertEqual(checked, output)
                self.assertEqual(frame_inspector._metadata(output)["frames"], 12)
                stat = output.stat()
                self.assertEqual(self.render(method).stat().st_mtime_ns, stat.st_mtime_ns)
                probe = subprocess.run([server.local_tool("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(output)], capture_output=True, text=True, check=True)
                self.assertTrue(probe.stdout.strip())
        self.assertEqual(self.source.read_bytes(), original)

    def test_deflicker_rejects_a_result_when_its_selected_denoise_predecessor_is_unsigned(self):
        import denoise_video
        noise = self.root / 'intermediate' / 'denoised' / 'selected.mkv'
        denoise_video.run(self.source, noise, method='hqdn3d', preset='gentle',
            intermediate_profile='lossless', ffmpeg=server.local_tool('ffmpeg'))
        self.settings['global']['denoise'] = 'true'
        self.settings['denoise'].update(method='hqdn3d', preset='gentle')
        self.app.denoise_output = self.app.denoised_source_for_downstream = lambda: str(noise)
        with patch.object(denoise_review, '_OUTPUTS', noise.parent):
            output = deflicker.backend.run(noise, {**self.settings['deflicker'], 'method': 'ffmpeg_median'}, 'lossless')
            denoise_review.review_pair(self.app, str(output), 'deflicker')
            noise.with_suffix('.mkv.sig.json').unlink()
            with self.assertRaisesRegex(ValueError, 'provenance'):
                denoise_review.review_pair(self.app, str(output), 'deflicker')

    def test_shared_exact_frame_reel_and_difference_use_the_deflicker_input(self):
        output = self.render("ffmpeg_median")
        still = denoise_review.inspect_denoise(self.app, str(output), 6, "deflicker")
        self.assertEqual(still["stage"], "deflicker")
        self.assertEqual(still["frame"], 6)
        reel = denoise_playback.playback_segment(self.app, str(output), 6, stage="deflicker")
        self.assertEqual(reel["frames"], 12)
        self.assertEqual((reel["width"],reel["height"]), (64,48))
        self.assertEqual(reel["difference_basis"], "decoded_processing_input_and_result")
        alone = denoise_review.inspect_denoise(self.app, "", 6, "deflicker")
        self.assertEqual(alone["before"], still["before"])

    def test_wrong_stage_stale_parameters_and_unconfirmed_experiments_are_rejected(self):
        output = self.render("ffmpeg_median")
        with self.assertRaises(ValueError):
            denoise_review.review_pair(self.app, str(output), "denoise")
        self.settings["deflicker"]["window"] = "9"
        with self.assertRaises(ValueError):
            denoise_review.review_pair(self.app, str(output), "deflicker")
        for method,key in (("localized_research","localized_confirm_single_shot"),("localized_stream","localized_confirm_film_experiment")):
            values = {**self.settings["deflicker"], "method":method, key:"false"}
            with self.subTest(method=method), self.assertRaises(ValueError):
                deflicker.backend.run(self.source, values, "lossless")
        with self.assertRaises(ValueError):
            denoise_review.inspect_denoise(self.app, "", 0, "cleanup")

    def test_failed_regeneration_cannot_keep_an_old_ready_signature(self):
        output = self.render("ffmpeg_median")
        self.settings["deflicker"]["method"] = "ffmpeg_median"
        sidecar = output.with_suffix(output.suffix + ".sig.json")
        self.assertTrue(sidecar.is_file())
        with patch.object(deflicker.backend, "resumable_geometry", return_value=False), self.assertRaises(ValueError):
            deflicker.backend.run(self.source, {**self.settings["deflicker"], "method":"ffmpeg_median"}, "lossless", force=True)
        self.assertFalse(sidecar.exists())
        with self.assertRaises(ValueError):
            denoise_review.review_pair(self.app, str(output), "deflicker")
        with patch.object(server.APP, "settings", self.settings):
            self.assertFalse(server.APP.deflicker_output_ready())
            self.assertEqual(server.APP.existing_outputs("deflicker"), [])
            self.settings["global"]["cleanup"] = "true"
            ok, message = server.APP.run_stage("cleanup")
            self.assertFalse(ok)
            self.assertIn("Run Light Deflicker first", message)

    def test_fractional_fps_keeps_frame_mapping_in_all_methods_and_comparison(self):
        subprocess.run([server.local_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=64x48:rate=30000/1001:duration=1.001", "-c:v", "ffv1", str(self.source)], check=True)
        before = frame_inspector._metadata(self.source)
        for method in deflicker.METHODS:
            with self.subTest(method=method):
                output = self.render(method)
                after = frame_inspector._metadata(output)
                self.assertEqual(after["frames"], before["frames"])
                self.assertEqual(after["fps_fraction"], before["fps_fraction"])
        still = denoise_review.inspect_denoise(self.app, str(output), 29, "deflicker")
        self.assertEqual(still["frame"], 29)
        reel = denoise_playback.playback_segment(self.app, str(output), 29, stage="deflicker")
        self.assertEqual(reel["fps_fraction"], before["fps_fraction"])

    def test_content_reuse_refreshes_lineage_after_an_unchanged_source_is_touched(self):
        output = self.render("ffmpeg_median")
        before = output.stat().st_mtime_ns
        stat = self.source.stat()
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with self.assertRaises(ValueError):
            denoise_review.review_pair(self.app, str(output), "deflicker")
        self.render("ffmpeg_median")
        self.assertEqual(output.stat().st_mtime_ns, before)
        denoise_review.review_pair(self.app, str(output), "deflicker")
