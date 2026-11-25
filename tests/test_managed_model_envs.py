"""CPU-only unit tests for Managed Isolated Model Environments & Pinned Subprocess Runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from ai_remaster_gui import http_handler, server, state
from ai_remaster_gui.commercial_policy import CommercialPolicyError
from ai_remaster_gui.image_segmenters import DEFAULT_IMAGE_SEGMENTER_REGISTRY, run_keyframe_segmentation
from ai_remaster_gui.managed_model_envs import (
    MANAGED_ENV_ACTIVE_VAR,
    MODEL_ENV_SPECS,
    OFFICIAL_SAM2_GIT_URL,
    TORCH_CPU_INDEX_URL,
    TORCH_CUDA_INDEX_URL,
    build_sanitized_subprocess_env,
    env_for_model,
    env_python_executable,
    inspect_all_envs,
    inspect_env,
    install_env,
    is_managed_env_active,
    remove_env,
    run_in_managed_env,
)
from ai_remaster_gui.rle_masks import MaskObjectMetadata, ShotMaskArchive
from ai_remaster_gui.runtime_settings import base_settings
from ai_remaster_gui.video_segmenters import DEFAULT_SEGMENTER_REGISTRY, run_shot_segmentation
import scripts.ddcolor_reference as ddcolor_script
import scripts.setup_model_env as setup_cli


class _EnvHttpHandler(http_handler.Handler):
    def __init__(self, path: str, body: dict | None = None, headers: dict[str, str] | None = None) -> None:
        self.path = path
        self.headers = headers or {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        self._body = body or {}
        self.responses: dict[str, object] = {}

    def read_json(self) -> dict:
        return self._body

    def send_json(self, payload):  # noqa: D102
        self.responses["json"] = payload

    def send_error(self, code, message=None, explain=None):  # noqa: D102
        self.responses["error"] = code


def _create_fake_ready_env(base_dir: Path, env_name: str, *, torch_flavor: str = "cpu") -> Path:
    env_dir = base_dir / env_name
    py_exe = env_python_executable(env_name, base_dir=base_dir)
    py_exe.parent.mkdir(parents=True, exist_ok=True)
    py_exe.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "env_name": env_name,
        "status": "ready",
        "python_version": MODEL_ENV_SPECS[env_name].python_version,
        "packages": list(MODEL_ENV_SPECS[env_name].packages),
        "torch_flavor": torch_flavor,
        "installer": "uv",
        "installed_at": "2026-11-25T14:00:00+00:00",
        "error": "",
    }
    (env_dir / "env_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return py_exe


class TestEnvRegistryCoverage(unittest.TestCase):
    """(a) All 5 env groups cover DDColor + 10 image segmenters + 9 video trackers; official SAM2 URL only."""

    def test_five_envs_cover_all_models_and_use_official_sam2(self) -> None:
        self.assertEqual(
            set(MODEL_ENV_SPECS.keys()),
            {"transformers_seg", "sam2_family", "ddcolor_env", "vos_matting", "noncommercial_research"},
        )

        # Every image segmenter and video tracker + ddcolor maps cleanly
        for spec in DEFAULT_IMAGE_SEGMENTER_REGISTRY.list_specs():
            with self.subTest(image_segmenter=spec.name):
                self.assertIsNotNone(env_for_model(spec.name))
        for spec in DEFAULT_SEGMENTER_REGISTRY.list_specs():
            with self.subTest(video_segmenter=spec.name):
                self.assertIsNotNone(env_for_model(spec.name))
        self.assertEqual(env_for_model("ddcolor"), "ddcolor_env")

        # Official facebookresearch/sam2 git URL required, never bare PyPI 'sam2'
        sam2_pkgs = MODEL_ENV_SPECS["sam2_family"].packages
        self.assertIn(OFFICIAL_SAM2_GIT_URL, sam2_pkgs)
        self.assertNotIn("sam2", sam2_pkgs)
        self.assertFalse(any(p.strip().lower().startswith("sam2==") for p in sam2_pkgs))

        # Strict non-commercial isolation
        nc_spec = MODEL_ENV_SPECS["noncommercial_research"]
        self.assertFalse(nc_spec.commercial_safe)
        self.assertEqual(set(nc_spec.models), {"sam3", "sam3_image", "sam31", "sam31_image"})


class TestInstallerUvAndVenvFallback(unittest.TestCase):
    """(b) Prefer uv when available, fall back cleanly to python -m venv, and handle CLI/manifests."""

    def test_install_prefers_uv_then_falls_back_to_venv_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "model_envs"

            # 1. Dry-run with uv present
            with mock.patch("ai_remaster_gui.managed_model_envs.shutil.which", return_value="uv.exe"), \
                 mock.patch("ai_remaster_gui.managed_model_envs.subprocess.run") as mock_run:
                plan = install_env("sam2_family", cuda=True, dry_run=True, base_dir=base)
            mock_run.assert_not_called()
            self.assertEqual(plan["installer"], "uv")
            self.assertEqual(plan["python_version"], "3.11")
            self.assertEqual(plan["torch_flavor"], "cuda")
            self.assertIn(TORCH_CUDA_INDEX_URL, plan["commands"][1])
            self.assertIn(OFFICIAL_SAM2_GIT_URL, plan["commands"][2])
            self.assertNotIn("sam2", plan["commands"][2])

            # 2. Full mocked install with uv -> creates ready env_manifest.json
            def _fake_uv_run(cmd, **_kwargs):
                if len(cmd) >= 2 and cmd[1] == "venv":
                    py = env_python_executable("transformers_seg", base_dir=base)
                    py.parent.mkdir(parents=True, exist_ok=True)
                    py.write_text("", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch("ai_remaster_gui.managed_model_envs.shutil.which", return_value="uv"), \
                 mock.patch("ai_remaster_gui.managed_model_envs.subprocess.run", side_effect=_fake_uv_run):
                info = install_env("transformers_seg", cuda=False, base_dir=base)
            self.assertEqual(info["status"], "ready")
            self.assertTrue(info["ready"])
            self.assertEqual(info["torch_flavor"], "cpu")
            self.assertEqual(info["installer"], "uv")

            # 3. Fallback to python -m venv when uv is missing
            def _fake_venv_run(cmd, **_kwargs):
                if "-m" in cmd and "venv" in cmd:
                    py = env_python_executable("ddcolor_env", base_dir=base)
                    py.parent.mkdir(parents=True, exist_ok=True)
                    py.write_text("", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch("ai_remaster_gui.managed_model_envs.shutil.which", return_value=None), \
                 mock.patch("ai_remaster_gui.managed_model_envs.subprocess.run", side_effect=_fake_venv_run) as run_venv:
                info_dd = install_env("ddcolor_env", cuda=False, base_dir=base)
            self.assertEqual(info_dd["installer"], "venv+pip")
            self.assertEqual(info_dd["status"], "ready")
            cmds = [c.args[0] for c in run_venv.call_args_list]
            self.assertIn("venv", cmds[0])
            self.assertIn("--isolated", cmds[1])
            self.assertIn(TORCH_CPU_INDEX_URL, cmds[1])

            # 4. Failed install records status=error
            with mock.patch("ai_remaster_gui.managed_model_envs.shutil.which", return_value=None), \
                 mock.patch("ai_remaster_gui.managed_model_envs.subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")):
                with self.assertRaises(subprocess.CalledProcessError):
                    install_env("vos_matting", base_dir=base)
            err_info = inspect_env("vos_matting", base_dir=base)
            self.assertEqual(err_info["status"], "error")
            self.assertFalse(err_info["ready"])

            # 5. CLI status + remove
            rc_status = setup_cli.main(["status", "--base-dir", str(base)])
            self.assertEqual(rc_status, 0)
            rc_remove = setup_cli.main(["remove", "ddcolor_env", "--base-dir", str(base)])
            self.assertEqual(rc_remove, 0)
            self.assertEqual(inspect_env("ddcolor_env", base_dir=base)["status"], "missing")


class TestNonCommercialResearchPolicyIsolation(unittest.TestCase):
    """(c) noncommercial_research hard-blocked in Commercial Mode; unlocked only via benchmark override."""

    def test_noncommercial_env_requires_explicit_override_and_guard_off(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "model_envs"
            _create_fake_ready_env(base, "noncommercial_research")

            guard_on = base_settings()  # commercial_model_guard="true"
            with self.assertRaises(CommercialPolicyError):
                install_env("noncommercial_research", dry_run=True, settings=guard_on, base_dir=base)
            with self.assertRaises(CommercialPolicyError):
                run_in_managed_env("noncommercial_research", "scripts/segment_keyframe_mask.py", settings=guard_on, base_dir=base)

            # Even guard="false" without non_commercial_benchmark_override="true" must fail closed
            guard_off_no_override = {
                "global": {"commercial_model_guard": "false", "non_commercial_benchmark_override": "false"}
            }
            with self.assertRaises(CommercialPolicyError):
                install_env("noncommercial_research", dry_run=True, settings=guard_off_no_override, base_dir=base)

            # CLI without --allow-non-commercial-benchmark exits with code 2
            rc_blocked = setup_cli.main(["install", "noncommercial_research", "--dry-run", "--base-dir", str(base)])
            self.assertEqual(rc_blocked, 2)

            # Unlocked only with commercial_model_guard="false" + non_commercial_benchmark_override="true"
            unlocked = {
                "global": {"commercial_model_guard": "false", "non_commercial_benchmark_override": "true"}
            }
            plan = install_env("noncommercial_research", dry_run=True, settings=unlocked, base_dir=base)
            self.assertTrue(plan["ok"])

            with mock.patch("ai_remaster_gui.managed_model_envs.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout="ok", stderr="")):
                proc = run_in_managed_env(
                    "noncommercial_research",
                    "scripts/segment_keyframe_mask.py",
                    ["--list-segmenters"],
                    settings=unlocked,
                    base_dir=base,
                )
            self.assertEqual(proc.returncode, 0)


class TestSubprocessEnvSanitizationAndRecursionGuard(unittest.TestCase):
    """(d) Strips VIRTUAL_ENV/PYTHONPATH/host venv PATH entries and enforces ARP_MANAGED_ENV_ACTIVE=1 guard."""

    def test_sanitized_env_and_recursion_guard(self) -> None:
        host_scripts = str(Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin"))
        fake_env = {
            "VIRTUAL_ENV": r"I:\host\.venv",
            "PYTHONHOME": r"C:\PythonOld",
            "PYTHONPATH": r"I:\host\.venv\Lib\site-packages",
            "UV_PROJECT_ENVIRONMENT": r"I:\other\.venv",
            "UV_PYTHON": "3.13",
            "PATH": os.pathsep.join([host_scripts, r"C:\Windows\System32"]),
            "SYSTEMROOT": r"C:\Windows",
        }
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            clean = build_sanitized_subprocess_env(repo_root=repo, base_env=fake_env, env_name="sam2_family")

            self.assertNotIn("VIRTUAL_ENV", clean)
            self.assertNotIn("PYTHONHOME", clean)
            self.assertNotIn("UV_PROJECT_ENVIRONMENT", clean)
            self.assertNotIn("UV_PYTHON", clean)
            self.assertEqual(clean["PYTHONPATH"], str(repo.resolve()))
            self.assertEqual(clean[MANAGED_ENV_ACTIVE_VAR], "1")
            self.assertEqual(clean["ARP_MANAGED_ENV_NAME"], "sam2_family")
            self.assertEqual(clean["PYTHONNOUSERSITE"], "1")
            if sys.prefix != sys.base_prefix:
                self.assertNotIn(host_scripts, clean["PATH"])

            base = Path(td) / "model_envs"
            _create_fake_ready_env(base, "sam2_family")
            with mock.patch.dict(os.environ, {MANAGED_ENV_ACTIVE_VAR: "1"}):
                self.assertTrue(is_managed_env_active())
                with self.assertRaisesRegex(RuntimeError, "Recursive managed environment"):
                    run_in_managed_env("sam2_family", "scripts/track_shot_masks.py", base_dir=base)


class TestTransparentSubprocessDispatch(unittest.TestCase):
    """(e) Transparent dispatch from main .venv to managed env when local imports fail."""

    def test_keyframe_segmentation_dispatches_to_managed_env_or_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            img_path = tmp_path / "still.png"
            out_mask = tmp_path / "out_mask.png"
            Image.fromarray(np.zeros((24, 32, 3), dtype=np.uint8)).save(img_path)

            # 1. Missing both locally and in managed env -> actionable RuntimeError
            with mock.patch("ai_remaster_gui.image_segmenters._module_available", return_value=False), \
                 mock.patch("ai_remaster_gui.image_segmenters.inspect_env", return_value={"ready": False, "status": "missing"}):
                with self.assertRaisesRegex(RuntimeError, r"setup_model_env\.py install transformers_seg"):
                    run_keyframe_segmentation(
                        image=img_path,
                        output_mask_path=out_mask,
                        segmenter="birefnet",
                    )

            # 2. Local missing + managed env ready -> dispatches cleanly
            def _fake_managed_run(env_name, script_path, argv, **_kwargs):
                self.assertEqual(env_name, "transformers_seg")
                self.assertTrue(str(script_path).endswith("segment_keyframe_mask.py"))
                mask_arr = np.full((24, 32), 255, dtype=np.uint8)
                Image.fromarray(mask_arr, mode="L").save(out_mask, format="PNG")
                out_mask.with_suffix(".json").write_text(
                    json.dumps({"foreground_pixels": 24 * 32, "width": 32, "height": 24, "labels": ["foreground"]}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess([], 0, stdout="{}", stderr="")

            with mock.patch("ai_remaster_gui.image_segmenters._module_available", return_value=False), \
                 mock.patch("ai_remaster_gui.image_segmenters.inspect_env", return_value={"ready": True, "status": "ready"}), \
                 mock.patch("ai_remaster_gui.image_segmenters.run_in_managed_env", side_effect=_fake_managed_run) as dispatch:
                res = run_keyframe_segmentation(
                    image=img_path,
                    output_mask_path=out_mask,
                    segmenter="birefnet",
                )
            dispatch.assert_called_once()
            self.assertEqual(res["runtime"], "managed_env:transformers_seg")
            self.assertEqual(res["foreground_pixels"], 24 * 32)
            self.assertTrue(out_mask.is_file())

    def test_video_segmentation_and_ddcolor_dispatch_to_managed_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            out_npz = tmp_path / "shot_0000.masks.npz"
            frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]
            seed_mask = np.full((16, 16), 255, dtype=np.uint8)

            def _fake_video_dispatch(env_name, script_path, argv, **_kwargs):
                self.assertEqual(env_name, "vos_matting")
                self.assertTrue(str(script_path).endswith("track_shot_masks.py"))
                prompts_idx = argv.index("--prompts-json") + 1
                prompts_data = json.loads(Path(argv[prompts_idx]).read_text(encoding="utf-8"))
                self.assertTrue(Path(prompts_data["1"]["mask_path"]).is_file())
                arch = ShotMaskArchive(
                    shot_id="shot_0000",
                    frame_count=2,
                    height=16,
                    width=16,
                    objects=[MaskObjectMetadata(obj_id=1, name="obj_1", entity_type="object")],
                )
                arch.set_mask(0, 1, seed_mask)
                arch.save_npz(out_npz)
                return subprocess.CompletedProcess([], 0, stdout="{}", stderr="")

            with mock.patch("ai_remaster_gui.video_segmenters._video_backend_locally_available", return_value=False), \
                 mock.patch("ai_remaster_gui.video_segmenters.inspect_env", return_value={"ready": True, "status": "ready"}), \
                 mock.patch("ai_remaster_gui.video_segmenters.run_in_managed_env", side_effect=_fake_video_dispatch):
                produced = run_shot_segmentation(
                    shot_id="shot_0000",
                    frames=frames,
                    prompts={1: {"frame_idx": 0, "mask": seed_mask}},
                    output_npz_path=out_npz,
                    segmenter="cutie",
                )
            self.assertEqual(produced, out_npz)
            loaded = ShotMaskArchive.load_npz(out_npz)
            np.testing.assert_array_equal(loaded.get_mask(0, 1), seed_mask)

            # DDColor dispatch when ddcolor pkg missing locally & ddcolor_env ready
            src_img = tmp_path / "dd_in.png"
            dst_img = tmp_path / "dd_out.png"
            ckpt = tmp_path / "ddcolor.pth"
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(src_img)
            ckpt.write_bytes(b"weights")

            def _fake_ddcolor_dispatch(env_name, script_path, argv, **_kwargs):
                self.assertEqual(env_name, "ddcolor_env")
                self.assertTrue(str(script_path).endswith("ddcolor_reference.py"))
                Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(dst_img)
                return subprocess.CompletedProcess([], 0, stdout="", stderr="")

            with mock.patch.object(ddcolor_script, "_ddcolor_package_importable", return_value=False), \
                 mock.patch.object(ddcolor_script, "inspect_managed_env", return_value={"ready": True, "status": "ready"}), \
                 mock.patch.object(ddcolor_script, "run_in_managed_env", side_effect=_fake_ddcolor_dispatch) as dd_run:
                res_path = ddcolor_script.colorize_reference_shot(
                    src_img,
                    dst_img,
                    model_path=ckpt,
                    dry_run=False,
                    settings=base_settings(),
                    enforce_policy=True,
                )
            dd_run.assert_called_once()
            self.assertEqual(res_path, dst_img)
            self.assertTrue(dst_img.is_file())


class TestManagedEnvHttpRoutes(unittest.TestCase):
    """(f) GET /api/model-envs, POST /api/model-envs/install, GET /api/segmenters."""

    def setUp(self) -> None:
        self._real_app = state.APP
        self.app = mock.MagicMock(spec=server.PipelineApp)
        self.app.log = []
        self.app.settings = base_settings()
        state.APP = self.app

    def tearDown(self) -> None:
        state.APP = self._real_app

    def test_routes_enforce_security_policy_and_report_managed_env_runtime(self) -> None:
        # 1. Host / Origin enforcement
        bad_host = _EnvHttpHandler("/api/model-envs", headers={"Host": "evil.example"})
        bad_host.do_GET()
        self.assertEqual(bad_host.responses.get("error"), 403)

        bad_origin = _EnvHttpHandler(
            "/api/model-envs/install",
            {"env_name": "sam2_family"},
            headers={"Host": "127.0.0.1:8765", "Origin": "https://evil.example"},
        )
        bad_origin.do_POST()
        self.assertEqual(bad_origin.responses.get("error"), 403)

        # 2. GET /api/model-envs returns all 5 envs
        h_get = _EnvHttpHandler("/api/model-envs")
        h_get.do_GET()
        payload = h_get.responses["json"]
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["commercial_guard_enabled"])
        self.assertEqual(len(payload["envs"]), 5)
        self.assertEqual(payload["model_to_env"]["ddcolor"], "ddcolor_env")

        # 3. POST /api/model-envs/install blocks noncommercial_research when Commercial Guard is ON
        h_nc_block = _EnvHttpHandler("/api/model-envs/install", {"env_name": "noncommercial_research"})
        h_nc_block.do_POST()
        self.assertFalse(h_nc_block.responses["json"]["ok"])
        self.assertIn("Commercial Model Guard", h_nc_block.responses["json"]["error"])

        # 4. POST /api/model-envs/install succeeds for commercial env & dispatches cleanly
        self.app.install_model_env.return_value = (True, "Started Install model env: sam2_family", {"env_name": "sam2_family", "async": True})
        h_ok = _EnvHttpHandler("/api/model-envs/install", {"env_name": "sam2_family"})
        h_ok.do_POST()
        self.assertTrue(h_ok.responses["json"]["ok"])
        self.assertEqual(h_ok.responses["json"]["env_name"], "sam2_family")

        # 5. GET /api/segmenters reports managed_env runtime when env is ready
        _inspect_side_effect = lambda name, **_kw: {
            "status": "ready" if name == "sam2_family" else "missing",
            "ready": name == "sam2_family",
        }
        with mock.patch("ai_remaster_gui.image_segmenters._module_available", return_value=False), \
             mock.patch("ai_remaster_gui.video_segmenters._video_backend_locally_available", return_value=False), \
             mock.patch("ai_remaster_gui.image_segmenters.inspect_env", side_effect=_inspect_side_effect), \
             mock.patch("ai_remaster_gui.video_segmenters.inspect_env", side_effect=_inspect_side_effect):
            h_seg = _EnvHttpHandler("/api/segmenters")
            h_seg.do_GET()
        seg_payload = h_seg.responses["json"]
        self.assertTrue(seg_payload["ok"])
        img_map = {item["name"]: item for item in seg_payload["image_segmenters"]}
        vid_map = {item["name"]: item for item in seg_payload["video_trackers"]}
        self.assertTrue(img_map["evf_sam2"]["available"])
        self.assertEqual(img_map["evf_sam2"]["runtime"], "managed_env:sam2_family")
        self.assertEqual(img_map["evf_sam2"]["env_name"], "sam2_family")
        self.assertTrue(vid_map["sam21"]["available"])
        self.assertEqual(vid_map["sam21"]["runtime"], "managed_env:sam2_family")
        self.assertFalse(img_map["birefnet"]["available"])
        self.assertEqual(img_map["birefnet"]["runtime"], "missing")


if __name__ == "__main__":
    unittest.main()
