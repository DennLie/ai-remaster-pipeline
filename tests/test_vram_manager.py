from __future__ import annotations

import gc
import io
import json
import sys
import threading
import types
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_free_comfy_vram_interrupts_frees_and_measures_freed_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_remaster_gui import vram_manager

    calls: list[tuple[str, dict | None]] = []
    vram_samples = iter([
        11_600 * 1024 * 1024,  # before
        400 * 1024 * 1024,     # after poll 1
    ])

    def fake_stats(_comfy_url: str, timeout: float = 1.0) -> dict:
        val = next(vram_samples, 400 * 1024 * 1024)
        return {"devices": [{"type": "cuda", "vram_total": 24_564 * 1024 * 1024, "torch_vram_total": val}]}

    def fake_post(url: str, payload: dict | None = None, timeout: float = 10.0) -> bool:
        calls.append((url, payload))
        return True

    monkeypatch.setattr(vram_manager, "comfy_system_stats", fake_stats)
    monkeypatch.setattr(vram_manager, "_post_comfy_empty", fake_post)
    monkeypatch.setattr(vram_manager.time, "sleep", lambda _s: None)

    result = vram_manager.free_comfy_vram(
        "http://127.0.0.1:8188",
        interrupt=True,
        verify_timeout=1.0,
        settle_seconds=0.0,
    )
    assert calls[0] == ("http://127.0.0.1:8188/interrupt", {})
    assert calls[1] == ("http://127.0.0.1:8188/free", {"unload_models": True, "free_memory": True})
    assert result["reachable"] is True
    assert result["interrupted"] is True
    assert result["freed_bytes"] == 11_200 * 1024 * 1024


def test_unload_inprocess_models_clears_sam2_and_defect_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_remaster_gui import sam_masks, vram_manager
    from scripts import neural_defects_adapter

    events: list[str] = []

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def to(self, device: str):
            events.append(f"to:{self.name}:{device}")
            return self

        def cpu(self):
            events.append(f"cpu:{self.name}")
            return self

    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            events.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            events.append("empty_cache")

        @staticmethod
        def ipc_collect() -> None:
            events.append("ipc_collect")

    fake_torch = types.SimpleNamespace(cuda=FakeTorchCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(sam_masks, "_PREDICTOR", types.SimpleNamespace(model=FakeModel("sam2")))

    seg_adapter = neural_defects_adapter.FilmArtefactSegmenterAdapter()
    seg_adapter._cached_model = FakeModel("seg")
    cnn_adapter = neural_defects_adapter.CNNBlotchesDetectorAdapter()
    cnn_adapter._cached_model = FakeModel("cnn")

    res = vram_manager.unload_inprocess_models()
    assert res["cuda_cache_cleared"] is True
    assert "sam2_image_predictor" in res["unloaded"]
    assert sam_masks._PREDICTOR is None
    assert seg_adapter._cached_model is None
    assert cnn_adapter._cached_model is None
    assert "to:sam2:cpu" in events
    assert "to:seg:cpu" in events
    assert "to:cnn:cpu" in events
    assert "empty_cache" in events


def test_shot_video_tracker_close_drops_owned_predictor_before_empty_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.sam2_video_tracker import ShotVideoTracker

    tracker = object.__new__(ShotVideoTracker)
    tracker.checkpoint = "sam2.1_hiera_large.pt"
    tracker.device = "cuda"
    tracker.inference_state = None
    tracker._temp_dir = None
    tracker._registered_objects = {}
    tracker._empty_background_objects = set()
    order: list[str] = []

    class FakePredictor:
        def to(self, device: str):
            order.append(f"predictor.to:{device}")
            return self

    tracker.predictor = FakePredictor()
    tracker._owns_predictor = True

    monkeypatch.setattr(gc, "collect", lambda: order.append("gc.collect") or 0)
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: order.append("cuda.empty_cache"),
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    tracker.close()
    assert tracker.predictor is None
    # close() runs _cleanup_shot() first (with predictor=None? No, predictor dropped after _cleanup_shot),
    # then moves owned predictor to CPU & clears self.predictor BEFORE its final gc.collect -> empty_cache.
    assert "predictor.to:cpu" in order
    cpu_idx = order.index("predictor.to:cpu")
    gc_after = order.index("gc.collect", cpu_idx)
    empty_after = order.index("cuda.empty_cache", gc_after)
    assert cpu_idx < gc_after < empty_after

    # External predictor injected by caller should NOT be cleared by close().
    ext = FakePredictor()
    tracker2 = object.__new__(ShotVideoTracker)
    tracker2.predictor = ext
    tracker2._owns_predictor = False
    tracker2.close()
    assert tracker2.predictor is ext


def test_system_status_cuda_probe_avoids_importing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_remaster_gui import system_status, vram_manager

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(
        system_status,
        "nvidia_smi_cuda_probe",
        lambda: {
            "available": True,
            "name": "NVIDIA GeForce RTX 4090",
            "capability": (8, 9),
            "cuda": "12.8",
            "driver": "572.60",
            "detail": "NVIDIA GeForce RTX 4090 (sm_89)",
        },
    )
    monkeypatch.setattr(vram_manager, "comfy_torch_vram_bytes", lambda _url, timeout=0.4: 11 * 1024 * 1024 * 1024)
    monkeypatch.setattr(vram_manager, "active_inprocess_models", lambda: ["sam2_image_predictor"])

    info = system_status.cuda_status()
    assert info["available"] is True
    assert info["capability"] == "8.9"
    assert "torch" not in sys.modules

    vram_payload = {
        "used": 12_000_000_000,
        "total": 24_000_000_000,
        "detail": "12.0 / 24.0 GB",
    }
    system_status._attach_vram_attribution(vram_payload)
    assert vram_payload["comfy_vram_used"] == 11 * 1024 * 1024 * 1024
    assert vram_payload["arp_inprocess_models"] == ["sam2_image_predictor"]
    assert "ComfyUI " in vram_payload["detail"]
    assert "ARP cached: sam2_image_predictor" in vram_payload["detail"]


def test_server_collect_output_and_stop_release_vram(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ai_remaster_gui import server, vram_manager

    app = object.__new__(server.PipelineApp)
    app.lock = threading.Lock()
    app.log = server.ConsoleLog()
    app.quitting = False
    app.running_stage = "Reference Generation"
    app.running_stage_key = "references"
    app.running_reference_manifest = ""
    app.running_reference_index = None
    app.run_started_at = 1.0
    app.settings = {
        "global": {"auto_free_vram": "true"},
        "references": {"method": "qwen", "compute": "local"},
        "cloud": {},
    }
    monkeypatch.setattr(app, "save", lambda: None)
    monkeypatch.setattr(app, "hydrate_stage_inputs", lambda _key: None)
    monkeypatch.setattr(app, "_record_stage_compliance", lambda _key: None)
    monkeypatch.setattr(server, "release_keep_awake", lambda: None)
    monkeypatch.setattr(server, "invalidate_system_status_cache", lambda: None)

    class FakeProc:
        def __init__(self, code: int | None = 0) -> None:
            self.stdout = io.StringIO("line1\n")
            self.returncode = code
            self.args = ["python", "qwen_colorize_references.py"]

        def poll(self):
            return None if self.returncode is None else self.returncode

        def wait(self) -> int:
            return 0 if self.returncode is None else self.returncode

    freed_calls: list[dict] = []
    monkeypatch.setattr(
        vram_manager,
        "free_comfy_vram",
        lambda url, **kw: freed_calls.append({"url": url, **kw}) or {"freed_bytes": 2048 * 1024 * 1024, "reachable": True},
    )

    # 1. Interactive single-still regeneration skips auto-free on exit
    app.process = FakeProc(0)
    app._collect_output("references", "", auto_free_vram_on_exit=False)
    assert freed_calls == []

    # 2. Full batch ComfyUI stage auto-frees on exit
    app.running_stage_key = "references"
    app.process = FakeProc(0)
    app._collect_output("references", "", auto_free_vram_on_exit=True)
    assert len(freed_calls) == 1
    assert any("Unloaded ComfyUI models after references" in line for line in app.log[-10:])

    # 3. Stop() interrupts ComfyUI and unloads in-process models in background
    done = threading.Event()
    stop_calls: list[dict] = []

    def fake_release_all(**kw):
        stop_calls.append(kw)
        done.set()
        return {"comfy": {"freed_bytes": 4096 * 1024 * 1024}, "inprocess": {"unloaded": ["sam2_image_predictor"]}}

    app.running_stage_key = "outpaint"
    app.process = FakeProc(None)
    monkeypatch.setattr(server, "terminate_process_tree", lambda _proc: None)
    monkeypatch.setattr(vram_manager, "release_all_vram", fake_release_all)
    app.stop()
    assert done.wait(timeout=3.0)
    assert stop_calls and stop_calls[0]["interrupt_comfy"] is True


def test_http_api_free_vram_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_remaster_gui import http_handler, state

    class FakeApp:
        def __init__(self) -> None:
            self.process = None
            self.calls: list[tuple[bool, bool]] = []

        def free_vram(self, *, interrupt_comfy: bool = False, include_ollama: bool = True) -> dict:
            self.calls.append((interrupt_comfy, include_ollama))
            return {"comfy": {"reachable": True, "freed_bytes": 5120 * 1024 * 1024}, "freed_mb": 5120.0}

        def state(self, view: str = "") -> dict:
            return {"running": False, "view": view}

    fake_app = FakeApp()
    monkeypatch.setattr(state, "APP", fake_app)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), http_handler.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        # While running without force -> rejected
        fake_app.process = types.SimpleNamespace(poll=lambda: None)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/free-vram",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is False
        assert "Stop it first" in body["error"]

        # While idle -> succeeds
        fake_app.process = None
        req2 = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/free-vram",
            data=json.dumps({"view": "colour"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=5) as resp2:
            body2 = json.loads(resp2.read().decode("utf-8"))
        assert body2["ok"] is True
        assert body2["freed_mb"] == 5120.0
        assert fake_app.calls == [(False, True)]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ollama_calls_send_keep_alive_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import audio_models, qwen_colorize_references

    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    captured_qwen: list[dict] = []
    monkeypatch.setattr(
        qwen_colorize_references,
        "http_json",
        lambda _method, _url, payload, timeout=180: captured_qwen.append(payload) or {"response": "a man"},
    )
    args = types.SimpleNamespace(
        ollama_vision_model="gemma3:4b",
        reference_description_prompt="Describe",
        ollama_url="http://127.0.0.1:11434",
    )
    assert qwen_colorize_references.describe_with_ollama(args, img) == "a man"
    assert captured_qwen[0]["keep_alive"] == 0

    captured_audio: list[dict] = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self) -> bytes:
            return json.dumps({"response": "quiet room"}).encode("utf-8")

    def fake_urlopen(req, timeout=180):
        captured_audio.append(json.loads(req.data.decode("utf-8")))
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert audio_models.run_ollama_caption("http://127.0.0.1:11434", "gemma3:4b", image_path=img, question="What?") == "quiet room"
    assert captured_audio[0]["keep_alive"] == 0
