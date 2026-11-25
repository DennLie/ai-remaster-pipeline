"""Centralized GPU/VRAM lifecycle management for ARP.

Handles safe unloading across all three sources of VRAM allocation:
1. ComfyUI server models & cache (via ``/interrupt``, ``/free``, ``/system_stats``).
2. In-process PyTorch/CUDA model caches inside the GUI server (SAM 2.1 predictor
   and cached neural defect detector adapters), without importing ``torch`` unless
   it is already loaded.
3. Local Ollama vision captioning models (via ``/api/ps`` and ``keep_alive: 0``).
"""

from __future__ import annotations

import gc
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

_VRAM_LOCK = threading.Lock()


def comfy_system_stats(comfy_url: str, timeout: float = 1.0) -> dict[str, Any] | None:
    """Query ComfyUI ``/system_stats`` cleanly, returning None if unreachable."""
    if not comfy_url:
        return None
    url = f"{comfy_url.rstrip('/')}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError):
        return None


def comfy_torch_vram_bytes(comfy_url: str, timeout: float = 1.0) -> int | None:
    """Return total CUDA VRAM bytes currently held by ComfyUI's PyTorch allocator."""
    stats = comfy_system_stats(comfy_url, timeout=timeout)
    if not isinstance(stats, dict):
        return None
    devices = stats.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    total = 0
    saw_metric = False
    for dev in devices:
        if not isinstance(dev, dict):
            continue
        dev_type = str(dev.get("type") or "").lower()
        if dev_type and dev_type != "cuda":
            continue
        torch_total = dev.get("torch_vram_total")
        if isinstance(torch_total, (int, float)) and torch_total >= 0:
            total += int(torch_total)
            saw_metric = True
            continue
        vram_total = dev.get("vram_total")
        vram_free = dev.get("vram_free")
        if isinstance(vram_total, (int, float)) and isinstance(vram_free, (int, float)):
            total += max(0, int(vram_total) - int(vram_free))
            saw_metric = True
    return total if saw_metric else None


def _comfy_queue_idle(comfy_url: str, timeout: float = 1.0) -> bool:
    if not comfy_url:
        return False
    try:
        with urllib.request.urlopen(f"{comfy_url.rstrip('/')}/queue", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            return False
        running = data.get("queue_running") or []
        pending = data.get("queue_pending") or []
        return len(running) == 0 and len(pending) == 0
    except (urllib.error.URLError, OSError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError):
        return False


def _post_comfy_empty(url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> bool:
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, TimeoutError, socket.timeout):
        return False


def interrupt_comfy(comfy_url: str, timeout: float = 5.0) -> bool:
    """Request ComfyUI to abort the active prompt execution immediately."""
    if not comfy_url:
        return False
    return _post_comfy_empty(f"{comfy_url.rstrip('/')}/interrupt", payload={}, timeout=timeout)


def free_comfy_vram(
    comfy_url: str,
    *,
    interrupt: bool = False,
    verify_timeout: float = 8.0,
    settle_seconds: float = 0.5,
) -> dict[str, Any]:
    """Interrupt (if requested), unload all ComfyUI models, and verify VRAM release."""
    if not comfy_url:
        return {
            "reachable": False,
            "interrupted": False,
            "before_bytes": None,
            "after_bytes": None,
            "freed_bytes": 0,
        }

    base = comfy_url.rstrip("/")
    before_bytes = comfy_torch_vram_bytes(base, timeout=1.0)
    reachable = before_bytes is not None

    interrupted = False
    if interrupt:
        interrupted = interrupt_comfy(base, timeout=5.0)
        if interrupted:
            reachable = True
            time.sleep(0.3)

    posted = _post_comfy_empty(
        f"{base}/free",
        payload={"unload_models": True, "free_memory": True},
        timeout=15.0,
    )
    reachable = reachable or posted
    if not posted:
        return {
            "reachable": reachable,
            "interrupted": interrupted,
            "before_bytes": before_bytes,
            "after_bytes": before_bytes,
            "freed_bytes": 0,
        }

    if settle_seconds > 0:
        time.sleep(settle_seconds)

    threshold_bytes = 512 * 1024 * 1024
    retry_threshold_bytes = 1024 * 1024 * 1024
    after_bytes = comfy_torch_vram_bytes(base, timeout=1.0)
    retried = False
    deadline = time.monotonic() + max(0.0, float(verify_timeout))
    last_seen = after_bytes
    stable_polls = 0

    while time.monotonic() < deadline:
        if after_bytes is None:
            break
        if after_bytes <= threshold_bytes:
            break
        if not retried and time.monotonic() + (verify_timeout * 0.5) >= deadline:
            if after_bytes > retry_threshold_bytes and _comfy_queue_idle(base, timeout=0.8):
                retried = True
                _post_comfy_empty(
                    f"{base}/free",
                    payload={"unload_models": True, "free_memory": True},
                    timeout=15.0,
                )
        time.sleep(0.4)
        after_bytes = comfy_torch_vram_bytes(base, timeout=1.0)
        if after_bytes == last_seen:
            stable_polls += 1
            if retried and stable_polls >= 2:
                break
        else:
            last_seen = after_bytes
            stable_polls = 0

    freed_bytes = 0
    if before_bytes is not None and after_bytes is not None:
        freed_bytes = max(0, before_bytes - after_bytes)

    return {
        "reachable": reachable,
        "interrupted": interrupted,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "freed_bytes": freed_bytes,
    }


def active_inprocess_models() -> list[str]:
    """Inspect ``sys.modules`` for resident in-process GPU caches without importing ``torch``."""
    active: list[str] = []
    sam_mod = sys.modules.get("ai_remaster_gui.sam_masks")
    if sam_mod is not None and getattr(sam_mod, "_PREDICTOR", None) is not None:
        active.append("sam2_image_predictor")

    seen_ids: set[int] = set()
    for mod_name in ("scripts.neural_defects_adapter", "neural_defects_adapter"):
        mod = sys.modules.get(mod_name)
        if mod is None or id(mod) in seen_ids:
            continue
        seen_ids.add(id(mod))
        instances = getattr(mod, "_ADAPTER_INSTANCES", None)
        if instances is not None:
            for adapter in list(instances):
                if getattr(adapter, "_cached_model", None) is not None:
                    name = str(getattr(adapter, "name", "neural_defect_detector"))
                    if name not in active:
                        active.append(name)
        reg = getattr(mod, "DEFECT_DETECTOR_REGISTRY", None)
        if reg is not None and hasattr(reg, "list_names"):
            try:
                for det_name in reg.list_names():
                    det = reg.get(det_name)
                    if det is not None and getattr(det, "_cached_model", None) is not None and det_name not in active:
                        active.append(str(det_name))
            except Exception:
                pass
    return active


def unload_inprocess_models() -> dict[str, Any]:
    """Unload any in-process SAM2 or neural defect GPU caches without eagerly importing ``torch``."""
    unloaded: list[str] = []
    with _VRAM_LOCK:
        sam_mod = sys.modules.get("ai_remaster_gui.sam_masks")
        if sam_mod is not None:
            unload_fn = getattr(sam_mod, "unload_sam2_predictor", None)
            if callable(unload_fn) and unload_fn():
                unloaded.append("sam2_image_predictor")

        seen_mods: set[int] = set()
        for mod_name in ("scripts.neural_defects_adapter", "neural_defects_adapter"):
            mod = sys.modules.get(mod_name)
            if mod is None or id(mod) in seen_mods:
                continue
            seen_mods.add(id(mod))
            unload_defects = getattr(mod, "unload_all_cached_defect_models", None)
            if callable(unload_defects):
                for item in unload_defects() or ():
                    if item not in unloaded:
                        unloaded.append(str(item))

        gc.collect()
        cuda_cleared = False
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                cuda = getattr(torch_mod, "cuda", None)
                if cuda is not None and cuda.is_available():
                    if hasattr(cuda, "synchronize"):
                        cuda.synchronize()
                    cuda.empty_cache()
                    ipc_collect = getattr(cuda, "ipc_collect", None)
                    if callable(ipc_collect):
                        ipc_collect()
                    cuda_cleared = True
            except Exception:
                pass

    return {
        "unloaded": unloaded,
        "cuda_cache_cleared": cuda_cleared,
    }


def free_ollama_models(ollama_url: str = "http://127.0.0.1:11434", timeout: float = 1.5) -> list[str]:
    """Evict loaded Ollama vision/LLM models immediately via ``keep_alive: 0``."""
    if not ollama_url:
        return []
    base = ollama_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, socket.timeout, json.JSONDecodeError, ValueError):
        return []

    evicted: list[str] = []
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model") or "").strip()
        if not name:
            continue
        req = urllib.request.Request(
            f"{base}/api/generate",
            data=json.dumps({"model": name, "keep_alive": 0}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                evicted.append(name)
        except (urllib.error.URLError, OSError, TimeoutError, socket.timeout):
            pass
    return evicted


def release_all_vram(
    comfy_url: str | None = None,
    *,
    interrupt_comfy: bool = False,
    include_ollama: bool = True,
    verify_timeout: float = 8.0,
) -> dict[str, Any]:
    """Release ComfyUI, in-process CUDA caches, and idle Ollama models; return VRAM telemetry."""
    from . import system_status

    system_status.invalidate_system_status_cache()
    before_status = system_status.vram_status()
    before_used = before_status.get("used") if isinstance(before_status.get("used"), int) else None

    inprocess_info = unload_inprocess_models()
    comfy_info = (
        free_comfy_vram(comfy_url, interrupt=interrupt_comfy, verify_timeout=verify_timeout)
        if comfy_url
        else {"reachable": False, "interrupted": False, "before_bytes": None, "after_bytes": None, "freed_bytes": 0}
    )
    ollama_evicted = free_ollama_models() if include_ollama else []

    system_status.invalidate_system_status_cache()
    after_status = system_status.vram_status()
    after_used = after_status.get("used") if isinstance(after_status.get("used"), int) else None

    freed_bytes = 0
    if before_used is not None and after_used is not None:
        freed_bytes = max(0, before_used - after_used)
    elif comfy_info.get("freed_bytes"):
        freed_bytes = int(comfy_info["freed_bytes"])

    return {
        "before_used": before_used,
        "after_used": after_used,
        "freed_bytes": freed_bytes,
        "before_used_mb": round(before_used / (1024 * 1024), 1) if before_used is not None else None,
        "after_used_mb": round(after_used / (1024 * 1024), 1) if after_used is not None else None,
        "freed_mb": round(freed_bytes / (1024 * 1024), 1),
        "comfy": comfy_info,
        "inprocess": inprocess_info,
        "ollama": ollama_evicted,
    }
