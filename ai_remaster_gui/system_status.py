from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time


_CACHE: dict[str, object] = {"at": 0.0, "data": {}}
_CACHE_SECONDS = 5.0
_CPU_TIMES: tuple[int, int] | None = None
FLASHVSR_MIN_CUDA_CAPABILITY = (7, 5)


def invalidate_system_status_cache() -> None:
    """Clear the cached header status sample so the next poll queries fresh VRAM metrics."""
    _CACHE["at"] = 0.0
    _CACHE["data"] = {}


def _nvml_cuda_probe() -> dict | None:
    """Query GPU name and compute capability via NVML ctypes without spawning a process or CUDA context."""
    initialized = False
    nvml = None
    try:
        nvml = ctypes.WinDLL("nvml.dll") if os.name == "nt" else ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            return None
        initialized = True
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
            return None
        major, minor = ctypes.c_int(), ctypes.c_int()
        if nvml.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor)) != 0:
            return None
        name_buf = ctypes.create_string_buffer(128)
        name = "NVIDIA GPU"
        if nvml.nvmlDeviceGetName(handle, name_buf, len(name_buf)) == 0:
            decoded = name_buf.value.decode("utf-8", errors="replace").strip()
            if decoded:
                name = decoded
        return {"name": name, "capability": (int(major.value), int(minor.value))}
    except Exception:
        return None
    finally:
        if initialized and nvml is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def nvidia_smi_cuda_probe() -> dict | None:
    """Query GPU name and compute capability without initializing CUDA."""
    nvml_probe = _nvml_cuda_probe()
    if nvml_probe is not None:
        return nvml_probe
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
        if not line or "," not in line:
            return None
        name_text, cap_text = [part.strip() for part in line.split(",", 1)]
        cap_parts = [part.strip() for part in cap_text.split(".") if part.strip().isdigit()]
        capability = (int(cap_parts[0]), int(cap_parts[1])) if len(cap_parts) >= 2 else None
        return {"name": name_text, "capability": capability}
    except Exception:
        return None


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _FileTime(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", ctypes.c_ulong),
        ("dwHighDateTime", ctypes.c_ulong),
    ]


def system_status() -> dict:
    now = time.monotonic()
    cached = _CACHE.get("data")
    if isinstance(cached, dict) and cached and now - float(_CACHE.get("at", 0.0)) < _CACHE_SECONDS:
        return cached

    vram = vram_status()
    data = {
        "cuda": cuda_status(),
        "cpu": cpu_status(),
        "gpu": gpu_status(vram),
        "ram": ram_status(),
        "vram": vram,
    }
    _CACHE["at"] = now
    _CACHE["data"] = data
    return data


def cuda_status() -> dict:
    probe = nvidia_smi_cuda_probe()
    if probe is not None:
        detail = str(probe.get("name") or "NVIDIA GPU")
        capability = probe.get("capability")
        warning = ""
        if capability is not None:
            detail = f"{detail} - compute capability {format_cuda_capability(capability)}"
            if capability < FLASHVSR_MIN_CUDA_CAPABILITY:
                warning = flashvsr_hardware_warning(detail, capability)
        torch_mod = sys.modules.get("torch")
        torch_ver = str(getattr(torch_mod, "__version__", "") or "") if torch_mod is not None else ""
        cuda_ver = str(getattr(getattr(torch_mod, "version", None), "cuda", "") or "") if torch_mod is not None else ""
        return {
            "available": True,
            "label": "CUDA available",
            "detail": detail,
            "torch": torch_ver,
            "cuda": cuda_ver,
            "capability": format_cuda_capability(capability) if capability else "",
            "flashvsr_supported": bool(capability) and capability >= FLASHVSR_MIN_CUDA_CAPABILITY,
            "warning": warning,
        }

    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        detail = torch.cuda.get_device_name(0) if available else f"torch {torch.__version__}"
        capability: tuple[int, int] | None = None
        warning = ""
        if available:
            try:
                major, minor = torch.cuda.get_device_capability(0)
                capability = (int(major), int(minor))
                detail = f"{detail} - compute capability {format_cuda_capability(capability)}"
                if capability < FLASHVSR_MIN_CUDA_CAPABILITY:
                    warning = flashvsr_hardware_warning(detail, capability)
            except Exception as exc:
                detail = f"{detail} - compute capability unknown: {exc}"
        return {
            "available": available,
            "label": "CUDA available" if available else "CUDA unavailable",
            "detail": detail,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or ""),
            "capability": format_cuda_capability(capability) if capability else "",
            "flashvsr_supported": available and bool(capability) and capability >= FLASHVSR_MIN_CUDA_CAPABILITY,
            "warning": warning,
        }
    except Exception as exc:
        return {
            "available": False,
            "label": "CUDA unknown",
            "detail": str(exc),
            "torch": "",
            "cuda": "",
            "capability": "",
            "flashvsr_supported": False,
            "warning": "",
        }


def flashvsr_hardware_warning(device_detail: str | None = None, capability: tuple[int, int] | None = None) -> str:
    if capability is None:
        probe = nvidia_smi_cuda_probe()
        if probe is not None and probe.get("capability") is not None:
            capability = probe["capability"]
            device_detail = device_detail or str(probe.get("name") or "")
        else:
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    return ""
                capability = tuple(int(part) for part in torch.cuda.get_device_capability(0))
                device_detail = torch.cuda.get_device_name(0)
            except Exception:
                return ""
    if capability >= FLASHVSR_MIN_CUDA_CAPABILITY:
        return ""
    device = f"{device_detail or 'This GPU'} reports compute capability {format_cuda_capability(capability)}"
    required = format_cuda_capability(FLASHVSR_MIN_CUDA_CAPABILITY)
    return f"{device}. FlashVSR upscaling requires NVIDIA compute capability {required}+; disable Upscale or use a newer NVIDIA GPU."


def format_cuda_capability(capability: tuple[int, int] | None) -> str:
    if not capability:
        return ""
    return f"{capability[0]}.{capability[1]}"


def ram_status() -> dict:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
            used = max(0, total - available)
            return memory_payload("RAM", used, total, int(status.dwMemoryLoad))
    except Exception:
        pass
    return {"label": "RAM unknown", "used": None, "total": None, "percent": None, "detail": "RAM status unavailable"}


def cpu_status() -> dict:
    try:
        percent = windows_cpu_percent()
        if percent is not None:
            return utilization_payload("CPU", percent, "Total processor utilization")
    except Exception:
        pass
    return {"name": "CPU", "label": "CPU unknown", "percent": None, "detail": "CPU utilization unavailable"}


def windows_cpu_percent() -> int | None:
    global _CPU_TIMES
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    except AttributeError:
        return None

    idle = _FileTime()
    kernel = _FileTime()
    user = _FileTime()
    if not kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    idle_now = filetime_to_int(idle)
    total_now = filetime_to_int(kernel) + filetime_to_int(user)
    previous = _CPU_TIMES
    _CPU_TIMES = (idle_now, total_now)
    if previous is None:
        time.sleep(0.05)
        return windows_cpu_percent()
    idle_delta = max(0, idle_now - previous[0])
    total_delta = max(0, total_now - previous[1])
    if total_delta <= 0:
        return None
    busy = max(0.0, min(100.0, 100.0 * (1.0 - (idle_delta / total_delta))))
    return int(round(busy))


def filetime_to_int(value: _FileTime) -> int:
    return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)


def gpu_status(vram: dict) -> dict:
    percent = vram.get("gpu_utilization")
    if isinstance(percent, int):
        name = str(vram.get("gpu_name") or "GPU")
        return utilization_payload("GPU", percent, f"{name} utilization")
    return {"name": "GPU", "label": "GPU unknown", "percent": None, "detail": "GPU utilization unavailable"}


def vram_status() -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=memory.used,memory.total,utilization.gpu,name",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
            if line:
                used_text, total_text, util_text, name = [part.strip() for part in line.split(",", 3)]
                used = int(used_text) * 1024 * 1024
                total = int(total_text) * 1024 * 1024
                percent = int(round((used / total) * 100)) if total else 0
                payload = memory_payload("VRAM", used, total, percent)
                payload["detail"] = f"{payload['detail']} - {name} - {util_text}% GPU utilization"
                payload["gpu_utilization"] = int(util_text)
                payload["gpu_name"] = name
                _attach_vram_attribution(payload)
                return payload
        except Exception as exc:
            return {"label": "VRAM unknown", "used": None, "total": None, "percent": None, "detail": str(exc)}

    return torch_vram_status()


def _attach_vram_attribution(payload: dict) -> None:
    try:
        from .config import current_config
        from .vram_manager import active_inprocess_models, comfy_torch_vram_bytes

        comfy_url = str(current_config().get("comfy_url") or "").strip()
        comfy_bytes = comfy_torch_vram_bytes(comfy_url, timeout=0.4) if comfy_url else None
        inprocess = active_inprocess_models()
        parts: list[str] = []
        if comfy_bytes is not None:
            payload["comfy_vram_used"] = comfy_bytes
            parts.append(f"ComfyUI {format_compact_gb(comfy_bytes)}")
        if inprocess:
            payload["arp_inprocess_models"] = inprocess
            parts.append("ARP cached: " + ", ".join(inprocess))
        if parts:
            payload["detail"] = f"{payload['detail']} ({'; '.join(parts)})"
    except Exception:
        pass


def torch_vram_status() -> dict:
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return {"label": "VRAM unavailable", "used": None, "total": None, "percent": None, "detail": "CUDA is unavailable"}
        free, total = torch.cuda.mem_get_info()
        used = max(0, int(total) - int(free))
        return memory_payload("VRAM", used, int(total), int(round((used / total) * 100)) if total else 0)
    except Exception as exc:
        return {"label": "VRAM unknown", "used": None, "total": None, "percent": None, "detail": str(exc)}


def memory_payload(name: str, used: int, total: int, percent: int) -> dict:
    return {
        "label": f"{name} {format_compact_gb(used)}/{format_compact_gb(total)} used",
        "used": used,
        "total": total,
        "percent": percent,
        "detail": f"{format_bytes(used)} / {format_bytes(total)} used ({percent}%)",
    }


def utilization_payload(name: str, percent: int, detail: str) -> dict:
    percent = max(0, min(100, int(percent)))
    return {
        "name": name,
        "label": f"{name} {percent}% used",
        "percent": percent,
        "detail": f"{detail}: {percent}%",
    }


def format_compact_gb(value: int) -> str:
    amount = value / (1024 ** 3)
    if amount >= 10 or amount.is_integer():
        return f"{amount:.0f}GB"
    return f"{amount:.1f}GB"


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"
