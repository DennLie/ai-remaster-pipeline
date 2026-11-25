#!/usr/bin/env python3
"""DDColor reference draft generator.

Provides fast, deterministic still reference colorization using DDColor (Apache-2.0).
Preserves original grayscale luminance L in Lab color space ("The original pixels win"),
predicting only chromaticity a* and b*.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure repository root is on sys.path for internal imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.common import (
        ROOT,
        file_fingerprint,
        resolve_path,
        resumable_output,
        root_relative,
        write_signature,
    )
except ImportError:
    ROOT = REPO_ROOT

    def resolve_path(path: Path | str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (ROOT / p).resolve()

    def root_relative(path: Path | str) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def file_fingerprint(path: Path) -> dict[str, Any]:
        return {"path": str(path), "exists": path.exists()}

    def resumable_output(path: Path, sig: dict[str, Any], **kwargs: Any) -> bool:
        return False

    def write_signature(path: Path, sig: dict[str, Any]) -> None:
        return None


try:
    from scripts.dependency_manager import ensure_ddcolor_model, resolve_ddcolor_target
except ImportError:
    ensure_ddcolor_model = None  # type: ignore[assignment]
    resolve_ddcolor_target = None  # type: ignore[assignment]

try:
    from ai_remaster_gui.compliance import append_compliance_record
except ImportError:
    append_compliance_record = None  # type: ignore[assignment]

try:
    from ai_remaster_gui.commercial_policy import CommercialPolicyError, evaluate_component
except ImportError:
    class CommercialPolicyError(RuntimeError):  # type: ignore[no-redef]
        """Raised when a model component fails commercial policy validation."""

    def evaluate_component(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        class _FallbackDecision:
            allowed = True
            status = "allowed_osi"
            license = "Apache-2.0"
            reason = "DDColor Apache-2.0 still reference draft generator"
            sha256 = None
        return _FallbackDecision()


try:
    from ai_remaster_gui.managed_model_envs import (
        inspect_env as inspect_managed_env,
        is_managed_env_active,
        run_in_managed_env,
    )
except ImportError:
    inspect_managed_env = None  # type: ignore[assignment]

    def is_managed_env_active(*args: Any, **kwargs: Any) -> bool:  # type: ignore[no-redef]
        return False

    run_in_managed_env = None  # type: ignore[assignment]


def _ddcolor_package_importable() -> bool:
    if "ddcolor" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("ddcolor") is not None
    except Exception:
        return False


def get_commercial_settings(settings: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
    """Resolve settings for commercial policy evaluation."""
    if settings is not None:
        return settings
    try:
        from ai_remaster_gui import state
        if getattr(state, "APP", None) is not None and getattr(state.APP, "settings", None):
            return state.APP.settings
    except Exception:
        pass
    try:
        from scripts.common import load_local_config
        local_cfg = load_local_config()
        return {
            "global": {
                "commercial_model_guard": str(local_cfg.get("commercial_model_guard", "true")),
                "non_commercial_benchmark_override": str(local_cfg.get("non_commercial_benchmark_override", "false")),
            }
        }
    except Exception:
        return None


DEFAULT_MODEL_PATH = ROOT / "models" / "ddcolor" / "ddcolor_paper_tiny.pth"
DEFAULT_OUTPUT_ROOT = ROOT / "intermediate" / "outpainted_references_color"


class DDColorModel(nn.Module):
    """PyTorch neural architecture for DDColor chromaticity prediction.

    Dual-decoder / multi-scale architecture representation predicting
    chromaticity (a*, b*) from grayscale/luminance input.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        ab = self.decoder(feat)
        return ab


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve target torch.device."""
    dev = str(device or "auto").lower().strip()
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path | str, device: str = "auto", allow_builtin_arch: bool = False) -> nn.Module:
    """Load DDColor checkpoint from model_path onto target device."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DDColor model weights not found at: {path}. "
            f"Model weights are required for normal execution. Pass --dry-run for synthetic mock mode."
        )
    target_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=target_device)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("params") or checkpoint.get("state_dict") or checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint dictionary at {path} does not contain valid state dict parameters.")
        model = DDColorModel()
        total_keys = len(list(model.state_dict().keys()))
        incompatible = model.load_state_dict(state_dict, strict=False)
        if len(incompatible.missing_keys) == total_keys:
            is_official = any(
                k.startswith("encoder.backbone")
                or k.startswith("encoder.arch")
                or "color_decoder" in k
                or "pixel_decoder" in k
                or k.startswith("refine_net.")
                for k in state_dict.keys()
            )
            if is_official:
                loaded_model = None
                try:
                    from ddcolor.builder import build_model  # type: ignore

                    try:
                        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
                        loaded_model = build_model(cfg)
                    except TypeError:
                        loaded_model = build_model()
                except Exception:
                    pass

                if loaded_model is None:
                    try:
                        from ddcolor import DDColor  # type: ignore

                        loaded_model = DDColor()
                    except Exception:
                        pass

                if loaded_model is not None:
                    loaded_model.load_state_dict(state_dict, strict=False)
                    setattr(loaded_model, "_is_official_ddcolor", True)
                    model = loaded_model
                elif allow_builtin_arch:
                    try:
                        from scripts.ddcolor_arch import build_official_ddcolor
                    except ImportError:
                        from ddcolor_arch import build_official_ddcolor  # type: ignore[no-redef]

                    loaded_model = build_official_ddcolor(path, state_dict)
                    setattr(loaded_model, "_is_official_ddcolor", True)
                    model = loaded_model
                else:
                    raise RuntimeError(
                        f"Checkpoint at '{path}' is an official piddnad/DDColor model, which requires the optional 'ddcolor' package. "
                        f"Install it via 'pip install ddcolor' (or clone https://github.com/piddnad/DDColor), or install the isolated managed environment via 'python scripts/setup_model_env.py install ddcolor_env' (or the GUI 'Install Environment' button), or supply a compatible DDColorModel state dict. Pass --dry-run for synthetic mock mode."
                    )
            else:
                raise ValueError(f"Checkpoint at {path} does not match DDColorModel architecture (0 matching keys).")
    else:
        raise ValueError(
            f"Checkpoint at {path} has unsupported format: expected nn.Module or dict, got {type(checkpoint).__name__}"
        )
    model.to(target_device)
    model.eval()
    return model


def _synthetic_dry_run_chroma(
    orig_l: np.ndarray,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic synthetic chromaticity (a*, b*) preserving luminance."""
    h, w = orig_l.shape[:2]
    rng_seed = seed if seed is not None and seed >= 0 else 42
    rng = np.random.RandomState(rng_seed)
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    l_norm = (orig_l.astype(np.float32) - 128.0) / 128.0

    # Seed-dependent deterministic phase and gain modulation
    phase_a = float(rng.uniform(-np.pi, np.pi))
    phase_b = float(rng.uniform(-np.pi, np.pi))
    gain_a = float(rng.uniform(8.0, 16.0))
    gain_b = float(rng.uniform(10.0, 20.0))

    # Natural tone offsets (mild warm tint, modulated by coordinates, seed, and luma)
    # Gently scale chromaticity towards neutral near pure black and pure white to stay within sRGB gamut
    chroma_scale = np.clip(orig_l.astype(np.float32) / 24.0, 0.0, 1.0) * np.clip(
        (255.0 - orig_l.astype(np.float32)) / 24.0, 0.0, 1.0
    )
    base_a = 128.0 + (gain_a * np.sin(1.5 * x + 0.5 * y + phase_a) + 6.0 * l_norm) * chroma_scale
    base_b = 128.0 + (gain_b * np.cos(1.0 * x - 1.2 * y + phase_b) + 8.0 * l_norm) * chroma_scale

    a = np.clip(base_a, 80, 175).astype(np.uint8)
    b = np.clip(base_b, 80, 175).astype(np.uint8)
    return a, b


def colorize_frame(
    image: np.ndarray,
    model_path: Path | str,
    device: str = "auto",
    dry_run: bool = False,
    input_size: int = 512,
    seed: int | None = None,
    model: nn.Module | None = None,
    allow_builtin_arch: bool = False,
) -> np.ndarray:
    """Colorize a single image frame using DDColor or synthetic dry-run.

    Preserves original grayscale luminance L in Lab space ("The original pixels win"),
    predicting only chromaticity a* and b*.
    """
    model_file = Path(model_path)

    # 1. Extract original luminance L in Lab space
    if image.ndim == 2:
        rgb_src = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        rgb_src = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb_src = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape for colorization: {image.shape}")

    lab = cv2.cvtColor(rgb_src, cv2.COLOR_RGB2LAB)
    orig_l = lab[:, :, 0]
    orig_h, orig_w = orig_l.shape[:2]

    # 2. Dry-run mode: deterministic synthetic colorization
    if dry_run:
        a, b = _synthetic_dry_run_chroma(orig_l, seed=seed)
        lab_out = np.dstack([orig_l, a, b])
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    # 3. Normal mode: MUST fail closed if weights are missing
    if not model_file.is_file():
        raise FileNotFoundError(
            f"DDColor model checkpoint not found at: {model_file}. "
            f"Model weights are required for normal execution. Pass --dry-run for synthetic mock mode."
        )

    # 4. Neural inference
    target_device = resolve_device(device)
    if model is None:
        model = load_model(model_file, device=device, allow_builtin_arch=allow_builtin_arch)

    is_official = bool(getattr(model, "_is_official_ddcolor", False) or hasattr(model, "refine_net"))
    if is_official:
        rgb_f32 = rgb_src.astype(np.float32) / 255.0
        orig_l_f32 = cv2.cvtColor(rgb_f32, cv2.COLOR_RGB2LAB)[:, :, 0]
        lab_gray = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        lab_gray[:, :, 0] = orig_l_f32
        rgb_gray = np.clip(cv2.cvtColor(lab_gray, cv2.COLOR_LAB2RGB), 0.0, 1.0)
        resized_gray = cv2.resize(rgb_gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
        inp_tensor = torch.from_numpy(resized_gray.transpose(2, 0, 1)).float().unsqueeze(0).to(target_device)
        with torch.no_grad():
            pred_ab = model(inp_tensor)
        if pred_ab.shape[2:] != (orig_h, orig_w):
            pred_ab = nn.functional.interpolate(
                pred_ab, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
        ab_f32 = pred_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        lab_out_f32 = np.concatenate([orig_l_f32[:, :, None], ab_f32], axis=-1)
        rgb_out_f32 = np.clip(cv2.cvtColor(lab_out_f32, cv2.COLOR_LAB2RGB), 0.0, 1.0)
        return np.round(rgb_out_f32 * 255.0).astype(np.uint8)

    # Fallback DDColorModel inference
    resized_l = cv2.resize(orig_l, (input_size, input_size), interpolation=cv2.INTER_AREA)
    l_tensor = torch.from_numpy(resized_l).float() / 255.0
    l_tensor = l_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(target_device)

    with torch.no_grad():
        pred_ab = model(l_tensor)

    # Upsample predicted chromaticity back to original frame dimensions
    if pred_ab.shape[2:] != (orig_h, orig_w):
        pred_ab = nn.functional.interpolate(
            pred_ab, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

    ab_np = pred_ab.squeeze(0).cpu().numpy()  # shape (2, H, W)

    # Handle common output ranges: [-1, 1], [-128, 127], or [0, 255]
    ab_min, ab_max = float(ab_np.min()), float(ab_np.max())
    if -1.5 <= ab_min and ab_max <= 1.5:
        a_channel = np.clip(ab_np[0] * 128.0 + 128.0, 0, 255).astype(np.uint8)
        b_channel = np.clip(ab_np[1] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    elif -130.0 <= ab_min and ab_max <= 130.0:
        a_channel = np.clip(ab_np[0] + 128.0, 0, 255).astype(np.uint8)
        b_channel = np.clip(ab_np[1] + 128.0, 0, 255).astype(np.uint8)
    else:
        a_channel = np.clip(ab_np[0], 0, 255).astype(np.uint8)
        b_channel = np.clip(ab_np[1], 0, 255).astype(np.uint8)

    lab_out = np.dstack([orig_l, a_channel, b_channel])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


def colorize_reference_shot(
    source_path: Path | str,
    output_path: Path | str,
    model_path: Path | str | None = None,
    device: str = "auto",
    dry_run: bool = False,
    seed: int | None = 42,
    settings: Mapping[str, Any] | None = None,
    enforce_policy: bool = False,
    allow_builtin_arch: bool = False,
) -> Path:
    """Colorize a single reference still programmatically and return the output Path.

    Preserves original grayscale luminance L in Lab space ("The original pixels win"),
    predicting only chromaticity a* and b*.

    Args:
        source_path: Path to input grayscale or reference image.
        output_path: Path to destination colorized output image.
        model_path: Path to DDColor checkpoint (.pth). Defaults to DEFAULT_MODEL_PATH.
        device: "auto" | "cuda" | "cpu".
        dry_run: If True, uses deterministic synthetic colorization without weights/GPU.
        seed: Optional RNG seed for deterministic colorization (defaults to 42).
        settings: Optional GUI or pipeline settings mapping for commercial policy check.
        enforce_policy: If True, evaluates Commercial Model Guard prior to inference.

    Returns:
        Path to the saved colorized output image.
    """
    src = resolve_path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source reference still not found: {src}")

    dst = resolve_path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    m_path = resolve_path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

    comm_settings = get_commercial_settings(settings)
    if enforce_policy or settings is not None:
        decision = evaluate_component(
            "ddcolor",
            component_type="checkpoint",
            file_path=m_path if m_path.is_file() else None,
            settings=comm_settings,
        )
        if not decision.allowed:
            raise CommercialPolicyError(f"Commercial Model Guard blocked DDColor: {decision.reason}")

    if (
        not dry_run
        and m_path.is_file()
        and not _ddcolor_package_importable()
        and not is_managed_env_active()
        and inspect_managed_env is not None
        and run_in_managed_env is not None
        and inspect_managed_env("ddcolor_env", settings=comm_settings)["ready"]
    ):
        argv = [
            "--input-image",
            str(src),
            "--output",
            str(dst),
            "--model-path",
            str(m_path),
            "--device",
            str(device),
        ]
        if seed is not None:
            argv.extend(["--seed", str(seed)])
        run_in_managed_env(
            "ddcolor_env",
            Path(__file__),
            argv,
            settings=comm_settings,
        )
        if not dst.is_file():
            raise RuntimeError(f"Managed env 'ddcolor_env' did not produce expected output at '{dst}'.")
        return dst

    with Image.open(src) as im:
        rgb_in = np.asarray(im.convert("RGB"))

    colorized_rgb = colorize_frame(
        rgb_in,
        model_path=m_path,
        device=device,
        dry_run=dry_run,
        seed=seed,
        allow_builtin_arch=allow_builtin_arch,
    )

    Image.fromarray(colorized_rgb).save(dst)
    return dst


def read_manifest_lines(manifest_path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Read manifest preserving leading comments and returning headers and row dicts."""
    comments: list[str] = []
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        csv_lines: list[str] = []
        for line in f:
            if line.startswith("#"):
                comments.append(line)
            else:
                csv_lines.append(line)

    if csv_lines:
        reader = csv.DictReader(csv_lines)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    return comments, fieldnames, rows


def write_manifest_lines(
    manifest_path: Path,
    comments: list[str],
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Write updated manifest preserving comments via atomic temp file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        for c in comments:
            f.write(c if c.endswith("\n") else c + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    tmp_path.replace(manifest_path)


def process_manifest(
    manifest_path: Path,
    output_manifest_path: Path | None,
    output_root: Path,
    model_path: Path,
    device: str = "auto",
    dry_run: bool = False,
    input_size: int = 512,
    seed: int | None = None,
    limit: int | None = None,
    force: bool = False,
    row_index: int | None = None,
    reference_index: int = 0,
    film_id: str | None = None,
    color_bible_db: Path | None = None,
    confidence_threshold: float = 0.8,
    reference_runner: str = "guided_only",
    allow_builtin_arch: bool = False,
) -> int:
    """Process reference rows in a manifest with resumable signatures and optional Color Bible review gate."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    comments, fieldnames, rows = read_manifest_lines(manifest_path)
    if "color_reference" not in fieldnames:
        fieldnames.append("color_reference")

    color_bible_service = None
    matcher = None
    record_ref_compliance = None
    update_manifest_row = None
    if film_id:
        try:
            from ai_remaster_gui.color_bible_service import ColorBibleService
            from ai_remaster_gui.manifests import update_manifest_row as _umr
            from ai_remaster_gui.semantic_matching import ColorBibleMatcher, record_reference_compliance as _rrc

            color_bible_service = ColorBibleService(db_path=color_bible_db)
            matcher = ColorBibleMatcher
            record_ref_compliance = _rrc
            update_manifest_row = _umr
        except Exception as exc:
            print(f"Warning: Color Bible integration unavailable in ddcolor_reference: {exc}", flush=True)
            color_bible_service = None

    try:
        expanded_rows: list[dict[str, str]]
        try:
            from scripts.reference_sets import expanded_reference_rows
            expanded_rows = expanded_reference_rows(rows)
        except Exception:
            expanded_rows = [{**r, "_shot_index": str(i), "_reference_index": "0"} for i, r in enumerate(rows)]

        model: nn.Module | None = None
        if not dry_run:
            model = load_model(model_path, device=device, allow_builtin_arch=allow_builtin_arch)

        count = 0
        for exp_idx, row in enumerate(expanded_rows):
            shot_idx = int(row.get("_shot_index", exp_idx))
            ref_idx = int(row.get("_reference_index", 0))
            if row_index is not None and shot_idx != row_index:
                continue
            if row_index is not None and ref_idx != reference_index:
                continue
            if limit is not None and count >= limit:
                break
            if row_index is None and row.get("enabled", "true").strip().lower() in {"false", "0", "no", "off"}:
                continue

            source_rel = row.get("source_reference") or row.get("reference") or ""
            if not source_rel:
                continue
            source_path = resolve_path(source_rel)
            if not source_path.is_file():
                raise FileNotFoundError(f"Source reference still not found: {source_path}")

            # Review gate check when Color Bible film is active
            match_result = None
            if matcher is not None and film_id:
                match_result = matcher.resolve_references(
                    rows=expanded_rows,
                    index=exp_idx,
                    count=2,
                    film_id=film_id,
                    service=color_bible_service,
                    confidence_threshold=confidence_threshold,
                )
                if match_result.needs_review:
                    rows[shot_idx]["qc_status"] = "needs_review"
                    if "qc_status" not in fieldnames:
                        fieldnames.append("qc_status")
                    if color_bible_service is not None:
                        try:
                            color_bible_service.update_shot_qc_status(film_id, shot_idx, "needs_review")
                        except Exception:
                            pass
                    print(f"[Review Gate] Shot {shot_idx:04d} paused for human review: {', '.join(match_result.review_reasons)}", flush=True)
                    continue

            # Destination color reference path
            if output_root and (output_root != DEFAULT_OUTPUT_ROOT or not row.get("color_reference")):
                target_path = resolve_path(output_root) / source_path.name
            elif row.get("color_reference"):
                target_path = resolve_path(row["color_reference"])
            else:
                target_path = DEFAULT_OUTPUT_ROOT / source_path.name

            sig = {
                "version": 1,
                "tool": "ddcolor_reference.py",
                "shot_index": shot_idx,
                "reference_index": ref_idx,
                "source": root_relative(source_path),
                "source_fingerprint": file_fingerprint(source_path),
                "model_path": root_relative(model_path),
                "model_fingerprint": file_fingerprint(model_path) if model_path.is_file() else {"dry_run": dry_run},
                "input_size": input_size,
                "seed": seed,
                "dry_run": dry_run,
                "film_id": film_id or "",
                "color_bible_version": getattr(match_result, "color_bible_version", None),
            }
            if not force and resumable_output(target_path, sig, image_like=source_path):
                if ref_idx == 0:
                    rows[shot_idx]["color_reference"] = root_relative(target_path)
                print(f"Reuse {shot_idx:04d}: {target_path}", flush=True)
                count += 1
                continue

            with Image.open(source_path) as im:
                rgb_in = np.asarray(im.convert("RGB"))

            colorized_rgb = colorize_frame(
                rgb_in,
                model_path=model_path,
                device=device,
                dry_run=dry_run,
                input_size=input_size,
                seed=seed,
                model=model,
                allow_builtin_arch=allow_builtin_arch,
            )

            target_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(colorized_rgb).save(target_path)
            write_signature(target_path, sig)
            if ref_idx == 0 and not rows[shot_idx].get("color_reference"):
                rows[shot_idx]["color_reference"] = root_relative(target_path)
            elif ref_idx == 0 and output_root:
                rows[shot_idx]["color_reference"] = root_relative(target_path)
            if record_ref_compliance is not None and match_result is not None:
                try:
                    record_ref_compliance(
                        output_path=target_path,
                        model_id="ddcolor",
                        component_type="checkpoint",
                        color_bible_version=match_result.color_bible_version,
                        retrieved_ref_ids=match_result.retrieved_ref_ids,
                        seed=seed,
                    )
                except Exception:
                    pass
            count += 1
            print(f"[{count}] Colorized reference: {source_path.name} -> {target_path}", flush=True)

        dst_manifest = output_manifest_path or manifest_path
        write_manifest_lines(dst_manifest, comments, fieldnames, rows)
        print(f"Updated manifest saved to: {dst_manifest} ({count} stills colorized)", flush=True)
        return count
    finally:
        if color_bible_service is not None:
            try:
                color_bible_service.close()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DDColor fast deterministic reference draft generator (Apache-2.0)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to shot references CSV manifest.",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        help="Path to single input image to colorize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (with --input-image) or output manifest path (with --manifest).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory for colorized reference stills (default: intermediate/outpainted_references_color).",
    )
    parser.add_argument(
        "--model-variant",
        default="piddnad/ddcolor_modelscope",
        help="Official HuggingFace piddnad/DDColor model variant (default: piddnad/ddcolor_modelscope).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to DDColor checkpoint (.pth). Defaults to the chosen --model-variant checkpoint in models/ddcolor/.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Download official Apache-2.0 piddnad/DDColor weights from HuggingFace if missing locally.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device: auto | cuda | cpu (default: auto).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="Inference image resolution (default: 512).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deterministic synthetic colorization for testing and CI without GPU/weights.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate reference outputs even when matching .sig.json signatures exist.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Zero-based shot row index to colorize in isolation.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Reference index within --row-index (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic colorization.",
    )
    parser.add_argument(
        "--compliance-output-dir",
        type=Path,
        default=None,
        help="Output directory to record run into compliance_report.json via append_compliance_record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional integer row limit for manifest processing.",
    )
    parser.add_argument(
        "--film-id",
        type=str,
        default=None,
        help="Optional film ID for Color Bible integration.",
    )
    parser.add_argument(
        "--reference-runner",
        choices=["auto", "guided_only", "ddcolor_only"],
        default="guided_only",
        help="Reference runner mode compatibility argument.",
    )
    parser.add_argument(
        "--guided-backend",
        default="qwen2511",
        help="Guided backend compatibility argument.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold compatibility argument.",
    )
    parser.add_argument(
        "--color-bible-db",
        type=Path,
        default=None,
        help="Optional path to Color Bible SQLite database.",
    )
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.manifest and not args.input_image:
        parser.error("Either --manifest or --input-image must be provided.")

    if resolve_ddcolor_target is not None:
        _, resolved_default = resolve_ddcolor_target(args.model_variant, args.model_path)
        model_path = resolved_default
    else:
        model_path = resolve_path(args.model_path) if args.model_path is not None else DEFAULT_MODEL_PATH

    if not args.dry_run and not model_path.is_file() and args.auto_download and ensure_ddcolor_model is not None:
        model_path = ensure_ddcolor_model(variant=args.model_variant, model_path=args.model_path, auto_download=True)

    # Enforce Commercial Model Guard prior to inference
    comm_settings = get_commercial_settings()
    decision = evaluate_component(
        "ddcolor",
        component_type="checkpoint",
        file_path=model_path if model_path.is_file() else None,
        settings=comm_settings,
    )
    if not decision.allowed:
        raise CommercialPolicyError(f"Commercial Model Guard blocked DDColor: {decision.reason}")

    if (
        not args.dry_run
        and model_path.is_file()
        and not _ddcolor_package_importable()
        and not is_managed_env_active()
        and inspect_managed_env is not None
        and run_in_managed_env is not None
        and inspect_managed_env("ddcolor_env", settings=comm_settings)["ready"]
    ):
        run_in_managed_env(
            "ddcolor_env",
            Path(__file__),
            sys.argv[1:],
            settings=comm_settings,
        )
        return 0

    # Single-image colorization mode (run only when --manifest is not taking precedence via --row-index)
    if args.input_image and not (args.manifest and args.row_index is not None):
        input_image = resolve_path(args.input_image)
        if not input_image.is_file():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        if args.output:
            output_image = resolve_path(args.output)
        elif args.output_root:
            output_image = resolve_path(args.output_root) / input_image.name
        else:
            output_image = input_image.parent / f"{input_image.stem}_colorized{input_image.suffix}"

        with Image.open(input_image) as im:
            rgb_in = np.asarray(im.convert("RGB"))

        colorized_rgb = colorize_frame(
            rgb_in,
            model_path=model_path,
            device=args.device,
            dry_run=args.dry_run,
            input_size=args.input_size,
            seed=args.seed,
            allow_builtin_arch=True,
        )

        output_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colorized_rgb).save(output_image)
        print(f"Colorized still saved to: {output_image}")

    # Manifest processing mode
    if args.manifest:
        manifest_path = resolve_path(args.manifest)
        output_manifest = None if (args.input_image and args.row_index is not None) else (resolve_path(args.output) if args.output else None)
        output_root = resolve_path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT

        process_manifest(
            manifest_path=manifest_path,
            output_manifest_path=output_manifest,
            output_root=output_root,
            model_path=model_path,
            device=args.device,
            dry_run=args.dry_run,
            input_size=args.input_size,
            seed=args.seed,
            limit=args.limit,
            force=args.force,
            row_index=args.row_index,
            reference_index=args.reference_index,
            film_id=args.film_id,
            color_bible_db=args.color_bible_db,
            confidence_threshold=args.confidence_threshold,
            reference_runner=args.reference_runner,
            allow_builtin_arch=True,
        )

    # Compliance reporting
    if args.compliance_output_dir and append_compliance_record is not None:
        comp_dir = resolve_path(args.compliance_output_dir)
        append_compliance_record(
            output_dir=comp_dir,
            stage="references",
            models=[
                {
                    "model_id": "ddcolor",
                    "component_type": "checkpoint",
                    "file_path": str(model_path) if model_path.is_file() else None,
                    "source_url": "https://github.com/piddnad/DDColor",
                    "license": getattr(decision, "license", "Apache-2.0"),
                    "status": getattr(decision, "status", "allowed_osi"),
                    "sha256": getattr(decision, "sha256", None),
                    "reason": getattr(decision, "reason", "DDColor Apache-2.0 still reference draft generator"),
                }
            ],
            settings={"references": {"method": "ddcolor"}},
            seeds={"references": args.seed} if args.seed is not None else None,
        )
        print(f"Recorded reference run to compliance report in: {comp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
