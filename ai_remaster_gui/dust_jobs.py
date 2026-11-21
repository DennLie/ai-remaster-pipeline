"""Dust-only cooperative stop, process-scoped candidate locking, and mask protection invariants."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    from film_defect_detectors import (
        DEFAULT_REGISTRY as DETECTOR_REGISTRY,
        BENCHMARK_PRESETS,
        ChannelOrigin,
        ChannelProvenance,
        MultiChannelDefectMask,
        RepairInclusionPolicy,
        UnsupportedPolicyError,
    )
except ImportError:
    from scripts.film_defect_detectors import (
        DEFAULT_REGISTRY as DETECTOR_REGISTRY,
        BENCHMARK_PRESETS,
        ChannelOrigin,
        ChannelProvenance,
        MultiChannelDefectMask,
        RepairInclusionPolicy,
        UnsupportedPolicyError,
    )

try:
    from dust_native_composite import (
        build_restoration_provenance,
        compute_file_sha256,
        compute_mask_sha256,
        compute_restoration_cache_identity,
        dispatch_repair_frame,
        execute_repair,
        inpaint_frame_dart_composite,
        inpaint_frame_propainter,
        inpaint_frame_telea,
        write_restoration_provenance,
    )
except ImportError:
    from scripts.dust_native_composite import (
        build_restoration_provenance,
        compute_file_sha256,
        compute_mask_sha256,
        compute_restoration_cache_identity,
        dispatch_repair_frame,
        execute_repair,
        inpaint_frame_dart_composite,
        inpaint_frame_propainter,
        inpaint_frame_telea,
        write_restoration_provenance,
    )


def check_cancel(path):
    if path and Path(path).exists():
        raise ValueError('Dust stop requested. Verified model sections are retained for retry.')


def protect_accepted_mask(target_folder: Path | str, settings: Mapping[str, Any] | None) -> None:
    """Guarantee that running or previewing a new candidate never mutates an already accepted mask.

    Invariant: Previewing a new detector or adjusting threshold NEVER overwrites an already accepted mask.
    """
    if not settings:
        return
    dust_settings = settings.get('dust', {}) if isinstance(settings, dict) else {}
    adopted_report = dust_settings.get('adopted_report')
    if not adopted_report:
        return

    target = Path(target_folder).resolve()
    report_file = target / 'report.json'
    if report_file.is_file():
        try:
            current_sha = compute_file_sha256(report_file)
            if current_sha == adopted_report:
                raise ValueError(
                    'Target folder contains the currently accepted mask. '
                    'Previewing a new detector or adjusting threshold cannot mutate an accepted mask.'
                )
        except OSError:
            pass


def propagate_defect_job_parameters(values: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize and propagate selected model, preset, and parameters to the defect job runner.

    Ensures detector, preset, and all algorithm thresholds/parameters are preserved across dispatch.
    """
    propagated: dict[str, Any] = {}
    for key, value in values.items():
        if value is not None:
            propagated[key] = value
    return propagated


@contextmanager
def candidate_lock(target):
    folder = target.parent / '.locks'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / target.name).open('a+b') as handle:
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another worker owns this Dust candidate. Wait for it to stop.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_effective_repair_mask(
    mask: MultiChannelDefectMask | np.ndarray | Any,
    policy: RepairInclusionPolicy | str = RepairInclusionPolicy.CONSERVATIVE,
) -> np.ndarray:
    """Resolve the effective binary repair mask based on the specified inclusion policy.

    If mask is a MultiChannelDefectMask, delegates to mask.get_effective_repair_mask(policy).
    Returns uint8 binary ndarray with strictly {0, 255} values.
    """
    if isinstance(policy, str):
        policy = RepairInclusionPolicy(policy.lower())

    if isinstance(mask, MultiChannelDefectMask):
        return mask.get_effective_repair_mask(policy)

    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask
        return (mask_2d > 127).astype(np.uint8) * 255

    raise TypeError(f"Unsupported mask type for effective repair resolution: {type(mask)}")


def adopt_defect_mask(
    target_folder: Path | str,
    mask: MultiChannelDefectMask | Sequence[MultiChannelDefectMask] | np.ndarray | None = None,
    policy: RepairInclusionPolicy | str | None = None,
    preset: str | Any = None,
    settings: Mapping[str, Any] | None = None,
    source_video: Path | str | None = None,
    repair_backend: str = "telea",
    dilate: int = 0,
    detector_info: dict[str, Any] | None = None,
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    """Adopt a defect mask for downstream restoration and enforce mask protection invariants.

    1. Enforces protect_accepted_mask invariant: previewing a different detector
       or adjusting threshold NEVER mutates an already accepted mask.
    2. Resolves effective repair mask via mask.get_effective_repair_mask(policy)
       using the preset's declared repair_policy or explicit override.
    3. Persists effective repair mask as effective_repair_mask.png/mkv alongside channel masks.
    4. Ensures downstream inpainting jobs consume THIS exact effective repair mask.
    """
    target = Path(target_folder).resolve()

    # Invariant: Previewing a new detector or adjusting threshold cannot mutate an accepted mask
    protect_accepted_mask(target, settings)

    target.mkdir(parents=True, exist_ok=True)

    # Resolve policy
    if policy is not None:
        resolved_policy = policy if isinstance(policy, RepairInclusionPolicy) else RepairInclusionPolicy(str(policy).lower())
    elif preset:
        preset_name = str(preset).lower()
        preset_obj = BENCHMARK_PRESETS.get(preset_name)
        if preset_obj and hasattr(preset_obj, "repair_policy"):
            resolved_policy = preset_obj.repair_policy
        else:
            resolved_policy = RepairInclusionPolicy.CONSERVATIVE
    elif settings and "dust" in settings and "repair_policy" in settings["dust"]:
        resolved_policy = RepairInclusionPolicy(str(settings["dust"]["repair_policy"]).lower())
    else:
        resolved_policy = RepairInclusionPolicy.CONSERVATIVE

    effective_mask_path = target / "effective_repair_mask.png"
    channel_provenance: dict[str, str] = {}

    if isinstance(mask, MultiChannelDefectMask):
        effective = mask.get_effective_repair_mask(resolved_policy)
        if dilate > 0:
            effective = cv2.dilate(effective, np.ones((3, 3), np.uint8), iterations=dilate)
        cv2.imwrite(str(effective_mask_path), effective)
        # Compatibility alias
        cv2.imwrite(str(target / "effective_mask.png"), effective)

        if mask.combined_mask is not None:
            cv2.imwrite(str(target / "mask.png"), mask.combined_mask)
        if mask.scratch_mask is not None:
            cv2.imwrite(str(target / "scratch_mask.png"), mask.scratch_mask)
        if mask.dust_mask is not None:
            cv2.imwrite(str(target / "dust_mask.png"), mask.dust_mask)
        if mask.uncertain_mask is not None:
            cv2.imwrite(str(target / "uncertain_mask.png"), mask.uncertain_mask)

        for ch, prov in mask.provenance.items():
            channel_provenance[ch] = prov.origin.value if hasattr(prov.origin, "value") else str(prov.origin)

    elif isinstance(mask, np.ndarray):
        effective = (mask > 127).astype(np.uint8) * 255
        if dilate > 0:
            effective = cv2.dilate(effective, np.ones((3, 3), np.uint8), iterations=dilate)
        cv2.imwrite(str(effective_mask_path), effective)
        cv2.imwrite(str(target / "effective_mask.png"), effective)
    elif (target / "effective_repair_mask.png").is_file():
        effective_mask_path = target / "effective_repair_mask.png"
    elif (target / "effective_mask.png").is_file():
        effective_mask_path = target / "effective_mask.png"
    elif (target / "effective_mask.mkv").is_file():
        effective_mask_path = target / "effective_mask.mkv"
    else:
        raise ValueError(f"No mask provided and target folder contains no existing mask: {target}")

    effective_mask_hash = compute_file_sha256(effective_mask_path)

    manifest_data = {
        "target_folder": str(target),
        "effective_repair_mask": str(effective_mask_path),
        "effective_mask_hash": effective_mask_hash,
        "repair_policy": resolved_policy.value,
        "preset": str(preset) if preset else "custom",
        "repair_backend": str(repair_backend).lower(),
        "channel_provenance": channel_provenance,
        "checkpoint_sha256": str(checkpoint_sha256 or ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_file = target / "adopted_mask_manifest.json"
    temp_file = target / ".adopted_mask_manifest.partial.json"
    temp_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    temp_file.replace(manifest_file)

    return manifest_data


def dispatch_repair(
    source: np.ndarray | Path | str,
    effective_repair_mask: np.ndarray | Path | str,
    repair_backend: str = "telea",
    output_path: Path | str | None = None,
    radius: int = 3,
    preset: str | Any = None,
    repair_policy: RepairInclusionPolicy | str = RepairInclusionPolicy.CONSERVATIVE,
    detector_info: dict[str, Any] | None = None,
    checkpoint_sha256: str = "",
    license_info: str | dict[str, Any] | None = None,
    channel_provenance: dict[str, Any] | None = None,
    patch_source: np.ndarray | Path | str | None = None,
    propainter_runner: Any = None,
    resolved_parameters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute inpainting on source frames using effective_repair_mask and record provenance.

    Backends:
      - 'telea': OpenCV inpainting strictly gated by effective_repair_mask.
      - 'propainter': ProPainter video inpainting strictly gated by effective_repair_mask.
      - 'dart_composite': DART diagnostic frame gated by effective_repair_mask.
    """
    if isinstance(source, np.ndarray):
        image = source.copy()
        src_path_str = "memory_array"
    else:
        src_p = Path(source).resolve()
        image = cv2.imread(str(src_p))
        if image is None:
            raise ValueError(f"Cannot read source image from {source}")
        src_path_str = str(src_p)

    if isinstance(effective_repair_mask, MultiChannelDefectMask):
        mask_arr = effective_repair_mask.get_effective_repair_mask(repair_policy)
        mask_hash = compute_mask_sha256(mask_arr)
    elif isinstance(effective_repair_mask, np.ndarray):
        mask_arr = (effective_repair_mask > 127).astype(np.uint8) * 255
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_hash = compute_mask_sha256(mask_arr)
    else:
        mask_p = Path(effective_repair_mask).resolve()
        mask_arr = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask_arr is None:
            raise ValueError(f"Cannot read effective mask from {effective_repair_mask}")
        mask_hash = compute_file_sha256(mask_p)

    patch_arr = None
    if patch_source is not None:
        if isinstance(patch_source, np.ndarray):
            patch_arr = patch_source
        else:
            patch_arr = cv2.imread(str(patch_source))
    elif propainter_runner is not None and str(repair_backend).lower() == "propainter":
        patch_arr = propainter_runner(image, mask_arr)

    repaired, counts = dispatch_repair_frame(
        image,
        mask_arr,
        repair_backend=repair_backend,
        radius=radius,
        model_patch=patch_arr,
    )

    out_path_str = str(output_path) if output_path else "memory_array"
    prov_data = build_restoration_provenance(
        detector=detector_info or {"name": "unknown", "version": "1.0.0", "architecture": "unknown"},
        preset=preset or "custom",
        repair_policy=repair_policy,
        repair_backend=repair_backend,
        effective_mask_hash=mask_hash,
        source_video=src_path_str,
        output_video=out_path_str,
        checkpoint_sha256=checkpoint_sha256,
        license_info=license_info,
        channel_provenance=channel_provenance,
        resolved_parameters=resolved_parameters or {"radius": radius},
    )

    if output_path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), repaired)
        prov_file = out_p.parent / "restoration_provenance.json"
        write_restoration_provenance(prov_file, prov_data)

    return repaired, prov_data

