from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

import artifact_ids as aid
import ltx_colorize
import openai_generate_reference as openai_images
from reference_sets import EXTRA_REFERENCES_FIELD, REFERENCE_KEYS, additional_references, reference_items
from comfy_api import ensure_node_types, extract_output_files, free_comfy_models, queue_prompt, wait_for_comfy, wait_for_prompt
from common import (
    ROOT,
    copy_to_comfy_input,
    file_fingerprint,
    find_ffmpeg,
    load_local_config,
    newest_output as newest_comfy_output,
    replace_with_retry,
    resolve_path,
    root_relative,
    safe_stem,
    resumable_output,
    video_info,
    write_signature,
)
from intermediate_video import codec_args as intermediate_codec_args, comfy_video_combine_inputs, container_args
from stream_copy import copy_join, count_video_packets, require_frames

try:
    from ai_remaster_gui.commercial_policy import (
        CommercialPolicyError,
        enabled as commercial_guard_enabled,
        evaluate_component,
    )
except ImportError:
    class CommercialPolicyError(RuntimeError):  # type: ignore[no-redef]
        """Raised when a model component fails commercial policy validation."""

    def commercial_guard_enabled(settings: Any = None) -> bool:  # type: ignore[misc]
        return False

    def evaluate_component(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        class _Decision:
            allowed = True
            reason = ""
        return _Decision()

try:
    from ai_remaster_gui.compliance import append_compliance_record
except ImportError:
    append_compliance_record = None  # type: ignore[assignment]


def get_commercial_settings() -> dict[str, Any]:
    try:
        from ai_remaster_gui import state
        if getattr(state, "APP", None) is not None and getattr(state.APP, "settings", None):
            return state.APP.settings
    except Exception:
        pass
    local_cfg = load_local_config()
    return {
        "global": {
            "commercial_model_guard": str(local_cfg.get("commercial_model_guard", "true")),
            "non_commercial_benchmark_override": str(local_cfg.get("non_commercial_benchmark_override", "false")),
        }
    }


def check_commercial_guard(method: str) -> None:
    settings = get_commercial_settings()
    if commercial_guard_enabled(settings):
        if method in {"cmnet2", "colormnet", "both"}:
            raise CommercialPolicyError(
                f"Commercial Model Guard blocked method '{method}': ColorMNet/CMNet2 non-commercial license (CC BY-NC-SA 4.0) prohibits commercial use."
            )
        if method == "osi_masked_lab":
            decision = evaluate_component("osi_masked_lab", settings=settings, component_type="backend")
            if not decision.allowed:
                raise CommercialPolicyError(f"Commercial Model Guard blocked osi_masked_lab: {decision.reason}")


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
REFERENCE_INPUT_COPY_STRATEGY = "content-keyed-v1"


def read_manifest(path: Path) -> tuple[str | None, list[dict[str, str]]]:
    source_video = None
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith("#"):
                if line.startswith("# source_video="):
                    source_video = line.split("=", 1)[1].strip()
                continue
            handle.seek(pos)
            for row in csv.DictReader(handle):
                if row.get("enabled", "true").strip().lower() not in {"false", "0", "no", "off"}:
                    rows.append(row)
            break
    return source_video, rows


def parse_time(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    seconds = float(parts[-1])
    minutes = int(parts[-2])
    hours = int(parts[-3]) if len(parts) > 2 else 0
    return hours * 3600 + minutes * 60 + seconds


def method_suffix(method: str) -> str:
    if method in {"colormnet", "cmnet2", "openai", "ltx", "osi_masked_lab"}:
        return method
    return "deepexemplar"


def processing_dimensions(width: int, height: int, processing_height: str) -> tuple[int, int]:
    text = str(processing_height or "source").strip().lower()
    if text in {"", "source", "original", "0"}:
        return width, height
    try:
        target_h = int(float(text.rstrip("p")))
    except ValueError:
        return width, height
    if target_h <= 0 or height <= target_h:
        return width, height
    scale = target_h / height
    target_w = max(2, int(round(width * scale / 2) * 2))
    target_h = max(2, int(round(target_h / 2) * 2))
    return target_w, target_h


CURRENT_INTERMEDIATE_PROFILE = "high"


def working_codec_args(*, fast: bool = False) -> list[str]:
    return intermediate_codec_args(CURRENT_INTERMEDIATE_PROFILE, fast=fast)


def working_container_args(path: Path) -> list[str]:
    return container_args(str(path), CURRENT_INTERMEDIATE_PROFILE)


def downscaled_video_signature(source: Path, width: int, height: int) -> dict[str, Any]:
    return {
        "version": 2,
        "tool": "colorize_video.py",
        "kind": "grayscale processing input",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "width": width,
        "height": height,
        "grayscale": True,
        "intermediate_profile": CURRENT_INTERMEDIATE_PROFILE,
    }


def prepare_processing_video(ffmpeg: str, source: Path, width: int, height: int, original_width: int, original_height: int) -> Path:
    """Create the shared, always-grayscale video input used by both colourisation backends.

    Dearchive can introduce colour, so this intermediate is produced even when the requested
    processing dimensions match the source. Shot ranges are selected from this file in ComfyUI.
    """
    digest = file_fingerprint(source)["sha256"][:12]
    output = ROOT / ".cache" / "colorize_inputs" / f"{safe_stem(source.name)}_{digest}_{width}x{height}_gray.mkv"
    sig = downscaled_video_signature(source, width, height)
    if resumable_output(output, sig, width=width, height=height):
        print(f"Reuse grayscale colourisation input: {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,hue=s=0,setsar=1",
        "-an",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)
    write_signature(output, sig)
    print(f"Wrote grayscale colourisation input: {output}", flush=True)
    return output


def prepare_processing_reference(reference: Path, width: int, height: int) -> Path:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return reference

    try:
        with Image.open(reference) as image:
            if image.width == width and image.height == height:
                return reference
            digest = file_fingerprint(reference)["sha256"][:12]
            output = ROOT / ".cache" / "colorize_refs" / f"{safe_stem(reference.stem)}_{digest}_{width}x{height}.png"
            sig = {
                "version": 1,
                "tool": "colorize_video.py",
                "kind": "processing reference",
                "source": root_relative(reference),
                "source_fingerprint": file_fingerprint(reference),
                "width": width,
                "height": height,
            }
            if resumable_output(output, sig, width=width, height=height):
                return output
            output.parent.mkdir(parents=True, exist_ok=True)
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            fitted = ImageOps.contain(image.convert("RGB"), (width, height), resampling)
            canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
            canvas.save(output, format="PNG")
            write_signature(output, sig)
            return output
    except Exception:
        return reference


def copy_reference_to_comfy_input(reference: Path, comfy_dir: Path, width: int | None = None, height: int | None = None) -> str:
    if width and height:
        reference = prepare_processing_reference(reference, width, height)
    target_dir = comfy_dir / "input" / "arp_colorize_refs"
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = file_fingerprint(reference)["sha256"][:12]
    suffix = reference.suffix.lower() or ".png"
    target = target_dir / f"{safe_stem(reference.stem)}_{digest}{suffix}"
    if (
        not target.exists()
        or target.stat().st_size != reference.stat().st_size
        or file_fingerprint(target)["sha256"] != file_fingerprint(reference)["sha256"]
    ):
        shutil.copy2(reference, target)
    return str(Path("arp_colorize_refs") / target.name).replace("\\", "/")


def default_output(manifest: Path, manifest_source: str | None, method: str) -> Path:
    # Match the artifact name the GUI locates by (references.colorized_output_for_manifest), so the
    # method="both" path — the only path that reaches default_output, since single methods get an
    # explicit --output — writes files the UI can find. The old "<stem>_<method>_colorized.mp4" name
    # drifted from the artifact-id scheme and left "both" outputs invisible to the GUI.
    ident = aid.colorized_identity(manifest.stem, method)
    name_src = Path(manifest_source).name if manifest_source else manifest.name
    return ROOT / "intermediate" / "outpainted_colorized" / aid.artifact_name(aid.source_word(name_src), "color", ident, "mkv")


def reference_signature(row: dict[str, str]) -> dict[str, Any]:
    refs = row_references(row)
    return {
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "start_frame": row.get("start_frame", ""),
        "end_frame": row.get("end_frame", ""),
        "references": [
            {"selected_frame": item["selected_frame"], "reference": root_relative(item["path"]), "reference_fingerprint": file_fingerprint(item["path"])}
            for item in refs
        ],
        "fade_to_next": row.get("fade_to_next", ""),
        "crossfade_seconds": row.get("crossfade_seconds", ""),
    }


def row_path_signature(row: dict[str, str], key: str) -> dict[str, Any]:
    text = row.get(key, "")
    if not text:
        return {"path": ""}
    path = resolve_path(text)
    sig: dict[str, Any] = {"path": root_relative(path)}
    if path.exists():
        sig["fingerprint"] = file_fingerprint(path)
    else:
        sig["missing"] = True
    return sig


def shot_input_signature(row: dict[str, str]) -> dict[str, Any]:
    """Everything about a manifest row that can make a cached shot segment stale."""
    refs = row_references(row)
    unsigned_row_keys = {"color_reference_previous", *LTX_ROW_KEYS}
    signed = {key: row.get(key, "") for key in sorted(row) if key not in unsigned_row_keys}
    if signed.get(EXTRA_REFERENCES_FIELD):
        # Extra references carry their own optional keys; only what they show is signed here.
        signed[EXTRA_REFERENCES_FIELD] = json.dumps(
            [{key: item.get(key, "") for key in REFERENCE_KEYS} for item in additional_references(row)], separators=(",", ":"),
        )
    return {
        "row": signed,
        "source_reference": row_path_signature(row, "source_reference"),
        "color_reference": row_path_signature(row, "color_reference"),
        "references": [
            {"selected_frame": item["selected_frame"], "reference": root_relative(item["path"]), "reference_fingerprint": file_fingerprint(item["path"])}
            for item in refs
        ],
    }


def method_settings_signature(args: argparse.Namespace) -> dict[str, Any]:
    settings = {
        "method": args.method,
        "use_torch_compile": args.use_torch_compile,
        # Retired ComfyUI encoder settings, frozen at their last defaults so existing
        # segments stay valid; ComfyUI renders now follow the intermediate profile.
        "video_format": "video/h264-mp4",
        "crf": 18,
        "processing_height": getattr(args, "processing_height", "source"),
        "intermediate_profile": getattr(args, "intermediate_profile", "high"),
    }
    if args.method == "openai":
        settings.update(
            {
                "openai_model": args.openai_model,
                "openai_previous_frames": args.openai_previous_frames,
                "openai_prompt": args.openai_prompt,
                "openai_size": args.openai_size,
                "openai_quality": args.openai_quality,
            }
        )
    elif args.method == "colormnet":
        settings.update(
            {
                "colormnet_memory_mode": args.colormnet_memory_mode,
                "colormnet_feature_encoder": args.colormnet_feature_encoder,
                "colormnet_text_guidance": args.colormnet_text_guidance,
                "colormnet_text_guidance_weight": args.colormnet_text_guidance_weight,
            }
        )
    elif args.method == "ltx":
        settings.update(ltx_colorize.settings_signature(args))
    elif args.method == "cmnet2":
        settings.update(
            {
                "cmnet2_top_k": args.cmnet2_top_k,
                "cmnet2_mem_every": args.cmnet2_mem_every,
                "cmnet2_runtime": root_relative(resolve_path(args.cmnet2_dir)),
                "cmnet2_model": root_relative(resolve_path(args.cmnet2_model)) if args.cmnet2_model else "auto",
            }
        )
    elif args.method == "osi_masked_lab":
        settings.update(
            {
                "masks_dir": getattr(args, "masks_dir", ""),
                "guided_radius": getattr(args, "guided_radius", 5),
                "guided_eps": getattr(args, "guided_eps", 1e-3),
                "flow_scale": getattr(args, "flow_scale", 0.5),
                "anchor_strength": getattr(args, "anchor_strength", 0.75),
            }
        )
    else:
        settings.update(
            {
                "frame_propagate": args.frame_propagate,
                "use_half_resolution": args.use_half_resolution,
                "use_sage_attention": args.use_sage_attention,
            }
        )
    return settings


def signature(args: argparse.Namespace, manifest: Path, source_video: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "version": 11,
        "tool": "colorize_video.py",
        "reference_input_copy": REFERENCE_INPUT_COPY_STRATEGY,
        "manifest": root_relative(manifest),
        "manifest_fingerprint": file_fingerprint(manifest),
        "source_video": root_relative(source_video),
        "source_fingerprint": file_fingerprint(source_video),
        "grayscale_video_input": args.method != "openai",
        "references": [reference_signature(row) for row in rows],
        "shot_inputs": [shot_input_signature(row) for row in rows],
        "settings": method_settings_signature(args),
        **({"ltx_shots": [ltx_shot_signature(row) for row in rows]} if args.method == "ltx" else {}),
    }


def row_reference(row: dict[str, str]) -> Path:
    ref = row.get("color_reference") or row.get("reference") or row.get("source_reference") or ""
    if not ref:
        raise RuntimeError("Manifest row has no color_reference/reference/source_reference.")
    path = resolve_path(ref)
    if not path.exists() and row.get("source_reference"):
        path = resolve_path(row["source_reference"])
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")
    return path


def row_references(row: dict[str, str]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for item in reference_items(row):
        effective = dict(row)
        effective.update(item)
        values.append({
            "path": row_reference(effective),
            "selected_frame": optional_int(item.get("selected_frame")),
            "guide_strength": optional_float(item.get("guide_strength")),
        })
    return values


def optional_float(value: str | None) -> float | None:
    try:
        text = str(value or "").strip()
        return float(text) if text else None
    except ValueError:
        return None


# LTX Colorizer per-shot and per-reference settings. Only the LTX method reads them, so they never
# make another method's cached segments stale.
LTX_ROW_KEYS = ("ltx_lora_strength", "guide_strength")


def ltx_shot_signature(row: dict[str, str]) -> dict[str, Any]:
    return {
        "ltx_lora_strength": row.get("ltx_lora_strength", ""),
        "guide_strengths": [item["guide_strength"] for item in row_references(row)],
    }


def build_prompt(
    video_name: str,
    ref_names: str | list[str],
    start_frame: int,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    args: argparse.Namespace,
    prefix: str,
) -> dict[str, Any]:
    if isinstance(ref_names, str):
        ref_names = [ref_names]
    if not ref_names:
        raise RuntimeError("At least one reference image is required.")
    prompt: dict[str, Any] = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": frame_count,
                "skip_first_frames": start_frame,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                **comfy_video_combine_inputs(args.intermediate_profile),
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    if len(ref_names) != 1:
        raise RuntimeError("ColorMNet and Deep Exemplar accept exactly one reference image per shot.")
    prompt["2"] = {"class_type": "LoadImage", "inputs": {"image": ref_names[0]}}
    prompt["3"] = colorization_node(args, width, height, ["2", 0])
    return prompt


def colorization_node(args: argparse.Namespace, width: int, height: int, reference_input: list[Any] | None = None) -> dict[str, Any]:
    reference_input = reference_input or ["2", 0]
    if args.method == "colormnet":
        return {
            "class_type": "ColorMNetVideo",
            "inputs": {
                "video_frames": ["1", 0],
                "reference_image": reference_input,
                "target_width": width,
                "target_height": height,
                "memory_mode": args.colormnet_memory_mode,
                "feature_encoder": args.colormnet_feature_encoder,
                "use_fp16": True,
                "use_torch_compile": args.use_torch_compile,
                "text_guidance": args.colormnet_text_guidance,
                "text_guidance_weight": args.colormnet_text_guidance_weight,
            },
        }
    return {
        "class_type": "DeepExColorVideoNode",
        "inputs": {
            "video_frames": ["1", 0],
            "reference_image": reference_input,
            "frame_propagate": args.frame_propagate,
            "use_half_resolution": args.use_half_resolution,
            "target_width": width,
            "target_height": height,
            "use_torch_compile": args.use_torch_compile,
            "use_sage_attention": args.use_sage_attention,
        },
    }


def newest_output(files: list[Path]) -> Path:
    return newest_comfy_output(files, VIDEO_EXTS, "video output")


def segment_signature(
    args: argparse.Namespace,
    source_video: Path,
    row: dict[str, str],
    references: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
    base_start_frame: int,
    base_end_frame: int,
    width: int,
    height: int,
    fps: float,
    *,
    mask_fingerprint: dict[str, Any] | None = None,
    color_bible_version: str | None = None,
) -> dict[str, Any]:
    if isinstance(references, Path):
        references = [{"path": references, "selected_frame": optional_int(row.get("selected_frame"))}]
    sig: dict[str, Any] = {
        "version": 11,
        "tool": "colorize_video.py",
        "kind": f"{args.method} segment",
        "reference_input_copy": REFERENCE_INPUT_COPY_STRATEGY,
        "source_video": root_relative(source_video),
        "source_fingerprint": file_fingerprint(source_video),
        "grayscale_video_input": True,
        "references": [
            {"selected_frame": item["selected_frame"], "reference": root_relative(item["path"]), "reference_fingerprint": file_fingerprint(item["path"])}
            for item in references
        ],
        "shot_input": shot_input_signature(row),
        "row_start": row.get("start", ""),
        "row_end": row.get("end", ""),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "base_start_frame": base_start_frame,
        "base_end_frame": base_end_frame,
        "fade_to_next": row.get("fade_to_next", ""),
        "crossfade_seconds": row.get("crossfade_seconds", ""),
        "width": width,
        "height": height,
        "fps": fps,
        "settings": method_settings_signature(args),
    }
    if mask_fingerprint is not None:
        sig["mask_fingerprint"] = mask_fingerprint
    if color_bible_version is not None:
        sig["color_bible_version"] = color_bible_version
    if row.get("color_bible_anchors"):
        sig["color_bible_anchors"] = row.get("color_bible_anchors")
    return sig


def segment_resumable(chunk: Path, chunk_sig: dict[str, Any], width: int, height: int, expected_frames: int) -> bool:
    if not resumable_output(chunk, chunk_sig, width=width, height=height):
        return False
    try:
        info = video_info(chunk)
    except Exception:
        return False
    return abs(int(info["frames"]) - expected_frames) <= 3


def normalize_clip(ffmpeg: str, source: Path, output: Path, fps: float, expected_frames: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    vf = f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},trim=end_frame={expected_frames},setpts=N/({fps:.8f}*TB),setsar=1"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        vf,
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def transition_seconds(row: dict[str, str]) -> float:
    if not truthy(row.get("fade_to_next", "")):
        return 0.0
    try:
        return max(0.0, float(row.get("crossfade_seconds", "") or 0.0))
    except ValueError:
        return 0.0

def optional_int(value: str | None) -> int | None:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return int(float(text))
    except ValueError:
        return None


def shot_plan(rows: list[dict[str, str]], total_frames: int, fps: float) -> tuple[list[dict[str, int]], list[int]]:
    base: list[tuple[int, int]] = []
    start_frame = 0
    for index, row in enumerate(rows):
        row_start = optional_int(row.get("start_frame"))
        row_end = optional_int(row.get("end_frame"))
        if row_start is not None:
            start_frame = max(0, min(total_frames - 1, row_start))
        if row_end is not None:
            end_frame = min(total_frames, max(start_frame + 1, row_end))
        else:
            end_frame = min(total_frames, max(start_frame + 1, round(parse_time(row.get("end", "")) * fps)))
        if index == len(rows) - 1:
            end_frame = total_frames
        base.append((start_frame, end_frame))
        start_frame = end_frame

    transitions = [0 for _ in rows]
    for index, row in enumerate(rows[:-1]):
        frames = int(round(transition_seconds(row) * fps))
        if frames <= 0:
            continue
        left = max(1, base[index][1] - base[index][0])
        right = max(1, base[index + 1][1] - base[index + 1][0])
        transitions[index] = max(1, min(frames, left, right))

    plan: list[dict[str, int]] = []
    for index, (base_start, base_end) in enumerate(base):
        prev_frames = transitions[index - 1] if index > 0 else 0
        next_frames = transitions[index] if index < len(transitions) else 0
        pre = prev_frames // 2
        post = next_frames - (next_frames // 2)
        actual_start = max(0, base_start - pre)
        actual_end = min(total_frames, base_end + post)
        plan.append(
            {
                "base_start": base_start,
                "base_end": base_end,
                "start": actual_start,
                "end": max(actual_start + 1, actual_end),
            }
        )
    return plan, transitions


def stitch(ffmpeg: str, chunks: list[Path], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # The segments are already in the working codec: join their packets rather than re-encoding
    # the whole film, falling back to the re-encode only if they cannot be joined as they are.
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    expected = sum(count_video_packets(ffmpeg, chunk) for chunk in chunks)
    print(f"Joining {len(chunks)} colourised segment(s) into {output.name} ({expected} frames)", flush=True)
    if copy_join(ffmpeg, chunks, partial, expected, working_container_args(partial)):
        replace_with_retry(partial, output)
        return
    concat = output.with_suffix(".concat.txt")
    concat.write_text("".join("file '" + str(chunk).replace("'", "'\\''") + "'\n" for chunk in chunks), encoding="utf-8")
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        # Segments can differ in colour tags; without -reinit_filter 0 each change rebuilds the
        # filter graph, restarting setpts' N, and the fps filter then drops thousands of frames.
        "-reinit_filter",
        "0",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-vf",
        f"setpts=N/({fps:.8f}*TB),fps={fps:.8f},setsar=1",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    concat.unlink(missing_ok=True)
    require_frames(ffmpeg, partial, expected, "The colourised video")
    replace_with_retry(partial, output)


def transition_groups(transitions: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = 0
    for index, frames in enumerate(transitions):
        if frames > 0:
            continue
        groups.append((start, index))
        start = index + 1
    if transitions:
        groups.append((start, len(transitions) - 1))
    elif not groups:
        groups.append((0, 0))
    return [(left, right) for left, right in groups if left <= right]


def xfade_group(ffmpeg: str, chunks: list[Path], transitions: list[int], output: Path, fps: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [ffmpeg, "-y"]
    for chunk in chunks:
        cmd += ["-i", str(chunk)]

    filters: list[str] = []
    for index in range(len(chunks)):
        filters.append(f"[{index}:v]setpts=N/({fps:.8f}*TB),fps=fps={fps:.8f}[v{index}]")

    previous = "v0"
    accumulated = video_info(chunks[0])["frames"] / fps
    for index in range(1, len(chunks)):
        duration = max(1 / fps, transitions[index - 1] / fps)
        offset = max(0.0, accumulated - duration)
        current = f"x{index}"
        filters.append(f"[{previous}][v{index}]xfade=transition=fade:duration={duration:.8f}:offset={offset:.8f},setpts=PTS-STARTPTS[{current}]")
        previous = current
        accumulated += video_info(chunks[index])["frames"] / fps - duration
    pixel_format = "yuv444p10le" if CURRENT_INTERMEDIATE_PROFILE == "lossless" else "yuv420p"
    filters.append(f"[{previous}]format={pixel_format}[vout]")

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def stitch_colorized(ffmpeg: str, chunks: list[Path], transitions: list[int], output: Path, fps: float, source_stem: str) -> None:
    if not any(transitions):
        stitch(ffmpeg, chunks, output, fps)
        return

    group_outputs: list[Path] = []
    group_dir = ROOT / ".cache" / "colorized_chunks" / "crossfaded" / source_stem
    group_dir.mkdir(parents=True, exist_ok=True)
    for group_index, (left, right) in enumerate(transition_groups(transitions)):
        group_chunks = chunks[left : right + 1]
        if len(group_chunks) == 1:
            group_outputs.append(group_chunks[0])
            continue
        group_transitions = transitions[left:right]
        group_output = group_dir / f"group_{group_index:04d}_{left:04d}_{right:04d}.mkv"
        # Only a group whose shots or fades changed is crossfaded again.
        group_sig = {
            "kind": "colourised crossfade group",
            "chunks": [file_fingerprint(chunk) for chunk in group_chunks],
            "transitions": group_transitions,
            "fps": fps,
            "profile": CURRENT_INTERMEDIATE_PROFILE,
        }
        if resumable_output(group_output, group_sig):
            print(f"Reuse crossfaded shots {left + 1}-{right + 1}: {group_output.name}", flush=True)
        else:
            xfade_group(ffmpeg, group_chunks, group_transitions, group_output, fps)
            write_signature(group_output, group_sig)
        group_outputs.append(group_output)
    stitch(ffmpeg, group_outputs, output, fps)


def prepare_openai_source_frames(
    ffmpeg: str,
    source: Path,
    width: int,
    height: int,
    total_frames: int,
) -> Path:
    """Extract colour-preserving source frames for cloud image editing.

    The directory is content-addressed, so it remains safe to reuse across interrupted jobs and
    never shares frames with the compulsory grayscale inputs used by the local colourizers.
    """
    digest = file_fingerprint(source)["sha256"][:16]
    frame_dir = ROOT / ".cache" / "openai_colorize" / f"{safe_stem(source.name)}_{digest}_{width}x{height}" / "source"
    expected = [frame_dir / f"frame_{index:08d}.png" for index in range(total_frames)]
    if expected and all(path.is_file() for path in expected):
        print(f"Reuse {total_frames} colour source frames for OpenAI Cloud: {frame_dir}", flush=True)
        return frame_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / "frame_%08d.png"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-an",
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(pattern),
    ]
    print(f"Extracting {total_frames} colour-preserving frames for OpenAI Cloud...", flush=True)
    subprocess.run(cmd, check=True)
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"FFmpeg extracted an incomplete frame sequence; first missing frame: {missing[0]}")
    print(f"Extracted {total_frames} source frames: {frame_dir}", flush=True)
    return frame_dir


def openai_frame_prompt(previous_count: int, reference_count: int) -> tuple[str, str]:
    parts = ["The first image is the current source frame and is the only image to edit."]
    if previous_count:
        if previous_count > 1:
            parts.append(
                f"Images 2-{previous_count + 1} are preceding generated frames in chronological order, oldest first; "
                "use them only for temporal continuity of colours, texture, grain, lighting and identity."
            )
        else:
            parts.append(
                "Image 2 is the immediately preceding generated frame; use it only for temporal continuity "
                "of colours, texture, grain, lighting and identity."
            )
    reference_start = previous_count + 2
    reference_end = previous_count + reference_count + 1
    if reference_count:
        if reference_count > 1:
            parts.append(
                f"Images {reference_start}-{reference_end} are the approved colour references for this shot; "
                "use all of them to infer stable object, clothing, skin, architecture and landscape colours throughout the pan."
            )
        else:
            parts.append(
                f"Image {reference_start} is the approved colour reference for this shot; use it to infer "
                "stable object, clothing, skin, architecture and landscape colours throughout the pan."
            )
    parts.append(
        "Never copy composition, people, poses or geometry from a preceding frame or reference into the current frame."
    )
    instructions = " ".join(parts)
    return "", instructions


def openai_frame_signature(
    args: argparse.Namespace,
    source_frame: Path,
    previous_frames: list[Path],
    references: list[Path],
    shot_index: int,
    frame_index: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "tool": "colorize_video.py",
        "kind": "OpenAI Cloud colourized frame",
        "frame": frame_index,
        "shot": shot_index,
        "source": file_fingerprint(source_frame),
        "previous_generated_frames": [file_fingerprint(path) for path in previous_frames],
        "references": [
            {"path": root_relative(path), "fingerprint": file_fingerprint(path)} for path in references
        ],
        "model": args.openai_model,
        "prompt": args.openai_prompt,
        "size": args.openai_size,
        "quality": args.openai_quality,
    }


def assemble_openai_frames(ffmpeg: str, frame_dir: Path, output: Path, fps: float, total_frames: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.12g}",
        "-start_number",
        "0",
        "-i",
        str(frame_dir / "frame_%08d.png"),
        "-frames:v",
        str(total_frames),
        "-an",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    print("Assembling OpenAI Cloud frames into the colorized video...", flush=True)
    subprocess.run(cmd, check=True)
    replace_with_retry(partial, output)


def run_openai_colorization(
    args: argparse.Namespace,
    manifest: Path,
    source_video: Path,
    rows: list[dict[str, str]],
    output: Path,
    output_sig: dict[str, Any],
    ffmpeg: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
) -> int:
    token = str(args.api_key or "").strip()
    if not token:
        raise RuntimeError("Missing OpenAI API key. Add it in ARP Settings before running OpenAI Cloud colorization.")
    previous_limit = max(0, min(12, int(args.openai_previous_frames)))
    plan, _transitions = shot_plan(rows, total_frames, fps)
    source_dir = prepare_openai_source_frames(ffmpeg, source_video, width, height, total_frames)
    # Keep the cache location stable when a project save rewrites identical files with new mtimes.
    # Per-frame signatures below still invalidate precisely when source, prompt, model, history or
    # reference content changes.
    cache_identity = {
        "source_sha256": output_sig["source_fingerprint"]["sha256"],
        "manifest": root_relative(manifest),
        "width": width,
        "height": height,
        "model": args.openai_model,
    }
    digest = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    generated_dir = ROOT / ".cache" / "openai_colorize" / f"{safe_stem(source_video.name)}_{digest}_{width}x{height}" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    row_references_cache = [[item["path"] for item in row_references(row)] for row in rows]
    print(
        f"OpenAI Cloud colorization: {total_frames} frames, model {args.openai_model}, "
        f"up to {previous_limit} preceding frame(s), all shot references.",
        flush=True,
    )
    print("Each completed frame is cached; this job can be stopped and resumed.", flush=True)
    history: list[Path] = []
    active_shot = -1
    reused = 0
    for frame_index in range(total_frames):
        shot_index = next(
            (index for index, item in enumerate(plan) if item["base_start"] <= frame_index < item["base_end"]),
            len(rows) - 1,
        )
        if shot_index != active_shot:
            history = []
            active_shot = shot_index
        previous = history[-previous_limit:] if previous_limit else []
        references = row_references_cache[shot_index]
        source_frame = source_dir / f"frame_{frame_index:08d}.png"
        generated = generated_dir / f"frame_{frame_index:08d}.png"
        frame_sig = openai_frame_signature(args, source_frame, previous, references, shot_index, frame_index)
        if not args.force and resumable_output(generated, frame_sig, width=width, height=height):
            reused += 1
        else:
            role_prompt, reference_instructions = openai_frame_prompt(len(previous), len(references))
            child = argparse.Namespace(
                source_image=source_frame,
                output=generated,
                api_key=token,
                model=args.openai_model,
                prompt=args.openai_prompt,
                prompt_suffix="",
                add_prompt=role_prompt,
                reference_instructions=reference_instructions,
                size=args.openai_size,
                quality=args.openai_quality,
                timeout=args.openai_timeout,
                max_retries=args.openai_max_retries,
                force=True,
                no_normalize_to_source_size=False,
                dry_run=False,
                quiet=True,
            )
            # Input order is significant and is described explicitly in the prompt.
            openai_images.generate(child, [*previous, *references])
            write_signature(generated, frame_sig)
        history.append(generated)
        print(
            f"OpenAI Cloud frame {frame_index + 1}/{total_frames}; "
            f"shot {shot_index + 1}/{len(rows)}; cached {reused}",
            flush=True,
        )
    assemble_openai_frames(ffmpeg, generated_dir, output, fps, total_frames)
    write_signature(output, output_sig)
    print(f"Wrote OpenAI Cloud colorized video: {output}", flush=True)
    return 0


def run_cmnet2_colorization(
    args: argparse.Namespace,
    source_video: Path,
    rows: list[dict[str, str]],
    output: Path,
    output_sig: dict[str, Any],
    ffmpeg: str,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
) -> int:
    """Colorize each true shot with every one of its timed CMNET2 reference anchors."""
    from cmnet2_runtime import CMNet2Session, resolve_model

    runtime_dir = resolve_path(args.cmnet2_dir)
    model_path = resolve_model(runtime_dir, args.cmnet2_model, args.comfy_dir)
    plan, transitions = shot_plan(rows, total_frames, fps)
    cache_dir = ROOT / ".cache" / "colorized_chunks" / "cmnet2" / safe_stem(source_video.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session: CMNet2Session | None = None
    chunks: list[Path] = []

    for index, row in enumerate(rows):
        item = plan[index]
        start_frame, end_frame = item["start"], item["end"]
        frame_count = max(1, end_frame - start_frame)
        references = row_references(row)
        prepared_references = [
            {
                **reference,
                "path": prepare_processing_reference(reference["path"], width, height),
            }
            for reference in references
        ]
        chunk = cache_dir / f"segment_{index:04d}_{start_frame:06d}_{end_frame:06d}.mkv"
        chunk_sig = segment_signature(
            args,
            source_video,
            row,
            references,
            start_frame,
            end_frame,
            item["base_start"],
            item["base_end"],
            width,
            height,
            fps,
        )
        if not args.force and segment_resumable(chunk, chunk_sig, width, height, frame_count):
            print(f"Reuse CMNET2 segment {index + 1}/{len(rows)}: {chunk}", flush=True)
            chunks.append(chunk)
            continue
        if session is None:
            session = CMNet2Session(
                runtime_dir,
                model_path,
                top_k=args.cmnet2_top_k,
                mem_every=args.cmnet2_mem_every,
                max_frames=total_frames,
            )
        frame_labels = ", ".join(
            str(reference["selected_frame"]) if reference["selected_frame"] is not None else "unspecified"
            for reference in references
        )
        print(
            f"Colorize segment {index + 1}/{len(rows)} with CMNET2: frames {start_frame}-{end_frame}; "
            f"{len(references)} permanent reference(s) at source frames {frame_labels}",
            flush=True,
        )
        session.colorize_segment(
            source_video,
            chunk,
            prepared_references,
            start_frame=start_frame,
            end_frame=end_frame,
            width=width,
            height=height,
            fps=fps,
            ffmpeg=ffmpeg,
            intermediate_profile=args.intermediate_profile,
        )
        write_signature(chunk, chunk_sig)
        chunks.append(chunk)

    stitch_colorized(ffmpeg, chunks, transitions, output, fps, safe_stem(source_video.name))
    write_signature(output, output_sig)
    print(f"Wrote CMNET2 colorized video: {output}", flush=True)
    return 0


def find_shot_mask_archive(masks_dir: Path, row: dict[str, str], index: int, film_id: str = "") -> Path | None:
    # 1. Direct row specification
    for key in ("masks", "mask_archive", "shot_mask", "shot_masks"):
        if row.get(key):
            p = resolve_path(row[key])
            if p.is_file():
                return p

    shot_id = str(row.get("shot_id") or row.get("shot") or row.get("id") or "").strip()
    search_dirs = [masks_dir, ROOT / "intermediate" / "shot_masks"]
    try:
        from ai_remaster_gui.color_bible_context import find_shot_mask_archive_candidates, shot_id_candidates
    except ImportError:
        find_shot_mask_archive_candidates = None
        shot_id_candidates = None

    for d in search_dirs:
        if not d.exists():
            continue
        if find_shot_mask_archive_candidates is not None:
            found = find_shot_mask_archive_candidates(d, index, film_id=film_id, shot_id=shot_id)
            if found is not None:
                return found
        extra_stems = shot_id_candidates(film_id, index) if shot_id_candidates is not None else [f"shot_{index:04d}", f"shot_{index:03d}", f"shot_{index}", f"{index:04d}", f"{index}"]
        if shot_id and shot_id not in extra_stems:
            extra_stems.insert(0, shot_id)
        for stem in extra_stems:
            for suffix in (".masks.npz", ".npz", "_masks.npz"):
                cand_path = d / f"{stem}{suffix}"
                if cand_path.is_file():
                    return cand_path
    return None


def extract_shot_grayscale_frames(
    processing_video: Path,
    start_frame: int,
    frame_count: int,
    width: int,
    height: int,
    ffmpeg_exe: str | None = None,
) -> list[Any]:
    import subprocess
    import cv2
    import numpy as np

    start_idx = max(0, int(start_frame))
    end_idx = start_idx + max(1, int(frame_count))
    frames: list[np.ndarray] = []

    ffmpeg_bin = ffmpeg_exe or shutil.which("ffmpeg")
    if ffmpeg_bin:
        vf = f"trim=start_frame={start_idx}:end_frame={end_idx},setpts=PTS-STARTPTS,scale={width}:{height}:flags=area,format=gray"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(processing_video),
            "-vf",
            vf,
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = width * height
        if proc.returncode == 0 and len(proc.stdout) >= frame_bytes:
            raw = np.frombuffer(proc.stdout, dtype=np.uint8)
            count = min(frame_count, len(raw) // frame_bytes)
            for idx in range(count):
                slice_arr = raw[idx * frame_bytes : (idx + 1) * frame_bytes].reshape((height, width))
                frames.append(slice_arr.copy())

    if not frames:
        cap = cv2.VideoCapture(str(processing_video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open processing video for frame extraction: {processing_video}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            for _ in range(frame_count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                if gray.shape != (height, width):
                    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
                frames.append(gray)
        finally:
            cap.release()

    if len(frames) < frame_count:
        if frames:
            while len(frames) < frame_count:
                frames.append(frames[-1].copy())
        else:
            for _ in range(frame_count):
                frames.append(np.zeros((height, width), dtype=np.uint8))
    return frames


def write_frames_to_video(
    ffmpeg: str,
    frames: list[Any],
    output: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    import numpy as np

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-r",
        f"{fps:.8f}",
        "-fps_mode",
        "cfr",
        *working_codec_args(),
        *working_container_args(partial),
        str(partial),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        arr = np.asarray(f, dtype=np.uint8)
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"FFmpeg failed writing segment {partial}: {err}")
    if proc.stderr:
        proc.stderr.close()
    replace_with_retry(partial, output)


def run_osi_masked_lab_colorization(
    args: argparse.Namespace,
    manifest: Path | None = None,
    source_video: Path | None = None,
    rows: list[dict[str, str]] | None = None,
    output: Path | None = None,
    sig: dict[str, Any] | None = None,
    ffmpeg: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    total_frames: int | None = None,
) -> int:
    try:
        from scripts.osi_masked_lab import propagate_shot_chroma
        from scripts.rle_masks import load_shot_masks_npz
    except ImportError:
        from osi_masked_lab import propagate_shot_chroma
        from rle_masks import load_shot_masks_npz
    import cv2
    import numpy as np

    # 1. Enforce commercial clearance
    check_commercial_guard(args.method)

    # 2. Setup inputs and manifest if not passed
    if manifest is None:
        manifest = resolve_path(args.manifest)
    if rows is None:
        source_from_manifest, rows = read_manifest(manifest)
    else:
        source_from_manifest = None
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No enabled rows in manifest: {manifest}")

    if source_video is None:
        source_video = resolve_path(args.source_video or source_from_manifest or "")
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found for colourisation: {source_video}")

    source_info = video_info(source_video)
    source_width, source_height = int(source_info["width"]), int(source_info["height"])
    target_width, target_height = processing_dimensions(source_width, source_height, args.processing_height)

    if output is None:
        output = resolve_path(args.output) if args.output else default_output(manifest, source_from_manifest, args.method)

    if sig is None:
        sig = signature(args, manifest, source_video, rows)

    if not args.force and resumable_output(output, sig, width=target_width, height=target_height):
        print(f"Reuse colorized video: {output}", flush=True)
        return 0

    if args.dry_run:
        counts = [len(row_references(row)) for row in rows]
        print(
            f"Would colorize {len(rows)} shot segment(s) with OSI Masked LAB: {source_video} -> {output}; "
            f"references per shot: {counts}.",
            flush=True,
        )
        return 0

    if ffmpeg is None:
        ffmpeg = find_ffmpeg(args.ffmpeg)

    processing_video = prepare_processing_video(
        ffmpeg, source_video, target_width, target_height, source_width, source_height
    )
    info = video_info(processing_video)
    width = int(info["width"])
    height = int(info["height"])
    fps = float(info["fps"])
    total_frames = int(info["frames"])

    # 3. Divide processing video into shots
    plan, transitions = shot_plan(rows, total_frames, fps)
    cache_dir = ROOT / ".cache" / "colorized_chunks" / "osi_masked_lab" / safe_stem(source_video.name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = resolve_path(args.masks_dir) if getattr(args, "masks_dir", None) else ROOT / "intermediate" / "shot_masks"

    film_id = getattr(args, "film_id", None)
    if not film_id and manifest and manifest.is_file():
        try:
            with manifest.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.startswith("#"):
                        break
                    if line.startswith("# color_bible_film_id="):
                        film_id = line.split("=", 1)[1].strip()
                        break
                    if line.startswith("# color_bible="):
                        film_id = line.split("=", 1)[1].strip()
                        break
        except Exception:
            pass

    color_bible_db_path = (
        Path(args.color_bible_db)
        if getattr(args, "color_bible_db", None)
        else ROOT / "manifests" / "color_bible.db"
    )

    service = None
    if film_id and color_bible_db_path.is_file():
        try:
            from ai_remaster_gui.color_bible_service import ColorBibleService
            service = ColorBibleService(db_path=color_bible_db_path)
        except Exception:
            try:
                from color_bible_service import ColorBibleService  # type: ignore[no-redef]
                service = ColorBibleService(db_path=color_bible_db_path)
            except Exception as exc:
                print(f"Warning: could not initialize ColorBibleService: {exc}", flush=True)
                service = None

    only_shots: set[int] = set()
    raw_only_shots = str(getattr(args, "only_shots", "") or "").strip()
    if raw_only_shots:
        for tok in raw_only_shots.split(","):
            tok = tok.strip()
            if tok.isdigit():
                only_shots.add(int(tok))

    cb_snapshot_hash: str | None = None
    if service is not None and film_id:
        try:
            cb_snapshot_hash = service.get_snapshot_hash(film_id)
        except Exception:
            cb_snapshot_hash = None

    chunks: list[Path] = []
    for index, row in enumerate(rows):
        item = plan[index]
        start_frame = item["start"]
        end_frame = item["end"]
        frame_count = max(1, end_frame - start_frame)
        references = row_references(row)
        mask_path = find_shot_mask_archive(masks_dir, row, index, film_id=film_id or "")

        chunk = cache_dir / f"segment_{index:04d}_{start_frame:06d}_{end_frame:06d}.mkv"
        chunk_sig = segment_signature(
            args,
            source_video,
            row,
            references,
            start_frame,
            end_frame,
            item["base_start"],
            item["base_end"],
            width,
            height,
            fps,
            mask_fingerprint=file_fingerprint(mask_path) if mask_path is not None and mask_path.is_file() else None,
            color_bible_version=cb_snapshot_hash,
        )

        if only_shots and index not in only_shots and chunk.is_file():
            print(f"Reuse untouched segment {index + 1}/{len(rows)}: {chunk}", flush=True)
            chunks.append(chunk)
            continue

        force_this_shot = bool(args.force) or (bool(only_shots) and index in only_shots)
        if not force_this_shot and segment_resumable(chunk, chunk_sig, width, height, frame_count):
            print(f"Reuse colorized segment {index + 1}/{len(rows)}: {chunk}", flush=True)
            chunks.append(chunk)
            continue

        # Extract grayscale frames for this shot via frame-exact FFmpeg trim
        frames_gray = extract_shot_grayscale_frames(
            processing_video, start_frame, frame_count, width, height, ffmpeg_exe=ffmpeg
        )

        # Gather approved reference keyframe(s)
        keyframes_color: dict[int, np.ndarray] = {}
        for r_idx, ref in enumerate(references):
            prep_ref_path = prepare_processing_reference(ref["path"], width, height)
            bgr = cv2.imread(str(prep_ref_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)

            sel = ref.get("selected_frame")
            if sel is not None:
                local_k = max(0, min(frame_count - 1, int(sel) - start_frame))
            else:
                if len(references) == 1:
                    local_k = 0
                else:
                    local_k = int(round(r_idx * (frame_count - 1) / max(1, len(references) - 1)))
            keyframes_color[local_k] = rgb

        if not keyframes_color and frames_gray:
            keyframes_color[0] = cv2.cvtColor(frames_gray[0], cv2.COLOR_GRAY2RGB)

        # Load shot mask archive if available
        mask_archive = None
        if mask_path is not None and load_shot_masks_npz is not None:
            try:
                mask_archive = load_shot_masks_npz(mask_path)
            except Exception as exc:
                print(f"Warning: could not load mask archive {mask_path}: {exc}", flush=True)
                mask_archive = None

        # Resolve color bible anchors if provided in row
        anchors = None
        if row.get("color_bible_anchors"):
            try:
                anchors = json.loads(row["color_bible_anchors"])
            except Exception:
                anchors = None

        if anchors is None and film_id and color_bible_db_path.is_file():
            if service is None:
                try:
                    from ai_remaster_gui.color_bible_service import ColorBibleService
                    service = ColorBibleService(db_path=color_bible_db_path)
                except Exception:
                    try:
                        from color_bible_service import ColorBibleService  # type: ignore[no-redef]
                        service = ColorBibleService(db_path=color_bible_db_path)
                    except Exception as exc:
                        print(f"Warning: could not initialize ColorBibleService: {exc}", flush=True)
                        service = None

            if service is not None:
                try:
                    assignments = service.get_shot_palette_assignments(film_id, index)
                    anchors_dict: dict[Any, Any] = {}
                    for asgn_item in assignments:
                        refs = asgn_item.get("color_references") or []
                        target_chroma = None
                        for ref in refs:
                            lab_stats = ref.get("lab_stats")
                            if not lab_stats and ref.get("lab_stats_json"):
                                try:
                                    lab_stats = json.loads(ref["lab_stats_json"])
                                except Exception:
                                    lab_stats = None
                            if isinstance(lab_stats, dict):
                                med_a, med_b = None, None
                                if "a" in lab_stats and "b" in lab_stats:
                                    a_val, b_val = lab_stats["a"], lab_stats["b"]
                                    if isinstance(a_val, (list, tuple)) and isinstance(b_val, (list, tuple)):
                                        if len(a_val) > 0 and len(b_val) > 0:
                                            med_a = float(a_val[len(a_val) // 2])
                                            med_b = float(b_val[len(b_val) // 2])
                                    else:
                                        try:
                                            med_a = float(a_val)
                                            med_b = float(b_val)
                                        except (ValueError, TypeError):
                                            pass
                                elif "median_a" in lab_stats and "median_b" in lab_stats:
                                    try:
                                        med_a = float(lab_stats["median_a"])
                                        med_b = float(lab_stats["median_b"])
                                    except (ValueError, TypeError):
                                        pass
                                elif "p50_a" in lab_stats and "p50_b" in lab_stats:
                                    try:
                                        med_a = float(lab_stats["p50_a"])
                                        med_b = float(lab_stats["p50_b"])
                                    except (ValueError, TypeError):
                                        pass

                                if med_a is not None and med_b is not None:
                                    target_chroma = (med_a, med_b)
                                    break
                        if target_chroma is not None:
                            entity_id = asgn_item.get("entity_id")
                            variant_id = asgn_item.get("variant_id")
                            name = (asgn_item.get("entity") or {}).get("name")
                            obj_id = asgn_item.get("obj_id") or (asgn_item.get("assignment") or {}).get("obj_id") or (asgn_item.get("entity") or {}).get("obj_id")
                            if entity_id:
                                anchors_dict[entity_id] = target_chroma
                            if variant_id:
                                anchors_dict[variant_id] = target_chroma
                            if name:
                                anchors_dict[name] = target_chroma
                            if obj_id is not None:
                                anchors_dict[obj_id] = target_chroma
                                anchors_dict[str(obj_id)] = target_chroma
                            if mask_archive is not None:
                                for m_obj in mask_archive.objects:
                                    if (entity_id and m_obj.entity_id == entity_id) or \
                                       (variant_id and m_obj.variant_id == variant_id) or \
                                       (name and m_obj.name == name):
                                        anchors_dict[m_obj.obj_id] = target_chroma
                                        anchors_dict[str(m_obj.obj_id)] = target_chroma
                    if anchors_dict:
                        anchors = anchors_dict
                except Exception as exc:
                    print(f"Warning: could not query ColorBible assignments for shot {index}: {exc}", flush=True)

        print(
            f"Colorize segment {index + 1}/{len(rows)} with osi_masked_lab: frames {start_frame}-{end_frame} "
            f"using {len(keyframes_color)} reference(s)"
            + (f" with mask archive {mask_path.name}" if mask_archive is not None else " (unmasked)"),
            flush=True,
        )

        output_frames = propagate_shot_chroma(
            frames_gray=frames_gray,
            keyframes_color=keyframes_color,
            mask_archive=mask_archive,
            color_bible_anchors=anchors,
            radius=getattr(args, "guided_radius", 5),
            eps=getattr(args, "guided_eps", 1e-3),
            flow_scale=float(getattr(args, "flow_scale", 0.5) or 0.5),
            anchor_strength=float(getattr(args, "anchor_strength", 0.75) or 0.75),
        )

        write_frames_to_video(ffmpeg, output_frames, chunk, fps, width, height)
        write_signature(chunk, chunk_sig)
        chunks.append(chunk)

    # 4. Combine / stitch chunks
    stitch_colorized(ffmpeg, chunks, transitions, output, fps, safe_stem(source_video.name))
    write_signature(output, sig)

    # 5. Append compliance record
    if append_compliance_record is not None and getattr(args, "compliance_output_dir", None):
        comp_dir = resolve_path(args.compliance_output_dir)
        osi_script = ROOT / "scripts" / "osi_masked_lab.py"
        models_entry = [{
            "model_id": "osi_masked_lab",
            "component_type": "backend",
        }]
        if osi_script.is_file():
            models_entry[0]["file_path"] = str(osi_script)
            try:
                from ai_remaster_gui.commercial_policy import compute_sha256
                models_entry[0]["sha256"] = compute_sha256(osi_script)
            except Exception:
                pass

        cb_version = None
        for r in rows:
            if r.get("color_bible_version"):
                cb_version = str(r["color_bible_version"]).strip()
                break
        if not cb_version and manifest and manifest.is_file():
            try:
                with manifest.open("r", encoding="utf-8-sig") as f:
                    for line in f:
                        if not line.startswith("#"):
                            break
                        if line.startswith("# color_bible_version="):
                            cb_version = line.split("=", 1)[1].strip()
                            break
                        if line.startswith("# color_bible="):
                            cb_version = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass

        try:
            append_compliance_record(
                output_dir=comp_dir,
                stage="colour",
                models=models_entry,
                settings=get_commercial_settings(),
                color_bible_version=cb_version,
            )
        except Exception as exc:
            print(f"Warning: could not append compliance record: {exc}", flush=True)

    if service is not None:
        try:
            service.close()
        except Exception:
            pass

    print(f"Wrote OSI Masked LAB colorized video: {output}", flush=True)

    if getattr(args, "run_qc", False) and not args.dry_run:
        try:
            try:
                from scripts import qc_color_consistency
            except ImportError:
                import qc_color_consistency  # type: ignore[no-redef]

            qc_argv = [
                "--manifest",
                str(manifest),
                "--colorized-video",
                str(output),
                "--masks-dir",
                str(masks_dir),
                "--stride",
                "4",
                "--update-manifest",
                "--record-db",
            ]
            if film_id:
                qc_argv.extend(["--film-id", film_id])
            if color_bible_db_path:
                qc_argv.extend(["--color-bible-db", str(color_bible_db_path)])
            qc_code = qc_color_consistency.main(qc_argv)
            print(f"Post-render Color QC finished (exit code {qc_code}).", flush=True)
        except Exception as exc:
            print(f"Warning: post-render Color QC failed: {exc}", flush=True)

    return 0


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description="Colorize outpainted video shots with reference-guided ComfyUI colorization.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-video", help="Override the # source_video from the manifest.")
    parser.add_argument("--output")
    parser.add_argument("--method", choices=["deepexemplar", "colormnet", "cmnet2", "openai", "ltx", "osi_masked_lab", "both"], default="deepexemplar")
    parser.add_argument("--masks-dir", default=str(ROOT / "intermediate" / "shot_masks"), help="Directory containing per-shot .masks.npz archives.")
    parser.add_argument("--guided-radius", type=int, default=8, help="Guided filter window radius for OSI Masked LAB colorization.")
    parser.add_argument("--guided-eps", type=float, default=1e-3, help="Guided filter regularization epsilon for OSI Masked LAB colorization.")
    parser.add_argument("--flow-scale", type=float, default=0.5, help="Optical flow computation downscale factor (0.25-1.0) for OSI Masked LAB.")
    parser.add_argument("--anchor-strength", type=float, default=0.75, help="Blend factor (0.0-1.0) for Color Bible median chroma shift inside masks.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker hint for multi-shot colorization.")
    parser.add_argument("--only-shots", default="", help="Comma-separated zero-based shot indices to re-render while reusing cached segments for other shots.")
    parser.add_argument("--run-qc", action="store_true", help="Run automated Color Consistency QC after OSI Masked LAB rendering.")
    parser.add_argument("--compliance-output-dir", default="output", help="Directory where compliance_report.json is recorded.")
    parser.add_argument("--color-bible-db", default=str(ROOT / "manifests" / "color_bible.db"), help="Path to Color Bible SQLite database.")
    parser.add_argument("--film-id", default=None, type=str, help="Film ID for Color Bible entity matching.")
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-output-root", default="")
    parser.add_argument("--processing-height", default="source", help="Downscale input video before ComfyUI processing. Use source/original or a target height such as 1080.")
    parser.add_argument("--frame-propagate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-half-resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-sage-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--colormnet-memory-mode", choices=["balanced", "low_memory", "high_quality"], default="balanced")
    parser.add_argument("--colormnet-feature-encoder", choices=["resnet50", "vgg19", "dinov2_vits", "dinov2_vitb", "dinov2_vitl", "clip_vitb"], default="resnet50")
    parser.add_argument("--colormnet-text-guidance", default="")
    parser.add_argument("--colormnet-text-guidance-weight", type=float, default=0.3)
    parser.add_argument("--cmnet2-dir", default=str(ROOT / "vendor" / "cmnet2"))
    parser.add_argument("--cmnet2-model", default="")
    parser.add_argument("--cmnet2-top-k", type=int, default=30)
    parser.add_argument("--cmnet2-mem-every", type=int, default=5)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--openai-model", default=openai_images.DEFAULT_MODEL)
    parser.add_argument("--openai-previous-frames", type=int, default=3)
    parser.add_argument(
        "--openai-prompt",
        default=(
            "Enhance and colourise the first image while preserving its exact composition, camera framing, "
            "geometry, identities, faces, hands, poses, objects, fine texture, film grain and luminance. "
            "Use its existing colour as evidence rather than removing it. Make the colour vivid, natural, "
            "period-appropriate and consistent with the supplied shot references. Do not add, remove, move, "
            "redesign or sharpen objects. Output one edited version of the first image only."
        ),
    )
    parser.add_argument("--openai-size", default="auto")
    parser.add_argument("--openai-quality", default="auto")
    parser.add_argument("--openai-timeout", type=float, default=900.0)
    parser.add_argument("--openai-max-retries", type=int, default=5)
    ltx_colorize.add_arguments(parser)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--intermediate-profile", choices=["low", "medium", "high", "lossless"], default="high")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-comfy-loaded", action="store_true", help="Skip unloading ComfyUI models after finishing.")
    return parser


def main() -> int:
    global CURRENT_INTERMEDIATE_PROFILE
    args = build_parser().parse_args()
    CURRENT_INTERMEDIATE_PROFILE = args.intermediate_profile
    check_commercial_guard(args.method)
    if args.method == "both":
        for method in ("deepexemplar", "colormnet"):
            child = argparse.Namespace(**vars(args))
            child.method = method
            child.output = ""
            run(child)
        return 0
    if args.method == "osi_masked_lab":
        return run_osi_masked_lab_colorization(args)
    return run(args)


def run(args: argparse.Namespace) -> int:
    global CURRENT_INTERMEDIATE_PROFILE
    CURRENT_INTERMEDIATE_PROFILE = getattr(args, "intermediate_profile", "high")
    check_commercial_guard(args.method)
    if args.method == "osi_masked_lab":
        return run_osi_masked_lab_colorization(args)
    manifest = resolve_path(args.manifest)
    source_from_manifest, rows = read_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No enabled rows in manifest: {manifest}")
    source_video = resolve_path(args.source_video or source_from_manifest or "")
    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found for colourisation: {source_video}")
    output = resolve_path(args.output) if args.output else default_output(manifest, source_from_manifest, args.method)
    source_info = video_info(source_video)
    source_width, source_height = int(source_info["width"]), int(source_info["height"])
    if args.method == "ltx":
        target_width, target_height = ltx_colorize.ltx_dimensions(source_width, source_height, args.ltx_height)
    else:
        target_width, target_height = processing_dimensions(source_width, source_height, args.processing_height)
    sig = signature(args, manifest, source_video, rows)
    if not args.force and resumable_output(output, sig, width=target_width, height=target_height):
        print(f"Reuse colorized video: {output}", flush=True)
        return 0
    if args.dry_run:
        if args.method == "openai":
            counts = [len(row_references(row)) for row in rows]
            print(
                f"Would send {int(source_info['frames'])} frame edit request(s) to {args.openai_model}: "
                f"{source_video} -> {output}; references per shot: {counts}; "
                f"up to {max(0, int(args.openai_previous_frames))} preceding frame(s).",
                flush=True,
            )
        elif args.method == "ltx":
            plan, _transitions = shot_plan(rows, int(source_info["frames"]), float(source_info["fps"]))
            chunk_frames = ltx_colorize.chunk_frames_for(args.ltx_chunk_seconds, float(source_info["fps"]))
            chunks = sum(len(ltx_colorize.shot_chunks(item["start"], item["end"], chunk_frames, args.ltx_overlap_frames)) for item in plan)
            print(
                f"Would colorize {len(rows)} shot(s) in {chunks} LTX chunk(s) at {target_width}x{target_height}: "
                f"{source_video} -> {output}",
                flush=True,
            )
        elif args.method == "cmnet2":
            counts = [len(row_references(row)) for row in rows]
            print(
                f"Would colorize {len(rows)} shot segment(s) with CMNET2: {source_video} -> {output}; "
                f"permanent references per shot: {counts}.",
                flush=True,
            )
        else:
            print(f"Would colorize {len(rows)} shot segment(s): {source_video} -> {output}", flush=True)
        return 0

    ffmpeg = find_ffmpeg(args.ffmpeg)
    if args.method == "openai":
        return run_openai_colorization(
            args,
            manifest,
            source_video,
            rows,
            output,
            sig,
            ffmpeg,
            target_width,
            target_height,
            float(source_info["fps"]),
            int(source_info["frames"]),
        )

    processing_video = prepare_processing_video(ffmpeg, source_video, target_width, target_height, source_width, source_height)
    info = video_info(processing_video)
    width, height, fps, total_frames = int(info["width"]), int(info["height"]), float(info["fps"]), int(info["frames"])
    if args.method == "ltx":
        plan, transitions = shot_plan(rows, total_frames, fps)
        segments = ltx_colorize.run_ltx_colorization(args, processing_video, rows, plan, row_references, ffmpeg, width, height, fps)
        stitch_colorized(ffmpeg, segments, transitions, output, fps, safe_stem(source_video.name))
        write_signature(output, sig)
        print(f"Wrote LTX Colorizer video: {output}", flush=True)
        return 0
    if args.method == "cmnet2":
        return run_cmnet2_colorization(
            args,
            processing_video,
            rows,
            output,
            sig,
            ffmpeg,
            width,
            height,
            fps,
            total_frames,
        )

    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    video_name = copy_to_comfy_input(processing_video, comfy_dir, "arp_colorize")
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    required_nodes = {
        "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
        "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
    }
    if args.method == "colormnet":
        required_nodes["ColorMNetVideo"] = "ComfyUI-Reference-Based-Video-Colorization"
    else:
        required_nodes["DeepExColorVideoNode"] = "ComfyUI-Reference-Based-Video-Colorization"
    ensure_node_types(args.comfy_url, required_nodes, f"{args.method} colourisation")

    chunks: list[Path] = []
    plan, transitions = shot_plan(rows, total_frames, fps)
    cache_dir = ROOT / ".cache" / "colorized_chunks" / method_suffix(args.method) / safe_stem(source_video.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, row in enumerate(rows):
            item = plan[index]
            start_frame = item["start"]
            end_frame = item["end"]
            frame_count = max(1, end_frame - start_frame)
            references = row_references(row)
            active_references = references[:1]
            if len(references) > 1:
                display_method = "ColorMNet" if args.method == "colormnet" else "Deep Exemplar"
                print(
                    f"{display_method} supports one reference per shot; using reference 1 of {len(references)} "
                    f"for shot {index + 1}. Choose CMNET2 or OpenAI Cloud to use every reference.",
                    flush=True,
                )
            source_reference_indices = [
                max(start_frame, min(end_frame - 1, int(reference["selected_frame"] or start_frame)))
                for reference in active_references
            ]
            ref_names = [copy_reference_to_comfy_input(item["path"], comfy_dir, width, height) for item in active_references]
            chunk = cache_dir / f"segment_{index:04d}_{start_frame:06d}_{end_frame:06d}.mkv"
            chunk_sig = segment_signature(args, source_video, row, active_references, start_frame, end_frame, item["base_start"], item["base_end"], width, height, fps)
            if not args.force and segment_resumable(chunk, chunk_sig, width, height, frame_count):
                print(f"Reuse colorized segment {index + 1}/{len(rows)}: {chunk}", flush=True)
                chunks.append(chunk)
                continue
            prefix = f"arp_colorize/{method_suffix(args.method)}_{safe_stem(source_video.name)}_segment_{index:04d}_{start_frame:06d}_{end_frame:06d}"
            keyframe_note = ", ".join(str(value) for value in source_reference_indices)
            print(f"Colorize segment {index + 1}/{len(rows)} with {args.method}: frames {start_frame}-{end_frame} using {len(active_references)} reference(s) at source frames {keyframe_note}", flush=True)
            prompt = build_prompt(video_name, ref_names, start_frame, frame_count, width, height, fps, args, prefix)
            prompt_id = queue_prompt(args.comfy_url, prompt)
            history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
            produced = newest_output(extract_output_files(history, comfy_output_root))
            normalize_clip(ffmpeg, produced, chunk, fps, frame_count)
            write_signature(chunk, chunk_sig)
            chunks.append(chunk)

        stitch_colorized(ffmpeg, chunks, transitions, output, fps, safe_stem(source_video.name))
        write_signature(output, sig)
        print(f"Wrote colorized video: {output}", flush=True)
        return 0
    finally:
        if not getattr(args, "keep_comfy_loaded", False):
            free_comfy_models(args.comfy_url)


if __name__ == "__main__":
    raise SystemExit(main())
