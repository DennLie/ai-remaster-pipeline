"""Pluggable Keyframe / Still Image Segmenter protocol, registry, and adapters.

Complements ai_remaster_gui/video_segmenters.py (shot-wide video mask tracking) with
dedicated single-image segmentation for reference stills. Generated binary PNG masks
and JSON sidecars feed directly into:
  1. ColorBibleService.register_reference_image(..., mask_uri=...)
  2. VideoSegmenterRegistry seed masks (add_prompts(..., mask=...)) -> .masks.npz

Commercial-safe OSI (ALLOWED_OSI):
  1. evf_sam2       - EVF-SAM-2 referring-expression segmentation (Apache-2.0)
  2. florence2      - Microsoft Florence-2 referring-expression segmentation (MIT)
  3. birefnet       - BiRefNet / BiRefNet_lite sharp foreground matting (MIT)
  4. mask2former    - Mask2Former panoptic/semantic scene parsing (MIT / Apache-2.0)
  5. oneformer      - SHI-Labs OneFormer universal panoptic/instance segmentation (MIT)
  6. grounding_dino - Grounding DINO open-vocab text-to-box + optional SAM 2.1 (Apache-2.0)
  7. rf_detr_seg    - Roboflow RF-DETR-Seg real-time instance segmentation (Apache-2.0)
  8. sam21_image    - Official facebookresearch/sam2 SAM 2.1 single-image mode (Apache-2.0)

Non-Commercial / Benchmark Only (HARD_BLOCK_COMMERCIAL unless explicitly overridden):
  9.  sam3_image    - Meta SAM 3 still-image text/prompt segmentation (Meta SAM License)
  10. sam31_image   - Meta SAM 3.1 still-image text/prompt segmentation (Meta SAM License)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import cv2
import numpy as np
from PIL import Image

from .commercial_policy import (
    CommercialPolicyError,
    PolicyDecision,
    compute_sha256,
    enabled as commercial_guard_enabled,
    evaluate_component,
)
from .config import ROOT
from .managed_model_envs import (
    env_for_model,
    inspect_env,
    is_managed_env_active,
    run_in_managed_env,
)
from .video_segmenters import binarize_mask, release_gpu_memory

VALID_IMAGE_CATEGORIES: frozenset[str] = frozenset({"promptable", "panoptic", "matting", "detection"})


@dataclass
class ImageSegmentationResult:
    """Normalized output from any Keyframe / Still Image Segmenter."""

    mask: np.ndarray
    segmenter: str
    labels: list[str] = field(default_factory=list)
    masks_by_label: dict[str, np.ndarray] = field(default_factory=dict)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.mask)
        if arr.ndim != 2:
            raise ValueError(f"ImageSegmentationResult.mask must be 2D (H, W), got shape {arr.shape}")
        self.mask = np.where(arr > 0, 255, 0).astype(np.uint8)
        self.segmenter = str(self.segmenter)
        self.labels = [str(lbl) for lbl in self.labels]


@runtime_checkable
class ImageSegmenterBackend(Protocol):
    """Protocol implemented by every pluggable still-image keyframe segmentation backend."""

    name: str
    component_id: str
    category: str
    supports_text_prompts: bool
    supports_point_box_prompts: bool

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        """Segment a single RGB image and return a binary mask plus label metadata."""
        ...

    def close(self) -> None:
        """Release model handles and clear GPU/CUDA caches."""
        ...


def normalize_target_classes(target_classes: list[str] | str | None) -> list[str]:
    """Normalize comma/period-separated class names into lowercase tokens."""
    if target_classes is None:
        return []
    if isinstance(target_classes, str):
        raw = [part.strip().lower() for part in target_classes.replace(";", ",").split(",")]
        return [c for c in raw if c]
    return [str(c).strip().lower() for c in target_classes if str(c).strip()]


def load_image_rgb_array(image: Path | str | np.ndarray) -> np.ndarray:
    """Load an image file or array cleanly into uint8 RGB (H, W, 3)."""
    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3].astype(np.uint8)
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

    p = Path(image)
    if not p.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {p}")
    with Image.open(p) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def load_seed_mask_for_tracker(mask_path: Path | str) -> np.ndarray:
    """Load a binary PNG keyframe mask (H, W) uint8 (0/255) ready for VideoSegmenterBackend."""
    p = Path(mask_path)
    if not p.is_file():
        raise FileNotFoundError(f"Keyframe seed mask does not exist: {p}")
    with Image.open(p) as im:
        gray = np.array(im.convert("L"), dtype=np.uint8)
    return np.where(gray > 0, 255, 0).astype(np.uint8)


def _to_numpy(val: Any) -> np.ndarray:
    if hasattr(val, "detach"):
        return val.detach().cpu().numpy()
    if hasattr(val, "cpu") and hasattr(val, "numpy"):
        return val.cpu().numpy()
    if hasattr(val, "numpy"):
        return val.numpy()
    return np.asarray(val)


def _inference_context() -> Any:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and hasattr(torch_mod, "inference_mode"):
        return torch_mod.inference_mode()
    return nullcontext()


def _require_module(module_name: str, install_hint: str) -> Any:
    mod = sys.modules.get(module_name)
    if mod is not None:
        return mod
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional image segmenter dependency '{module_name}' is not installed. {install_hint}"
        ) from exc


def _module_available(module_names: Sequence[str]) -> bool:
    for name in module_names:
        if name in sys.modules:
            return True
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except Exception:
            continue
    return False


class BaseImageSegmenter:
    """Shared lifecycle, device resolution, and GPU VRAM cleanup for image segmenters."""

    name: str = "base"
    component_id: str = "base"
    category: str = "promptable"
    supports_text_prompts: bool = False
    supports_point_box_prompts: bool = False
    supports_target_classes: bool = False

    def __init__(self, *, checkpoint_path: Path | str | None = None, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._is_closed = False

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            try:
                import torch as torch_mod  # noqa: F401
            except ImportError:
                return "cpu"
        try:
            return "cuda" if torch_mod.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _release_gpu_memory(self) -> None:
        release_gpu_memory()

    def close(self) -> None:
        self._is_closed = True
        self._release_gpu_memory()

    def __enter__(self) -> BaseImageSegmenter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if not getattr(self, "_is_closed", False):
                self.close()
        except Exception:
            pass


class EVFSAM2ImageSegmenter(BaseImageSegmenter):
    """EVF-SAM-2 natural-language referring-expression still segmentation adapter (Apache-2.0)."""

    name = "evf_sam2"
    component_id = "evf_sam2"
    category = "promptable"
    supports_text_prompts = True
    supports_point_box_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()

    def _build_runner(self) -> Any:
        try:
            mod = _require_module(
                "evf_sam",
                "Install EVF-SAM-2 (Apache-2.0, https://github.com/hustvl/EVF-SAM) or HuggingFace transformers.",
            )
            factory = getattr(mod, "build_evf_sam2", None) or getattr(mod, "EVFSAM2Pipeline", None)
            if factory is not None:
                return factory(str(self.checkpoint_path), device=self._resolve_device()) if self.checkpoint_path else factory(device=self._resolve_device())
            return mod
        except RuntimeError:
            transformers_mod = _require_module(
                "transformers",
                "Install EVF-SAM-2 (Apache-2.0, https://github.com/hustvl/EVF-SAM) or transformers.",
            )
            model_id = str(self.checkpoint_path or "YxZhang/evf-sam2")
            tok = transformers_mod.AutoTokenizer.from_pretrained(model_id)
            model = transformers_mod.AutoModel.from_pretrained(model_id, trust_remote_code=False)
            return {"tokenizer": tok, "model": model}

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        if not text_prompt or not str(text_prompt).strip():
            raise ValueError("EVF-SAM-2 requires a non-empty text_prompt.")
        prompt_str = str(text_prompt).strip()
        h, w = image_rgb.shape[:2]
        try:
            with _inference_context():
                if hasattr(self.runner, "predict"):
                    raw = self.runner.predict(image_rgb, text_prompt=prompt_str)
                elif callable(self.runner):
                    raw = self.runner(image_rgb, text_prompt=prompt_str)
                else:
                    raise RuntimeError("EVF-SAM-2 runner does not expose predict() or callable inference.")

            raw_mask = raw.get("mask") if isinstance(raw, Mapping) else raw
            binary = binarize_mask(raw_mask, min_mask_area=1, threshold=alpha_threshold)
            score = float(raw.get("score", 1.0)) if isinstance(raw, Mapping) else 1.0
            if binary.shape != (h, w):
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            return ImageSegmentationResult(
                mask=binary,
                segmenter=self.name,
                labels=[prompt_str] if np.any(binary) else [],
                masks_by_label=({prompt_str: binary} if np.any(binary) else {}),
                scores=[score] if np.any(binary) else [],
                metadata={"prompt": prompt_str},
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_runner and self.runner is not None:
            self.runner = None
        super().close()


class Florence2ImageSegmenter(BaseImageSegmenter):
    """Microsoft Florence-2 referring-expression image segmentation adapter (MIT)."""

    name = "florence2"
    component_id = "florence2"
    category = "promptable"
    supports_text_prompts = True
    supports_point_box_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()

    def _build_runner(self) -> Any:
        transformers_mod = _require_module(
            "transformers",
            "Install transformers (for Microsoft Florence-2, MIT, https://huggingface.co/microsoft/Florence-2-large).",
        )
        model_id = str(self.checkpoint_path or "microsoft/Florence-2-large")
        processor = transformers_mod.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = transformers_mod.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        dev = self._resolve_device()
        if hasattr(model, "to"):
            model = model.to(dev)
        return {"processor": processor, "model": model, "device": dev}

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        if not text_prompt or not str(text_prompt).strip():
            raise ValueError("Florence-2 segmentation requires a non-empty text_prompt.")
        prompt_str = str(text_prompt).strip()
        h, w = image_rgb.shape[:2]
        task_tag = "<REFERRING_EXPRESSION_SEGMENTATION>"
        try:
            with _inference_context():
                if hasattr(self.runner, "predict"):
                    parsed = self.runner.predict(image_rgb, text_prompt=prompt_str)
                elif isinstance(self.runner, Mapping) and "model" in self.runner:
                    proc = self.runner["processor"]
                    model = self.runner["model"]
                    pil_img = Image.fromarray(image_rgb)
                    full_prompt = f"{task_tag}{prompt_str}"
                    inputs = proc(text=full_prompt, images=pil_img, return_tensors="pt")
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.runner.get("device", "cpu"))
                    gen_ids = model.generate(**inputs, max_new_tokens=1024, num_beams=3)
                    gen_text = proc.batch_decode(gen_ids, skip_special_tokens=False)[0]
                    parsed = proc.post_process_generation(gen_text, task=task_tag, image_size=(w, h))
                else:
                    raise RuntimeError("Invalid Florence-2 runner configuration.")

            union_mask = np.zeros((h, w), dtype=np.uint8)
            labels: list[str] = []
            masks_by_label: dict[str, np.ndarray] = {}

            if isinstance(parsed, Mapping) and "mask" in parsed:
                union_mask = binarize_mask(parsed["mask"], min_mask_area=1, threshold=alpha_threshold)
                if np.any(union_mask):
                    labels.append(prompt_str)
                    masks_by_label[prompt_str] = union_mask
            elif isinstance(parsed, Mapping):
                seg_block = parsed.get(task_tag, parsed)
                polygons = seg_block.get("polygons", []) if isinstance(seg_block, Mapping) else []
                seg_labels = seg_block.get("labels", []) if isinstance(seg_block, Mapping) else []
                for idx, poly_group in enumerate(polygons):
                    lbl = str(seg_labels[idx]) if idx < len(seg_labels) else prompt_str
                    obj_mask = np.zeros((h, w), dtype=np.uint8)
                    for poly in poly_group:
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                        if len(pts) >= 3:
                            cv2.fillPoly(obj_mask, [pts], 255)
                    if np.any(obj_mask):
                        union_mask = np.maximum(union_mask, obj_mask)
                        masks_by_label[lbl] = obj_mask
                        if lbl not in labels:
                            labels.append(lbl)
            else:
                union_mask = binarize_mask(parsed, min_mask_area=1, threshold=alpha_threshold)
                if np.any(union_mask):
                    labels.append(prompt_str)
                    masks_by_label[prompt_str] = union_mask

            return ImageSegmentationResult(
                mask=union_mask,
                segmenter=self.name,
                labels=labels,
                masks_by_label=masks_by_label,
                metadata={"prompt": prompt_str, "task": task_tag},
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_runner and isinstance(self.runner, Mapping):
            model = self.runner.get("model")
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception:
                    pass
        self.runner = None
        super().close()


class BiRefNetImageSegmenter(BaseImageSegmenter):
    """BiRefNet / BiRefNet_lite sharp foreground matting & segmentation adapter (MIT)."""

    name = "birefnet"
    component_id = "birefnet"
    category = "matting"
    supports_text_prompts = False
    supports_point_box_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()

    def _build_runner(self) -> Any:
        try:
            mod = _require_module(
                "birefnet",
                "Install BiRefNet (MIT, https://github.com/ZhengPeng7/BiRefNet) or transformers.",
            )
            factory = getattr(mod, "BiRefNetPipeline", None) or getattr(mod, "build_birefnet", None)
            if factory is not None:
                return factory(str(self.checkpoint_path), device=self._resolve_device()) if self.checkpoint_path else factory(device=self._resolve_device())
            return mod
        except RuntimeError:
            transformers_mod = _require_module(
                "transformers",
                "Install BiRefNet (MIT, https://github.com/ZhengPeng7/BiRefNet) or HuggingFace transformers.",
            )
            model_id = str(self.checkpoint_path or "ZhengPeng7/BiRefNet_lite")
            model = transformers_mod.AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
            dev = self._resolve_device()
            if hasattr(model, "to"):
                model = model.to(dev)
            return {"hf_model": model, "device": dev}

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        h, w = image_rgb.shape[:2]
        try:
            with _inference_context():
                if hasattr(self.runner, "predict"):
                    alpha = self.runner.predict(image_rgb)
                elif hasattr(self.runner, "matte"):
                    alpha = self.runner.matte(image_rgb)
                elif callable(self.runner):
                    alpha = self.runner(image_rgb)
                elif isinstance(self.runner, Mapping) and "hf_model" in self.runner:
                    import torch

                    tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                    tensor = tensor.to(self.runner.get("device", "cpu"))
                    preds = self.runner["hf_model"](tensor)[-1]
                    alpha = torch.sigmoid(preds).squeeze()
                else:
                    raise RuntimeError("Invalid BiRefNet runner configuration.")

            binary = binarize_mask(alpha, min_mask_area=1, threshold=float(alpha_threshold), is_logit=False)
            if binary.shape != (h, w):
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            labels = ["foreground"] if np.any(binary) else []
            return ImageSegmentationResult(
                mask=binary,
                segmenter=self.name,
                labels=labels,
                masks_by_label=({"foreground": binary} if labels else {}),
                metadata={"alpha_threshold": float(alpha_threshold)},
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_runner and isinstance(self.runner, Mapping):
            model = self.runner.get("hf_model")
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception:
                    pass
        self.runner = None
        super().close()


class _PanopticSceneSegmenterBase(BaseImageSegmenter):
    """Shared panoptic / semantic / instance class-filtering adapter for Mask2Former & OneFormer."""

    category = "panoptic"
    supports_text_prompts = False
    supports_point_box_prompts = False
    supports_target_classes = True
    _default_hf_id: str = ""
    _install_hint: str = ""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()

    def _build_runner(self) -> Any:
        transformers_mod = _require_module("transformers", self._install_hint)
        model_id = str(self.checkpoint_path or self._default_hf_id)
        processor = transformers_mod.AutoImageProcessor.from_pretrained(model_id)
        model = transformers_mod.AutoModelForUniversalSegmentation.from_pretrained(model_id)
        dev = self._resolve_device()
        if hasattr(model, "to"):
            model = model.to(dev)
        return {"processor": processor, "model": model, "device": dev}

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        h, w = image_rgb.shape[:2]
        requested = set(normalize_target_classes(target_classes))
        try:
            with _inference_context():
                if hasattr(self.runner, "predict"):
                    res = self.runner.predict(image_rgb, target_classes=list(requested) if requested else None)
                elif isinstance(self.runner, Mapping) and "model" in self.runner:
                    proc = self.runner["processor"]
                    model = self.runner["model"]
                    pil_img = Image.fromarray(image_rgb)
                    try:
                        inputs = proc(images=pil_img, task_inputs=["panoptic"], return_tensors="pt")
                    except TypeError:
                        inputs = proc(images=pil_img, return_tensors="pt")
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.runner.get("device", "cpu"))
                    outputs = model(**inputs)
                    pan = proc.post_process_panoptic_segmentation(outputs, target_sizes=[(h, w)])[0]
                    id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
                    seg_map = _to_numpy(pan["segmentation"])
                    segments_info = [
                        {
                            "id": int(s["id"]),
                            "label": str(id2label.get(int(s["label_id"]), s.get("label_id", "unknown"))),
                            "score": float(s.get("score", 1.0)),
                        }
                        for s in pan.get("segments_info", [])
                    ]
                    res = {"segmentation": seg_map, "segments_info": segments_info}
                else:
                    raise RuntimeError(f"Invalid {self.name} runner configuration.")

            union_mask = np.zeros((h, w), dtype=np.uint8)
            labels: list[str] = []
            masks_by_label: dict[str, np.ndarray] = {}
            scores: list[float] = []

            if isinstance(res, Mapping) and "segmentation" in res and "segments_info" in res:
                seg_map = _to_numpy(res["segmentation"])
                for seg in res["segments_info"]:
                    lbl = str(seg.get("label") or seg.get("label_name") or "").strip()
                    lbl_low = lbl.lower()
                    if requested and lbl_low not in requested and not any(rc in lbl_low for rc in requested):
                        continue
                    seg_id = int(seg["id"])
                    m = np.where(seg_map == seg_id, 255, 0).astype(np.uint8)
                    if not np.any(m):
                        continue
                    union_mask = np.maximum(union_mask, m)
                    if lbl in masks_by_label:
                        masks_by_label[lbl] = np.maximum(masks_by_label[lbl], m)
                    else:
                        masks_by_label[lbl] = m
                        labels.append(lbl)
                    scores.append(float(seg.get("score", 1.0)))
            elif isinstance(res, Mapping) and "masks_by_label" in res:
                for lbl, raw_m in res["masks_by_label"].items():
                    lbl_low = str(lbl).strip().lower()
                    if requested and lbl_low not in requested and not any(rc in lbl_low for rc in requested):
                        continue
                    m = binarize_mask(raw_m, min_mask_area=1, threshold=alpha_threshold)
                    if np.any(m):
                        union_mask = np.maximum(union_mask, m)
                        masks_by_label[str(lbl)] = m
                        labels.append(str(lbl))
            else:
                binary = binarize_mask(res, min_mask_area=1, threshold=alpha_threshold)
                if np.any(binary):
                    union_mask = binary
                    lbl = ",".join(sorted(requested)) or "foreground"
                    labels = [lbl]
                    masks_by_label = {lbl: binary}

            return ImageSegmentationResult(
                mask=union_mask,
                segmenter=self.name,
                labels=labels,
                masks_by_label=masks_by_label,
                scores=scores,
                metadata={"target_classes": sorted(requested)},
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_runner and isinstance(self.runner, Mapping):
            model = self.runner.get("model")
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception:
                    pass
        self.runner = None
        super().close()


class Mask2FormerImageSegmenter(_PanopticSceneSegmenterBase):
    """Mask2Former panoptic & semantic scene parser (MIT / Apache-2.0)."""

    name = "mask2former"
    component_id = "mask2former"
    _default_hf_id = "facebook/mask2former-swin-large-coco-panoptic"
    _install_hint = "Install transformers (for Mask2Former, MIT/Apache-2.0, https://github.com/facebookresearch/Mask2Former)."


class OneFormerImageSegmenter(_PanopticSceneSegmenterBase):
    """SHI-Labs OneFormer universal panoptic/instance/semantic segmenter (MIT)."""

    name = "oneformer"
    component_id = "oneformer"
    _default_hf_id = "shi-labs/oneformer_coco_swin_large"
    _install_hint = "Install transformers (for SHI-Labs OneFormer, MIT, https://github.com/SHI-Labs/OneFormer)."


class GroundingDINOImageSegmenter(BaseImageSegmenter):
    """Grounding DINO open-vocabulary text-to-box detector with optional SAM 2.1 refinement (Apache-2.0)."""

    name = "grounding_dino"
    component_id = "grounding_dino"
    category = "detection"
    supports_text_prompts = True
    supports_point_box_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        runner: Any = None,
        refiner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()
        self.refiner = refiner

    def _build_runner(self) -> Any:
        gd_mod = sys.modules.get("groundingdino")
        if gd_mod is not None:
            return gd_mod
        try:
            return importlib.import_module("groundingdino.util.inference")
        except ImportError:
            pass
        transformers_mod = _require_module(
            "transformers",
            "Install GroundingDINO (Apache-2.0, https://github.com/IDEA-Research/GroundingDINO) or HuggingFace transformers.",
        )
        model_id = str(self.checkpoint_path or "IDEA-Research/grounding-dino-tiny")
        proc = transformers_mod.AutoProcessor.from_pretrained(model_id)
        model = transformers_mod.AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        dev = self._resolve_device()
        if hasattr(model, "to"):
            model = model.to(dev)
        return {"processor": proc, "model": model, "device": dev}

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        if not text_prompt or not str(text_prompt).strip():
            raise ValueError("Grounding DINO requires a non-empty text_prompt.")
        prompt_str = str(text_prompt).strip()
        h, w = image_rgb.shape[:2]
        detections: list[tuple[str, tuple[float, float, float, float], float]] = []

        try:
            with _inference_context():
                if hasattr(self.runner, "predict_boxes"):
                    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                    raw = self.runner.predict_boxes(
                        bgr,
                        [p.strip() for p in prompt_str.split(".") if p.strip()],
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                    )
                    for item in raw:
                        if isinstance(item, Mapping):
                            lbl = str(item.get("label") or prompt_str)
                            b = [float(v) for v in item["box"][:4]]
                            sc = float(item.get("score", 1.0))
                        else:
                            lbl, b = str(item[0]), [float(v) for v in item[1][:4]]
                            sc = float(item[2]) if len(item) > 2 else 1.0
                        detections.append((lbl, (b[0], b[1], b[2], b[3]), sc))
                elif isinstance(self.runner, Mapping) and "model" in self.runner:
                    proc = self.runner["processor"]
                    model = self.runner["model"]
                    pil_img = Image.fromarray(image_rgb)
                    caption = prompt_str if prompt_str.endswith(".") else prompt_str + "."
                    inputs = proc(images=pil_img, text=caption, return_tensors="pt")
                    if hasattr(inputs, "to"):
                        inputs = inputs.to(self.runner.get("device", "cpu"))
                    outputs = model(**inputs)
                    parsed = proc.post_process_grounded_object_detection(
                        outputs,
                        inputs["input_ids"],
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        target_sizes=[(h, w)],
                    )[0]
                    for lbl, box_t, sc in zip(parsed["labels"], parsed["boxes"], parsed["scores"]):
                        b = _to_numpy(box_t).tolist()
                        detections.append((str(lbl), (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(sc)))
                else:
                    raise RuntimeError("Invalid Grounding DINO runner configuration.")

                union_mask = np.zeros((h, w), dtype=np.uint8)
                labels: list[str] = []
                masks_by_label: dict[str, np.ndarray] = {}
                det_boxes: list[tuple[float, float, float, float]] = []
                scores: list[float] = []

                if detections and self.refiner is not None and hasattr(self.refiner, "set_image"):
                    self.refiner.set_image(image_rgb)

                for lbl, box, sc in detections:
                    det_boxes.append(box)
                    scores.append(sc)
                    if self.refiner is not None and hasattr(self.refiner, "predict"):
                        masks, r_scores, _ = self.refiner.predict(box=np.array(box, dtype=np.float32), multimask_output=True)
                        best_i = int(np.argmax(r_scores)) if len(r_scores) else 0
                        m = binarize_mask(masks[best_i], min_mask_area=1, threshold=alpha_threshold)
                    else:
                        m = np.zeros((h, w), dtype=np.uint8)
                        x1, y1, x2, y2 = [int(round(v)) for v in box]
                        m[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 255

                    if np.any(m):
                        union_mask = np.maximum(union_mask, m)
                        if lbl in masks_by_label:
                            masks_by_label[lbl] = np.maximum(masks_by_label[lbl], m)
                        else:
                            masks_by_label[lbl] = m
                            labels.append(lbl)

            return ImageSegmentationResult(
                mask=union_mask,
                segmenter=self.name,
                labels=labels,
                masks_by_label=masks_by_label,
                boxes=det_boxes,
                scores=scores,
                metadata={"prompt": prompt_str, "refined_with_sam2": self.refiner is not None},
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_runner and isinstance(self.runner, Mapping):
            model = self.runner.get("model")
            if hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception:
                    pass
        self.runner = None
        super().close()


class RFDETRSegImageSegmenter(BaseImageSegmenter):
    """Roboflow RF-DETR-Seg real-time instance segmentation on reference stills (Apache-2.0)."""

    name = "rf_detr_seg"
    component_id = "rf_detr_seg"
    category = "detection"
    supports_text_prompts = False
    supports_point_box_prompts = False
    supports_target_classes = True

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        confidence_threshold: float = 0.4,
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.confidence_threshold = float(confidence_threshold)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()

    def _build_runner(self) -> Any:
        mod = _require_module(
            "rfdetr",
            "Install Roboflow rfdetr (Apache-2.0, https://github.com/roboflow/rf-detr) before using rf_detr_seg.",
        )
        cls = (
            getattr(mod, "RFDETRSegPreview", None)
            or getattr(mod, "RFDETRSegMedium", None)
            or getattr(mod, "RFDETRSegLarge", None)
        )
        if cls is None:
            raise RuntimeError("Installed 'rfdetr' package does not expose RFDETRSeg* segmentation classes.")
        if self.checkpoint_path:
            return cls(pretrain_weights=str(self.checkpoint_path), device=self._resolve_device())
        return cls(device=self._resolve_device())

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        h, w = image_rgb.shape[:2]
        requested = set(normalize_target_classes(target_classes))
        try:
            with _inference_context():
                preds = self.runner.predict(image_rgb, threshold=self.confidence_threshold)

            masks_attr = getattr(preds, "mask", None)
            if masks_attr is None and isinstance(preds, Mapping):
                masks_attr = preds.get("masks")
            raw_boxes = getattr(preds, "xyxy", None) if not isinstance(preds, Mapping) else preds.get("boxes")
            raw_scores = getattr(preds, "confidence", None) if not isinstance(preds, Mapping) else preds.get("scores")
            raw_class_names = None
            if isinstance(preds, Mapping):
                raw_class_names = preds.get("labels") or preds.get("class_names")
            else:
                data = getattr(preds, "data", None)
                if isinstance(data, Mapping):
                    raw_class_names = data.get("class_name")

            union_mask = np.zeros((h, w), dtype=np.uint8)
            labels: list[str] = []
            masks_by_label: dict[str, np.ndarray] = {}
            det_boxes: list[tuple[float, float, float, float]] = []
            scores: list[float] = []

            if masks_attr is not None:
                masks_np = _to_numpy(masks_attr)
                for i in range(masks_np.shape[0]):
                    lbl = str(raw_class_names[i]) if raw_class_names is not None and i < len(raw_class_names) else f"instance_{i + 1}"
                    if requested and lbl.strip().lower() not in requested:
                        continue
                    m = binarize_mask(masks_np[i], min_mask_area=1, threshold=alpha_threshold)
                    if not np.any(m):
                        continue
                    union_mask = np.maximum(union_mask, m)
                    if lbl in masks_by_label:
                        masks_by_label[lbl] = np.maximum(masks_by_label[lbl], m)
                    else:
                        masks_by_label[lbl] = m
                        labels.append(lbl)
                    if raw_boxes is not None and i < len(raw_boxes):
                        b = _to_numpy(raw_boxes[i]).tolist()
                        det_boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
                    if raw_scores is not None and i < len(raw_scores):
                        scores.append(float(raw_scores[i]))

            return ImageSegmentationResult(
                mask=union_mask,
                segmenter=self.name,
                labels=labels,
                masks_by_label=masks_by_label,
                boxes=det_boxes,
                scores=scores,
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_runner and self.runner is not None:
            if hasattr(self.runner, "model") and hasattr(self.runner.model, "to"):
                try:
                    self.runner.model.to("cpu")
                except Exception:
                    pass
            self.runner = None
        super().close()


class SAM21ImageSegmenter(BaseImageSegmenter):
    """Official facebookresearch/sam2 SAM 2.1 single-image promptable segmenter (Apache-2.0)."""

    name = "sam21_image"
    component_id = "sam21_image"
    category = "promptable"
    supports_text_prompts = False
    supports_point_box_prompts = True

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        predictor: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._owns_predictor = predictor is None
        self.predictor = predictor if predictor is not None else self._build_predictor()

    def _build_predictor(self) -> Any:
        from scripts.sam2_video_tracker import verify_sam2_installation

        verify_sam2_installation(self.checkpoint_path)
        if self.checkpoint_path is None:
            from .sam_masks import _sam2_predictor

            return _sam2_predictor()
        img_pred_mod = _require_module(
            "sam2.sam2_image_predictor",
            "Install official facebookresearch/sam2 (Apache-2.0) to use SAM2ImagePredictor.",
        )
        cls = getattr(img_pred_mod, "SAM2ImagePredictor")
        return cls.from_pretrained(str(self.checkpoint_path), device=self._resolve_device())

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        point_labels: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        if text_prompt and str(text_prompt).strip():
            raise ValueError("SAM 2.1 single-image mode does not accept free-form text_prompt; provide points or boxes.")
        h, w = image_rgb.shape[:2]
        if points is None and boxes is None:
            raise ValueError("SAM 2.1 single-image mode requires at least points or boxes.")

        pts_arr = np.asarray(points, dtype=np.float32) if points is not None else None
        lbl_arr = None
        if pts_arr is not None:
            lbl_arr = (
                np.asarray(point_labels, dtype=np.int32)
                if point_labels is not None
                else np.ones(len(pts_arr), dtype=np.int32)
            )
        box_arr = np.asarray(boxes, dtype=np.float32) if boxes is not None else None

        try:
            with _inference_context():
                self.predictor.set_image(image_rgb)
                masks, scores, _ = self.predictor.predict(
                    point_coords=pts_arr,
                    point_labels=lbl_arr,
                    box=box_arr[0] if (box_arr is not None and box_arr.ndim == 2 and len(box_arr) == 1) else box_arr,
                    multimask_output=True,
                )
            masks_np = _to_numpy(masks)
            scores_np = _to_numpy(scores) if scores is not None else np.ones(len(masks_np))
            best = int(np.argmax(scores_np)) if len(scores_np) else 0
            cand = masks_np[best] if masks_np.ndim == 3 else masks_np
            binary = binarize_mask(cand, min_mask_area=1, threshold=alpha_threshold)
            if binary.shape != (h, w):
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            labels = ["selected_region"] if np.any(binary) else []
            return ImageSegmentationResult(
                mask=binary,
                segmenter=self.name,
                labels=labels,
                masks_by_label=({"selected_region": binary} if labels else {}),
                scores=[float(scores_np[best])] if len(scores_np) else [],
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_predictor and self.predictor is not None:
            self.predictor = None
        super().close()


class SAM3ImageSegmenter(BaseImageSegmenter):
    """Meta SAM 3 single-image text/promptable segmenter (Meta SAM License - non-commercial)."""

    name = "sam3_image"
    component_id = "sam3_image"
    category = "promptable"
    supports_text_prompts = True
    supports_point_box_prompts = True
    _default_cfg = "configs/sam3/sam3_image.yaml"

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        predictor: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._owns_predictor = predictor is None
        self.predictor = predictor if predictor is not None else self._build_predictor()

    def _build_predictor(self) -> Any:
        mod = _require_module(
            "sam3",
            "Install official Meta SAM 3 (https://github.com/facebookresearch/sam3, Meta SAM License, benchmark-only).",
        )
        builder = getattr(mod, "build_sam3_image_predictor", None) or getattr(mod, "SAM3ImagePredictor", None)
        if builder is None:
            raise RuntimeError("Installed 'sam3' package does not expose SAM 3 image predictor.")
        if self.checkpoint_path:
            return builder(str(self.checkpoint_path), device=self._resolve_device())
        return builder(device=self._resolve_device())

    def segment_image(
        self,
        image_rgb: np.ndarray,
        *,
        text_prompt: str | None = None,
        points: Any | None = None,
        point_labels: Any | None = None,
        boxes: Any | None = None,
        target_classes: list[str] | str | None = None,
        alpha_threshold: float = 0.5,
    ) -> ImageSegmentationResult:
        h, w = image_rgb.shape[:2]
        prompt_str = str(text_prompt).strip() if text_prompt else ""
        if not prompt_str and points is None and boxes is None:
            raise ValueError(f"{self.name} requires text_prompt, points, or boxes.")
        try:
            with _inference_context():
                if hasattr(self.predictor, "set_image"):
                    self.predictor.set_image(image_rgb)
                masks, scores, _ = self.predictor.predict(
                    text_prompt=prompt_str or None,
                    point_coords=np.asarray(points, dtype=np.float32) if points is not None else None,
                    point_labels=np.asarray(point_labels, dtype=np.int32) if point_labels is not None else None,
                    box=np.asarray(boxes, dtype=np.float32) if boxes is not None else None,
                )
            masks_np = _to_numpy(masks)
            scores_np = _to_numpy(scores) if scores is not None else np.ones(len(masks_np))
            best = int(np.argmax(scores_np)) if len(scores_np) else 0
            cand = masks_np[best] if masks_np.ndim == 3 else masks_np
            binary = binarize_mask(cand, min_mask_area=1, threshold=alpha_threshold)
            if binary.shape != (h, w):
                binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
            lbl = prompt_str or "sam3_region"
            return ImageSegmentationResult(
                mask=binary,
                segmenter=self.name,
                labels=[lbl] if np.any(binary) else [],
                masks_by_label=({lbl: binary} if np.any(binary) else {}),
                scores=[float(scores_np[best])] if len(scores_np) else [],
            )
        finally:
            self._release_gpu_memory()

    def close(self) -> None:
        if self._owns_predictor and self.predictor is not None:
            self.predictor = None
        super().close()


class SAM31ImageSegmenter(SAM3ImageSegmenter):
    """Meta SAM 3.1 single-image text/promptable segmenter (Meta SAM License - non-commercial)."""

    name = "sam31_image"
    component_id = "sam31_image"
    _default_cfg = "configs/sam3/sam3.1_image.yaml"


@dataclass(frozen=True)
class ImageSegmenterSpec:
    """Descriptor for a registered single-image keyframe segmenter."""

    name: str
    label: str
    component_id: str
    category: str
    license: str
    commercial_tier: str  # "allowed_osi" | "hard_blocked"
    supports_text_prompts: bool
    supports_point_box_prompts: bool
    factory: Callable[..., ImageSegmenterBackend]
    probe_modules: tuple[str, ...]
    supports_target_classes: bool = False
    aliases: tuple[str, ...] = ()
    description: str = ""


class ImageSegmenterRegistry:
    """Pluggable registry mapping still-image keyframe segmenters to backends."""

    def __init__(self) -> None:
        self._specs: dict[str, ImageSegmenterSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: ImageSegmenterSpec) -> None:
        if spec.category not in VALID_IMAGE_CATEGORIES:
            raise ValueError(f"Invalid image segmenter category {spec.category!r}")
        canonical = spec.name.strip().lower()
        self._specs[canonical] = spec
        self._aliases[canonical] = canonical
        for alias in spec.aliases:
            self._aliases[alias.strip().lower()] = canonical

    def resolve_name(self, name: str) -> str:
        direct = str(name or "").strip().lower()
        norm = direct.replace("-", "_")
        for cand in (direct, norm):
            if cand in self._aliases:
                return self._aliases[cand]
        available = ", ".join(self._specs.keys())
        raise ValueError(f"Unknown keyframe image segmenter {name!r}. Supported: {available}")

    def has(self, name: str) -> bool:
        try:
            self.resolve_name(name)
            return True
        except ValueError:
            return False

    def get_spec(self, name: str) -> ImageSegmenterSpec:
        return self._specs[self.resolve_name(name)]

    def list_specs(self) -> list[ImageSegmenterSpec]:
        return list(self._specs.values())

    def is_available(self, name: str) -> bool:
        return self.availability_info(name)["available"]

    def availability_info(
        self,
        name: str,
        *,
        settings: Mapping[str, Mapping[str, str]] | None = None,
    ) -> dict[str, Any]:
        spec = self.get_spec(name)
        env_name = env_for_model(spec.name) or ""
        local_ok = _module_available(spec.probe_modules)
        if local_ok:
            return {
                "available": True,
                "runtime": "in_process",
                "env_name": env_name,
                "env_status": inspect_env(env_name, settings=settings)["status"] if env_name else "missing",
            }
        if env_name:
            env_info = inspect_env(env_name, settings=settings)
            if env_info["ready"]:
                return {
                    "available": True,
                    "runtime": f"managed_env:{env_name}",
                    "env_name": env_name,
                    "env_status": "ready",
                }
            return {
                "available": False,
                "runtime": "missing",
                "env_name": env_name,
                "env_status": env_info["status"],
            }
        return {
            "available": False,
            "runtime": "missing",
            "env_name": "",
            "env_status": "missing",
        }

    def check_policy(
        self,
        name: str,
        *,
        settings: Mapping[str, Mapping[str, str]] | None = None,
        allow_non_commercial_benchmark: bool = False,
        checkpoint_path: Path | str | None = None,
    ) -> PolicyDecision:
        spec = self.get_spec(name)
        decision = evaluate_component(
            spec.component_id,
            file_path=checkpoint_path,
            settings=settings,
            component_type="tool",
        )
        guard_on = (not allow_non_commercial_benchmark) and commercial_guard_enabled(settings)
        if guard_on and not decision.allowed:
            raise CommercialPolicyError(
                f"Commercial Model Guard blocked keyframe image segmenter '{spec.name}': {decision.reason}"
            )
        return decision

    def create(
        self,
        name: str,
        *,
        settings: Mapping[str, Mapping[str, str]] | None = None,
        allow_non_commercial_benchmark: bool = False,
        **kwargs: Any,
    ) -> ImageSegmenterBackend:
        spec = self.get_spec(name)
        self.check_policy(
            spec.name,
            settings=settings,
            allow_non_commercial_benchmark=allow_non_commercial_benchmark,
            checkpoint_path=kwargs.get("checkpoint_path"),
        )
        return spec.factory(**kwargs)


def _build_default_image_registry() -> ImageSegmenterRegistry:
    reg = ImageSegmenterRegistry()
    reg.register(
        ImageSegmenterSpec(
            name="evf_sam2",
            label="EVF-SAM-2 (Text-Prompt Still, Apache-2.0)",
            component_id="evf_sam2",
            category="promptable",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=True,
            supports_point_box_prompts=False,
            factory=EVFSAM2ImageSegmenter,
            probe_modules=("evf_sam", "transformers"),
            aliases=("evf-sam2", "evf-sam-2"),
            description="Referring-expression segmentation on still frames using natural language prompts.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="florence2",
            label="Florence-2 (Text Referring Seg, MIT)",
            component_id="florence2",
            category="promptable",
            license="MIT",
            commercial_tier="allowed_osi",
            supports_text_prompts=True,
            supports_point_box_prompts=False,
            factory=Florence2ImageSegmenter,
            probe_modules=("transformers",),
            aliases=("florence-2",),
            description="Microsoft Florence-2 referring-expression polygon segmentation for stills.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="birefnet",
            label="BiRefNet / BiRefNet_lite (Sharp Foreground Matting, MIT)",
            component_id="birefnet",
            category="matting",
            license="MIT",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            supports_point_box_prompts=False,
            factory=BiRefNetImageSegmenter,
            probe_modules=("birefnet", "transformers"),
            aliases=("birefnet-lite", "birefnet_lite"),
            description="High-resolution edge-accurate foreground segmentation/matting.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="mask2former",
            label="Mask2Former (Panoptic / Scene Classes, MIT/Apache-2.0)",
            component_id="mask2former",
            category="panoptic",
            license="MIT / Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            supports_point_box_prompts=False,
            supports_target_classes=True,
            factory=Mask2FormerImageSegmenter,
            probe_modules=("transformers",),
            aliases=("mask2-former",),
            description="Panoptic/semantic parser for sky, road, buildings, foliage, and people.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="oneformer",
            label="OneFormer (Universal Panoptic / Instance, MIT)",
            component_id="oneformer",
            category="panoptic",
            license="MIT",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            supports_point_box_prompts=False,
            supports_target_classes=True,
            factory=OneFormerImageSegmenter,
            probe_modules=("transformers",),
            aliases=("one-former",),
            description="SHI-Labs OneFormer universal panoptic/instance/semantic segmentation.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="grounding_dino",
            label="Grounding DINO (Open-Vocab Text Boxes, Apache-2.0)",
            component_id="grounding_dino",
            category="detection",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=True,
            supports_point_box_prompts=False,
            factory=GroundingDINOImageSegmenter,
            probe_modules=("groundingdino", "transformers"),
            aliases=("grounding-dino", "groundingdino"),
            description="Open-vocabulary text-to-box detection with optional SAM 2.1 mask refinement.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="rf_detr_seg",
            label="RF-DETR-Seg (Real-Time Instance Still, Apache-2.0)",
            component_id="rf_detr_seg",
            category="detection",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            supports_point_box_prompts=False,
            supports_target_classes=True,
            factory=RFDETRSegImageSegmenter,
            probe_modules=("rfdetr",),
            aliases=("rf-detr-seg", "rfdetr"),
            description="Fast Apache-2.0 instance segmentation on single reference frames.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="sam21_image",
            label="SAM 2.1 Single-Image (Points / Boxes, Apache-2.0)",
            component_id="sam21_image",
            category="promptable",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            supports_point_box_prompts=True,
            factory=SAM21ImageSegmenter,
            probe_modules=("sam2",),
            aliases=("sam21-image", "sam2.1-image"),
            description="Official Apache-2.0 SAM 2.1 single-image point and box predictor.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="sam3_image",
            label="SAM 3 Still-Image (Meta SAM License — Non-Commercial / Benchmark Only)",
            component_id="sam3_image",
            category="promptable",
            license="Meta SAM License",
            commercial_tier="hard_blocked",
            supports_text_prompts=True,
            supports_point_box_prompts=True,
            factory=SAM3ImageSegmenter,
            probe_modules=("sam3",),
            aliases=("sam3-image",),
            description="Meta SAM 3 promptable still segmentation restricted to non-commercial benchmarks.",
        )
    )
    reg.register(
        ImageSegmenterSpec(
            name="sam31_image",
            label="SAM 3.1 Still-Image (Meta SAM License — Non-Commercial / Benchmark Only)",
            component_id="sam31_image",
            category="promptable",
            license="Meta SAM License",
            commercial_tier="hard_blocked",
            supports_text_prompts=True,
            supports_point_box_prompts=True,
            factory=SAM31ImageSegmenter,
            probe_modules=("sam3",),
            aliases=("sam31-image", "sam3.1-image"),
            description="Meta SAM 3.1 promptable still segmentation restricted to non-commercial benchmarks.",
        )
    )
    return reg


DEFAULT_IMAGE_SEGMENTER_REGISTRY: ImageSegmenterRegistry = _build_default_image_registry()


def _record_keyframe_compliance(
    *,
    compliance_output_dir: Path | str,
    spec: ImageSegmenterSpec,
    mask_path: Path,
    sidecar_path: Path,
    checkpoint_path: Path | str | None,
    settings: Mapping[str, Any] | None,
    allow_non_commercial_benchmark: bool,
) -> None:
    from .compliance import append_compliance_record

    eff_settings: dict[str, Any] = {
        k: dict(v) if isinstance(v, Mapping) else v
        for k, v in (settings.items() if settings else [])
    }
    global_cfg = dict(eff_settings.get("global", {})) if isinstance(eff_settings.get("global"), Mapping) else {}
    if allow_non_commercial_benchmark:
        global_cfg["non_commercial_benchmark_override"] = "true"
    elif "commercial_model_guard" not in global_cfg:
        global_cfg["commercial_model_guard"] = "true"
    eff_settings["global"] = global_cfg

    is_nc_run = bool(
        allow_non_commercial_benchmark
        or spec.commercial_tier == "hard_blocked"
        or str(global_cfg.get("non_commercial_benchmark_override", "false")).lower() == "true"
        or str(global_cfg.get("commercial_model_guard", "true")).lower() == "false"
    )

    append_compliance_record(
        output_dir=compliance_output_dir,
        stage="keyframe_segmentation",
        models=[
            {
                "model_id": spec.component_id,
                "component_type": "tool",
                "file_path": str(checkpoint_path) if checkpoint_path else None,
            }
        ],
        settings=eff_settings,
        metadata={
            "image_segmenter": spec.name,
            "category": spec.category,
            "mask_path": str(mask_path),
            "sidecar_path": str(sidecar_path),
            "non_commercial_benchmark_run": is_nc_run,
        },
    )


def _dispatch_keyframe_in_managed_env(
    *,
    env_name: str,
    spec: ImageSegmenterSpec,
    image: Path | str | np.ndarray,
    mask_file: Path,
    sidecar_file: Path,
    text_prompt: str | None,
    points: Any | None,
    boxes: Any | None,
    target_classes: list[str] | str | None,
    alpha_threshold: float,
    checkpoint_path: Path | str | None,
    device: str,
    settings: Mapping[str, Mapping[str, str]] | None,
    allow_non_commercial_benchmark: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="arp-kf-env-") as tmpdir:
        tmp_path = Path(tmpdir)
        if isinstance(image, np.ndarray):
            img_arg = tmp_path / "keyframe_input.png"
            Image.fromarray(load_image_rgb_array(image)).save(img_arg, format="PNG")
        else:
            img_arg = Path(image)

        argv: list[str] = [
            "--segmenter",
            spec.name,
            "--image",
            str(img_arg),
            "--output",
            str(mask_file),
            "--alpha-threshold",
            str(alpha_threshold),
            "--device",
            str(device),
            "--compliance-output-dir",
            "",
        ]
        if text_prompt is not None and str(text_prompt).strip():
            argv.extend(["--text-prompt", str(text_prompt).strip()])
        cls_list = normalize_target_classes(target_classes)
        if cls_list:
            argv.extend(["--target-classes", ",".join(cls_list)])
        if points is not None:
            pts_file = tmp_path / "points.json"
            pts_file.write_text(json.dumps(np.asarray(points).tolist()), encoding="utf-8")
            argv.extend(["--points-json", str(pts_file)])
        if boxes is not None:
            boxes_file = tmp_path / "boxes.json"
            boxes_file.write_text(json.dumps(np.asarray(boxes).tolist()), encoding="utf-8")
            argv.extend(["--boxes-json", str(boxes_file)])
        if checkpoint_path is not None:
            argv.extend(["--checkpoint", str(checkpoint_path)])
        if allow_non_commercial_benchmark or spec.commercial_tier == "hard_blocked":
            argv.append("--allow-non-commercial-benchmark")

        run_in_managed_env(
            env_name,
            ROOT / "scripts" / "segment_keyframe_mask.py",
            argv,
            settings=settings,
            allow_non_commercial=bool(allow_non_commercial_benchmark or spec.commercial_tier == "hard_blocked"),
        )

    if not mask_file.is_file():
        raise RuntimeError(
            f"Managed env '{env_name}' did not produce expected keyframe mask at '{mask_file}'."
        )

    mask_sha256 = compute_sha256(mask_file)
    if sidecar_file.is_file():
        sidecar_data = json.loads(sidecar_file.read_text(encoding="utf-8"))
        labels = list(sidecar_data.get("labels", []))
        fg_pixels = int(sidecar_data.get("foreground_pixels", 0))
        w = int(sidecar_data.get("width", 0))
        h = int(sidecar_data.get("height", 0))
    else:
        with Image.open(mask_file) as im:
            arr = np.array(im.convert("L"), dtype=np.uint8)
        h, w = arr.shape[:2]
        fg_pixels = int(np.count_nonzero(arr))
        labels = [str(text_prompt).strip()] if text_prompt and str(text_prompt).strip() else []

    return {
        "mask_path": str(mask_file),
        "sidecar_path": str(sidecar_file),
        "mask_sha256": mask_sha256,
        "segmenter": spec.name,
        "category": spec.category,
        "labels": labels,
        "foreground_pixels": fg_pixels,
        "width": int(w),
        "height": int(h),
        "runtime": f"managed_env:{env_name}",
    }


def run_keyframe_segmentation(
    *,
    image: Path | str | np.ndarray,
    output_mask_path: Path | str,
    segmenter: str = "birefnet",
    text_prompt: str | None = None,
    points: Any | None = None,
    boxes: Any | None = None,
    target_classes: list[str] | str | None = None,
    alpha_threshold: float = 0.5,
    checkpoint_path: Path | str | None = None,
    device: str = "auto",
    settings: Mapping[str, Mapping[str, str]] | None = None,
    allow_non_commercial_benchmark: bool = False,
    compliance_output_dir: Path | str | None = None,
    registry: ImageSegmenterRegistry | None = None,
    **backend_kwargs: Any,
) -> dict[str, Any]:
    """Run a registered ImageSegmenterBackend and write binary PNG mask + JSON sidecar."""
    reg = registry or DEFAULT_IMAGE_SEGMENTER_REGISTRY
    spec = reg.get_spec(segmenter)

    # Always evaluate commercial policy FIRST (fail-closed).
    reg.check_policy(
        spec.name,
        settings=settings,
        allow_non_commercial_benchmark=allow_non_commercial_benchmark,
        checkpoint_path=checkpoint_path,
    )

    init_kwargs: dict[str, Any] = dict(backend_kwargs)
    if checkpoint_path is not None:
        init_kwargs.setdefault("checkpoint_path", checkpoint_path)
    init_kwargs.setdefault("device", device)

    mask_file = Path(output_mask_path)
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    sidecar_file = mask_file.with_suffix(".json")

    has_injected_runner = any(init_kwargs.get(k) is not None for k in ("runner", "predictor", "refiner"))
    local_ok = _module_available(spec.probe_modules)

    if not has_injected_runner and not local_ok:
        env_name = env_for_model(spec.name)
        if env_name and not is_managed_env_active():
            env_info = inspect_env(env_name, settings=settings)
            if env_info["ready"]:
                out = _dispatch_keyframe_in_managed_env(
                    env_name=env_name,
                    spec=spec,
                    image=image,
                    mask_file=mask_file,
                    sidecar_file=sidecar_file,
                    text_prompt=text_prompt,
                    points=points,
                    boxes=boxes,
                    target_classes=target_classes,
                    alpha_threshold=alpha_threshold,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    settings=settings,
                    allow_non_commercial_benchmark=allow_non_commercial_benchmark,
                )
                if compliance_output_dir is not None:
                    _record_keyframe_compliance(
                        compliance_output_dir=compliance_output_dir,
                        spec=spec,
                        mask_path=mask_file,
                        sidecar_path=sidecar_file,
                        checkpoint_path=checkpoint_path,
                        settings=settings,
                        allow_non_commercial_benchmark=allow_non_commercial_benchmark,
                    )
                return out
        target_env = env_name or "transformers_seg"
        raise RuntimeError(
            f"Keyframe image segmenter '{spec.name}' dependencies ({', '.join(spec.probe_modules)}) are not installed locally "
            f"and managed environment '{target_env}' is not ready. Install it via "
            f"'python scripts/setup_model_env.py install {target_env}' or the GUI 'Install Environment' button."
        )

    image_rgb = load_image_rgb_array(image)
    backend = reg.create(
        spec.name,
        settings=settings,
        allow_non_commercial_benchmark=allow_non_commercial_benchmark,
        **init_kwargs,
    )
    with backend:
        result = backend.segment_image(
            image_rgb,
            text_prompt=text_prompt,
            points=points,
            boxes=boxes,
            target_classes=target_classes,
            alpha_threshold=alpha_threshold,
        )

    Image.fromarray(result.mask, mode="L").save(mask_file, format="PNG")
    mask_sha256 = compute_sha256(mask_file)
    h, w = result.mask.shape[:2]
    fg_pixels = int(np.count_nonzero(result.mask))

    sidecar_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segmenter": spec.name,
        "component_id": spec.component_id,
        "category": spec.category,
        "license": spec.license,
        "commercial_tier": spec.commercial_tier,
        "image_path": (str(image) if isinstance(image, (Path, str)) else None),
        "mask_path": str(mask_file),
        "mask_sha256": mask_sha256,
        "width": int(w),
        "height": int(h),
        "foreground_pixels": fg_pixels,
        "text_prompt": text_prompt,
        "target_classes": normalize_target_classes(target_classes),
        "labels": result.labels,
        "boxes": [list(b) for b in result.boxes],
        "scores": [float(s) for s in result.scores],
        "metadata": result.metadata,
    }
    sidecar_file.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")

    if compliance_output_dir is not None:
        _record_keyframe_compliance(
            compliance_output_dir=compliance_output_dir,
            spec=spec,
            mask_path=mask_file,
            sidecar_path=sidecar_file,
            checkpoint_path=checkpoint_path,
            settings=settings,
            allow_non_commercial_benchmark=allow_non_commercial_benchmark,
        )

    return {
        "mask_path": str(mask_file),
        "sidecar_path": str(sidecar_file),
        "mask_sha256": mask_sha256,
        "segmenter": spec.name,
        "category": spec.category,
        "labels": result.labels,
        "foreground_pixels": fg_pixels,
        "width": int(w),
        "height": int(h),
    }


__all__ = [
    "BiRefNetImageSegmenter",
    "BaseImageSegmenter",
    "DEFAULT_IMAGE_SEGMENTER_REGISTRY",
    "EVFSAM2ImageSegmenter",
    "Florence2ImageSegmenter",
    "GroundingDINOImageSegmenter",
    "ImageSegmentationResult",
    "ImageSegmenterBackend",
    "ImageSegmenterRegistry",
    "ImageSegmenterSpec",
    "Mask2FormerImageSegmenter",
    "OneFormerImageSegmenter",
    "RFDETRSegImageSegmenter",
    "SAM21ImageSegmenter",
    "SAM31ImageSegmenter",
    "SAM3ImageSegmenter",
    "load_image_rgb_array",
    "load_seed_mask_for_tracker",
    "normalize_target_classes",
    "run_keyframe_segmentation",
]
