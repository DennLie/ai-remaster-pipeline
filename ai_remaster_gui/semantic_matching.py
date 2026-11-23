"""Semantic reference matching subsystem for Color Bible guided colorization.

Provides film-wide semantic reference retrieval, human-in-the-loop review gating,
structured prompt enrichment with CIE-LAB and RGB palette statistics, two-stage
reference backend routing, scoped cache signatures, and commercial compliance logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .paths import ROOT, resolve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols & Data Structures for Guided Reference Backends
# ---------------------------------------------------------------------------


@dataclass
class ReferenceRequest:
    """Request payload passed to a GuidedReferenceBackend."""

    shot_index: int
    source_image_path: Path | str
    output_path: Path | str
    reference_image_paths: list[Path | str] = field(default_factory=list)
    prompt: str = ""
    prompt_suffix: str = ""
    workflow: str | None = None
    comfy_url: str | None = None
    comfy_dir: str | None = None
    model_backend: str = "gguf"
    gguf_model: str = ""
    seed: int = -1
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceResult:
    """Execution result returned by a GuidedReferenceBackend."""

    output_path: Path | str
    backend_name: str
    command: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GuidedReferenceBackend(Protocol):
    """Protocol for pluggable guided colorization reference backends."""

    name: str

    def colorize_shot(self, request: ReferenceRequest) -> ReferenceResult:
        """Execute or prepare colorization for a single shot reference."""
        ...


class ReferenceBackendRegistry:
    """Registry maintaining available GuidedReferenceBackend implementations."""

    def __init__(self) -> None:
        self._backends: dict[str, GuidedReferenceBackend | type[GuidedReferenceBackend] | Callable[[], GuidedReferenceBackend]] = {}

    def register(
        self,
        name_or_backend: str | GuidedReferenceBackend | type[GuidedReferenceBackend] | None = None,
        backend: GuidedReferenceBackend | type[GuidedReferenceBackend] | Callable[[], GuidedReferenceBackend] | None = None,
    ) -> Any:
        """Register a backend instance, class, or factory. Supports decorator usage."""
        if callable(name_or_backend) and backend is None:
            # Decorator without arguments: @registry.register
            target = name_or_backend
            name = getattr(target, "name", target.__name__.lower())
            self._backends[str(name)] = target
            return target

        if isinstance(name_or_backend, str):
            if backend is None:
                # Decorator with name: @registry.register("my_backend")
                def decorator(fn_or_cls: Any) -> Any:
                    self._backends[name_or_backend] = fn_or_cls
                    return fn_or_cls

                return decorator
            # Direct call: registry.register("my_backend", backend)
            self._backends[name_or_backend] = backend
            return backend

        if name_or_backend is not None and backend is None:
            name = getattr(name_or_backend, "name", name_or_backend.__class__.__name__.lower())
            self._backends[str(name)] = name_or_backend
            return name_or_backend

        raise ValueError("Invalid backend registration arguments.")

    def get(self, name: str) -> GuidedReferenceBackend:
        """Retrieve an initialized backend instance by name."""
        key = str(name).strip().lower()
        if key not in self._backends:
            raise KeyError(f"Guided reference backend '{name}' is not registered. Registered: {self.list_backends()}")
        candidate = self._backends[key]
        if isinstance(candidate, type) or callable(candidate):
            instance = candidate()
            return instance
        return candidate

    def list_backends(self) -> list[str]:
        """Return all registered backend names."""
        return sorted(list(self._backends.keys()))

    def clear(self) -> None:
        """Clear all registered backends."""
        self._backends.clear()


registry = ReferenceBackendRegistry()
register_backend = registry.register


# ---------------------------------------------------------------------------
# Default Guided Reference Backend: Qwen2511Backend
# ---------------------------------------------------------------------------


@registry.register("qwen2511")
class Qwen2511Backend:
    """Built-in GuidedReferenceBackend interfacing with Qwen-Image-Edit-2511 ComfyUI workflow."""

    name: str = "qwen2511"

    def colorize_shot(self, request: ReferenceRequest) -> ReferenceResult:
        """Build execution command or run ComfyUI reference regeneration for Qwen 2511."""
        scripts_dir = ROOT / "scripts"
        generator_script = scripts_dir / "generate_single_reference.py"

        source_path = str(request.source_image_path)
        out_path = str(request.output_path)

        cmd: list[str] = [
            sys.executable,
            "-u",
            str(generator_script),
            "--source-image",
            source_path,
            "--output",
            out_path,
        ]

        if request.workflow:
            cmd.extend(["--workflow", str(request.workflow)])
        if request.comfy_url:
            cmd.extend(["--comfy-url", str(request.comfy_url)])
        if request.comfy_dir:
            cmd.extend(["--comfy-dir", str(request.comfy_dir)])

        comfy_root = request.extra_params.get("comfy_output_root")
        if comfy_root:
            cmd.extend(["--comfy-output-root", str(comfy_root)])

        cmd.extend([
            "--model-backend",
            str(request.model_backend or "gguf"),
            "--gguf-model",
            str(request.gguf_model or "qwen-image-edit-2511"),
            "--prompt",
            str(request.prompt or ""),
            "--prompt-suffix",
            str(request.prompt_suffix or ""),
        ])

        if request.seed >= 0:
            cmd.extend(["--seed", str(request.seed)])

        for ref_img in request.reference_image_paths:
            cmd.extend(["--reference-image", str(ref_img)])

        cmd.append("--force")

        executor = request.extra_params.get("executor")
        success = True
        err_msg = None
        if callable(executor):
            try:
                executor(cmd)
            except Exception as e:
                success = False
                err_msg = str(e)
                logger.error("Qwen2511Backend execution failed: %s", e)

        return ReferenceResult(
            output_path=request.output_path,
            backend_name=self.name,
            command=cmd,
            success=success,
            error_message=err_msg,
            metadata={
                "seed": request.seed,
                "model_backend": request.model_backend,
                "gguf_model": request.gguf_model,
                "reference_images": [str(p) for p in request.reference_image_paths],
            },
        )


def get_backend(name: str) -> GuidedReferenceBackend:
    """Retrieve an initialized guided reference backend by name."""
    return registry.get(name)


def list_backends() -> list[str]:
    """List all registered guided reference backends."""
    return registry.list_backends()


# ---------------------------------------------------------------------------
# 3-Tier Semantic Reference Retrieval & Review Gate (ColorBibleMatcher)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticMatchResult:
    """Result of semantic reference resolution for a shot."""

    reference_paths: list[Path]
    retrieved_ref_ids: list[str]
    retrieved_ref_hashes: dict[str, str]
    entities: list[dict[str, Any]]
    needs_review: bool
    review_reasons: list[str]
    color_bible_version: str


def _safe_resolve_file(path_str: str) -> Path | None:
    """Attempt to resolve a path string to an existing file, handling workspace relativity."""
    if not path_str or not str(path_str).strip():
        return None
    raw = Path(path_str)
    try:
        if raw.is_file():
            return raw.resolve()
    except OSError:
        pass

    try:
        res = resolve(path_str)
        if res.is_file():
            return res.resolve()
    except (OSError, RuntimeError):
        pass

    return None


def _get_neighbor_references(rows: list[dict[str, str]], index: int, count: int) -> list[Path]:
    """Retrieve nearby shot color references using existing same-shot and adjacent-shot heuristics."""
    if not rows or count <= 0:
        return []

    target_shot = rows[index].get("_shot_index") if 0 <= index < len(rows) else None

    def shot_number(pos: int) -> int:
        try:
            return int(rows[pos].get("_shot_index", pos))
        except (ValueError, TypeError):
            return pos

    target_shot_num = shot_number(index) if 0 <= index < len(rows) else index

    own: list[tuple[int, Path]] = []
    others: list[tuple[int, int, Path]] = []

    for other_index, row in enumerate(rows):
        if other_index == index:
            continue
        candidate_str = row.get("color_reference") or row.get("reference") or ""
        if not candidate_str:
            continue
        cand_path = _safe_resolve_file(candidate_str)
        if not cand_path:
            continue

        other_shot = row.get("_shot_index")
        if target_shot is not None and other_shot == target_shot:
            own.append((abs(other_index - index), cand_path))
        else:
            dist = abs(shot_number(other_index) - target_shot_num)
            tie_breaker = 0 if other_index < index else 1
            others.append((dist, tie_breaker, cand_path))

    chosen = [p for _d, p in sorted(own, key=lambda item: item[0])][:count]
    fill = max(0, count - len(chosen))
    chosen += [p for _d, _tie, p in sorted(others, key=lambda item: (item[0], item[1]))][:fill]
    return chosen


class ColorBibleMatcher:
    """Matches shots to approved Color Bible master references with neighbor fallback and review gating."""

    @staticmethod
    def resolve_references(
        rows: list[dict[str, str]],
        index: int,
        count: int = 2,
        film_id: str | None = None,
        service: Any | None = None,
        confidence_threshold: float = 0.8,
    ) -> SemanticMatchResult:
        """Resolve up to `count` color references for a shot using the 3-Tier Semantic Retrieval engine.

        Priority A: Extract approved master color references from the Color Bible database.
                    Prioritizes character appearance variants first, then characters without variants,
                    then objects, then locations. Within each category, sorted by assignment confidence descending.
                    Deduplicates normalized file paths and skips missing files cleanly.
        Priority B: If fewer than `count` references retrieved, fills remaining slots with adjacent-shot
                    neighbor references, deduplicated against Priority A.
        Priority C (Review Gate): Evaluates whether any assigned entity has approval_status == 'pending_review'
                    or confidence < confidence_threshold. If so, marks needs_review = True with human-readable reasons.
        Backward Compatibility: If film_id or service is not provided, cleanly falls back 100% to neighbor logic.
        """
        # Determine actual shot index
        shot_index = index
        if 0 <= index < len(rows):
            s_val = rows[index].get("_shot_index")
            if s_val is not None:
                try:
                    shot_index = int(s_val)
                except (ValueError, TypeError):
                    pass

        # Backward compatibility fallback when Color Bible service or film_id is missing
        if not film_id or not str(film_id).strip() or service is None:
            neighbor_refs = _get_neighbor_references(rows, index, count)
            return SemanticMatchResult(
                reference_paths=neighbor_refs,
                retrieved_ref_ids=[],
                retrieved_ref_hashes={},
                entities=[],
                needs_review=False,
                review_reasons=[],
                color_bible_version="",
            )

        target_film_id = str(film_id).strip()

        # Query snapshot hash
        color_bible_version = ""
        try:
            snapshot_fn = getattr(service, "get_snapshot_hash", None)
            if callable(snapshot_fn):
                color_bible_version = str(snapshot_fn(target_film_id) or "")
        except Exception as e:
            logger.warning("ColorBibleMatcher: Failed to query snapshot hash for film '%s': %s", target_film_id, e)
            color_bible_version = ""

        # Query shot palette assignments
        assignments: list[dict[str, Any]] = []
        try:
            get_asgns_fn = getattr(service, "get_shot_palette_assignments", None)
            if callable(get_asgns_fn):
                assignments = list(get_asgns_fn(target_film_id, shot_index))
        except Exception as e:
            logger.warning("ColorBibleMatcher: Error querying shot palette assignments: %s", e)
            assignments = []

        needs_review = False
        review_reasons: list[str] = []

        # Priority C: Review Gate Check
        for item in assignments:
            asgn_dict = item.get("assignment") if isinstance(item.get("assignment"), dict) else item
            raw_status = str(asgn_dict.get("approval_status") or item.get("approval_status") or "").strip().lower()
            if raw_status == "rejected":
                continue

            raw_conf = asgn_dict.get("confidence", item.get("confidence", 1.0))
            try:
                conf = float(raw_conf) if raw_conf is not None else 1.0
            except (ValueError, TypeError):
                conf = 1.0

            entity_id = str(item.get("entity_id") or asgn_dict.get("entity_id") or "unknown")
            variant_id = str(item.get("variant_id") or asgn_dict.get("variant_id") or "")
            entity_info = item.get("entity") if isinstance(item.get("entity"), dict) else {}
            entity_name = str(entity_info.get("name") or entity_id)

            label = f"Entity '{entity_name}'"
            if variant_id:
                label += f" costume '{variant_id}'"

            if raw_status == "pending_review":
                needs_review = True
                review_reasons.append(f"{label} assignment status is 'pending_review'")
            elif conf < confidence_threshold:
                needs_review = True
                review_reasons.append(f"{label} assignment has confidence {conf:.2f} < {confidence_threshold:.2f}")

        # Filter out rejected entity assignments so rejected palette constraints are never injected
        valid_assignments: list[dict[str, Any]] = []
        for item in assignments:
            asgn_dict = item.get("assignment") if isinstance(item.get("assignment"), dict) else item
            raw_status = str(asgn_dict.get("approval_status") or item.get("approval_status") or "").strip().lower()
            if raw_status == "rejected":
                continue
            valid_assignments.append(item)

        # Priority A: Extract Approved Master Color References
        def _group_priority(asgn_item: dict[str, Any]) -> int:
            etype = str(asgn_item.get("entity_type") or "").strip().lower()
            var_id = asgn_item.get("variant_id")
            if etype == "character":
                return 0 if var_id else 1
            if etype == "object":
                return 2
            if etype == "location":
                return 3
            return 4

        def _item_confidence(asgn_item: dict[str, Any]) -> float:
            asgn = asgn_item.get("assignment") if isinstance(asgn_item.get("assignment"), dict) else asgn_item
            try:
                return float(asgn.get("confidence", 1.0) or 1.0)
            except (ValueError, TypeError):
                return 1.0

        selected_assignments = sorted(
            valid_assignments,
            key=lambda it: (_group_priority(it), -_item_confidence(it)),
        )

        selected_paths: list[Path] = []
        seen_normalized_paths: set[str] = set()
        retrieved_ref_ids: list[str] = []
        retrieved_ref_hashes: dict[str, str] = {}

        for asgn_item in selected_assignments:
            if len(selected_paths) >= count:
                break

            color_refs = asgn_item.get("color_references", [])
            if not isinstance(color_refs, list):
                continue

            # Filter approved references only
            approved_refs = [
                r for r in color_refs
                if isinstance(r, dict) and str(r.get("approval_status", "")).strip().lower() == "approved"
            ]

            # Sort by version descending
            def _ref_version(r: dict[str, Any]) -> int:
                try:
                    return int(r.get("version", 0))
                except (ValueError, TypeError):
                    return 0

            sorted_refs = sorted(approved_refs, key=_ref_version, reverse=True)

            for ref in sorted_refs:
                if len(selected_paths) >= count:
                    break

                raw_path = (
                    ref.get("source_keyframe_path")
                    or ref.get("keyframe_path")
                    or ref.get("path")
                    or ""
                )
                if not raw_path:
                    continue

                resolved_file = _safe_resolve_file(str(raw_path))
                if not resolved_file:
                    logger.warning("ColorBibleMatcher: Reference image file does not exist: %s", raw_path)
                    continue

                norm_key = os.path.normpath(str(resolved_file))
                if norm_key in seen_normalized_paths:
                    continue

                seen_normalized_paths.add(norm_key)
                selected_paths.append(resolved_file)

                ref_id = str(ref.get("ref_id") or "").strip()
                if ref_id:
                    retrieved_ref_ids.append(ref_id)
                    ref_hash = str(ref.get("sha256") or f"v{ref.get('version', 1)}")
                    retrieved_ref_hashes[ref_id] = ref_hash

        # Priority B: Fill remaining slots with neighbor references
        if len(selected_paths) < count:
            slots_needed = count - len(selected_paths)
            neighbor_candidates = _get_neighbor_references(rows, index, count=count + len(seen_normalized_paths))
            for cand in neighbor_candidates:
                norm_cand = os.path.normpath(str(cand.resolve()))
                if norm_cand not in seen_normalized_paths:
                    seen_normalized_paths.add(norm_cand)
                    selected_paths.append(cand)
                    if len(selected_paths) >= count:
                        break

        return SemanticMatchResult(
            reference_paths=selected_paths,
            retrieved_ref_ids=retrieved_ref_ids,
            retrieved_ref_hashes=retrieved_ref_hashes,
            entities=selected_assignments,
            needs_review=needs_review,
            review_reasons=review_reasons,
            color_bible_version=color_bible_version,
        )


def resolve_references(
    rows: list[dict[str, str]],
    index: int,
    count: int = 2,
    film_id: str | None = None,
    service: Any | None = None,
    confidence_threshold: float = 0.8,
) -> SemanticMatchResult:
    """Convenience function calling ColorBibleMatcher.resolve_references."""
    return ColorBibleMatcher.resolve_references(
        rows=rows,
        index=index,
        count=count,
        film_id=film_id,
        service=service,
        confidence_threshold=confidence_threshold,
    )


# ---------------------------------------------------------------------------
# Structured Prompt Enrichment (PromptEnricher)
# ---------------------------------------------------------------------------


class PromptEnricher:
    """Enriches neural colorization prompts with structured Color Bible constraints."""

    @staticmethod
    def enrich_prompt(
        base_prompt: str,
        entities: list[dict[str, Any]],
        lighting_context: str = "",
    ) -> str:
        """Inject character names, costume IDs, target RGB swatches, and CIE-LAB statistics into prompt.

        Example output:
        "Base prompt text [Color Bible Constraints: Character Mickey (costume: red_shorts, target RGB: #D32F2F / [211, 47, 47], LAB L*: 45.2, a*: 58.1, b*: 39.4); Scene Lighting: warm afternoon daylight]."
        """
        entity_clauses: list[str] = []
        extracted_lighting = str(lighting_context or "").strip()

        for item in entities:
            etype = str(item.get("entity_type") or "entity").strip().lower()
            e_info = item.get("entity") if isinstance(item.get("entity"), dict) else {}
            v_info = item.get("variant") if isinstance(item.get("variant"), dict) else {}

            name = str(
                e_info.get("name")
                or e_info.get("set_name")
                or item.get("name")
                or item.get("entity_id")
                or "Unknown"
            ).strip()

            costume_id = str(
                v_info.get("costume_id")
                or item.get("costume_id")
                or item.get("variant_id")
                or ""
            ).strip()

            if not extracted_lighting:
                found_light = v_info.get("lighting_context") or e_info.get("lighting_context") or item.get("lighting_context")
                if found_light:
                    extracted_lighting = str(found_light).strip()

            # Extract target RGB swatch
            rgb_str = ""
            rgb_list: list[int] | None = None
            hex_val: str | None = None

            # 1. From color_references rgb_palette_json
            color_refs = item.get("color_references", [])
            if isinstance(color_refs, list) and color_refs:
                first_ref = color_refs[0]
                palette_raw = first_ref.get("rgb_palette_json")
                if palette_raw:
                    try:
                        palette_data = json.loads(palette_raw) if isinstance(palette_raw, str) else palette_raw
                        if isinstance(palette_data, list) and palette_data:
                            top_color = palette_data[0]
                            hex_val = str(top_color.get("hex", "")).strip().upper()
                            c_rgb = top_color.get("rgb")
                            if isinstance(c_rgb, list) and len(c_rgb) >= 3:
                                rgb_list = [int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2])]
                    except Exception:
                        pass

            # 2. From entity or item palette
            if not hex_val:
                raw_pal = e_info.get("palette") or item.get("palette") or item.get("rgb_palette")
                if isinstance(raw_pal, list) and raw_pal:
                    first_p = raw_pal[0]
                    if isinstance(first_p, str) and first_p.startswith("#"):
                        hex_val = first_p.strip().upper()
                        if len(hex_val) == 7:
                            try:
                                rgb_list = [
                                    int(hex_val[1:3], 16),
                                    int(hex_val[3:5], 16),
                                    int(hex_val[5:7], 16),
                                ]
                            except ValueError:
                                pass
                    elif isinstance(first_p, dict):
                        hex_val = str(first_p.get("hex", "")).strip().upper() or None
                        c_rgb = first_p.get("rgb")
                        if isinstance(c_rgb, list) and len(c_rgb) >= 3:
                            rgb_list = [int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2])]

            # 3. Direct rgb / hex keys
            if not hex_val and item.get("hex"):
                hex_val = str(item["hex"]).strip().upper()
            if not rgb_list and isinstance(item.get("rgb"), list) and len(item["rgb"]) >= 3:
                rgb_list = [int(item["rgb"][0]), int(item["rgb"][1]), int(item["rgb"][2])]

            if hex_val and rgb_list:
                rgb_str = f"target RGB: {hex_val} / [{rgb_list[0]}, {rgb_list[1]}, {rgb_list[2]}]"
            elif hex_val:
                rgb_str = f"target RGB: {hex_val}"
            elif rgb_list:
                rgb_str = f"target RGB: [{rgb_list[0]}, {rgb_list[1]}, {rgb_list[2]}]"

            # Extract LAB percentiles (median 50th percentile)
            lab_str = ""
            if isinstance(color_refs, list) and color_refs:
                first_ref = color_refs[0]
                lab_raw = first_ref.get("lab_stats_json")
                if lab_raw:
                    try:
                        lab_data = json.loads(lab_raw) if isinstance(lab_raw, str) else lab_raw
                        if isinstance(lab_data, dict):
                            L_vals = lab_data.get("L")
                            a_vals = lab_data.get("a")
                            b_vals = lab_data.get("b")

                            def _extract_median(vals: Any) -> float | None:
                                if isinstance(vals, list) and vals:
                                    # Pick middle element (50th percentile in standard 5-point list)
                                    mid = len(vals) // 2
                                    return float(vals[mid])
                                if isinstance(vals, (int, float)):
                                    return float(vals)
                                return None

                            L_med = _extract_median(L_vals)
                            a_med = _extract_median(a_vals)
                            b_med = _extract_median(b_vals)
                            if L_med is not None and a_med is not None and b_med is not None:
                                lab_str = f"LAB L*: {L_med:.1f}, a*: {a_med:.1f}, b*: {b_med:.1f}"
                    except Exception:
                        pass

            # Direct lab key fallback
            if not lab_str and isinstance(item.get("lab"), dict):
                lab_d = item["lab"]
                if "L" in lab_d and "a" in lab_d and "b" in lab_d:
                    lab_str = f"LAB L*: {float(lab_d['L']):.1f}, a*: {float(lab_d['a']):.1f}, b*: {float(lab_d['b']):.1f}"

            # Format entity descriptor
            category_title = etype.capitalize()
            descriptors: list[str] = []
            if costume_id:
                descriptors.append(f"costume: {costume_id}")
            if rgb_str:
                descriptors.append(rgb_str)
            if lab_str:
                descriptors.append(lab_str)

            if descriptors:
                entity_clauses.append(f"{category_title} {name} ({', '.join(descriptors)})")
            else:
                entity_clauses.append(f"{category_title} {name}")

        clauses_joined = "; ".join(entity_clauses)
        if extracted_lighting:
            if clauses_joined:
                clauses_joined += f"; Scene Lighting: {extracted_lighting}"
            else:
                clauses_joined = f"Scene Lighting: {extracted_lighting}"

        if not clauses_joined:
            return base_prompt

        constraint_clause = f"[Color Bible Constraints: {clauses_joined}]."
        trimmed_base = base_prompt.strip()
        if trimmed_base:
            return f"{trimmed_base} {constraint_clause}"
        return constraint_clause


def enrich_prompt(
    base_prompt: str,
    entities: list[dict[str, Any]],
    lighting_context: str = "",
) -> str:
    """Convenience function calling PromptEnricher.enrich_prompt."""
    return PromptEnricher.enrich_prompt(
        base_prompt=base_prompt,
        entities=entities,
        lighting_context=lighting_context,
    )


# ---------------------------------------------------------------------------
# Two-Stage Reference Router (TwoStageReferenceRouter)
# ---------------------------------------------------------------------------


class TwoStageReferenceRouter:
    """Routes reference colorization between fast deterministic drafts and guided neural models."""

    @staticmethod
    def select_backend(
        row: dict[str, str],
        has_color_bible_entities: bool,
        reference_runner: str = "auto",
        default_guided_backend: str = "qwen2511",
    ) -> str:
        """Select backend for a shot based on runner mode, row overrides, and Color Bible entities."""
        # 1. Per-row explicit override takes absolute precedence
        per_row_override = str(row.get("reference_backend", "") or "").strip()
        if per_row_override:
            return per_row_override

        runner_mode = str(reference_runner or "auto").strip().lower()

        # 2. Strict single-backend modes
        if runner_mode == "ddcolor_only":
            return "ddcolor"
        if runner_mode == "guided_only":
            return default_guided_backend

        # 3. Two-stage auto mode
        escalate_guided = str(row.get("escalate_guided", "") or "").strip().lower() in ("true", "1", "yes")
        escalate_qwen = str(row.get("escalate_qwen", "") or "").strip().lower() in ("true", "1", "yes")

        if has_color_bible_entities or escalate_guided or escalate_qwen:
            return default_guided_backend

        return "ddcolor"


def select_backend(
    row: dict[str, str],
    has_color_bible_entities: bool,
    reference_runner: str = "auto",
    default_guided_backend: str = "qwen2511",
) -> str:
    """Convenience function calling TwoStageReferenceRouter.select_backend."""
    return TwoStageReferenceRouter.select_backend(
        row=row,
        has_color_bible_entities=has_color_bible_entities,
        reference_runner=reference_runner,
        default_guided_backend=default_guided_backend,
    )


# ---------------------------------------------------------------------------
# Scoped Cache Key & Signature Builders
# ---------------------------------------------------------------------------


def compute_scoped_cache_key(
    shot_index: int,
    source_fingerprint: str | dict[str, Any] | Any,
    retrieved_ref_hashes: dict[str, str],
    prompt: str,
    backend_params: dict[str, Any],
) -> str:
    """Compute deterministic SHA-256 hash across shot-specific parameters ONLY.

    Crucially excludes global film-wide color_bible_version from the equality hash so
    updates to one character variant do not invalidate reference caches for other shots.
    """
    fp = source_fingerprint
    if isinstance(fp, dict):
        fp = fp.get("sha256") or fp.get("size") or str(fp.get("mtime_ns"))
    fp_str = str(fp)

    sorted_ref_hashes = {str(k): str(v) for k, v in sorted(retrieved_ref_hashes.items())}
    payload: dict[str, Any] = {
        "shot_index": int(shot_index),
        "source_fingerprint": fp_str,
        "retrieved_ref_hashes": sorted_ref_hashes,
        "prompt": str(prompt).strip(),
        "backend_params": backend_params,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_reference_signature(
    shot_index: int,
    source_fingerprint: str | dict[str, Any] | Any,
    retrieved_ref_hashes: dict[str, str],
    prompt: str,
    backend_params: dict[str, Any],
    color_bible_version: str | None = None,
    retrieved_ref_ids: list[str] | None = None,
    scoped_cache_key: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build full reference signature dictionary containing scoped cache key and provenance metadata."""
    fp = source_fingerprint
    if isinstance(fp, dict):
        fp = fp.get("sha256") or fp.get("size") or str(fp.get("mtime_ns"))
    fp_str = str(fp)

    if scoped_cache_key is None:
        scoped_cache_key = compute_scoped_cache_key(
            shot_index=shot_index,
            source_fingerprint=fp_str,
            retrieved_ref_hashes=retrieved_ref_hashes,
            prompt=prompt,
            backend_params=backend_params,
        )

    if retrieved_ref_ids is None:
        retrieved_ref_ids = sorted(list(retrieved_ref_hashes.keys()))

    sig: dict[str, Any] = {
        "scoped_cache_key": scoped_cache_key,
        "color_bible_version": str(color_bible_version or ""),
        "retrieved_ref_ids": list(retrieved_ref_ids),
        "retrieved_ref_hashes": {str(k): str(v) for k, v in sorted(retrieved_ref_hashes.items())},
        "shot_index": int(shot_index),
        "source_fingerprint": fp_str,
        "prompt": str(prompt).strip(),
        "backend_params": backend_params,
    }

    if extra_metadata:
        sig.update(extra_metadata)

    return sig


# ---------------------------------------------------------------------------
# Commercial Model Guard & Compliance Recording
# ---------------------------------------------------------------------------


def record_reference_compliance(
    output_path: Path | str,
    model_id: str,
    component_type: str,
    color_bible_version: str | None,
    retrieved_ref_ids: list[str],
    settings: dict | None = None,
    seed: int | None = None,
    execution_time_ms: float | None = None,
    fail_closed: bool = True,
    conditional_attestations: Mapping[str, Any] | None = None,
) -> Path:
    """Evaluate commercial model policy and record compliance provenance to output/compliance_report.json."""
    from .commercial_policy import CommercialPolicyError, evaluate_component
    from .compliance import append_compliance_record

    out_p = Path(output_path)
    if out_p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"} or out_p.is_file():
        out_dir = out_p.parent
    else:
        out_dir = out_p

    settings_map = settings if isinstance(settings, Mapping) else None

    attestations: dict[str, Any] = {}
    if conditional_attestations:
        attestations.update(conditional_attestations)
    elif settings_map:
        global_comm = settings_map.get("global", {}) if isinstance(settings_map.get("global"), Mapping) else {}
        if "qwen_2511_audited" in global_comm:
            attestations["qwen_2511_audited"] = global_comm["qwen_2511_audited"]
        else:
            attestations["qwen_2511_audited"] = True
    else:
        attestations["qwen_2511_audited"] = True

    decision = evaluate_component(
        component_id=model_id,
        component_type=component_type,
        settings=settings_map,  # type: ignore[arg-type]
        conditional_attestations=attestations,
    )

    if fail_closed and not decision.allowed:
        raise CommercialPolicyError(f"Commercial policy blocked model '{model_id}': {decision.reason}")

    model_record: dict[str, Any] = {
        "model_id": model_id,
        "component_type": component_type,
        "sha256": decision.sha256,
        "license": decision.license,
        "status": decision.status,
        "reason": decision.reason,
    }

    seeds = {"reference_seed": int(seed)} if seed is not None and seed >= 0 else None
    metadata: dict[str, Any] = {
        "retrieved_ref_ids": list(retrieved_ref_ids),
        "color_bible_version": str(color_bible_version or ""),
    }
    if execution_time_ms is not None:
        metadata["execution_time_ms"] = float(execution_time_ms)

    return append_compliance_record(
        output_dir=out_dir,
        stage="color_reference",
        models=[model_record],
        settings=settings,
        seeds=seeds,
        color_bible_version=color_bible_version,
        conditional_attestations=attestations,
        metadata=metadata,
    )
