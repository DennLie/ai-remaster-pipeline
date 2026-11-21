from __future__ import annotations

import json
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

from .config import IMAGE_EXTS, ROOT, VIDEO_EXTS
from . import state
from .dust_mask_edits import (
    compute_file_sha256,
    normalize_source_ref,
    verify_archive_pair,
)
from .manifests import read_manifest, read_outpaint_chunk_rows
from .paths import resolve, resolve_video_source, safe_stem
from .runtime_settings import default_settings, load_settings, normalize_settings, migrate_deflicker_settings
from .cloud import SECRET_SETTING_KEYS

PROJECT_SCHEMA_VERSION = 2
PROJECT_JSON_NAME = "project.json"

# Extensions allowed inside a project bundle. .json is included so outpaint guide sidecars
# (resume signatures, edit metadata) travel with the guides and aren't treated as stale on load.
BUNDLE_EXTS = IMAGE_EXTS | {".csv", ".txt", ".json"}
# Per-frame outpaint masks and sparse Dust edit bitmasks are stored as ZIPs under these
# manifest directories only. Zips are accepted exclusively from these prefixes so a project
# file cannot drop arbitrary archives elsewhere in the install.
FRAME_MASK_BUNDLE_DIR = "manifests/outpaint_masks/"
DUST_MASK_BUNDLE_DIR = "manifests/dust_masks/"


def bundle_name_allowed(name: str) -> bool:
    name = name.replace("\\", "/")
    if not name or name.startswith("/") or ".." in Path(name).parts:
        return False
    suffix = Path(name).suffix.lower()
    return suffix in BUNDLE_EXTS or (
        suffix == ".zip" and name.startswith((FRAME_MASK_BUNDLE_DIR, DUST_MASK_BUNDLE_DIR))
    )


def bind_context(context: dict) -> None:
    """Bind GUI helpers used to identify the active outpaint chunk manifest and dust edit stores."""
    globals().update({key: context[key] for key in (
        "outpaint_chunk_manifest_for", "pipeline_source_text",
        "outpaint_source_for_settings", "custom_outpaint_mask_for", "frame_outpaint_masks_for",
        "dust_mask_archives_for",
    ) if key in context})


def source_signature(source_text: str) -> tuple[str, int, int] | None:
    if not source_text:
        return None
    try:
        source = resolve_video_source(source_text)
        if not source.exists() or source.suffix.lower() not in VIDEO_EXTS:
            return None
        stat = source.stat()
        return str(source), stat.st_size, stat.st_mtime_ns
    except OSError:
        return None

def source_analysis_key(signature: tuple[str, int, int]) -> str:
    return "\0".join(str(part) for part in signature)

def project_payload(settings: dict[str, dict[str, str]]) -> dict:
    stored_settings = {
        stage: {key: value for key, value in values.items() if key not in SECRET_SETTING_KEYS}
        for stage, values in settings.items()
    }
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app": "AI Remaster Pipeline",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": stored_settings,
    }

def write_project_file(path: Path, settings: dict[str, dict[str, str]]) -> None:
    payload = project_payload(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PROJECT_JSON_NAME, json.dumps(payload, indent=2) + "\n")
        for asset in project_asset_paths(settings):
            archive.write(asset, asset.relative_to(ROOT).as_posix())

def read_project_file(path: Path) -> dict[str, dict[str, str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            try:
                data = json.loads(archive.read(PROJECT_JSON_NAME).decode("utf-8-sig"))
            except KeyError as exc:
                raise RuntimeError("Project bundle does not contain project.json.") from exc
            extract_project_assets(archive)
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Project file is not valid JSON: {exc}") from exc
    return load_project_payload(data)

def load_project_payload(data: dict) -> dict[str, dict[str, str]]:
    try:
        version = int(data.get("schema_version", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Project file has an invalid schema_version.") from exc
    if version < 1:
        raise RuntimeError("Project file does not include a supported schema_version.")
    if version > PROJECT_SCHEMA_VERSION:
        raise RuntimeError(f"Project schema {version} is newer than this ARP build supports.")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise RuntimeError("Project file does not contain settings.")
    settings = migrate_deflicker_settings(settings)
    # A project must hydrate from clean defaults, not from the settings of the project that is
    # currently open.  In particular, projects saved before image-sequence support do not contain
    # source_images/source_sequence_preview; inheriting those fields could make their conventional
    # video source get replaced by the previously-open sequence during restore_image_sequence_source.
    # Keep only machine-local secrets which are deliberately excluded from project bundles.
    current = load_settings()
    loaded = default_settings(include_newest_source=False)
    for stage, values in current.items():
        if stage not in loaded:
            continue
        for key in SECRET_SETTING_KEYS:
            if key in values:
                loaded[stage][key] = values[key]
    for stage, values in settings.items():
        if stage in loaded and isinstance(values, dict):
            loaded[stage].update({str(key): str(value) for key, value in values.items()})
    if not loaded.get("global", {}).get("source"):
        loaded.setdefault("global", {})["expand_outpaint"] = "true"
    loaded.setdefault("global", {}).setdefault("upscale", "false")
    return normalize_settings(loaded, include_newest_source=False)

def project_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    candidates: list[str] = []
    for key in ("outpaint", "shots", "references", "colour", "recomp"):
        manifest = settings.get(key, {}).get("manifest", "")
        if manifest:
            candidates.append(manifest)
    assets: list[Path] = []
    seen: set[Path] = set()
    for text in candidates:
        manifest = resolve(text)
        if not project_asset_is_bundleable(manifest) or manifest in seen:
            continue
        seen.add(manifest)
        assets.append(manifest)
        for row in read_manifest(manifest):
            reference_values = [row]
            try:
                extras = json.loads(row.get("additional_references", "") or "[]")
                if isinstance(extras, list):
                    reference_values.extend(item for item in extras if isinstance(item, dict))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            for reference in reference_values:
                for field in ("source_reference", "color_reference"):
                    image = resolve(str(reference.get(field, "") or ""))
                    if project_asset_is_bundleable(image) and image not in seen:
                        seen.add(image)
                        assets.append(image)
    # Outpaint chunk manifests and the hand-edited guide frames they reference, so a project
    # keeps the user's manual guide work for safekeeping.
    for asset in outpaint_guide_asset_paths(settings):
        if asset not in seen:
            seen.add(asset)
            assets.append(asset)
    outpaint_source = globals().get("outpaint_source_for_settings")
    for mask_for_name in ("custom_outpaint_mask_for", "frame_outpaint_masks_for"):
        mask_for = globals().get(mask_for_name)
        if not (outpaint_source and mask_for):
            continue
        try:
            mask = mask_for(outpaint_source(settings), settings.get("outpaint", {}))
            if mask not in seen and project_asset_is_bundleable(mask):
                seen.add(mask)
                assets.append(mask)
        except Exception:
            pass
    for asset in dust_mask_asset_paths(settings):
        if asset not in seen:
            seen.add(asset)
            assets.append(asset)
    return assets


def _candidate_dust_sources(settings: dict[str, dict[str, str]]) -> tuple[set[str], list[Path]]:
    raw_texts: list[str] = []
    global_values = settings.get("global", {}) if isinstance(settings, dict) else {}
    dust_values = settings.get("dust", {}) if isinstance(settings, dict) else {}
    for value in (
        global_values.get("source", ""),
        dust_values.get("source", ""),
        dust_values.get("stage_source", ""),
        dust_values.get("original_source", ""),
    ):
        if value:
            raw_texts.append(str(value))
    for fn_name in ("pipeline_source_text", "outpaint_source_for_settings"):
        fn = globals().get(fn_name)
        if fn:
            try:
                value = fn(settings)
                if value:
                    raw_texts.append(str(value))
            except Exception:
                pass

    norm_refs: set[str] = set()
    existing_files: list[Path] = []
    seen_files: set[Path] = set()
    for text in raw_texts:
        norm_refs.add(normalize_source_ref(text, root=ROOT))
        cand_path = Path(text)
        if not cand_path.is_absolute():
            local_path = ROOT / cand_path
            if local_path.is_file():
                resolved_local = local_path.resolve()
                norm_refs.add(normalize_source_ref(resolved_local, root=ROOT))
                if resolved_local not in seen_files:
                    seen_files.add(resolved_local)
                    existing_files.append(resolved_local)
        try:
            resolved = resolve_video_source(text)
            norm_refs.add(normalize_source_ref(resolved, root=ROOT))
            if resolved.is_file():
                resolved_canon = resolved.resolve()
                if resolved_canon not in seen_files:
                    seen_files.add(resolved_canon)
                    existing_files.append(resolved_canon)
        except OSError:
            pass
    norm_refs.discard("")
    return norm_refs, existing_files


def dust_mask_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    """Collect active project Dust & Scratches bitmask ZIP archives and verified .meta.json sidecars."""
    masks_dir = ROOT / "manifests" / "dust_masks"
    candidate_pairs: list[tuple[Path, Path]] = []
    seen_pairs: set[tuple[Path, Path]] = set()

    def _add_pair(zip_path: Path, meta_path: Path) -> None:
        key = (zip_path.resolve(strict=False), meta_path.resolve(strict=False))
        if key not in seen_pairs:
            seen_pairs.add(key)
            candidate_pairs.append((zip_path, meta_path))

    archives_for = globals().get("dust_mask_archives_for")
    if archives_for:
        try:
            for item in archives_for(settings) or ():
                p = Path(item) if Path(item).is_absolute() else ROOT / item
                if p.name.endswith(".meta.json"):
                    tid = p.name[: -len(".meta.json")]
                    _add_pair(p.with_name(f"{tid}.zip"), p)
                elif p.suffix.lower() == ".zip":
                    _add_pair(p, p.with_suffix(".meta.json"))
        except Exception:
            pass

    dust_values = settings.get("dust", {}) if isinstance(settings, dict) else {}
    for key in ("timeline_id", "edit_timeline_id", "stage_timeline_id", "original_timeline_id"):
        tid = str(dust_values.get(key, "") or "").strip()
        if tid and "/" not in tid and "\\" not in tid and ".." not in tid:
            _add_pair(masks_dir / f"{tid}.zip", masks_dir / f"{tid}.meta.json")

    if masks_dir.is_dir():
        norm_refs, existing_files = _candidate_dust_sources(settings)
        file_shas_by_size: dict[int, set[str]] = {}
        if norm_refs or existing_files:
            for meta_path in sorted(masks_dir.glob("*.meta.json")):
                if ".tmp" in meta_path.name or ".partial" in meta_path.name:
                    continue
                try:
                    raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(raw_meta, dict):
                    continue
                binding_data = raw_meta.get("timeline_binding")
                if not isinstance(binding_data, dict):
                    continue
                tid = meta_path.name[: -len(".meta.json")]
                zip_path = masks_dir / f"{tid}.zip"
                binding_src = normalize_source_ref(str(binding_data.get("source_path", "") or ""), root=ROOT)
                matched = bool(binding_src and binding_src in norm_refs)
                if not matched and existing_files:
                    binding_sha = str(binding_data.get("source_sha256", "") or "").strip().lower()
                    try:
                        binding_size = int(binding_data.get("source_size", -1))
                    except (TypeError, ValueError):
                        binding_size = -1
                    if binding_sha and binding_size >= 0:
                        if binding_size not in file_shas_by_size:
                            shas: set[str] = set()
                            for src_file in existing_files:
                                try:
                                    if src_file.stat().st_size == binding_size:
                                        shas.add(compute_file_sha256(src_file))
                                except OSError:
                                    continue
                            file_shas_by_size[binding_size] = shas
                        matched = binding_sha in file_shas_by_size[binding_size]
                if matched:
                    _add_pair(zip_path, meta_path)

    assets: list[Path] = []
    seen_assets: set[Path] = set()
    for zip_path, meta_path in candidate_pairs:
        if not zip_path.exists() and not meta_path.exists():
            continue
        state = verify_archive_pair(zip_path, meta_path)
        if not state.get("exists") or not state.get("edited_frames"):
            continue
        for asset in (zip_path, meta_path):
            if asset not in seen_assets and project_asset_is_bundleable(asset):
                seen_assets.add(asset)
                assets.append(asset)
    return assets

def chunk_guide_image_texts(manifest: Path) -> list[str]:
    """Guide image paths referenced by a chunk manifest's rows (the active guide per frame plus
    one undo step), covering both hand-edited guides (outpaint_guides/) and auto seed guides
    (outpaint_seed_guides/). Mirrors outpaint_guides._parse_guide_frames, including the legacy
    guide_image / guide_end_image fields, without importing that module."""
    texts: list[str] = []
    try:
        rows = read_outpaint_chunk_rows(manifest)
    except Exception:
        return texts
    for row in rows.values():
        frames: list = []
        raw = (row.get("guide_frames") or "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    frames = parsed
            except (json.JSONDecodeError, TypeError):
                frames = []
        for legacy in ("guide_image", "guide_end_image"):
            if row.get(legacy):
                frames.append({"image": row[legacy]})
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            for key in ("image", "image_previous"):
                value = frame.get(key)
                if value:
                    texts.append(str(value))
    return texts

def outpaint_guide_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    # The outpaint chunk manifest is user-authored project state: custom lengths, seeds, offsets,
    # guide strengths, and guide image references all live there. Prefer any explicit settings
    # pointer, then use the same GUI helper that names the active manifest so Save Project can
    # preserve guide work even when the cache itself has been cleared.
    manifests: list[Path] = []
    seen_manifests: set[Path] = set()
    explicit = settings.get("outpaint", {}).get("manifest", "")
    if explicit:
        manifest = resolve(explicit)
        if project_asset_is_bundleable(manifest):
            seen_manifests.add(manifest)
            manifests.append(manifest)

    chunk_manifest_for = globals().get("outpaint_chunk_manifest_for")
    # The plan is named after the video Outpainting consumes (the cleaned or stabilized one when
    # those stages are on), not the raw section clip.
    outpaint_source = globals().get("outpaint_source_for_settings")
    if chunk_manifest_for and outpaint_source:
        try:
            manifest_text = chunk_manifest_for(outpaint_source(settings), settings.get("outpaint", {}))
            manifest = resolve(manifest_text) if manifest_text else None
            if manifest and project_asset_is_bundleable(manifest) and manifest not in seen_manifests:
                seen_manifests.add(manifest)
                manifests.append(manifest)
        except Exception:
            pass

    assets: list[Path] = []
    seen_assets: set[Path] = set()
    for manifest in manifests:
        if manifest not in seen_assets:
            seen_assets.add(manifest)
            assets.append(manifest)
        for image_text in chunk_guide_image_texts(manifest):
            image = resolve(image_text)
            # Bundle the guide image and its resume/edit sidecars so reloaded guides are used
            # as-is and not treated as stale.
            for candidate in (image, Path(str(image) + ".sig.json"), Path(str(image) + ".json")):
                if candidate not in seen_assets and project_asset_is_bundleable(candidate):
                    seen_assets.add(candidate)
                    assets.append(candidate)
            seen_assets.add(image)
    return assets

def project_asset_is_bundleable(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_file():
            return False
    except OSError:
        return False
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return False
    return bundle_name_allowed(relative.as_posix())

def asset_already_on_disk(info: zipfile.ZipInfo, target: Path) -> bool:
    """True when the target file already holds the archived bytes. Restoring it anyway would
    only refresh its mtime, which used to invalidate resume signatures of dependent outputs
    (e.g. outpaint chunks re-rendering after every project load)."""
    try:
        if not target.is_file() or target.stat().st_size != info.file_size:
            return False
        return zlib.crc32(target.read_bytes()) == info.CRC
    except OSError:
        return False

def extract_project_assets(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name == PROJECT_JSON_NAME or name.startswith("/") or ".." in Path(name).parts:
            continue
        if not bundle_name_allowed(name):
            continue
        target = ROOT / name
        try:
            target.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        # Restore each asset independently so one failure (e.g. a Windows MAX_PATH limit on a
        # deeply nested guide-edit file) can't abort restoring the rest of the project.
        try:
            if asset_already_on_disk(info, target):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as dest:
                dest.write(source.read())
        except OSError as exc:
            print(f"Warning: could not restore project asset {name}: {exc}", flush=True)

def project_default_path(settings: dict[str, dict[str, str]]) -> Path:
    source = resolve_video_source(settings.get("global", {}).get("source", ""))
    stem = safe_stem(source.name if source.name else "arp_project")
    return ROOT / "projects" / f"{stem}.arpp"

def project_save_suggestion(settings: dict[str, dict[str, str]], project_path: Path | None = None) -> Path:
    if project_path:
        return project_path
    default_path = project_default_path(settings)
    last_dir = last_browse_dir(settings)
    return (last_dir / default_path.name) if last_dir else default_path

def last_browse_dir(settings: dict[str, dict[str, str]] | None = None) -> Path | None:
    values = settings or (state.APP.settings if state.APP is not None else {})
    text = values.get("global", {}).get("last_browse_dir", "") if isinstance(values, dict) else ""
    if not text:
        return None
    path = resolve(str(text))
    return path if path.exists() and path.is_dir() else None
