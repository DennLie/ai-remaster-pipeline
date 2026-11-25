from __future__ import annotations

import json
import mimetypes
import subprocess
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

from . import state
from .color_bible_context import canonical_shot_id, resolve_color_bible_context
from .frame_inspector import inspection as inspect_frame
from .frame_differences import cleanup_frame_differences
from .denoise_review import inspect_denoise, denoise_frame_differences
from .denoise_diff_video import diff_video_path
from .cache import delete_cache_category, delete_cache_file
from .config import STATIC_DIR
from .file_dialogs import browse_path, browse_source_paths
from .manifests import manifest_source_video, update_manifest_row
from .media import (
    aspect_preview_at_for_settings,
    aspect_preview_frame_for_settings,
    section_relative_seconds,
    auto_crop_for_settings,
    existing_browser_playback,
    export_media_file,
    media_clip_path,
    pipeline_source_text,
)
from .outpaint_guides import (
    _guide_source_seconds,
    accept_guide_edit,
    add_guide_frame,
    chunk_frame_preview,
    clear_guide_frame_image,
    remove_guide_frame,
    revert_guide_edit,
    sam_guide_mask,
    save_guide_frame,
    save_guide_paint,
    upload_guide_frame_image,
)
from .paths import is_unfinished_media_artifact, resolve, resolve_served
from .references import (
    accept_reference_edit,
    update_reference_guide_strength,
    update_reference_prompt,
    update_shot_ltx_strength,
    add_reference_frame,
    delete_color_reference,
    extract_reference_frame,
    install_custom_color_reference,
    merge_manifest_shots,
    preview_reference_frame,
    remove_additional_reference,
    revert_reference_edit,
    sam_reference_mask,
    save_reference_paint,
    split_manifest_shot,
    shot_rows_for_indices,
    update_shot_boundary,
    update_shot_cq_guides,
    update_shot_fade,
)
from .scrub_sheets import SHEETS_ROOT, sheet_status


def redact_command_arguments(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value in {"--api-key", "--openai-api-key"}:
            redacted[index + 1] = "[redacted]"
    return redacted

# These outpaint helpers stay in server.py because they are wired into its chunk/guide internals;
# importing server here would be circular (see state.py's note). server.py injects them at startup
# via bind_context. Declared here so the names resolve statically and tooling can see them.
ensure_outpaint_prepared_canvas = None
outpaint_chunks_state = None
outpaint_chunk_preview = None
update_outpaint_chunk = None
install_outpaint_guide = None
clear_outpaint_guide = None
install_outpaint_end_guide = None
clear_outpaint_end_guide = None
save_custom_outpaint_mask = None
clear_custom_outpaint_mask = None
frame_outpaint_mask_editor = None
frame_outpaint_mask_image = None
save_frame_outpaint_mask = None
clear_frame_outpaint_masks = None

_SERVER_OUTPAINT_OPS = (
    "ensure_outpaint_prepared_canvas",
    "outpaint_chunks_state",
    "outpaint_chunk_preview",
    "update_outpaint_chunk",
    "install_outpaint_guide",
    "clear_outpaint_guide",
    "install_outpaint_end_guide",
    "clear_outpaint_end_guide",
    "save_custom_outpaint_mask",
    "clear_custom_outpaint_mask",
    "frame_outpaint_mask_editor",
    "frame_outpaint_mask_image",
    "save_frame_outpaint_mask",
    "clear_frame_outpaint_masks",
)


def bind_context(context: dict) -> None:
    """Wire in the few server.py-defined outpaint helpers the request handlers call (see above)."""
    globals().update({name: context[name] for name in _SERVER_OUTPAINT_OPS})


# The GUI only ever serves the local user through their own browser. These are the host names a
# loopback address answers to; anything else in the Host header means a remote site is trying to
# reach us by name (a DNS-rebinding attack), so we refuse it.
LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def served_source_paths() -> list[str]:
    """The selected source video lives anywhere on disk, so its folder is added to the set of
    directories the file-serving endpoints are allowed to read from."""
    if state.APP is None or not hasattr(state.APP, "settings"):
        return []
    source = state.APP.settings.get("global", {}).get("source", "")
    return [source] if source else []


def _parse_color_bible_ranges(raw: object) -> list[tuple[int, int]]:
    """Parse shot ranges from a string ('12-18,410-435') or list of pairs/ints."""
    result: list[tuple[int, int]] = []
    if isinstance(raw, str):
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                s, e = part.split("-", 1)
                result.append((int(s.strip()), int(e.strip())))
            else:
                val = int(part.strip())
                result.append((val, val))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                result.append((int(item[0]), int(item[1])))
            elif isinstance(item, (int, str)) and str(item).strip().lstrip("+-").isdigit():
                val = int(str(item).strip())
                result.append((val, val))
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "AIRemasterGUI/1.0"

    def request_host_is_local(self) -> bool:
        """Reject requests whose Host header is not a loopback name. Blocks DNS rebinding, where a
        page on attacker.com re-points its own domain at 127.0.0.1 to talk to this server."""
        host = self.headers.get("Host", "")
        if not host:
            return True  # a missing Host cannot carry an attacker-controlled domain
        return (urlparse("//" + host).hostname or "").lower() in LOOPBACK_HOSTNAMES

    def request_origin_is_local(self) -> bool:
        """Reject state-changing requests carrying a cross-origin Origin/Referer. Blocks CSRF, where
        another site silently POSTs to our localhost endpoints from the user's browser."""
        for header in ("Origin", "Referer"):
            value = self.headers.get(header, "")
            if value:
                return (urlparse(value).hostname or "").lower() in LOOPBACK_HOSTNAMES
        return True  # same-origin requests may omit both; the Host check still applies

    def do_GET(self) -> None:  # noqa: N802
        if not self.request_host_is_local():
            self.send_error(403)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif parsed.path.startswith("/static/"):
            static_path = STATIC_DIR / unquote(parsed.path.removeprefix("/static/"))
            try:
                static_path.resolve().relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(404)
                return
            self.send_static(static_path)
        elif parsed.path == "/site.webmanifest":
            self.send_json(
                {
                    "name": "ARP - AI Remaster Pipeline",
                    "short_name": "ARP",
                    "start_url": "/",
                    "display": "standalone",
                    "background_color": "#101316",
                    "theme_color": "#2d8f7d",
                    "icons": [
                        {"src": "/media?path=assets/branding/arp-app-icon-192.png", "sizes": "192x192", "type": "image/png"},
                        {"src": "/media?path=assets/branding/arp-app-icon-512.png", "sizes": "512x512", "type": "image/png"},
                    ],
                }
            )
        elif parsed.path == "/api/frame-inspector":
            query = parse_qs(parsed.query)
            try:
                stage = query.get("stage", [""])[0]
                frame = int(query.get("frame", ["0"])[0])
                self.send_json({"ok": True, **inspect_frame(state.APP, stage, frame)})
            except (ValueError, IndexError, FileNotFoundError, RuntimeError, OSError,
                    subprocess.SubprocessError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/dust-mask-frame":
            query = parse_qs(parsed.query)

            def load_dust_mask() -> dict:
                raw_frame = query.get("frame", ["0"])[0]
                if not str(raw_frame).strip().lstrip("+-").isdigit():
                    raise ValueError(f"Invalid frame index: {raw_frame!r}")
                frame_idx = int(str(raw_frame).strip())
                return state.APP.load_dust_mask_frame(
                    frame_idx,
                    analysis_source=query.get("analysis_source", ["stage"])[0],
                    timeline_id=query.get("timeline_id", [None])[0],
                    project_expected=query.get("project", [None])[0],
                )

            self._send_dust_result("Dust mask frame", load_dust_mask)
        elif parsed.path == "/api/cleanup-frame-differences":
            # No client-selected filesystem paths or method overrides.
            try:
                self.send_json({"ok": True, **cleanup_frame_differences(state.APP)})
            except (ValueError, FileNotFoundError, RuntimeError, OSError,
                    subprocess.SubprocessError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/denoise-inspector":
            # Candidate is revalidated against the current source and the
            # completed denoiser's provenance; never serve arbitrary paths.
            query = parse_qs(parsed.query)
            try:
                candidate = query.get("variant", [""])[0]
                frame = int(query.get("frame", ["0"])[0])
                self.send_json({"ok": True, **inspect_denoise(state.APP, candidate, frame, query.get("stage", ["denoise"])[0], query.get('reference', ['input'])[0])})
            except (ValueError, IndexError, FileNotFoundError, RuntimeError, OSError,
                    subprocess.SubprocessError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/stabilization-footprint":
            from .stabilization import footprint
            query = parse_qs(parsed.query)
            try:
                self.send_json({'ok': True, **footprint(state.APP, query.get('variant', [''])[0], int(query.get('frame', ['0'])[0]))})
            except (ValueError, IndexError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                self.send_json({'ok': False, 'error': str(exc)})
        elif parsed.path == "/api/denoise-frame-differences":
            query = parse_qs(parsed.query)
            try:
                candidate = query.get("variant", [""])[0]
                self.send_json({"ok": True, **denoise_frame_differences(state.APP, candidate, query.get("stage", ["denoise"])[0])})
            except (ValueError, FileNotFoundError, RuntimeError, OSError,
                    subprocess.SubprocessError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/denoise-grid-download":
            # Zero path parameters: only the last validated grid for this active source.
            path = state.APP.denoise_grid_download()
            if path is None:
                self.send_error(404, "No completed grid for this source.")
            else:
                self.send_attachment(path)
        elif parsed.path == "/api/denoise-diff-video":
            token = parse_qs(parsed.query).get("token", [""])[0]
            path = diff_video_path(token)
            if path is None:
                self.send_error(404, "No rendered difference preview for this token.")
            else:
                self.send_media(path)
        elif parsed.path == "/api/denoise-playback-video":
            from .denoise_playback import segment_video_path
            path = segment_video_path(parse_qs(parsed.query).get("token", [""])[0])
            if path is None:
                self.send_error(404, "No comparison segment for this token.")
            else:
                self.send_media(path)
        elif parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            self.send_json(state.APP.state(query.get("active", [""])[0], query.get("retry_section", [""])[0] == "1"))
        elif parsed.path == "/api/command":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"command": redact_command_arguments(state.APP.command_for(stage)) if stage else []})
        elif parsed.path == "/api/existing-outputs":
            stage = parse_qs(parsed.query).get("stage", [""])[0]
            self.send_json({"paths": state.APP.existing_outputs(stage) if stage else []})
        elif parsed.path == "/api/existing-shot-list":
            self.send_json({"path": state.APP.existing_shot_list()})
        elif parsed.path == "/api/media-status":
            query = parse_qs(parsed.query)
            path_text = query.get("path", [""])[0]
            path = resolve_served(path_text, served_source_paths())
            exists = bool(path and path.is_file())
            self.send_json(
                {
                    "ok": True,
                    "path": path_text,
                    "exists": exists,
                    "mtime": path.stat().st_mtime if exists else 0,
                    "running": state.APP.process is not None and state.APP.process.poll() is None,
                    "running_stage": state.APP.running_stage,
                    "log_count": len(state.APP.log),
                }
            )
        elif parsed.path == "/api/comfy":
            url = parse_qs(parsed.query).get("url", ["http://127.0.0.1:8188"])[0].rstrip("/")
            # Only reach out over HTTP(S). urlopen also speaks file://, ftp://, etc., which would
            # turn this connectivity check into a way to read local files or probe other services.
            if urlparse(url).scheme not in ("http", "https"):
                self.send_json({"ok": False, "error": "ComfyUI URL must be http or https."})
                return
            try:
                with urlopen(url + "/queue", timeout=3) as response:
                    self.send_json({"ok": True, "queue": json.loads(response.read().decode("utf-8"))})
            except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/logfile":
            path = resolve_served(parse_qs(parsed.query).get("path", [""])[0], served_source_paths())
            text = path.read_text(encoding="utf-8", errors="replace")[-12000:] if path and path.exists() else ""
            self.send_json({"text": text})
        elif parsed.path == "/api/openai-models":
            token = state.APP.settings.get("references", {}).get("openai_api_key", "").strip()
            if not token:
                self.send_json({"ok": False, "error": "Add your OpenAI API key first."})
                return
            try:
                request = Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {token}"})
                with urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                models = sorted(
                    item.get("id", "")
                    for item in payload.get("data", [])
                    if isinstance(item, dict) and model_looks_image_capable(str(item.get("id", "")))
                )
                self.send_json({"ok": True, "models": models})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-preview":
            query = parse_qs(parsed.query)
            try:
                frame_text = query.get("frame", [""])[0]
                frame = int(frame_text) if str(frame_text).strip() != "" else None
                path = preview_reference_frame(query.get("manifest", [""])[0], int(query.get("index", ["0"])[0]), float(query.get("time", ["0"])[0]), frame=frame)
                self.send_json({"ok": True, "path": path})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/shot-sheets":
            try:
                source_text = manifest_source_video(resolve(parse_qs(parsed.query).get("manifest", [""])[0]))
                if not source_text or not resolve(source_text).is_file():
                    raise FileNotFoundError("The shot manifest's source video is missing.")
                self.send_json({"ok": True, **sheet_status(resolve(source_text))})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/upscale-shot-comparison":
            query = parse_qs(parsed.query)
            try:
                frames = state.APP.upscale_shot_comparison_frames(
                    int(query.get("index", ["0"])[0]),
                    float(query.get("time", ["0"])[0]),
                )
                self.send_json({"ok": True, **frames})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/aspect-preview":
            query = parse_qs(parsed.query)
            try:
                seconds = section_relative_seconds(state.APP.settings, float(query.get("time", ["0"])[0]))
                path, frame = aspect_preview_frame_for_settings(state.APP.settings, seconds)
                self.send_json({"ok": True, "path": path, "frame": frame})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-auto-crop":
            query = parse_qs(parsed.query)
            try:
                result = auto_crop_for_settings(state.APP.settings, float(query.get("time", ["0"])[0]))
                self.send_json({"ok": True, **result, "state": state.APP.state("outpaint")})
            except Exception as exc:
                state.APP.log.append(f"Auto Trim failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-custom-mask":
            self.send_json({"ok": True, **state.APP.state("outpaint").get("custom_outpaint_mask", {})})
        elif parsed.path == "/api/outpaint-frame-mask-frames":
            query = parse_qs(parsed.query)
            try:
                self.send_json({"ok": True, **frame_outpaint_mask_editor(int(query.get("first", ["0"])[0]), int(query.get("count", ["24"])[0]))})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-frame-mask":
            query = parse_qs(parsed.query)
            try:
                self.send_json({"ok": True, **frame_outpaint_mask_image(int(query.get("frame", ["0"])[0]))})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-chunk-preview":
            query = parse_qs(parsed.query)
            try:
                preview = outpaint_chunk_preview(
                    state.APP.settings,
                    int(query.get("chunk_index", ["0"])[0]),
                    query.get("kind", ["source"])[0],
                    query.get("position", ["middle"])[0],
                )
                self.send_json({"ok": True, "preview": preview})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/outpaint-guide-preview":
            query = parse_qs(parsed.query)
            try:
                chunk_index = int(query.get("chunk_index", ["0"])[0])
                frame_idx = int(query.get("frame_idx", ["0"])[0])
                source_text = state.APP.outpaint_source_for()
                if not source_text:
                    self.send_json({"ok": False, "error": "No source material"})
                else:
                    chunks_state = outpaint_chunks_state(state.APP.settings)
                    rows = chunks_state.get("rows", [])
                    row = next((r for r in rows if r.get("index") == chunk_index), None)
                    if row is None:
                        self.send_json({"ok": False, "error": "Chunk not found"})
                    else:
                        range_source = ensure_outpaint_prepared_canvas(source_text, state.APP.settings.get("outpaint", {}))
                        fps = float(row.get("fps", 24) or 24)
                        secs = _guide_source_seconds(row, frame_idx, fps)
                        cache_key = f"gfprev_{chunk_index}_{frame_idx}"
                        preview = chunk_frame_preview(range_source, secs, cache_key)
                        self.send_json({"ok": True, "preview": preview})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/color-bible/state":
            query = parse_qs(parsed.query)
            req_film = query.get("film_id", [""])[0].strip()
            req_db = query.get("color_bible_db", [""])[0].strip()
            app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
            cb_ctx = resolve_color_bible_context(
                app_settings,
                state.ROOT if hasattr(state, "ROOT") else Path.cwd(),
                stage="references",
                explicit_db=req_db,
                explicit_film_id=req_film,
            )
            film_id = cb_ctx["film_id"]
            if not film_id and app_settings:
                source = app_settings.get("global", {}).get("source", "")
                if source:
                    film_id = Path(source).stem
            try:
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    repo = service.repo
                    films = repo.list_films()
                    if not film_id and films:
                        film_id = films[0].film_id

                    characters = [c.to_dict() for c in repo.list_characters(film_id)] if film_id else []
                    appearance_variants = [v.to_dict() for v in repo.list_variants_for_film(film_id)] if film_id else []
                    objects = [o.to_dict() for o in repo.list_objects(film_id)] if film_id else []
                    locations = [l.to_dict() for l in repo.list_locations(film_id)] if film_id else []
                    assignments = [a.to_dict() for a in repo.list_assignments(film_id)] if film_id else []
                    references = [r.to_dict() for r in repo.list_references(film_id)] if film_id else []
                    shots = [s.to_dict() for s in repo.list_shots(film_id)] if film_id else []
                    snapshot_hash = service.get_snapshot_hash(film_id) if film_id else None

                    self.send_json({
                        "ok": True,
                        "film_id": film_id,
                        "db_path": cb_ctx["db_rel"],
                        "films": [f.to_dict() for f in films],
                        "characters": characters,
                        "appearance_variants": appearance_variants,
                        "objects": objects,
                        "locations": locations,
                        "assignments": assignments,
                        "references": references,
                        "shots": shots,
                        "snapshot_hash": snapshot_hash,
                    })
            except Exception as exc:
                state.APP.log.append(f"Color Bible state failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/color-bible/shot-assignments":
            query = parse_qs(parsed.query)
            try:
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(
                    app_settings,
                    state.ROOT if hasattr(state, "ROOT") else Path.cwd(),
                    stage="references",
                    explicit_db=query.get("color_bible_db", [""])[0].strip(),
                    explicit_film_id=query.get("film_id", [""])[0].strip(),
                    allow_auto_film=False,
                )
                film_id = cb_ctx["film_id"]
                raw_index = query.get("shot_index", [""])[0]
                if not str(raw_index).strip().lstrip("+-").isdigit():
                    raise ValueError(f"Invalid shot_index: {raw_index!r}")
                shot_index = int(str(raw_index).strip())

                raw_conf = query.get("confidence_threshold", [""])[0].strip()
                conf_threshold = float(raw_conf) if raw_conf else 0.8

                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    assignments = service.get_shot_palette_assignments(film_id, shot_index)
                    shot_record = service.get_shot_by_index(film_id, shot_index)
                    qc_status = shot_record.qc_status if shot_record else ""

                    needs_review = (qc_status == "needs_review")
                    review_reasons: list[str] = [f"Shot QC status is '{qc_status}'"] if needs_review else []
                    for item in assignments:
                        asgn = item.get("assignment", {})
                        status = str(asgn.get("approval_status", "")).strip().lower()
                        try:
                            conf = float(asgn.get("confidence", 1.0))
                        except (ValueError, TypeError):
                            conf = 1.0
                        entity_name = (item.get("entity") or {}).get("name") or item.get("entity_id") or "unknown"
                        variant_id = item.get("variant_id") or ""
                        label = f"Entity '{entity_name}'"
                        if variant_id:
                            label += f" costume '{variant_id}'"
                        if status == "pending_review":
                            needs_review = True
                            reason = f"{label} assignment status is 'pending_review'"
                            if reason not in review_reasons:
                                review_reasons.append(reason)
                        elif conf < conf_threshold:
                            needs_review = True
                            reason = f"{label} assignment has confidence {conf:.2f} < {conf_threshold:.2f}"
                            if reason not in review_reasons:
                                review_reasons.append(reason)

                    approval_status = qc_status if qc_status else ("pending_review" if needs_review else "approved")

                    self.send_json({
                        "ok": True,
                        "shot_index": shot_index,
                        "assignments": assignments,
                        "needs_review": needs_review,
                        "review_reasons": review_reasons,
                        "qc_status": qc_status,
                        "approval_status": approval_status,
                        "shot": shot_record.to_dict() if shot_record else None,
                    })
            except Exception as exc:
                state.APP.log.append(f"Color Bible shot assignments failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/color-bible/snapshot-hash":
            query = parse_qs(parsed.query)
            try:
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(
                    app_settings,
                    state.ROOT if hasattr(state, "ROOT") else Path.cwd(),
                    stage="references",
                    explicit_db=query.get("color_bible_db", [""])[0].strip(),
                    explicit_film_id=query.get("film_id", [""])[0].strip(),
                    allow_auto_film=False,
                )
                film_id = cb_ctx["film_id"]
                if not film_id:
                    raise ValueError("film_id is required.")
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    snapshot_hash = service.get_snapshot_hash(film_id)
                    self.send_json({
                        "ok": True,
                        "snapshot_hash": snapshot_hash,
                    })
            except Exception as exc:
                state.APP.log.append(f"Color Bible snapshot hash failed: {exc}")
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/color-bible/qc-summary":
            query = parse_qs(parsed.query)
            app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
            cb_ctx = resolve_color_bible_context(
                app_settings,
                state.ROOT if hasattr(state, "ROOT") else Path.cwd(),
                stage="colour",
                explicit_db=query.get("color_bible_db", [""])[0].strip(),
                explicit_film_id=query.get("film_id", [""])[0].strip(),
            )
            film_id = cb_ctx["film_id"]
            if not film_id and app_settings:
                source = app_settings.get("global", {}).get("source", "")
                if source:
                    film_id = Path(source).stem
            if not film_id:
                self.send_error(400, "film_id is required.")
                return

            def qc_summary_action() -> dict:
                from .color_bible_service import ColorBibleService

                shot_id = query.get("shot_id", [None])[0]
                if shot_id is not None:
                    shot_id = shot_id.strip() or None

                qc_kwargs: dict[str, Any] = {"shot_id": shot_id}
                limit_str = query.get("limit", [None])[0]
                if limit_str is not None:
                    try:
                        qc_kwargs["limit"] = int(limit_str)
                    except ValueError:
                        pass

                with ColorBibleService(cb_ctx["db_path"]) as service:
                    summary = service.get_qc_summary(film_id, **qc_kwargs)
                    return {"status": "ok", "summary": summary}

            self._send_result("Color Bible QC summary", qc_summary_action)
        elif parsed.path == "/api/segmenters":
            def list_segmenters_action() -> dict:
                from .commercial_policy import enabled as commercial_guard_enabled, evaluate_component
                from .image_segmenters import DEFAULT_IMAGE_SEGMENTER_REGISTRY
                from .video_segmenters import DEFAULT_SEGMENTER_REGISTRY

                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                guard_on = commercial_guard_enabled(app_settings)

                img_items = []
                for spec in DEFAULT_IMAGE_SEGMENTER_REGISTRY.list_specs():
                    dec = evaluate_component(spec.component_id, settings=app_settings, component_type="tool")
                    avail_info = DEFAULT_IMAGE_SEGMENTER_REGISTRY.availability_info(spec.name, settings=app_settings)
                    img_items.append({
                        "name": spec.name,
                        "label": spec.label,
                        "component_id": spec.component_id,
                        "category": spec.category,
                        "license": spec.license,
                        "commercial_tier": spec.commercial_tier,
                        "allowed": bool(dec.allowed) or (not guard_on),
                        "reason": str(dec.reason or ""),
                        "supports_text_prompts": spec.supports_text_prompts,
                        "supports_point_box_prompts": spec.supports_point_box_prompts,
                        "supports_target_classes": spec.supports_target_classes,
                        "available": avail_info["available"],
                        "runtime": avail_info["runtime"],
                        "env_name": avail_info["env_name"],
                        "env_status": avail_info["env_status"],
                    })

                vid_items = []
                for spec in DEFAULT_SEGMENTER_REGISTRY.list_specs():
                    decs = [
                        evaluate_component(cid, settings=app_settings, component_type="tool")
                        for cid in (spec.component_id, *spec.extra_component_ids)
                    ]
                    blocked = next((d for d in decs if not d.allowed), None)
                    avail_info = DEFAULT_SEGMENTER_REGISTRY.availability_info(spec.name, settings=app_settings)
                    vid_items.append({
                        "name": spec.name,
                        "label": spec.label,
                        "component_id": spec.component_id,
                        "license": spec.license,
                        "commercial_tier": spec.commercial_tier,
                        "allowed": (blocked is None) or (not guard_on),
                        "reason": str((blocked.reason if blocked else decs[0].reason) or ""),
                        "supports_text_prompts": spec.supports_text_prompts,
                        "available": avail_info["available"],
                        "runtime": avail_info["runtime"],
                        "env_name": avail_info["env_name"],
                        "env_status": avail_info["env_status"],
                    })

                return {
                    "commercial_guard_enabled": guard_on,
                    "image_segmenters": img_items,
                    "video_trackers": vid_items,
                }

            self._send_result("Segmenter registry query", list_segmenters_action)
        elif parsed.path == "/api/model-envs":
            def list_model_envs_action() -> dict:
                from .commercial_policy import enabled as commercial_guard_enabled
                from .managed_model_envs import inspect_all_envs, model_to_env_map

                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                return {
                    "commercial_guard_enabled": commercial_guard_enabled(app_settings),
                    "envs": inspect_all_envs(settings=app_settings),
                    "model_to_env": model_to_env_map(),
                }

            self._send_result("Model environments status", list_model_envs_action)
        elif parsed.path == "/media":
            query = parse_qs(parsed.query)
            path = resolve_served(unquote(query.get("path", [""])[0]), served_source_paths())
            if path is None or is_unfinished_media_artifact(path):
                # Do not attempt to read .partial.mkv while an encoder writes it.
                self.send_error(404)
                return
            if "clip_start" in query or "clip_end" in query:
                try:
                    start = float(query.get("clip_start", ["0"])[0])
                    end = float(query.get("clip_end", [str(start + 0.041)])[0])
                    path = media_clip_path(path, start, end, query.get("clip_key", [""])[0])
                except FileNotFoundError:
                    self.send_error(404)
                    return
                except Exception as exc:
                    state.APP.log.append(f"Shot video preview failed: {exc}")
                    self.send_error(404)
                    return
            else:
                # UI playback only: processing reads the original file from disk, never via /media.
                path = existing_browser_playback(path) or path
            self.send_media(path)
        else:
            self.send_error(404)

    def _send_result(self, label: str, produce) -> None:
        """Success path for endpoints answering {"ok": True, **<dict>}. produce() returns that
        dict; on any error, log "<label> failed: …" and answer {"ok": False, "error": …}."""
        try:
            self.send_json({"ok": True, **produce()})
        except Exception as exc:
            state.APP.log.append(f"{label} failed: {exc}")
            self.send_json({"ok": False, "error": str(exc)})

    def _send_dust_result(self, label: str, produce) -> None:
        """Dust endpoint wrapper returning {"ok": False, "error": ..., "error_type": ...} on failure."""
        try:
            self.send_json({"ok": True, **produce()})
        except Exception as exc:
            state.APP.log.append(f"{label} failed: {exc}")
            self.send_json({"ok": False, "error": str(exc), "error_type": type(exc).__name__})

    def _send_action(self, label: str, run, view: str = "", extra_key: str | None = None) -> None:
        """Success path for APP.run_* endpoints returning (ok, message) or (ok, message, value):
        answer {ok, message, state, error}, including fresh state only when ok and the optional
        third value under extra_key. On error, same log + error response as _send_result."""
        try:
            result = run()
            ok, message = result[0], result[1]
            payload = {"ok": ok, "message": message, "state": state.APP.state(view) if ok else None, "error": "" if ok else message}
            if extra_key is not None:
                payload[extra_key] = result[2]
            self.send_json(payload)
        except Exception as exc:
            state.APP.log.append(f"{label} failed: {exc}")
            self.send_json({"ok": False, "error": str(exc), "error_type": type(exc).__name__})

    def do_POST(self) -> None:  # noqa: N802
        if not self.request_host_is_local() or not self.request_origin_is_local():
            self.send_error(403)
            return
        parsed = urlparse(self.path)
        data = self.read_json()
        if parsed.path == "/api/settings":
            self._send_result('Settings', lambda: state.APP.update_settings(str(data.get('stage', '')), data.get('values', {}), project_expected=data.get('project')) or {})
        elif parsed.path == "/api/run":
            ok, message = state.APP.run_all(keep_shots=bool(data.get("keep_shots"))) if data.get("all") else state.APP.run_stage(str(data.get("stage", "")))
            self.send_json({"ok": ok, "message": message})
        elif parsed.path == "/api/denoise-batch":
            self._send_action("Remove Noise batch",
                              lambda: state.APP.run_denoise_batch(data.get("variants")), view="denoise")
        elif parsed.path == "/api/enhancement-batch":
            self._send_action('Detail Enhancement / Deblur batch',
                               lambda: state.APP.run_enhancement_batch(data.get('variants')), view='enhancement')
        elif parsed.path == "/api/dust-status":
            self._send_dust_result("Dust & Scratches status", lambda: state.APP.dust_status(data.get("values", {}), data.get("source"), data.get("project")))
        elif parsed.path == "/api/dust-inspect":
            self._send_dust_result("Dust & Scratches frame", lambda: state.APP.dust_inspect(data.get("values", {}), data.get("frame", 0), data.get("source"), data.get("project")))
        elif parsed.path == "/api/dust-run":
            self._send_action("Dust & Scratches", lambda: state.APP.run_dust(data.get("values", {}), data.get("kind"), data.get("source"), data.get("project")), view="dust")
        elif parsed.path == "/api/dust-adopt":
            self._send_dust_result("Dust & Scratches adoption", lambda: state.APP.adopt_dust(data.get("values", {}), data.get("source"), data.get("project"), data.get("report")))
        elif parsed.path == "/api/dust-mask-frame":
            def save_dust_mask() -> dict:
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object.")
                if any(k in data for k in ("path", "zip_path", "meta_path")):
                    raise ValueError("Client-supplied filesystem paths are not allowed.")
                result = state.APP.save_dust_mask_frame(
                    data.get("frame"),
                    data.get("edit_mask_b64"),
                    analysis_source=data.get("analysis_source", "stage"),
                    expected_revision=data.get("expected_revision"),
                    timeline_id=data.get("timeline_id"),
                    project_expected=data.get("project"),
                    source_expected=data.get("source"),
                )
                return {**result, "state": state.APP.state("dust")}

            self._send_dust_result("Dust mask save", save_dust_mask)
        elif parsed.path == "/api/dust-mask-reset":
            def reset_dust_mask() -> dict:
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object.")
                if any(k in data for k in ("path", "zip_path", "meta_path")):
                    raise ValueError("Client-supplied filesystem paths are not allowed.")
                result = state.APP.reset_dust_masks(
                    data.get("scope", ""),
                    frame=data.get("frame"),
                    analysis_source=data.get("analysis_source", "stage"),
                    expected_revision=data.get("expected_revision"),
                    timeline_id=data.get("timeline_id"),
                    project_expected=data.get("project"),
                    source_expected=data.get("source"),
                )
                return {**result, "state": state.APP.state("dust")}

            self._send_dust_result("Dust mask reset", reset_dust_mask)
        elif parsed.path == "/api/dust-mask-export":
            def export_dust_mask() -> dict:
                if not isinstance(data, dict):
                    raise ValueError("Request body must be a JSON object.")
                if any(k in data for k in ("export_dir", "output_dir", "path", "dir")):
                    raise ValueError("Client-supplied export paths are not allowed.")
                return state.APP.export_dust_provenance(
                    analysis_source=data.get("analysis_source"),
                    values=data.get("values"),
                    include_auto_proposals=data.get("include_auto_proposals", True),
                    frames=data.get("frames"),
                    project_expected=data.get("project"),
                )

            self._send_dust_result("Dust mask provenance export", export_dust_mask)
        elif parsed.path == "/api/stabilize-batch":
            self._send_action('Stabilization batch',
                              lambda: state.APP.run_stabilization_batch(data.get('variants')), view='stabilize')
        elif parsed.path == "/api/stabilize-adopt":
            self._send_result('Stabilization adoption',
                              lambda: state.APP.adopt_stabilization(data.get('method'), data.get('variant')))
        elif parsed.path == "/api/deflicker-batch":
            self._send_action("Light Deflicker batch",
                              lambda: state.APP.run_deflicker_batch(data.get("variants")), view="deflicker")
        elif parsed.path == "/api/denoise-playback":
            from .denoise_playback import playback_segment
            self._send_result("Remove Noise playback", lambda: playback_segment(
                state.APP, data.get("variant", ""), data.get("frame", 0),
                data.get("mode", "direction"), data.get("contrast", 32), data.get("stage", "denoise"), data.get('reference', 'input'),
                **({'mask_values': data['mask_values']} if 'mask_values' in data else {})))
        elif parsed.path == "/api/denoise-grid":
            self._send_action("Remove Noise grid",
                              lambda: state.APP.run_denoise_grid(str(data.get("frames", ""))), view="denoise")
        elif parsed.path == "/api/denoise-diff":
            try:
                result = state.APP.denoise_diff_preview(
                    str(data.get("first", "")),
                    str(data.get("second", "")),
                    float(data.get("start", 0)),
                    float(data.get("end", 0)),
                    str(data.get("mode", "direction")),
                    data.get("contrast", 32),
                )
                self.send_json({"ok": True, **result})
            except (ValueError, FileNotFoundError, RuntimeError, OSError,
                    subprocess.SubprocessError) as exc:
                self.send_json({"ok": False, "error": str(exc)})
        elif parsed.path == "/api/upscale-preview":
            ok, message = state.APP.run_upscale_preview()
            self.send_json({"ok": ok, "message": message})
        elif parsed.path == "/api/stop":
            state.APP.stop()
            self.send_json({"ok": True})
        elif parsed.path == "/api/quit":
            # Kill any running stage, answer, then stop the server so the launching shell
            # returns to its prompt with nothing left running. Answer before shutdown so
            # this response still reaches the browser. request_quit is imported here rather
            # than at module load because lifecycle imports http_handler (a cycle).
            from .lifecycle import request_quit

            state.APP.stop_for_quit()
            self.send_json({"ok": True})
            request_quit(self.server)
        elif parsed.path == "/api/shot-scrub":
            self._send_result("Shot scrub", lambda: {
                **extract_reference_frame(
                    str(data.get("manifest", "")),
                    int(data.get("index", 0)),
                    float(data.get("time", 0)),
                    int(data.get("frame")) if data.get("frame") is not None and str(data.get("frame")).strip() != "" else None,
                    int(data.get("reference_index", 0)),
                ),
                "state": state.APP.state("references"),
            })
        elif parsed.path == "/api/shot-prompt":
            self._send_result("Reference prompt save", lambda: update_reference_prompt(
                str(data.get("manifest", "")), int(data.get("index", 0)), int(data.get("reference_index", 0)), str(data.get("prompt", "")),
            ))
        elif parsed.path == "/api/shot-enabled":
            self._send_result("Shot enabled save", lambda: update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"enabled": "true" if data.get("enabled") else "false"}) or {})
        elif parsed.path == "/api/shot-merge":
            self._send_result("Shot merge", lambda: {**merge_manifest_shots(str(data.get("manifest", "")), int(data.get("index", 0))), "state": state.APP.state("shots")})
        elif parsed.path == "/api/shot-split":
            self._send_result("Shot split", lambda: {**split_manifest_shot(str(data.get("manifest", "")), int(data.get("index", 0))), "state": state.APP.state("shots")})
        elif parsed.path == "/api/shot-boundary":
            def update_boundary_result():
                manifest = str(data.get("manifest", ""))
                index = int(data.get("index", 0))
                edge = str(data.get("edge", ""))
                result = update_shot_boundary(
                    manifest,
                    index,
                    edge,
                    float(data.get("time", 0)),
                    int(data.get("frame")) if data.get("frame") is not None and str(data.get("frame")).strip() != "" else None,
                )
                return {
                    **result,
                    "rows": shot_rows_for_indices(manifest, (index - 2, index - 1, index, index + 1)),
                    "log": "\n".join(state.APP.log[-800:]),
                    "log_count": len(state.APP.log),
                }
            self._send_result("Shot boundary update", update_boundary_result)
        elif parsed.path == "/api/shot-fade":
            self._send_result("Shot fade update", lambda: {**update_shot_fade(str(data.get("manifest", "")), int(data.get("index", 0)), bool(data.get("enabled")), str(data.get("crossfade_seconds", ""))), "state": state.APP.state("shots")})
        elif parsed.path == "/api/shot-upscale-strength":
            def update_upscale_strength():
                manifest = str(data.get("manifest", ""))
                index = int(data.get("index", 0))
                raw = str(data.get("strength", "")).strip()
                if raw:
                    strength = max(0.0, min(100.0, float(raw)))
                    raw = f"{strength:.3f}".rstrip("0").rstrip(".")
                update_manifest_row(resolve(manifest), index, {"upscale_strength": raw})
                return {"strength": raw, "state": state.APP.state("upscale")}
            self._send_result("Shot upscale strength update", update_upscale_strength)
        elif parsed.path == "/api/shot-ltx-strength":
            self._send_result("Shot LTX strength update", lambda: {
                **update_shot_ltx_strength(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("key", "")), str(data.get("value", ""))),
                "state": state.APP.state("colour"),
            })
        elif parsed.path == "/api/reference-ltx-strength":
            self._send_result("Reference LTX strength update", lambda: {
                **update_reference_guide_strength(str(data.get("manifest", "")), int(data.get("index", 0)), int(data.get("reference_index", 0)), str(data.get("value", ""))),
                "state": state.APP.state("colour"),
            })
        elif parsed.path == "/api/shot-cq-luma-only":
            def update_cq_luma_only():
                # "" inherits the CQ colour default; "true"/"false" is this shot's own choice.
                raw = str(data.get("luma_only", "")).strip().lower()
                raw = raw if raw in {"true", "false"} else ""
                update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"cq_luma_only": raw})
                return {"luma_only": raw, "state": state.APP.state("upscale")}
            self._send_result("Shot CQ colour update", update_cq_luma_only)
        elif parsed.path == "/api/shot-cq-skip":
            def update_cq_skip():
                skip = "true" if str(data.get("skip", "")).strip().lower() == "true" else ""
                update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"cq_skip": skip})
                return {"skip": skip, "state": state.APP.state("upscale")}
            self._send_result("Shot CQ skip update", update_cq_skip)
        elif parsed.path == "/api/shot-cq-guides":
            def update_cq_guides():
                numbers = update_shot_cq_guides(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("shots", "")))
                return {"guide_shots": numbers, "state": state.APP.state("upscale")}
            self._send_result("Shot CQ guide shots update", update_cq_guides)
        elif parsed.path == "/api/shot-cq-anchor":
            def update_cq_anchor():
                # "" or 1 anchors the shot on its first chunk, as before.
                raw = str(data.get("chunk", "")).strip()
                raw = raw if raw.isdigit() and int(raw) > 1 else ""
                update_manifest_row(resolve(str(data.get("manifest", ""))), int(data.get("index", 0)), {"cq_anchor_chunk": raw})
                return {"chunk": raw, "state": state.APP.state("upscale")}
            self._send_result("Shot CQ anchor chunk update", update_cq_anchor)
        elif parsed.path == "/api/reference-regenerate":
            self._send_action("Reference regeneration", lambda: state.APP.run_reference_regeneration(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("provider", "qwen")), int(data.get("reference_index", 0))))
        elif parsed.path == "/api/reference-add":
            self._send_result("Reference add", lambda: {**add_reference_frame(
                str(data.get("manifest", "")),
                int(data.get("index", 0)),
                float(data.get("time", 0)),
                int(data.get("frame")) if data.get("frame") is not None and str(data.get("frame")).strip() != "" else None,
            ), "state": state.APP.state("references")})
        elif parsed.path == "/api/reference-remove":
            self._send_result("Reference remove", lambda: {**remove_additional_reference(str(data.get("manifest", "")), int(data.get("index", 0)), int(data.get("reference_index", 0))), "state": state.APP.state("references")})
        elif parsed.path == "/api/reference-delete":
            self._send_result("Reference delete", lambda: {**delete_color_reference(str(data.get("manifest", "")), int(data.get("index", 0)), int(data.get("reference_index", 0))), "state": state.APP.state()})
        elif parsed.path == "/api/reference-custom":
            self._send_result("Custom reference install", lambda: {**install_custom_color_reference(str(data.get("manifest", "")), int(data.get("index", 0)), int(data.get("reference_index", 0))), "state": state.APP.state()})
        elif parsed.path == "/api/reference-mask-sam":
            self._send_result("Reference smart mask", lambda: {**sam_reference_mask(
                str(data.get("manifest", "")),
                int(data.get("index", 0)),
                data.get("points", []) if isinstance(data.get("points", []), list) else [],
                int(data.get("width", 1)),
                int(data.get("height", 1)),
                int(data.get("tolerance", 10)),
                int(data.get("reference_index", 0)),
            )})
        elif parsed.path == "/api/guide-frame-mask-sam":
            self._send_result("Guide smart mask", lambda: {**sam_guide_mask(
                int(data.get("chunk_index", 0)),
                int(data.get("guide_index", 0)),
                data.get("points", []) if isinstance(data.get("points", []), list) else [],
                int(data.get("width", 1)),
                int(data.get("height", 1)),
                str(data.get("fallback_path", "")),
            )})
        elif parsed.path == "/api/reference-edit-preview":
            self._send_action("Reference edit preview", lambda: state.APP.run_reference_edit_preview(
                str(data.get("manifest", "")),
                int(data.get("index", 0)),
                str(data.get("instruction", "")),
                str(data.get("mask", "")),
                str(data.get("sampled_color", "")),
                int(data.get("reference_index", 0)),
            ), view="references", extra_key="preview")
        elif parsed.path == "/api/reference-edit-accept":
            self._send_result("Reference edit accept", lambda: {**accept_reference_edit(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("preview", "")), int(data.get("reference_index", 0))), "state": state.APP.state("references")})
        elif parsed.path == "/api/reference-edit-revert":
            self._send_result("Reference edit revert", lambda: {**revert_reference_edit(str(data.get("manifest", "")), int(data.get("index", 0)), int(data.get("reference_index", 0))), "state": state.APP.state("references")})
        elif parsed.path == "/api/reference-paint-save":
            self._send_result("Reference paint save", lambda: {**save_reference_paint(str(data.get("manifest", "")), int(data.get("index", 0)), str(data.get("image", "")), int(data.get("reference_index", 0))), "state": state.APP.state("references")})
        elif parsed.path == "/api/export-media":
            self._send_result("Media export", lambda: {**export_media_file(str(data.get("path", "")))})
        elif parsed.path == "/api/outpaint-chunk":
            self._send_result("Outpaint chunk save", lambda: update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")), str(data.get("guide_strength", "")), str(data.get("guide_end_strength", "")), data.get("custom_length", None), str(data.get("offset_x", "0")), str(data.get("offset_y", "0")), data.get("auto_start_guide", True), data.get("offset_override", None)) or {"state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-custom-mask-save":
            self._send_result("Custom outpaint mask save", lambda: {**save_custom_outpaint_mask(str(data.get("image", ""))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-custom-mask-clear":
            self._send_result("Custom outpaint mask clear", lambda: {**clear_custom_outpaint_mask(), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-frame-mask-save":
            self._send_result("Frame mask save", lambda: {"frame_outpaint_masks": save_frame_outpaint_mask(int(data.get("frame", -1)), str(data.get("image", "") or ""))})
        elif parsed.path == "/api/outpaint-frame-mask-clear":
            self._send_result("Frame masks clear", lambda: {"frame_outpaint_masks": clear_frame_outpaint_masks(), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-chunk-regenerate":
            self._send_action("Outpaint chunk regeneration", lambda: (
                update_outpaint_chunk(int(data.get("index", 0)), str(data.get("seed", "")), str(data.get("prompt_suffix", "")), str(data.get("custom_seconds", "")), str(data.get("negative_suffix", "")), str(data.get("guide_strength", "")), str(data.get("guide_end_strength", "")), data.get("custom_length", None), str(data.get("offset_x", "0")), str(data.get("offset_y", "0")), data.get("auto_start_guide", True), data.get("offset_override", None)),
                state.APP.run_outpaint_chunk(int(data.get("index", 0))),
            )[1], view="outpaint")
        elif parsed.path == "/api/outpaint-anchor":
            self._send_result("Outpaint guide install", lambda: {**install_outpaint_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-anchor-clear":
            self._send_result("Outpaint guide clear", lambda: {**clear_outpaint_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-anchor-generate":
            self._send_action("Outpaint guide generation", lambda: state.APP.run_outpaint_guide_generation(int(data.get("index", 0)), str(data.get("prompt", ""))), view="outpaint")
        elif parsed.path == "/api/outpaint-end-anchor":
            self._send_result("Outpaint end guide install", lambda: {**install_outpaint_end_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-end-anchor-clear":
            self._send_result("Outpaint end guide clear", lambda: {**clear_outpaint_end_guide(int(data.get("index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/outpaint-end-anchor-generate":
            self._send_action("Outpaint end guide generation", lambda: state.APP.run_outpaint_end_guide_generation(int(data.get("index", 0)), str(data.get("prompt", ""))), view="outpaint")
        elif parsed.path == "/api/guide-frame-add":
            self._send_result("Guide frame add", lambda: {**add_guide_frame(int(data.get("chunk_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-remove":
            self._send_result("Guide frame remove", lambda: {**remove_guide_frame(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-save":
            self._send_result("Guide frame save", lambda: {**save_guide_frame(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), int(data.get("frame_idx", 0)), float(data.get("strength", 0.7))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-upload":
            self._send_result("Guide frame upload", lambda: {**upload_guide_frame_image(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-clear":
            self._send_result("Guide frame clear", lambda: {**clear_guide_frame_image(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-generate":
            self._send_action("Guide frame generation", lambda: state.APP.run_guide_frame_generation(
                int(data.get("chunk_index", 0)),
                int(data.get("guide_index", 0)),
                int(data.get("frame_idx", 0)),
                str(data.get("prompt", "")),
            ), view="outpaint")
        elif parsed.path == "/api/guide-frame-edit-preview":
            self._send_action("Guide frame edit preview", lambda: state.APP.run_guide_edit_preview(
                int(data.get("chunk_index", 0)),
                int(data.get("guide_index", 0)),
                str(data.get("instruction", "")),
                str(data.get("mask", "")),
                str(data.get("sampled_color", "")),
                str(data.get("engine", "")),
            ), view="outpaint", extra_key="preview")
        elif parsed.path == "/api/guide-frame-edit-accept":
            self._send_result("Guide frame edit accept", lambda: {**accept_guide_edit(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), str(data.get("preview", ""))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-frame-edit-revert":
            self._send_result("Guide frame edit revert", lambda: {**revert_guide_edit(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/guide-paint-save":
            self._send_result("Guide paint save", lambda: {**save_guide_paint(int(data.get("chunk_index", 0)), int(data.get("guide_index", 0)), str(data.get("image", ""))), "state": state.APP.state("outpaint")})
        elif parsed.path == "/api/browse":
            self._send_result("Browse", lambda: {"path": browse_path(str(data.get("kind", "file")), str(data.get("current", "")))})
        elif parsed.path == "/api/browse-global-source":
            def browse_global_source() -> dict:
                selected = browse_source_paths(str(data.get("current", "")))
                result = state.APP.select_source_paths(selected, str(data.get("fps", "24")))
                return {**result, "state": state.APP.state("global")}

            self._send_result("Browse", browse_global_source)
        elif parsed.path == "/api/source-sequence-fps":
            self._send_result(
                "Image-sequence frame rate",
                lambda: {**state.APP.update_source_sequence_fps(str(data.get("fps", "24"))), "state": state.APP.state("global")},
            )
        elif parsed.path == "/api/overview-clear":
            state.APP.clear_overview()
            self.send_json({"ok": True, "state": state.APP.state("global")})
        elif parsed.path == "/api/project-save":
            self._send_result("Project save", lambda: {**state.APP.save_project(bool(data.get("save_as"))), "state": state.APP.state("global")})
        elif parsed.path == "/api/project-load":
            self._send_result("Project load", lambda: {**state.APP.load_project(), "state": state.APP.state("global")})
        elif parsed.path == "/api/cache-delete":
            def cache_delete() -> dict:
                if data.get("all"):
                    result = delete_cache_category("all")
                elif data.get("category"):
                    result = delete_cache_category(str(data.get("category", "")))
                else:
                    result = delete_cache_file(str(data.get("path", "")))
                return {**result, "state": state.APP.state()}

            self._send_result("Cache delete", cache_delete)
        elif parsed.path == "/api/color-bible/sync-manifest":
            manifest_str = str(data.get("manifest", "")).strip()
            resolved_manifest = resolve_served(manifest_str, served_source_paths()) if manifest_str else None
            if not resolved_manifest or not resolved_manifest.is_file():
                self.send_error(400, "Invalid manifest path or path traversal detected.")
                return

            def sync_manifest_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(
                    app_settings, ROOT, stage="references",
                    explicit_db=str(data.get("color_bible_db", "")),
                    explicit_film_id=str(data.get("film_id", "")),
                )
                film_id = cb_ctx["film_id"]
                title = str(data.get("title", "")).strip() or None
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    result = service.sync_manifest(film_id, resolved_manifest, title=title)
                if film_id and state.APP is not None and hasattr(state.APP, "update_settings"):
                    state.APP.update_settings("global", {"color_bible_film_id": film_id, "color_bible_db": cb_ctx["db_raw"]})
                return {"shots_synced": result.get("shots_synced", 0), "film_id": film_id}

            self._send_result("Color Bible sync manifest", sync_manifest_action)
        elif parsed.path == "/api/color-bible/select-film":
            def select_film_action():
                from .paths import ROOT
                film_id = str(data.get("film_id", "")).strip()
                explicit_db = str(data.get("color_bible_db", "")).strip()
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=explicit_db, explicit_film_id=film_id)
                updates = {"color_bible_film_id": film_id}
                if explicit_db:
                    updates["color_bible_db"] = cb_ctx["db_raw"]
                if state.APP is not None and hasattr(state.APP, "update_settings"):
                    state.APP.update_settings("global", updates)
                return {"film_id": film_id, "db_path": cb_ctx["db_rel"]}

            self._send_result("Color Bible select film", select_film_action)
        elif parsed.path == "/api/color-bible/film":
            def create_film_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                raw_film_id = str(data.get("film_id", "")).strip()
                if not raw_film_id:
                    raise ValueError("film_id must not be empty.")
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")), explicit_film_id=raw_film_id, allow_auto_film=False)
                film_id = cb_ctx["film_id"]
                title = str(data.get("title", "")).strip() or None
                source_video = str(data.get("source_video", "")).strip() or None
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    film = service.create_film(film_id=film_id, title=title, source_video=source_video)
                if film.film_id and state.APP is not None and hasattr(state.APP, "update_settings"):
                    state.APP.update_settings("global", {"color_bible_film_id": film.film_id, "color_bible_db": cb_ctx["db_raw"]})
                return {"film": film.to_dict()}

            self._send_result("Color Bible film", create_film_action)
        elif parsed.path == "/api/color-bible/character":
            def create_char_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")), explicit_film_id=str(data.get("film_id", "")))
                name = str(data.get("name", "")).strip()
                notes = str(data.get("notes", "")).strip()
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    char = service.create_character(film_id=cb_ctx["film_id"], name=name, notes=notes)
                    return {"character": char.to_dict()}

            self._send_result("Color Bible character", create_char_action)
        elif parsed.path == "/api/color-bible/variant":
            def create_var_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")))
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    var = service.create_appearance_variant(
                        character_id=str(data.get("character_id", "")).strip(),
                        costume_id=str(data.get("costume_id", "")).strip(),
                        hair_profile=str(data.get("hair_profile", "")).strip(),
                        skin_reference=str(data.get("skin_reference", "")).strip(),
                        lighting_context=str(data.get("lighting_context", "")).strip(),
                        notes=str(data.get("notes", "")).strip(),
                    )
                    return {"variant": var.to_dict()}

            self._send_result("Color Bible variant", create_var_action)
        elif parsed.path == "/api/color-bible/object":
            def create_obj_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")), explicit_film_id=str(data.get("film_id", "")))
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    obj = service.create_object(
                        film_id=cb_ctx["film_id"],
                        kind=str(data.get("kind", "")).strip(),
                        name=str(data.get("name", "")).strip(),
                        palette=data.get("palette"),
                        notes=str(data.get("notes", "")).strip(),
                    )
                    return {"object": obj.to_dict()}

            self._send_result("Color Bible object", create_obj_action)
        elif parsed.path == "/api/color-bible/location":
            def create_loc_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")), explicit_film_id=str(data.get("film_id", "")))
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    loc = service.create_location(
                        film_id=cb_ctx["film_id"],
                        set_name=str(data.get("set_name", "")).strip(),
                        lighting_context=str(data.get("lighting_context", "")).strip(),
                        palette=data.get("palette"),
                        notes=str(data.get("notes", "")).strip(),
                    )
                    return {"location": loc.to_dict()}

            self._send_result("Color Bible location", create_loc_action)
        elif parsed.path == "/api/color-bible/assign-ranges":
            def assign_ranges_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")), explicit_film_id=str(data.get("film_id", "")))
                film_id = cb_ctx["film_id"]
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    if data.get("action") == "delete" or "delete_assignment_id" in data:
                        asgn_id = str(data.get("delete_assignment_id") or data.get("assignment_id", "")).strip()
                        if asgn_id:
                            service.delete_assignment(asgn_id)
                            return {"assignments": [a.to_dict() for a in service.list_assignments(film_id)]}

                    entity_type = str(data.get("entity_type", "")).strip()
                    entity_id = str(data.get("entity_id", "")).strip()
                    variant_id = str(data.get("variant_id", "")).strip() or None
                    parsed_ranges = _parse_color_bible_ranges(data.get("ranges", []))
                    confidence = float(data.get("confidence", 1.0))
                    approval_status = str(data.get("approval_status", "approved")).strip()
                    provenance = str(data.get("provenance", "manual")).strip()

                    assignments = service.assign_entity_ranges(
                        film_id=film_id,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        ranges=parsed_ranges,
                        variant_id=variant_id,
                        confidence=confidence,
                        approval_status=approval_status,
                        provenance=provenance,
                    )
                    return {"assignments": [a.to_dict() for a in assignments]}

            self._send_result("Color Bible assign ranges", assign_ranges_action)
        elif parsed.path == "/api/color-bible/reference":
            keyframe_str = str(data.get("keyframe_path", "")).strip()
            resolved_kf = resolve_served(keyframe_str, served_source_paths()) if keyframe_str else None
            if not resolved_kf or not resolved_kf.is_file():
                self.send_error(400, "Invalid keyframe path or path traversal detected.")
                return

            mask_str = data.get("mask_uri")
            resolved_mask = None
            if mask_str is not None and str(mask_str).strip() != "":
                resolved_mask = resolve_served(str(mask_str).strip(), served_source_paths())
                if not resolved_mask or not resolved_mask.is_file():
                    self.send_error(400, "Invalid mask path or path traversal detected.")
                    return

            def register_ref_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")), explicit_film_id=str(data.get("film_id", "")))
                film_id = cb_ctx["film_id"]
                entity_type = str(data.get("entity_type", "")).strip()
                entity_id = str(data.get("entity_id", "")).strip()
                variant_id = str(data.get("variant_id", "")).strip() or None
                provenance = str(data.get("provenance", "manual")).strip()
                source_shot_id = str(data.get("source_shot_id", "")).strip() or None
                auto_approve = bool(data.get("auto_approve", False))
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    ref = service.register_reference_image(
                        film_id=film_id,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        keyframe_path=resolved_kf,
                        mask_uri=resolved_mask,
                        variant_id=variant_id,
                        provenance=provenance,
                        source_shot_id=source_shot_id,
                        auto_approve=auto_approve,
                    )
                    return {"reference": ref.to_dict()}

            self._send_result("Color Bible reference", register_ref_action)
        elif parsed.path == "/api/color-bible/reference-approval":
            def ref_approval_action():
                from .paths import ROOT
                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                cb_ctx = resolve_color_bible_context(app_settings, ROOT, stage="references", explicit_db=str(data.get("color_bible_db", "")))
                ref_id = str(data.get("ref_id", "")).strip()
                action = str(data.get("action", "")).strip().lower()
                approved_by = str(data.get("approved_by", "colorist")).strip()
                from .color_bible_service import ColorBibleService
                with ColorBibleService(db_path=cb_ctx["db_path"]) as service:
                    if action == "approve":
                        ref = service.approve_reference(ref_id, approved_by=approved_by)
                    elif action == "reject":
                        ref = service.reject_reference(ref_id)
                    else:
                        raise ValueError(f"Invalid approval action: '{action}'. Must be 'approve' or 'reject'.")
                    return {"reference": ref.to_dict()}

            self._send_result("Color Bible reference approval", ref_approval_action)
        elif parsed.path == "/api/free-vram":
            force = bool(data.get("force", False))
            if state.APP.process and state.APP.process.poll() is None and not force:
                self.send_json({
                    "ok": False,
                    "error": "A stage is currently running. Stop it first before freeing VRAM.",
                })
                return
            result = state.APP.free_vram(
                interrupt_comfy=bool(data.get("interrupt_comfy", force)),
                include_ollama=bool(data.get("include_ollama", True)),
            )
            view_name = str(data.get("view") or "")
            self.send_json({"ok": True, **result, "state": state.APP.state(view_name)})
        elif parsed.path == "/api/color-bible/run-qc":
            ok, msg = state.APP.run_color_qc(
                film_id=str(data.get("film_id") or ""),
                manifest=str(data.get("manifest") or ""),
                colorized_video=str(data.get("colorized_video") or ""),
                stride=int(data.get("stride") or 4),
            )
            self.send_json({"ok": ok, "message": msg, **({"error": msg} if not ok else {})})
        elif parsed.path == "/api/colour-rerender-failed":
            raw_indices = data.get("shot_indices")
            indices = [int(i) for i in raw_indices] if isinstance(raw_indices, list) else []
            manifest_str = str(data.get("manifest") or state.APP.settings.get("colour", {}).get("manifest") or "").strip()
            if not indices and manifest_str:
                from .manifests import read_manifest
                man_path = resolve_served(manifest_str, served_source_paths()) or resolve(manifest_str)
                if man_path.is_file():
                    for idx, row in enumerate(read_manifest(man_path)):
                        if str(row.get("qc_status") or "").strip().lower() in {"fail", "warn"}:
                            indices.append(idx)
            ok, msg = state.APP.rerender_failed_colour_shots(shot_indices=indices, manifest=manifest_str)
            self.send_json({"ok": ok, "message": msg, **({"error": msg} if not ok else {})})
        elif parsed.path == "/api/sam2-track-shot" or parsed.path == "/api/track-shot":
            def sam2_track_shot_action() -> dict:
                from .commercial_policy import (
                    CommercialPolicyError,
                    enabled as commercial_guard_enabled,
                    evaluate_component,
                )
                from .paths import ROOT, is_within, resolve, served_roots
                from .sam2_tracker import run_shot_tracking_pipeline, verify_sam2_installation
                from .video_segmenters import DEFAULT_SEGMENTER_REGISTRY

                settings = getattr(state.APP, "settings", {}) if state.APP else {}
                raw_segmenter = str(
                    data.get("segmenter")
                    or settings.get("colour", {}).get("segmenter")
                    or "sam21"
                ).strip()
                spec = DEFAULT_SEGMENTER_REGISTRY.get_spec(raw_segmenter)

                guard_on = commercial_guard_enabled(settings)
                for cid in (spec.component_id, *spec.extra_component_ids):
                    decision = evaluate_component(cid, settings=settings, component_type="tool")
                    if guard_on and not decision.allowed:
                        raise CommercialPolicyError(f"Commercial Model Guard blocked {cid}: {decision.reason}")

                if spec.name in {"sam21", "samurai"}:
                    try:
                        verify_sam2_installation()
                    except CommercialPolicyError:
                        raise
                    except RuntimeError:
                        from .managed_model_envs import inspect_env, is_managed_env_active

                        if is_managed_env_active() or not inspect_env("sam2_family", settings=settings)["ready"]:
                            raise

                cb_ctx = resolve_color_bible_context(settings, ROOT, stage="colour")
                shot_index = data.get("shot_index")
                shot_id = str(data.get("shot_id") or "").strip()
                if not shot_id and shot_index is not None:
                    shot_id = canonical_shot_id(cb_ctx["film_id"], int(shot_index))
                frames = data.get("frames")
                prompts = data.get("prompts", {})
                if not isinstance(prompts, dict):
                    raise ValueError("prompts must be a dictionary.")

                text_prompts = data.get("text_prompts")
                if text_prompts is not None and not isinstance(text_prompts, (str, list)):
                    raise ValueError("text_prompts must be a string or list of strings.")

                allowed_roots = served_roots(served_source_paths())
                resolved_frames = frames
                if (resolved_frames is None or resolved_frames == "") and shot_index is not None:
                    manifest_str = str(data.get("manifest") or settings.get("colour", {}).get("manifest") or "").strip()
                    if manifest_str:
                        from .manifests import manifest_source_video
                        man_path = resolve(manifest_str)
                        src_vid_str = manifest_source_video(man_path) or settings.get("global", {}).get("source", "")
                        if src_vid_str:
                            resolved_vid = resolve(src_vid_str)
                            if any(is_within(resolved_vid, r) for r in allowed_roots):
                                resolved_frames = resolved_vid

                if isinstance(frames, str) and frames.strip():
                    frame_path = resolve(frames.strip())
                    if not any(is_within(frame_path, r) for r in allowed_roots):
                        raise ValueError("Invalid frames path or path traversal detected.")
                    resolved_frames = frame_path
                elif isinstance(frames, list):
                    checked_list = []
                    for item in frames:
                        item_path = resolve(str(item).strip())
                        if not any(is_within(item_path, r) for r in allowed_roots):
                            raise ValueError("Invalid frames path or path traversal detected.")
                        checked_list.append(item_path)
                    resolved_frames = checked_list

                raw_output_npz = data.get("output_npz_path")
                if raw_output_npz and str(raw_output_npz).strip():
                    out_path = resolve(str(raw_output_npz).strip())
                    if not is_within(out_path, ROOT):
                        raise ValueError("Invalid output_npz_path or path traversal detected.")
                else:
                    clean_shot_id = Path(shot_id).name
                    out_path = ROOT / "intermediate" / "shot_masks" / f"{clean_shot_id}.masks.npz"
                    if not is_within(out_path, ROOT):
                        raise ValueError("Invalid shot_id or path traversal detected.")

                ckpt_raw = data.get("checkpoint_path")
                resolved_ckpt = None
                if ckpt_raw is not None and str(ckpt_raw).strip():
                    ckpt_str = str(ckpt_raw).strip()
                    ckpt_path = resolve(ckpt_str)
                    if Path(ckpt_str).is_absolute() or ".." in ckpt_str.replace("\\", "/").split("/"):
                        if not is_within(ckpt_path, ROOT):
                            raise ValueError("Invalid checkpoint_path or path traversal detected.")
                        resolved_ckpt = ckpt_path
                    else:
                        resolved_ckpt = ckpt_str

                seed_mask_raw = data.get("seed_mask_path")
                if seed_mask_raw is not None and str(seed_mask_raw).strip():
                    from .image_segmenters import load_seed_mask_for_tracker
                    resolved_seed = resolve_served(str(seed_mask_raw).strip(), served_source_paths())
                    if not resolved_seed or not resolved_seed.is_file():
                        raise ValueError("Invalid seed_mask_path or path traversal detected.")
                    seed_arr = load_seed_mask_for_tracker(resolved_seed)
                    prompts = dict(prompts)
                    if 1 not in prompts and "1" not in prompts:
                        prompts[1] = {
                            "frame_idx": int(data.get("seed_frame_idx", 0)),
                            "mask": seed_arr,
                            "name": resolved_seed.stem,
                            "entity_type": "object",
                        }

                alpha_thresh = float(data.get("alpha_threshold", 0.5))
                produced_path = run_shot_tracking_pipeline(
                    shot_id=shot_id,
                    frames=resolved_frames,
                    prompts=prompts,
                    output_npz_path=out_path,
                    checkpoint_path=resolved_ckpt,
                    segmenter=spec.name,
                    text_prompts=text_prompts,
                    alpha_threshold=alpha_thresh,
                    settings=settings,
                    compliance_output_dir=ROOT / "output",
                )
                return {"output_npz_path": str(produced_path)}

            self._send_result("SAM2 track shot", sam2_track_shot_action)
        elif parsed.path == "/api/segment-keyframe":
            def segment_keyframe_action() -> dict:
                from .commercial_policy import (
                    CommercialPolicyError,
                    enabled as commercial_guard_enabled,
                    evaluate_component,
                )
                from .image_segmenters import DEFAULT_IMAGE_SEGMENTER_REGISTRY, run_keyframe_segmentation
                from .paths import ROOT, is_within, resolve

                settings = getattr(state.APP, "settings", {}) if state.APP else {}
                raw_segmenter = str(
                    data.get("segmenter")
                    or settings.get("colour", {}).get("image_segmenter")
                    or "birefnet"
                ).strip()
                spec = DEFAULT_IMAGE_SEGMENTER_REGISTRY.get_spec(raw_segmenter)

                guard_on = commercial_guard_enabled(settings)
                decision = evaluate_component(spec.component_id, settings=settings, component_type="tool")
                if guard_on and not decision.allowed:
                    raise CommercialPolicyError(f"Commercial Model Guard blocked {spec.component_id}: {decision.reason}")

                image_str = str(data.get("image_path") or data.get("image") or "").strip()
                resolved_img = resolve_served(image_str, served_source_paths()) if image_str else None
                if not resolved_img or not resolved_img.is_file():
                    raise ValueError("Invalid image_path or path traversal detected.")

                raw_output_mask = data.get("output_mask_path")
                if raw_output_mask and str(raw_output_mask).strip():
                    out_mask = resolve(str(raw_output_mask).strip())
                    if not is_within(out_mask, ROOT):
                        raise ValueError("Invalid output_mask_path or path traversal detected.")
                else:
                    clean_stem = Path(resolved_img.stem).name
                    out_mask = ROOT / "intermediate" / "keyframe_masks" / f"{clean_stem}_{spec.name}.png"
                    if not is_within(out_mask, ROOT):
                        raise ValueError("Invalid output mask path.")

                ckpt_raw = data.get("checkpoint_path")
                resolved_ckpt = None
                if ckpt_raw is not None and str(ckpt_raw).strip():
                    ckpt_str = str(ckpt_raw).strip()
                    ckpt_path = resolve(ckpt_str)
                    if Path(ckpt_str).is_absolute() or ".." in ckpt_str.replace("\\", "/").split("/"):
                        if not is_within(ckpt_path, ROOT):
                            raise ValueError("Invalid checkpoint_path or path traversal detected.")
                        resolved_ckpt = ckpt_path
                    else:
                        resolved_ckpt = ckpt_str

                return run_keyframe_segmentation(
                    image=resolved_img,
                    output_mask_path=out_mask,
                    segmenter=spec.name,
                    text_prompt=data.get("text_prompt"),
                    points=data.get("points"),
                    boxes=data.get("boxes"),
                    target_classes=data.get("target_classes"),
                    alpha_threshold=float(data.get("alpha_threshold", 0.5)),
                    checkpoint_path=resolved_ckpt,
                    settings=settings,
                    compliance_output_dir=ROOT / "output",
                )

            self._send_result("Keyframe segmentation", segment_keyframe_action)
        elif parsed.path == "/api/model-envs/install":
            def install_model_env_action() -> dict:
                from .managed_model_envs import get_env_spec, require_env_policy

                app_settings = getattr(state.APP, "settings", {}) if state.APP else {}
                env_name = str(data.get("env_name") or "").strip()
                spec = get_env_spec(env_name)
                require_env_policy(spec.name, settings=app_settings, allow_non_commercial=False)
                ok, msg, extra = state.APP.install_model_env(
                    spec.name,
                    cpu=bool(data.get("cpu", False)),
                    dry_run=bool(data.get("dry_run", False)),
                )
                if not ok:
                    raise RuntimeError(msg)
                return {"message": msg, **extra}

            self._send_result("Model environment install", install_model_env_action)
        else:
            self.send_error(404)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_attachment(self, path: Path) -> None:
        """Stream only server-selected finished comparison archives to the browser."""
        if not path.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def send_static(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            self.send_error(404)
            return
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = 200
        if range_header.startswith("bytes="):
            raw_range = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
            left, _, right = raw_range.partition("-")
            try:
                if left:
                    start = int(left)
                    if right:
                        end = int(right)
                elif right:
                    suffix = int(right)
                    start = max(0, file_size - suffix)
                if start < 0 or end < start or start >= file_size:
                    raise ValueError
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if path.is_relative_to(SHEETS_ROOT):
            # Sheet folders are keyed by the video's size and mtime, so a URL never changes content.
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except FileNotFoundError:
            return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def model_looks_image_capable(model_id: str) -> bool:
    text = model_id.lower()
    return text.startswith("gpt-image") or text.startswith("dall-e")

