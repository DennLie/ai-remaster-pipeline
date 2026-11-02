#!/usr/bin/env python3
"""Prepare a small, traceable multi-film archival restoration benchmark.

Downloads ONLY catalog-listed Wikimedia Commons videos via MediaWiki API,
or accepts a user-provided local source. Stores short audio-free 8-bit
FFV1 excerpts, gentle Unsharp/CAS, and artificial ground-truth blur
stress tests. All original source files stay outside Git/artifacts.

Research rights review required. Never interpret synthetic blur as original
film damage or pixel-difference scores as restored archival detail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from fractions import Fraction
from pathlib import Path

MAX_BYTES = 250 * 1024 * 1024
TIMEOUT = 90
UA = "ARP-PublicDomainFilm-Corpus/0.1 (research; https://github.com/DennLie/ai-remaster-pipeline)"
FILM_ID = re.compile(r"^[a-z][a-z0-9_]{1,36}$")


def hash_file(path: Path, algo: str = "sha256") -> str:
    digest = hashlib.new(algo)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("films"), list):
        raise ValueError("Unsupported film catalog schema")
    ids = set()
    for film in data["films"]:
        key = film.get("id")
        if not isinstance(key, str) or not FILM_ID.fullmatch(key) or key in ids:
            raise ValueError(f"Invalid or duplicated film id: {key}")
        ids.add(key)
        shots = film.get("shots")
        if not isinstance(shots, list) or not 1 <= len(shots) <= 4:
            raise ValueError(f"Expected 1–4 curated shots for {key}")
        for shot in shots:
            if not FILM_ID.fullmatch(shot.get("name", "")):
                raise ValueError(f"Unsafe shot name for {key}")
            if not isinstance(shot.get("frames"), int) or not 12 <= shot["frames"] <= 96:
                raise ValueError(f"Unsafe frame count for {key}")
            if not isinstance(shot.get("seconds"), (int, float)) or not math.isfinite(shot["seconds"]) or not 0 <= shot["seconds"] <= 86400:
                raise ValueError(f"Unsafe shot start for {key}")
        if film.get("acquire") not in ("original", "vp9_360_or_480"):
            raise ValueError(f"Unsupported Commons variant for {key}")
        cap = film.get("max_download_megabytes", 250)
        if not isinstance(cap, int) or not 1 <= cap <= 400:
            raise ValueError(f"Per-film download safety cap must be 1..400 MiB: {key}")
        if not film.get("source_page", "").startswith("https://commons.wikimedia.org/wiki/File:"):
            raise ValueError(f"Not a Wikimedia Commons file page: {key}")
    return data


def request_json(url: str) -> dict:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == 3:
                raise RuntimeError(f"Commons metadata query failed: {error}") from error
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def remote_video(film: dict) -> dict:
    title = "File:" + film["commons_file"]
    # The Commons API returns source SHA1/size and the server's actual
    # transcoded video URLs. Do not synthesize transcode URLs from filenames.
    api = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "videoinfo", "viprop": "url|size|sha1|derivatives",
        "titles": title,
    })
    response = request_json(api)
    if "error" in response:
        raise ValueError(f"Commons API video metadata error: {response['error']}")
    pages = response.get("query", {}).get("pages", [])
    if len(pages) != 1 or "missing" in pages[0]:
        raise FileNotFoundError(f"Commons film file not found: {title}")
    items = pages[0].get("videoinfo", [])
    if not items:
        raise ValueError(f"Commons returned no original videoinfo: {title}")
    meta = items[0]
    if not isinstance(meta.get("sha1"), str) or not re.fullmatch("[a-f0-9]{40}", meta["sha1"].lower()):
        raise ValueError(f"Missing Commons original SHA-1 for {title}")
    expected = film.get("source_original_sha1")
    if expected is not None and expected.lower() != meta["sha1"].lower():
        raise ValueError(f"Commons original SHA-1 changed for {title}")
    if film["acquire"] == "original":
        target = meta.get("url")
        format_note = "Commons original source file"
    else:
        options = meta.get("derivatives", [])
        desired = []
        for d in options:
            key = str(d.get("transcodekey", "")).lower()
            src = str(d.get("src", ""))
            if (("360p" in key or "480p" in key) and "vp9" in key
                    and ".webm" in src):
                desired.append(d)
        if not desired:
            raise ValueError(f"No VP9 360p/480p WebM derivative for {title}; found {[(x.get('transcodekey'),x.get('height')) for x in options]}")
        # Explicit bounded policy: choose 360p, else 480p; never silently
        # download the multi-gigabyte original when both are missing.
        desired.sort(key=lambda d: (0 if "360p" in str(d.get("transcodekey", "")) else 1,
                                    str(d.get("transcodekey", ""))))
        chosen = desired[0]
        target = chosen["src"]
        format_note = f"Commons-generated {chosen.get('transcodekey')} proxy, NOT original master"
    if not isinstance(target, str) or not target.startswith("https://upload.wikimedia.org/"):
        raise ValueError(f"Unsafe Commons media URL for {title}: {target}")
    return {
        "commons_title": title, "commons_api": api, "commons_original_sha1": meta["sha1"],
        "commons_original_bytes": int(meta.get("size", 0) or 0),
        "selected_media_url": target, "selected_format": format_note,
        "original_sha1_pinned": expected is not None,
    }


def download(url: str, destination: Path, *, max_bytes: int = MAX_BYTES) -> dict:
    if not url.startswith("https://upload.wikimedia.org/"):
        raise ValueError("Only HTTPS Wikimedia upload media allowed")
    last_error = None
    for attempt in range(3):
        destination.unlink(missing_ok=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as stream, destination.open("wb") as out:
                final = stream.geturl()
                if not final.startswith("https://upload.wikimedia.org/"):
                    raise ValueError(f"Disallowed redirect for Commons media: {final}")
                declared = stream.headers.get("Content-Length")
                if declared and int(declared) > max_bytes:
                    raise ValueError(f"Source too large: {declared} bytes exceeds {max_bytes}")
                total = 0
                while data := stream.read(1024 * 1024):
                    total += len(data)
                    if total > max_bytes:
                        raise ValueError(f"Downloaded source exceeds safety cap {max_bytes}")
                    out.write(data)
                if total < 1024:
                    raise ValueError("Downloaded video is suspiciously small")
            return {"downloaded_sha256": hash_file(destination), "downloaded_bytes": total, "actual_url": final}
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        except Exception:
            destination.unlink(missing_ok=True)
            raise
    raise RuntimeError(f"Wikimedia source download failed: {last_error}")


def probe(ffprobe: str, video: Path, *, count_frames: bool = False) -> dict:
    keys = "stream=width,height,avg_frame_rate,r_frame_rate"
    if count_frames:
        keys += ",nb_read_frames"
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        cmd += ["-count_frames"]
    cmd += ["-show_entries", keys, "-of", "json", str(video)]
    obj = subprocess.run(cmd, capture_output=True, text=True, check=True)
    streams = json.loads(obj.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"Video source has no stream: {video}")
    s = streams[0]
    w, h = int(s["width"]), int(s["height"])
    fps = Fraction(s["avg_frame_rate"])
    if w < 100 or h < 100 or fps <= 0 or fps > 120:
        raise ValueError(f"Unsupported archival video geometry/cadence: {w}x{h}, {fps}")
    if Fraction(s.get("r_frame_rate", "0/1")) != fps:
        raise ValueError(f"Variable frame rate: {video}")
    result = {"width": w, "height": h, "fps": f"{fps.numerator}/{fps.denominator}"}
    if count_frames:
        result["frames"] = int(s["nb_read_frames"])
    return result


def compatible_vsr_geometry(w: int, h: int, limit: int = 2048) -> dict | None:
    gcd = math.gcd(w, h)
    a, b = w // gcd, h // gcd
    multiple = max(1, math.ceil(w / (128 * a)), math.ceil(h / (128 * b)))
    tw, th = 128 * a * multiple, 128 * b * multiple
    if max(tw, th) > limit:
        return None
    return {"width": tw, "height": th, "ratio_preserved": True}


def encode(ffmpeg: str, source: Path, destination: Path, *, start: float = 0.,
           frames: int | None = None, filter_expr: str | None = None) -> None:
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn"]
    if frames is not None:
        args += ["-frames:v", str(frames)]
    if filter_expr:
        args += ["-vf", filter_expr]
    args += ["-fps_mode", "passthrough", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(destination)]
    subprocess.run(args, capture_output=True, text=True, check=True, timeout=180)


def render_one(catalog: dict, film: dict, source: Path, out_dir: Path,
               *, source_provenance: dict, ffmpeg: str, ffprobe: str,
               include_classical: bool = True) -> dict:
    from compare_denoising import compare

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite corpus folder: {out_dir}")
    info = probe(ffprobe, source)
    output = {
        "schema_version": 1, "id": film["id"], "title": film["title"],
        "year": film["year"], "creator": film["creator"],
        "source_page": film["source_page"], "commons_file": film["commons_file"],
        "source_provenance": source_provenance, "decoded_source": info,
        "rights_status": film["right_status"], "transfer_note": film["transfer_note"],
        "original_audio_redistributed": False, "clips": [],
        "warnings": [
            "These are short extracts of a digitized transfer, not verified original film scans.",
            "Synthetic blur variants are ARTIFICIAL known degradations, not historical damage.",
            "Perceptual quality and historical authenticity require human review. No commercial clearance is claimed."
        ],
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arp-film-staging-", dir=out_dir.parent) as td:
        stage = Path(td)
        for scene in film["shots"]:
            root = stage / scene["name"]
            root.mkdir()
            orig = root / "original_native.mkv"
            encode(ffmpeg, source, orig, start=float(scene["seconds"]), frames=scene["frames"])
            expected = {**info, "frames": scene["frames"]}
            if probe(ffprobe, orig, count_frames=True) != expected:
                raise ValueError(f"Frame count/geometry/cadence mismatch: {film['id']}/{scene['name']}")
            candidate_paths = {}
            for sigma, name in ((0.7, "synthetic_blur_mild"), (1.3, "synthetic_blur_medium")):
                target = root / (name + ".mkv")
                encode(ffmpeg, orig, target, filter_expr=f"gblur=sigma={sigma}:steps=2")
                if probe(ffprobe, target, count_frames=True) != expected:
                    raise ValueError(f"Generated synthetic blur variant invalid: {target}")
                candidate_paths[name] = target
            if include_classical:
                for method in ("unsharp", "cas"):
                    target = root / (method + "_gentle.mkv")
                    subprocess.run(
                        ["python", "scripts/detail_enhance_video.py",
                         "--source", str(orig), "--output", str(target),
                         "--method", method, "--preset", "gentle", "--mix", "100",
                         "--intermediate-profile", "lossless"],
                        capture_output=True, text=True, check=True, timeout=180,
                    )
                    if probe(ffprobe, target, count_frames=True) != expected:
                        raise ValueError(f"Native detail variant invalid: {target}")
                    candidate_paths[method + "_gentle"] = target
            diagnostics = root / "diagnostics"
            compare(orig, candidate_paths, diagnostics)
            vsr = compatible_vsr_geometry(info["width"], info["height"])
            if vsr is not None:
                baseline = root / "lanczos_reference.mkv"
                encode(ffmpeg, orig, baseline,
                       filter_expr=f"scale={vsr['width']}:{vsr['height']}:flags=lanczos")
                probe_result = probe(ffprobe, baseline, count_frames=True)
                if (probe_result["width"], probe_result["height"], probe_result["frames"]) != (
                    vsr["width"], vsr["height"], scene["frames"]
                ):
                    raise ValueError(f"VSR reference geometry/frame mismatch: {baseline}")
            entry = {
                "name": scene["name"], "start_seconds": scene["seconds"],
                "frames": scene["frames"], "geometry": info,
                "vsr_target": vsr,
                "purpose_tags": film["tests"],
                "files": {},
            }
            for f in sorted(root.glob("*.mkv")):
                entry["files"][f.name] = {"sha256": hash_file(f), "bytes": f.stat().st_size}
            output["clips"].append(entry)
        (stage / "corpus_manifest.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (stage / "README.txt").write_text(
            "ARP historical film corpus, research only. Each scene has native FFV1, "
            "synthetic gaussian blur variants and classic non-generative outputs. "
            "No audio. License and transfer review REQUIRED. "
            "Never equate a sharper or higher-scoring result with historical truth.\n",
            encoding="utf-8",
        )
        if out_dir.exists():
            raise FileExistsError(f"Output appeared during processing: {out_dir}")
        stage.rename(out_dir)
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=root / "benchmarks/public_domain_films/catalog.json")
    p.add_argument("--film", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-file", type=Path, help="Use own local transfer instead of Commons; provenance marked local")
    p.add_argument("--metadata-only", action="store_true", help="Verify Commons API media identity without download")
    p.add_argument("--no-classical", action="store_true")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args()
    catalog = load_catalog(args.catalog)
    film = next((f for f in catalog["films"] if f["id"] == args.film), None)
    if film is None:
        raise ValueError(f"Unknown curated film {args.film}")
    if args.output_dir.exists():
        raise FileExistsError("Output corpus directory exists; refusing overwrite")
    if args.metadata_only and args.source_file:
        raise ValueError("--metadata-only is not useful with --source-file")
    metadata = remote_video(film) if not args.source_file else {
        "selected_format": "local-file-override", "commons_original_sha1": None,
        "original_sha1_pinned": False
    }
    if args.metadata_only:
        print(json.dumps({"film": film["id"], "metadata": metadata}, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="arp-pd-source-") as td:
        if args.source_file:
            source = args.source_file.resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            transfer = {"downloaded_sha256": hash_file(source),
                        "downloaded_bytes": source.stat().st_size,
                        "actual_url": "local-file-override"}
        else:
            source = Path(td) / "wikipedia_transfer.webm"
            transfer = download(metadata["selected_media_url"], source, max_bytes=film.get("max_download_megabytes", 250) * 1024 * 1024)
            if film["acquire"] == "original" and metadata.get("commons_original_sha1"):
                if hash_file(source, "sha1") != metadata["commons_original_sha1"]:
                    raise ValueError("Downloaded Commons original SHA-1 does not match Commons API")
        payload = render_one(
            catalog, film, source, args.output_dir,
            source_provenance={**metadata, **transfer}, ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe, include_classical=not args.no_classical,
        )
    print(json.dumps({"film": film["id"], "clips": len(payload["clips"]),
                      "output": str(args.output_dir), "sha256": payload["source_provenance"]["downloaded_sha256"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
