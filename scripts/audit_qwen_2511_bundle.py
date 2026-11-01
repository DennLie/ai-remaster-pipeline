"""Offline SHA-256 and source-inventory audit of ARP's Qwen 2511 model bundle.

No downloads, GPU inference, licence grants or changes to the model allowlist.
The optional Qwen-Image-2.1 reference-description encoder is NOT in this
four-file manifest and must be reviewed separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from dependency_manager import QWEN_IMAGE_EDIT_MODELS
from model_paths import resolve_comfy_model_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "qwen-2511-file-manifest.json"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("commercially_approved") is not False:
        raise ValueError("Expected Qwen manifest schema v1 and commercially_approved=false")
    entries = payload.get("artifacts")
    if not isinstance(entries, list) or len(entries) != len(QWEN_IMAGE_EDIT_MODELS):
        raise ValueError("Qwen manifest must list every downloader model exactly once")

    recorded: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Qwen manifest entry must be a JSON object")
        destination = entry.get("destination", "")
        candidate = Path(destination) if isinstance(destination, str) else Path()
        if (candidate.is_absolute() or ".." in candidate.parts
                or len(candidate.parts) < 3 or candidate.parts[0] != "models"):
            raise ValueError(f"Unsafe model destination: {destination!r}")
        sha = entry.get("sha256")
        if not isinstance(sha, str) or HEX_SHA256.fullmatch(sha) is None:
            raise ValueError(f"Invalid expected SHA-256 for {destination!r}")
        if entry.get("declared_license") != "apache-2.0":
            raise ValueError(f"Unrecognised declared licence for {destination!r}")
        evidence = entry.get("evidence", "")
        if not isinstance(evidence, str) or not evidence.startswith("https://huggingface.co/"):
            raise ValueError(f"Missing source evidence for {destination!r}")
        identity = (entry.get("repo"), entry.get("source_path"), destination)
        if identity in recorded:
            raise ValueError(f"Duplicate model: {identity}")
        recorded.add(identity)

    downloader = {(model.repo, model.file, model.destination) for model in QWEN_IMAGE_EDIT_MODELS}
    if recorded != downloader:
        raise ValueError(f"Qwen manifest drifted from the downloader: missing={sorted(downloader-recorded)}, unexpected={sorted(recorded-downloader)}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(comfy_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify downloaded file identity; not a commercial licensing decision."""
    items = []
    for item in manifest["artifacts"]:
        local = resolve_comfy_model_path(comfy_dir, item["destination"])
        row = {
            "id": item["id"],
            "source_repo": item["repo"],
            "source_path": item["source_path"],
            "destination": item["destination"],
            "local_path": str(local),
            "expected_sha256": item["sha256"],
            "status": "missing",
        }
        if local.is_file():
            row["size_bytes"] = local.stat().st_size
            row["actual_sha256"] = sha256_file(local)
            row["status"] = "verified" if row["actual_sha256"] == row["expected_sha256"] else "mismatch"
        items.append(row)
    return {
        "schema_version": 1,
        "purpose": "file identity only; not commercial licensing approval",
        "commercially_approved": False,
        "all_files_verified": all(row["status"] == "verified" for row in items),
        "artifacts": items,
        "unreviewed_dependencies": manifest["additional_dependencies_not_covered"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--comfy-dir", type=Path, default=ROOT / "tools" / "comfyui")
    parser.add_argument("--report", type=Path, help="Write a JSON report of installed file hashes")
    parser.add_argument("--check-manifest-only", action="store_true", help="Validate source inventory without hashing installed models")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.check_manifest_only:
        print("Qwen 2511 manifest matches the four ARP downloader models. Commercial licence clearance remains pending.")
        return 0
    report = verify_bundle(args.comfy_dir, manifest)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["all_files_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
