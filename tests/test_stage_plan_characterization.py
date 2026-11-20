"""Characterization (golden) test for stage order, upstream, input and dependency resolution.

It pins the observable behaviour of ``PipelineApp`` across the three order regimes (legacy
``preprocessing_order``, ``video_order`` and complete ``pipeline_order``) with every producer
and readiness probe replaced by deterministic stubs, so it needs no media, GPU or FFmpeg.

Regenerate only on a commit whose behaviour is intentionally the new reference:

    python tests/test_stage_plan_characterization.py --write [--dump full.json]
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import random
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import config, paths, server
from ai_remaster_gui.runtime_settings import base_settings

GOLDEN = Path(__file__).parent / "fixtures" / "stage_plan_golden.json"
VIDEO = ("denoise", "deflicker", "enhancement", "cleanup", "stabilize", "dust", "detail")
LEGACY3 = VIDEO[:3]
ARTIFACT = ("outpaint", "shots", "references", "colour", "recomp", "audio", "upscale")
ALL14 = VIDEO + ARTIFACT
OTHER_FLAGS = ("colorize", "expand_outpaint", "upscale", "add_soundtrack")
SEED = 20261120


def _stub(tag):
    return lambda source, *args, **kwargs: f"{source}/{tag}" if source else ""


def _ready(mode, rng_bits, key):
    if mode == "all":
        return True
    if mode == "none":
        return False
    return bool(rng_bits.get(key, False))


def scenarios():
    """Deterministic scenario descriptors; identical on every run and every commit."""
    rng = random.Random(SEED)
    out = []

    def add(regime, order, flags, queue, mode, bits):
        out.append({"regime": regime, "order": order, "flags": flags, "queue": queue, "mode": mode, "bits": bits})

    # Exhaustive: legacy three-stage orders x their enable subsets x queue x readiness.
    for order in itertools.permutations(LEGACY3):
        for enabled in itertools.product((False, True), repeat=3):
            flags = {key: False for key in (*VIDEO, *OTHER_FLAGS)}
            flags.update(zip(order, enabled))
            for queue in (False, True):
                for mode in ("all", "none"):
                    add("legacy", list(order), flags, queue, mode, {})
    # Seeded random coverage of every regime.
    for regime, count in (("legacy", 300), ("video", 400), ("pipeline", 500)):
        for _ in range(count):
            flags = {key: rng.random() < 0.55 for key in (*VIDEO, *OTHER_FLAGS)}
            if regime == "legacy":
                order = list(LEGACY3)
                rng.shuffle(order)
            elif regime == "video":
                order = list(VIDEO)
                rng.shuffle(order)
            else:
                order = list(ALL14) if rng.random() < 0.3 else rng.sample(ALL14, len(ALL14))
            mode = rng.choice(("all", "none", "mixed"))
            bits = {key: rng.random() < 0.5 for key in ALL14}
            add(regime, order, flags, rng.random() < 0.5, mode, bits)
    for index, item in enumerate(out):
        item["id"] = f"{index:04d}-{item['regime']}"
    return out


def build_settings(item):
    settings = base_settings()
    glob = settings["global"]
    glob["source"] = "input/film.mkv"
    glob.pop("video_order", None)
    glob.pop("pipeline_order", None)
    for key, value in item["flags"].items():
        glob[key] = "true" if value else "false"
    if item["regime"] == "legacy":
        glob["preprocessing_order"] = ",".join(item["order"])
    elif item["regime"] == "video":
        glob["video_order"] = ",".join(item["order"])
    else:
        glob["pipeline_order"] = ",".join(item["order"])
    settings["upscale"]["auto_target"] = "false"
    return settings


def capture(function, *args):
    try:
        return {"ok": function(*args)}
    except Exception as exc:  # exceptions are behaviour too
        return {"err": f"{type(exc).__name__}: {exc}"}


def without_mode(result):
    """``mode`` is an additive key introduced after the golden reference was recorded."""
    value = result.get("ok")
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "mode"}
        return {"ok": value}
    return result


def observe(item):
    settings = build_settings(item)
    before = json.dumps(settings, sort_keys=True)
    app = server.APP
    bits, mode = item["bits"], item["mode"]
    stubs = {
        "denoise_output_for": _stub("denoise"), "cleanup_output_for": _stub("cleanup"),
        "stabilize_output_for": _stub("stabilize"), "detail_output_for": _stub("detail"),
        "outpaint_output_for": _stub("outpaint.mkv"), "manifest_for_outpainted": _stub("shots.json"),
        "recomposition_output_for": _stub("recomp.mp4"), "upscale_output_for": _stub("upscale.mkv"),
        "soundtrack_output_for": _stub("audio.mp4"),
        "colorized_output_for_manifest": lambda manifest, *a, **k: f"{manifest}/colour.mkv" if manifest else "",
    }
    golden_root = Path(r"I:\ai-remaster-pipeline-stage-plan")
    with ExitStack() as stack:
        stack.enter_context(patch.object(paths, "ROOT", golden_root))
        stack.enter_context(patch.object(config, "ROOT", golden_root))
        stack.enter_context(patch.object(server, "ROOT", golden_root))
        stack.enter_context(patch.object(app, "settings", settings))
        stack.enter_context(patch.object(app, "queue_running", item["queue"], create=True))
        stack.enter_context(patch.object(app, "apply_auto_upscale_target"))
        stack.enter_context(patch.object(server, "comparison_output_ready", side_effect=lambda _app, key: _ready(mode, bits, key)))
        stack.enter_context(patch.object(app, "stage_is_current", side_effect=lambda key: _ready(mode, bits, key)))
        stack.enter_context(patch.object(server.deflicker_stage, "output_for", side_effect=_stub("deflicker")))
        stack.enter_context(patch.object(server.enhancement_stage, "output_for", side_effect=_stub("enhancement")))
        stack.enter_context(patch.object(server.dust_stage, "output_for", side_effect=_stub("dust")))
        for name, function in stubs.items():
            stack.enter_context(patch.object(server, name, side_effect=function))
        record = {}
        keys = (None, *ALL14)
        for key in keys:
            label = key or "-"
            record[f"input[{label}]"] = capture(app.processing_input, key)
            record[f"upstream[{label}]"] = capture(app.preprocessing_upstream, key)
            record[f"configured_upstream[{label}]"] = capture(app.configured_preprocessing_upstream, key)
            record[f"uses_original[{label}]"] = capture(app.preprocessing_uses_original, key)
            record[f"problem[{label}]"] = capture(app.preprocessing_problem, key)
            record[f"state[{label}]"] = without_mode(capture(app.processing_input_state, key))
        record["active_stages"] = capture(lambda: [stage.key for stage in app.active_stages()])
        if item["regime"] == "pipeline":
            record["pipeline_plan"] = capture(app.pipeline_plan)
            record["complete_plan"] = capture(server.complete_pipeline_plan, settings)
        for name in ("cleaned_source_for_downstream", "stabilized_source_for_downstream", "dust_source_for_downstream",
                     "detailed_source_for_downstream", "enhanced_source_for_downstream", "denoised_source_for_downstream",
                     "deflickered_source_for_downstream", "upscale_input_for", "soundtrack_source_for",
                     "recomposition_base_input", "outpaint_source_for"):
            record[name] = capture(getattr(app, name))
        record["outpaint_source_for_settings"] = capture(server.outpaint_source_for_settings, settings)
        record["preprocessing_input_for[-]"] = capture(server.preprocessing_input_for, settings)
        record["output_selection_state"] = capture(app.output_selection_state)
        record["settings_unchanged"] = json.dumps(settings, sort_keys=True) == before
    return record


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def compute(dump=None):
    hashes, full = {}, {}
    for item in scenarios():
        record = observe(item)
        hashes[item["id"]] = digest(record)
        if dump is not None:
            full[item["id"]] = record
    return hashes, full


class StagePlanCharacterization(unittest.TestCase):
    def test_behaviour_matches_golden_reference(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(golden["seed"], SEED)
        current, _ = compute()
        self.assertEqual(len(current), len(golden["scenarios"]))
        changed = [key for key, value in current.items() if golden["scenarios"].get(key) != value]
        self.assertFalse(changed, f"{len(changed)} scenarios changed behaviour, first: {changed[:10]}")

    def test_scenario_matrix_is_stable_and_covers_every_regime(self):
        regimes = {}
        for item in scenarios():
            regimes[item["regime"]] = regimes.get(item["regime"], 0) + 1
        self.assertEqual(set(regimes), {"legacy", "video", "pipeline"})
        self.assertEqual(sum(regimes.values()), len(scenarios()))


if __name__ == "__main__":
    if "--write" in sys.argv:
        dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
        hashes, full = compute(dump)
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"seed": SEED, "scenarios": hashes}, indent=0, sort_keys=True) + "\n", encoding="utf-8")
        if dump:
            Path(dump).write_text(json.dumps(full, sort_keys=True, default=str), encoding="utf-8")
        print(f"wrote {len(hashes)} scenarios to {GOLDEN}")
    else:
        unittest.main()
