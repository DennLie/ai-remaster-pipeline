"""Stage order, upstream resolution, input planning and dependency checks.

This module is the single owner of "which stages run, in which order, on which input, and is
everything upstream of a stage ready". It replaces three parallel order regimes that callers
used to branch on:

* ``legacy``   - only ``preprocessing_order`` (denoise/deflicker/enhancement); later video stages
                 follow in a fixed order. Standalone stage previews from the original source exist
                 only in this regime.
* ``video``    - ``video_order`` covers all seven video stages.
* ``pipeline`` - ``pipeline_order`` covers all fourteen phases and binds artifact producers.

Compatibility contract
----------------------
Stored settings are never rewritten here. ``canonical_order`` is a read-only adapter that derives
the complete fourteen-phase order from whichever keys a project actually stores, so old project
files keep their exact semantics until the user saves them with a new order. Contradictory or
malformed stored orders raise ``ValueError`` (strict callers) or fall back exactly as before; no
new meaning is invented.

Output *producers* (the ``*_output_for`` naming functions) and readiness probes stay with the
application and are injected, so this module does no I/O of its own beyond the probes it is given.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

VIDEO_STAGES = ('denoise', 'deflicker', 'enhancement', 'cleanup', 'stabilize', 'dust', 'detail')
PREPROCESSING_STAGES = ('denoise', 'deflicker', 'enhancement')
ARTIFACT_STAGES = ('outpaint', 'shots', 'references', 'colour', 'recomp', 'audio', 'upscale')
STAGES = (*VIDEO_STAGES, *ARTIFACT_STAGES)

INPUT_PARAMETER_FIELDS = ('method', 'preset', 'mix', 'zoom', 'smoothing', 'detector', 'repair')
CHAIN_PARAMETER_FIELDS = ('method', 'preset', 'mix', 'strength', 'max_correction', 'window', 'zoom',
                          'smoothing', 'detector', 'repair')


class Regime(str, Enum):
    LEGACY = 'legacy'
    VIDEO = 'video'
    PIPELINE = 'pipeline'


# --------------------------------------------------------------------------- regime


def pipeline_configured(settings) -> bool:
    return 'pipeline_order' in settings.get('global', {})


def video_configured(settings) -> bool:
    return 'video_order' in settings.get('global', {})


def regime(settings) -> Regime:
    if pipeline_configured(settings):
        return Regime.PIPELINE
    if video_configured(settings):
        return Regime.VIDEO
    return Regime.LEGACY


def uses_stage_order(settings) -> bool:
    """True when the project stores an explicit video or pipeline order (not legacy)."""
    return regime(settings) is not Regime.LEGACY


def allows_original_preview(settings) -> bool:
    """Standalone previews from the original source exist only in the legacy regime."""
    return regime(settings) is Regime.LEGACY


def flag(settings, key: str) -> bool:
    return settings.get('global', {}).get(key) == 'true'


def with_enabled(settings, key: str):
    """A copy of settings with one global stage flag forced on (settings are never mutated)."""
    return {**settings, 'global': {**settings.get('global', {}), key: 'true'}}


# --------------------------------------------------------------------------- order


def preprocessing_order(settings, *, strict=False):
    stored = settings.get('global', {}).get('preprocessing_order', ','.join(PREPROCESSING_STAGES))
    keys = stored.split(',') if isinstance(stored, str) else []
    if len(keys) != 3 or set(keys) != set(PREPROCESSING_STAGES):
        if strict:
            raise ValueError('Pipeline order must contain each of the three preprocessing stages exactly once.')
        return list(PREPROCESSING_STAGES)
    return keys


def ordered_video_stages(settings, legacy, *, strict=False):
    stored = settings.get('global', {}).get('video_order')
    if stored is None:
        if strict or video_configured(settings):
            raise ValueError('Choose a video processing order.')
        return [*legacy, *VIDEO_STAGES[3:]]
    keys = stored.split(',') if isinstance(stored, str) else []
    if len(keys) != len(VIDEO_STAGES) or set(keys) != set(VIDEO_STAGES):
        raise ValueError('Video order must contain each of the seven video stages exactly once.')
    return keys


def ordered_stages(settings, video, *, strict=False):
    if not pipeline_configured(settings):
        if strict:
            raise ValueError('Choose a complete pipeline order.')
        return [*video, *STAGES[len(VIDEO_STAGES):]]
    value = settings['global']['pipeline_order']
    keys = value.split(',') if isinstance(value, str) else []
    if len(keys) != len(STAGES) or set(keys) != set(STAGES):
        raise ValueError('Pipeline order must contain every one of the fourteen phases exactly once.')
    return keys


def video_stages_in_order(settings):
    """All seven video stages in stored order, for the legacy and video regimes (read-only)."""
    return ordered_video_stages(settings, preprocessing_order(settings))


def chain_order(settings):
    """The video stages whose outputs chain into a stage's input (legacy: the three classic ones)."""
    if video_configured(settings):
        return video_stages_in_order(settings)
    return preprocessing_order(settings)


def canonical_order(settings, *, strict=False):
    """Compatibility adapter: the complete fourteen-phase order for any regime, read-only."""
    if pipeline_configured(settings):
        return ordered_stages(settings, (), strict=strict)
    if strict:
        raise ValueError('Choose a complete pipeline order.')
    return ordered_stages(settings, video_stages_in_order(settings))


def enabled(settings, key):
    flags = settings.get('global', {})
    yes = lambda name: flags.get(name) == 'true'
    if key == 'shots':
        return yes('colorize') or yes('upscale')
    if key in ('references', 'colour'):
        return yes('colorize')
    if key == 'recomp':
        return yes('expand_outpaint') or yes('colorize')
    return yes({'outpaint': 'expand_outpaint', 'audio': 'add_soundtrack'}.get(key, key))


# --------------------------------------------------------------------------- plan (pipeline regime)


def build(settings, source, ordered, output):
    """No I/O or settings mutation. An invalid placement remains visible/editable."""
    rows = {}
    current = source
    shots = None
    references = None
    canvas = None
    colour = None
    for key in ordered:
        if not enabled(settings, key):
            continue
        bindings = {}
        problem = '' if current else 'Choose a source video or resolve earlier missing video bindings.'
        if key == 'shots':
            shots = {'source': current, 'manifest': output(key, current, {})}
            bindings = shots.copy()
            target = current
        elif key == 'references':
            if not shots or shots['source'] != current:
                problem = 'References require earlier Shot Detection on this exact video input.'
            bindings = shots.copy() if shots else {}
            references = bindings.copy()
            target = current
        elif key == 'colour':
            if not references or references.get('source') != current:
                problem = 'Colorization requires earlier Shots and References for this exact video; move or regenerate them after intervening video transforms.'
            bindings = references.copy() if references else {}
            target = output(key, current, bindings)
            if not target and not problem:
                problem = 'Colorization has no unambiguous selected output. Choose the recomposition colour method.'
            colour = {'source': current, 'output': target, **bindings}
        elif key == 'recomp':
            candidate = colour or canvas
            if not candidate or candidate['output'] != current:
                problem = 'Recomposition requires a preceding matching colour/outpaint result with no intervening video transform. Move Recomposition before that transform.'
            bindings = {'source': canvas['source'] if canvas else colour['source'] if colour else current,
                        'outpainted_video': canvas['output'] if canvas else '',
                        'colorized_video': colour['output'] if colour else '',
                        'manifest': colour.get('manifest', '') if colour else ''}
            if canvas and colour and colour['source'] != canvas['output']:
                problem = 'Colour and outpaint branches do not share the same canvas. Regenerate Shots/References/Colorization on the outpainted video, or recompose colour before Outpainting.'
            target = output(key, current, bindings)
            canvas = None
            colour = None
        else:
            if key == 'upscale' and shots:
                bindings = shots.copy()
            target = output(key, current, {})
            if key == 'outpaint':
                canvas = {'source': current, 'output': target}
        rows[key] = {'input': current, 'output': target, 'bindings': bindings, 'problem': problem}
        current = target
    return {'nodes': rows, 'output': current,
            'problems': [f'{key}: {row["problem"]}' for key, row in rows.items() if row['problem']]}


def plan_for(settings, stage_key, plan, build_plan: Callable[[Mapping], dict]):
    """The plan to read for a stage. Dust may be requested while disabled: plan it as enabled."""
    if stage_key == 'dust' and 'dust' not in plan['nodes']:
        return build_plan(with_enabled(settings, 'dust'))
    return plan


def plan_input(plan, stage_key):
    return plan['nodes'].get(stage_key, {}).get('input', '') if stage_key else plan['output']


# --------------------------------------------------------------------------- input chain


def input_chain(settings, stage_key, source, ordered, output):
    """Apply each enabled stage before ``stage_key`` to ``source`` (``output(key, current)``)."""
    for key in ordered:
        if key == stage_key:
            break
        if source and settings.get('global', {}).get(key) == 'true':
            source = output(key, source)
    return source


# --------------------------------------------------------------------------- upstream


def upstream(settings, stage_key, plan=None):
    """Enabled stages before ``stage_key`` in the configured order (``plan`` for pipeline regime)."""
    if pipeline_configured(settings):
        keys = list(plan['nodes'])
        return keys[:keys.index(stage_key)] if stage_key in keys else keys
    result = []
    for key in chain_order(settings):
        if key == stage_key:
            break
        if flag(settings, key):
            result.append(key)
    return result


def original_preview(settings, stage_key, *, queue_running: bool,
                     upstream_keys: Callable[[], list], output_ready: Callable[[str], bool]) -> bool:
    """True when a stage runs as an explicit standalone preview from the original source.

    Previews exist only in the legacy regime, only for the three classic stages, never inside a
    whole-remaster queue, and only while some upstream result is missing. A preview is reported
    through ``input_state(...)['mode'] == 'preview'`` and is never an upstream result itself.
    """
    if not allows_original_preview(settings):
        return False
    return (stage_key in PREPROCESSING_STAGES and not queue_running
            and any(not output_ready(key) for key in upstream_keys()))


# --------------------------------------------------------------------------- dependencies


@dataclass(frozen=True)
class Probes:
    """Late-bound readiness probes supplied by the application (no I/O in this module)."""
    output_ready: Callable[[str], bool]
    stage_current: Callable[[str], bool]
    is_file: Callable[[Any], bool]
    resolve: Callable[[str], Any]
    manifest_source_video: Callable[[Any], str]
    metadata: Callable[[Any], Mapping]


def pipeline_problem(plan, stage_key, probes: Probes) -> str:
    for key, row in plan['nodes'].items():
        if row['problem']:
            return key + ': ' + row['problem']
        bindings = row['bindings']
        manifest = bindings.get('manifest', '')
        if key in ('references', 'colour') and manifest and probes.is_file(manifest):
            try:
                actual = probes.manifest_source_video(probes.resolve(manifest))
            except (OSError, ValueError, KeyError, TypeError):
                return key + ': the shot manifest source could not be validated.'
            if not actual or probes.resolve(actual).resolve() != probes.resolve(bindings.get('source', '')).resolve():
                return key + ': the saved shot manifest belongs to another video. Review/regenerate Shot Detection; old references are not substituted.'
        if key == 'recomp':
            paths = [bindings.get(name, '') for name in ('source', 'outpainted_video', 'colorized_video')]
            paths = [probes.resolve(path) for path in paths if path]
            if paths and all(probes.is_file(path) for path in paths):
                try:
                    metadata = [probes.metadata(path) for path in paths]
                    if any(any(value[name] != metadata[0][name] for name in ('frames', 'fps_fraction')) for value in metadata[1:]):
                        return 'Recomposition branch frame counts/cadence do not match; regenerate matching branches.'
                except (OSError, ValueError, KeyError):
                    return 'Recomposition branch metadata could not be validated.'
        if key == stage_key:
            break
        ready = probes.output_ready(key) if key in VIDEO_STAGES else probes.stage_current(key)
        if not ready:
            return f'Run {key} first for the current pipeline bindings; old artifacts are not substituted.'
    return ''


def upstream_problem(upstream_keys, titles: Mapping[str, str], output_ready: Callable[[str], bool]) -> str:
    for key in upstream_keys:
        if not output_ready(key):
            return f'Run {titles[key]} first so this stage has its current upstream video.'
    return ''


# --------------------------------------------------------------------------- input state


@dataclass(frozen=True)
class InputHooks:
    """Application callbacks used to describe a stage's input (all late-bound)."""
    titles: Mapping[str, str]
    plan_keys: Callable[[], list]
    upstream: Callable[[Any], list]
    processing_input: Callable[[Any], str]
    problem: Callable[[Any], str]
    uses_original: Callable[[Any], bool]
    downstream_source: Callable[[str], str]   # legacy getters for cleanup/dust/detail/stabilize
    cleanup_enabled: Callable[[], bool]
    is_file: Callable[[Any], bool]
    stage_options: Callable[[str, Mapping], Mapping]


def input_state(settings, stage_key, hooks: InputHooks) -> dict:
    """What feeds a stage and why. ``mode`` is ``'preview'`` for a standalone original-source
    preview and ``'pipeline'`` otherwise; a preview never counts as an upstream result."""
    titles = hooks.titles
    if pipeline_configured(settings):
        keys = hooks.plan_keys()
        upstream_keys = hooks.upstream(stage_key)
        previous = upstream_keys[-1] if upstream_keys else None
        return {'source': hooks.processing_input(stage_key), 'from_stage': previous,
                'label': previous or 'Original source', 'original_fallback': False, 'mode': 'pipeline',
                'parameters': {}, 'problem': hooks.problem(stage_key),
                'chain': [{'key': key, 'title': titles[key]} for key in keys]}
    if not video_configured(settings) and stage_key in ('cleanup', 'dust', 'detail'):
        result = input_state(settings, None, hooks)
        ordered = video_stages_in_order(settings)
        previous = [key for key in ordered[:ordered.index(stage_key)] if flag(settings, key)]
        key = previous[-1] if previous else None
        result.update(source=hooks.downstream_source(stage_key), from_stage=key, original_fallback=False, mode='pipeline',
                      label=titles[key] if key in titles else 'Original source · selected processing input',
                      parameters={name: value for name, value in settings.get(key, {}).items() if name in INPUT_PARAMETER_FIELDS})
        return result
    if stage_key == 'stabilize' and not video_configured(settings):
        result = input_state(settings, None, hooks)
        result.update(source=hooks.downstream_source('stabilize'), original_fallback=False, mode='pipeline')
        if hooks.cleanup_enabled():
            result.update(from_stage='cleanup', label='Clean Up output', parameters={},
                          problem=result['problem'] or ('' if hooks.is_file(result['source']) else 'Run Clean Up first.'))
        return result
    upstream_keys = hooks.upstream(stage_key)
    previous = upstream_keys[-1] if upstream_keys else None
    values = settings.get(previous, {}) if previous else {}
    if previous in ('deflicker', 'enhancement'):
        values = hooks.stage_options(previous, values)
    preview = hooks.uses_original(stage_key)
    return {'source': hooks.processing_input(stage_key), 'from_stage': previous,
            'label': titles[previous] if previous else ('Original source · upstream result unavailable' if preview else 'Original source · selected processing input'),
            'original_fallback': hooks.uses_original(stage_key), 'mode': 'preview' if preview else 'pipeline',
            'parameters': {key: values[key] for key in CHAIN_PARAMETER_FIELDS if key in values},
            'problem': hooks.problem(stage_key),
            'chain': [{'key': key, 'title': titles[key]}
                      for key in video_stages_in_order(settings) if flag(settings, key)]}
