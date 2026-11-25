from __future__ import annotations

import argparse
import base64
import copy
import csv
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

from comfy_api import extract_output_files, free_comfy_models, node_by_id, queue_prompt, set_widget, wait_for_comfy, wait_for_prompt, widget_fallback_inputs, workflow_to_prompt, http_json
from common import QWEN_IMAGE_EDIT_MODEL, ROOT, copy_to_comfy_input, file_fingerprint, load_local_config, newest_output, resolve_path, root_relative, resumable_output, write_signature
from dependency_manager import ensure_ddcolor_model, ensure_qwen_image_edit_models
from ddcolor_reference import DEFAULT_MODEL_PATH as DDCOLOR_DEFAULT_MODEL_PATH, colorize_reference_shot
from reference_sets import expanded_reference_rows

DEFAULT_PROMPT = (
    'Colorize this image. Preserve the drawing and composition. '
    'Use clean modern cartoon colours, not sepia. Do not add text or new objects.'
)
DEFAULT_PROMPT_SUFFIX = (
    'Keep black ink deep, whites clean, and props/backgrounds naturally coloured.'
)
DEFAULT_OUTPUT_ROOT = ROOT / 'intermediate' / 'outpainted_references_color'
# Qwen 2511 takes up to three images: the frame being coloured plus two colour references.
MAX_REFERENCE_IMAGES = 2
# The result must keep the grey frame's composition; below this it copied a reference instead
# (the measure and threshold are Qwen-Image-2.1's, see qwen21_generate_reference.py).
COPY_THRESHOLD = 0.45
DEFAULT_OLLAMA_URL = 'http://127.0.0.1:11434'
DEFAULT_OLLAMA_VISION_MODEL = 'qwen2.5vl:7b'
REFERENCE_DESCRIPTION_PROMPT = (
    'In one short sentence, describe only the reusable colour palette of this already-colourised film frame. '
    'Mention skin, clothing, walls/wood/fabric/metal, and lighting only if clearly visible. '
    'No markdown, no headings, no bullets, no film-history commentary, no composition summary. Maximum 35 words.'
)


def read_manifest(path: Path):
    source_video = None
    rows = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.startswith('#'):
                if line.startswith('# source_video='):
                    source_video = line.split('=', 1)[1].strip()
                continue
            handle.seek(pos)
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get('enabled', 'true').strip().lower() in {'false', '0', 'no', 'off'}:
                    continue
                rows.append(row)
            break
    return source_video, rows


def load_workflow(path: Path):
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def row_source(row: dict[str, str]) -> str:
    return row.get('source_reference') or row.get('reference') or ''


def row_target(row: dict[str, str]) -> str:
    return row.get('color_reference') or row.get('reference') or ''


def output_for_row(args: argparse.Namespace, row: dict[str, str]) -> Path:
    target = row_target(row)
    if not target:
        raise ValueError('Manifest row has no color_reference/reference column.')
    target_path = resolve_path(target)
    if args.output_root:
        source_path = Path(row_source(row))
        target_path = resolve_path(args.output_root) / source_path.name
    return target_path


def read_text_file(path: Path | None) -> str:
    if not path:
        return ''
    return resolve_path(path).read_text(encoding='utf-8-sig').strip()


def describe_with_ollama(args: argparse.Namespace, image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    payload = {
        'model': args.ollama_vision_model,
        'prompt': args.reference_description_prompt,
        'images': [base64.b64encode(image_bytes).decode('ascii')],
        'stream': False,
        'keep_alive': 0,
    }
    response = http_json('POST', f"{args.ollama_url.rstrip('/')}/api/generate", payload, timeout=180)
    return str(response.get('response') or '').strip()


def description_cache_path(args: argparse.Namespace, image_path: Path) -> Path:
    cache_dir = image_path.parent / '_reference_descriptions'
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = args.reference_description_provider.replace('-', '_')
    model = args.ollama_vision_model.replace(':', '_').replace('/', '_')
    return cache_dir / f'{image_path.stem}.{provider}.{model}.txt'


def describe_reference(args: argparse.Namespace, image_path: Path) -> str:
    cache_path = description_cache_path(args, image_path)
    if not args.force_reference_descriptions and cache_path.exists():
        return cache_path.read_text(encoding='utf-8-sig').strip()
    print(f'Describe colour reference with {args.ollama_vision_model}: {image_path}')
    if args.reference_description_provider == 'ollama':
        text = describe_with_ollama(args, image_path)
    else:
        raise RuntimeError(f'Unknown reference description provider: {args.reference_description_provider}')
    cache_path.write_text(text + '\n', encoding='utf-8')
    return text


def continuity_reference_paths(args: argparse.Namespace, rows: list[dict[str, str]], row_index: int) -> list[Path]:
    paths: list[Path] = []
    for manual in args.reference:
        path = resolve_path(manual)
        if path.exists():
            paths.append(path)
    if args.continuity_reference_count <= 0:
        return paths[: args.continuity_reference_count or len(paths)]
    for previous in reversed(rows[:row_index]):
        if len(paths) >= args.continuity_reference_count:
            break
        candidate = output_for_row(args, previous)
        if candidate.exists() and candidate not in paths:
            paths.append(candidate)
    return paths[: args.continuity_reference_count]


def compact_reference_description(text: str, max_chars: int) -> str:
    text = re.sub(r'(?m)^#+\s*', '', text)
    text = re.sub(r'(?m)^\s*[-*]\s*', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(?i)\b(colou?r palette and restoration style|colou?r palette|restoration style)\s*:?', '', text).strip()
    text = re.sub(
        r'(?i)\b(skin tones?|clothing colou?rs?|wall/fabric/metal colou?rs?|lighting temperature|shadows?|overall grade)\s*:\s*',
        '',
        text,
    )
    text = re.sub(r'\bThis description should help.*$', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\bThis palette and style should be applied.*$', '', text, flags=re.IGNORECASE).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(' ', 1)[0].rstrip(' ,;:')
    return cut + '.'


def continuity_description(args: argparse.Namespace, refs: list[Path]) -> str:
    if args.reference_description_provider == 'none' or not refs:
        return ''
    lines = ['Local colour descriptions from already-colourised nearby reference frames:']
    for index, ref in enumerate(refs, start=1):
        description = compact_reference_description(describe_reference(args, ref), args.reference_description_max_chars)
        if description:
            lines.append(f'{index}. {description}')
    return '\n'.join(lines) if len(lines) > 1 else ''


def build_prompt(args: argparse.Namespace, extra_description: str = '', row_prompt: str = '') -> str:
    parts = [args.prompt.strip()]
    suffix = (args.prompt_suffix or '').strip()
    static_description = read_text_file(args.reference_description_file)
    if args.reference_description:
        static_description = (static_description + '\n' + args.reference_description).strip() if static_description else args.reference_description.strip()
    if suffix:
        parts.append(suffix)
    if static_description:
        parts.append(static_description)
    if extra_description:
        parts.append(extra_description.strip())
    if row_prompt:
        parts.append(row_prompt.strip())
    # The user's one-off direction belongs last so it is not buried under generated palette text.
    if args.add_prompt:
        parts.append(args.add_prompt.strip())
    return ' '.join(part for part in parts if part).strip()


def reference_images_prompt(prompt: str, count: int) -> str:
    """Say which picture is coloured and that the others only lend their colours."""
    if not count:
        return prompt
    images, them, their = ("Image 2 is an already coloured frame", "it", "its") if count == 1 else ("Images 2 and 3 are already coloured frames", "them", "their")
    return (
        f"{prompt} Image 1 is the black-and-white frame to colourise: keep exactly its drawing, composition, "
        f"characters, poses and background. {images} from the same film: use {them} only as a guide to the colours "
        f"of the same characters, objects and scenery, and do not copy {their} composition or contents into image 1."
    )


def resolve_shot_references(
    rows: list[dict[str, str]],
    index: int,
    count: int = MAX_REFERENCE_IMAGES,
    film_id: str | None = None,
    color_bible_service: Any | None = None,
    confidence_threshold: float = 0.8,
) -> Any:
    """Resolve semantic color references for a shot, returning SemanticMatchResult."""
    try:
        from ai_remaster_gui.semantic_matching import ColorBibleMatcher
        return ColorBibleMatcher.resolve_references(
            rows=rows,
            index=index,
            count=count,
            film_id=film_id,
            service=color_bible_service,
            confidence_threshold=confidence_threshold,
        )
    except ImportError:
        from ai_remaster_gui.semantic_matching import SemanticMatchResult
        neighbor_refs = _neighbor_reference_images_for(rows, index, count)
        return SemanticMatchResult(
            reference_paths=neighbor_refs,
            retrieved_ref_ids=[],
            retrieved_ref_hashes={},
            entities=[],
            needs_review=False,
            review_reasons=[],
            color_bible_version="",
        )


def reference_images_for(
    rows: list[dict[str, str]],
    index: int,
    count: int,
    film_id: str | None = None,
    color_bible_service: Any | None = None,
    confidence_threshold: float = 0.8,
) -> list[Path]:
    """Colour references to show Qwen with one still. When color_bible_service and film_id
    are provided, delegates to ColorBibleMatcher.resolve_references(...).
    When color_bible_service or film_id is None, preserves 100% backward compatibility
    with the existing adjacent-shot neighbor selection. At most count slots."""
    if color_bible_service is not None and film_id:
        match_result = resolve_shot_references(
            rows=rows,
            index=index,
            count=count,
            film_id=film_id,
            color_bible_service=color_bible_service,
            confidence_threshold=confidence_threshold,
        )
        return match_result.reference_paths
    return _neighbor_reference_images_for(rows, index, count)


def _neighbor_reference_images_for(rows: list[dict[str, str]], index: int, count: int) -> list[Path]:
    shot = rows[index].get('_shot_index')

    def shot_number(position: int) -> int:
        try:
            return int(rows[position].get('_shot_index', position))
        except ValueError:
            return position

    own, others = [], []
    for other_index, row in enumerate(rows):
        if other_index == index:
            continue
        candidate = resolve_path(row_target(row)) if row_target(row) else None
        if not candidate or not candidate.is_file():
            continue
        if shot is not None and row.get('_shot_index') == shot:
            own.append((abs(other_index - index), candidate))
        else:
            # Nearness counts shots, not reference slots; the earlier shot wins a tie.
            distance = abs(shot_number(other_index) - shot_number(index))
            others.append((distance, 0 if other_index < index else 1, candidate))
    chosen = [path for _distance, path in sorted(own, key=lambda item: item[0])][:MAX_REFERENCE_IMAGES]
    fill = max(0, min(count, MAX_REFERENCE_IMAGES) - len(chosen))
    chosen += [path for _distance, _later, path in sorted(others, key=lambda item: (item[0], item[1]))][:fill]
    return chosen


def reference_colours(args: argparse.Namespace, references: list[Path]) -> str:
    """Each reference's colours as words, from Qwen3-VL (cached per image).

    Shown only as images, Qwen 2511 carried a reference's scenery colours but not its characters'
    (shot 3: Mickey's red shorts came out blue). With the colours also described it matched the
    approved reference: blue hat, red shorts, yellow shoes, green hills.
    """
    from dependency_manager import QWEN_IMAGE_21_SHARED_MODELS, ensure_hf_models
    from ltx_colorize import COLOUR_REQUEST, describe_image

    encoder = next(model for model in QWEN_IMAGE_21_SHARED_MODELS if 'text_encoders' in model.destination)
    ensure_hf_models(resolve_path(args.comfy_dir), [encoder])
    parts = []
    for position, path in enumerate(references, start=2):
        text = describe_image(args, path, COLOUR_REQUEST)
        if text:
            parts.append(f'The colours in image {position}, to use for the same things in image 1: {text}')
    return ' '.join(parts)


def prepare_reference_image(path: Path, comfy_dir: Path) -> str:
    """A colour reference as a ComfyUI input. The encoder resizes it itself (and reads it as a
    second picture of its own), so it is copied as it is."""
    return copy_to_comfy_input(path, comfy_dir, 'arp_qwen_ref_images')


def signature(
    args: argparse.Namespace,
    workflow_path: Path,
    source_path: Path,
    prompt: str,
    shot_index: int = 0,
    retrieved_ref_hashes: dict[str, str] | None = None,
    retrieved_ref_ids: list[str] | None = None,
    color_bible_version: str | None = None,
) -> dict[str, Any]:
    ref_hashes = retrieved_ref_hashes or {}
    ref_ids = retrieved_ref_ids or sorted(list(ref_hashes.keys()))
    version_str = color_bible_version or ""

    backend_params: dict[str, Any] = {
        'workflow': root_relative(workflow_path),
        'load_image_node_id': str(getattr(args, 'load_image_node_id', 'auto')),
        'prompt_node_id': str(args.prompt_node_id) if getattr(args, 'prompt_node_id', None) else None,
        'save_node_id': str(getattr(args, 'save_node_id', 'auto')),
        'reference_description_provider': getattr(args, 'reference_description_provider', 'none'),
        'continuity_reference_count': getattr(args, 'continuity_reference_count', 0),
        'reference_description_max_chars': getattr(args, 'reference_description_max_chars', 220),
        'normalize_to_source_size': not getattr(args, 'no_normalize_to_source_size', False),
        'model_backend': getattr(args, 'model_backend', 'gguf'),
        'gguf_model': getattr(args, 'gguf_model', None) if getattr(args, 'model_backend', 'gguf') == 'gguf' else None,
        **({'seed': args.seed} if getattr(args, 'seed', -1) >= 0 else {}),
        **({'reference_count': args.reference_count} if getattr(args, 'reference_count', 0) > 0 else {}),
    }

    legacy_metadata = {
        'version': 5,
        'tool': 'qwen_colorize_references.py',
        'source': root_relative(source_path),
        'source_fingerprint': file_fingerprint(source_path),
        'workflow': root_relative(workflow_path),
        'workflow_fingerprint': file_fingerprint(workflow_path),
        'prompt': prompt,
        'load_image_node_id': str(getattr(args, 'load_image_node_id', 'auto')),
        'prompt_node_id': str(args.prompt_node_id) if getattr(args, 'prompt_node_id', None) else None,
        'save_node_id': str(getattr(args, 'save_node_id', 'auto')),
        'reference_description_provider': getattr(args, 'reference_description_provider', 'none'),
        'continuity_reference_count': getattr(args, 'continuity_reference_count', 0),
        'reference_description_max_chars': getattr(args, 'reference_description_max_chars', 220),
        'normalize_to_source_size': not getattr(args, 'no_normalize_to_source_size', False),
        'model_backend': getattr(args, 'model_backend', 'gguf'),
        'gguf_model': getattr(args, 'gguf_model', None) if getattr(args, 'model_backend', 'gguf') == 'gguf' else None,
        **({'seed': args.seed} if getattr(args, 'seed', -1) >= 0 else {}),
        **({'reference_count': args.reference_count} if getattr(args, 'reference_count', 0) > 0 else {}),
    }

    try:
        from ai_remaster_gui.semantic_matching import build_reference_signature
        return build_reference_signature(
            shot_index=shot_index,
            source_fingerprint=file_fingerprint(source_path),
            retrieved_ref_hashes=ref_hashes,
            prompt=prompt,
            backend_params=backend_params,
            color_bible_version=version_str,
            retrieved_ref_ids=ref_ids,
            extra_metadata=legacy_metadata,
        )
    except ImportError:
        return legacy_metadata


def patch_workflow(args: argparse.Namespace, workflow: dict[str, Any], source_path: Path, output_path: Path, prompt: str) -> dict[str, Any]:
    if has_frontend_subgraphs(workflow):
        return subgraph_workflow_to_prompt(args, workflow, source_path, output_path, prompt)
    comfy_image = copy_to_comfy_input(source_path, resolve_path(args.comfy_dir), 'arp_qwen_refs')
    load_id = resolve_node_id(workflow, args.load_image_node_id, {'LoadImage'})
    save_id = resolve_node_id(workflow, args.save_node_id, {'SaveImage'})
    prompt_id = resolve_node_id(workflow, args.prompt_node_id, {'TextEncodeQwenImageEditPlus', 'CLIPTextEncode'}, prefer_title='positive') if args.prompt_node_id else ''
    load_node = node_by_id(workflow, load_id)
    set_widget(load_node, args.load_image_widget, comfy_image)
    if prompt_id:
        prompt_node = node_by_id(workflow, prompt_id)
        set_widget(prompt_node, args.prompt_widget, prompt)
    save_node = node_by_id(workflow, save_id)
    prefix = str(Path('ai_remaster_qwen') / output_path.stem).replace('\\', '/')
    set_widget(save_node, args.save_prefix_widget, prefix)
    for node in iter_workflow_nodes(workflow):
        inputs = node.get('inputs')
        if (node.get('type') or node.get('class_type')) in {'KSampler', 'RandomNoise'} and isinstance(inputs, dict):
            for name in ('seed', 'noise_seed'):
                if name in inputs:
                    inputs[name] = active_seed(args)
    return workflow_to_prompt(workflow, save_id)


def active_seed(args: argparse.Namespace) -> int:
    """The seed for the still being made now; main_with_args picks it per still."""
    return int(getattr(args, 'active_seed', 1))


def has_frontend_subgraphs(workflow: dict[str, Any]) -> bool:
    definitions = workflow.get('definitions')
    if not isinstance(definitions, dict):
        return False
    subgraphs = definitions.get('subgraphs')
    return isinstance(subgraphs, list) and bool(subgraphs)


def subgraph_workflow_to_prompt(args: argparse.Namespace, workflow: dict[str, Any], source_path: Path, output_path: Path, prompt: str) -> dict[str, Any]:
    subgraphs = workflow.get('definitions', {}).get('subgraphs') or []
    if not subgraphs:
        raise RuntimeError('Workflow does not contain a frontend subgraph definition.')
    subgraph = copy.deepcopy(subgraphs[0])
    nodes = subgraph.get('nodes') or []
    links = subgraph.get('links') or []
    subgraph_inputs = subgraph.get('inputs') or []
    load_id = 90001
    save_id = 90002
    comfy_image = copy_to_comfy_input(source_path, resolve_path(args.comfy_dir), 'arp_qwen_refs')
    link_lookup = {int(link.get('id')): link for link in links if isinstance(link, dict) and 'id' in link}
    source_image_link: dict[str, Any] | None = None
    reference_names = list(getattr(args, 'active_reference_names', []) or [])[:MAX_REFERENCE_IMAGES]
    reference_targets: dict[int, dict[str, Any]] = {}

    def subgraph_input_name(slot: int) -> str:
        if 0 <= slot < len(subgraph_inputs) and isinstance(subgraph_inputs[slot], dict):
            return str(subgraph_inputs[slot].get('name') or subgraph_inputs[slot].get('label') or '').lower()
        return ''

    for node in nodes:
        if not isinstance(node, dict):
            continue
        for item in node.get('inputs') or []:
            link_id = item.get('link')
            if link_id is None:
                continue
            link = link_lookup.get(int(link_id))
            if not link or int(link.get('origin_id', 0)) != -10:
                continue
            slot = int(link.get('origin_slot', -1))
            input_name = subgraph_input_name(slot)
            if input_name in {'image', 'image1'}:
                source_image_link = link
                item['link'] = 900001
            elif input_name in {'image2', 'image3'}:
                position = 0 if input_name == 'image2' else 1
                item['link'] = 900002 + position if position < len(reference_names) else None
                if item['link'] is not None:
                    reference_targets.setdefault(position, link)
            elif input_name == 'prompt':
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'prompt')}
                set_workflow_widget(node, item.get('name', 'prompt'), prompt)
            elif input_name == 'seed':
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'seed')}
                set_workflow_widget(node, item.get('name', 'seed'), active_seed(args))
            elif input_name in {'value', 'enable_turbo_mode'}:
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'value')}
                set_workflow_widget(node, item.get('name', 'value'), True)
            elif input_name == 'unet_name':
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'unet_name')}
                set_workflow_widget(node, item.get('name', 'unet_name'), args.gguf_model if args.model_backend == 'gguf' else 'qwen_image_edit_2511_bf16.safetensors')
            elif input_name == 'clip_name':
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'clip_name')}
                set_workflow_widget(node, item.get('name', 'clip_name'), 'qwen_2.5_vl_7b_fp8_scaled.safetensors')
            elif input_name == 'vae_name':
                item['link'] = None
                item['widget'] = {'name': item.get('name', 'vae_name')}
                set_workflow_widget(node, item.get('name', 'vae_name'), 'qwen_image_vae.safetensors')

    # Defensively clear any remaining links from the subgraph input node (-10) that
    # weren't explicitly handled above (e.g. inputs replaced by patch_qwen_model_backend).
    input_link_ids = {
        int(l.get('id'))
        for l in links
        if isinstance(l, dict) and int(l.get('origin_id', 0)) == -10
    }
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for item in node.get('inputs') or []:
            if isinstance(item, dict) and item.get('link') is not None and int(item['link']) in input_link_ids:
                item['link'] = None

    if not source_image_link:
        raise RuntimeError('Could not find subgraph source image input link.')

    nodes.append(
        {
            'id': load_id,
            'type': 'LoadImage',
            'inputs': [{'name': 'image', 'type': 'COMBO', 'widget': {'name': 'image'}}],
            'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': [900001]}],
            'widgets_values': [comfy_image],
        }
    )
    links.append({
        'id': 900001,
        'origin_id': load_id,
        'origin_slot': 0,
        'target_id': int(source_image_link.get('target_id')),
        'target_slot': int(source_image_link.get('target_slot', 0)),
        'type': 'IMAGE',
    })
    for position, target in sorted(reference_targets.items()):
        reference_load_id = 90003 + position
        nodes.append({
            'id': reference_load_id,
            'type': 'LoadImage',
            'inputs': [{'name': 'image', 'type': 'COMBO', 'widget': {'name': 'image'}}],
            'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': [900002 + position]}],
            'widgets_values': [reference_names[position]],
        })
        links.append({
            'id': 900002 + position,
            'origin_id': reference_load_id,
            'origin_slot': 0,
            'target_id': int(target.get('target_id')),
            'target_slot': int(target.get('target_slot', 0)),
            'type': 'IMAGE',
        })

    if reference_targets:
        vision_only_references(nodes, links)

    output_link = next((link for link in links if isinstance(link, dict) and int(link.get('target_id', 0)) == -20), None)
    if not output_link:
        raise RuntimeError('Could not find subgraph output image link.')
    output_link['target_id'] = save_id
    output_link['target_slot'] = 0
    prefix = str(Path('ai_remaster_qwen') / output_path.stem).replace('\\', '/')
    nodes.append(
        {
            'id': save_id,
            'type': 'SaveImage',
            'inputs': [{'name': 'images', 'type': 'IMAGE', 'link': int(output_link['id'])}, {'name': 'filename_prefix', 'type': 'STRING', 'widget': {'name': 'filename_prefix'}}],
            'widgets_values': [prefix],
        }
    )
    # Convert positional widgets_values lists to named dicts for nodes with known widget
    # layouts. This prevents misalignment caused by hidden widgets (e.g. KSampler's
    # control_after_generate) that occupy a slot in widgets_values but have no entry in
    # the inputs list, which would shift all subsequent widget indices by one.
    for node in nodes:
        if not isinstance(node, dict):
            continue
        wv = node.get('widgets_values')
        if not isinstance(wv, list):
            continue
        class_type = node.get('type') or node.get('class_type', '')
        named = widget_fallback_inputs(class_type, wv)
        if named:
            node['widgets_values'] = named
        if class_type == 'KSampler' and isinstance(node.get('widgets_values'), dict):
            node['widgets_values']['seed'] = active_seed(args)
            node['widgets_values']['control_after_generate'] = 'fixed'

    flat = {'nodes': nodes, 'links': [link_to_list(link) for link in links]}
    return workflow_to_prompt(flat, save_id)


def vision_only_references(nodes: list[dict[str, Any]], links: list[dict[str, Any]]) -> None:
    """Let the colour references be seen but not redrawn.

    TextEncodeQwenImageEditPlus turns every image into vision tokens and, when it has a VAE, also
    into a reference latent, which Qwen reproduces: with the shot's own reference attached it
    redrew that reference's pose instead of colouring the grey frame (shot 3, composition 0.32).
    So the encoders lose their VAE (every image is vision only) and a ReferenceLatent node gives
    each conditioning back the grey frame alone, from the graph's own VAEEncode of image 1.
    """
    encoders = [node for node in nodes if isinstance(node, dict) and node.get('type') == 'TextEncodeQwenImageEditPlus']
    encode = next((node for node in nodes if isinstance(node, dict) and node.get('type') == 'VAEEncode'), None)
    if not encoders or encode is None:
        raise RuntimeError('The Qwen 2511 workflow has no TextEncodeQwenImageEditPlus/VAEEncode pair to give the colour references.')
    for item in (inp for encoder in encoders for inp in encoder.get('inputs') or []):
        if item.get('name') == 'vae':
            item['link'] = None
    next_link = 900020
    for offset, encoder in enumerate(encoders):
        reference_id = 90010 + offset
        conditioning_link, latent_link = next_link, next_link + 1
        next_link += 2
        consumers = [link for link in links if isinstance(link, dict) and int(link.get('origin_id', 0)) == int(encoder['id'])]
        for link in consumers:
            link['origin_id'], link['origin_slot'] = reference_id, 0
        for output in encoder.get('outputs') or []:
            output['links'] = [conditioning_link]
        for output in encode.get('outputs') or []:
            output['links'] = [*(output.get('links') or []), latent_link]
        nodes.append({
            'id': reference_id,
            'type': 'ReferenceLatent',
            'inputs': [
                {'name': 'conditioning', 'type': 'CONDITIONING', 'link': conditioning_link},
                {'name': 'latent', 'type': 'LATENT', 'link': latent_link},
            ],
            'outputs': [{'name': 'CONDITIONING', 'type': 'CONDITIONING', 'links': [int(link['id']) for link in consumers]}],
            'widgets_values': [],
        })
        links.append({'id': conditioning_link, 'origin_id': int(encoder['id']), 'origin_slot': 0, 'target_id': reference_id, 'target_slot': 0, 'type': 'CONDITIONING'})
        links.append({'id': latent_link, 'origin_id': int(encode['id']), 'origin_slot': 0, 'target_id': reference_id, 'target_slot': 1, 'type': 'LATENT'})


def link_to_list(link: Any) -> Any:
    if not isinstance(link, dict):
        return link
    return [
        link.get('id'),
        link.get('origin_id'),
        link.get('origin_slot', 0),
        link.get('target_id'),
        link.get('target_slot', 0),
        link.get('type', '*'),
    ]


def set_workflow_widget(node: dict[str, Any], input_name: str, value: Any) -> None:
    widgets = node.setdefault('widgets_values', [])
    if isinstance(widgets, dict):
        widgets[input_name] = value
        return
    if not isinstance(widgets, list):
        widgets = [widgets]
        node['widgets_values'] = widgets
    widget_index = 0
    for item in node.get('inputs') or []:
        if 'widget' not in item:
            continue
        if item.get('name') == input_name or item.get('widget', {}).get('name') == input_name:
            while len(widgets) <= widget_index:
                widgets.append(None)
            widgets[widget_index] = value
            return
        widget_index += 1
    widgets.append(value)


def resolve_node_id(workflow: dict[str, Any], value: str | None, class_types: set[str], prefer_title: str = '') -> str:
    if value and str(value).lower() != 'auto':
        return str(value)
    matches = [node for node in iter_workflow_nodes(workflow) if (node.get('class_type') or node.get('type')) in class_types]
    if prefer_title:
        titled = [node for node in matches if prefer_title.lower() in str(node.get('title') or '').lower()]
        if titled:
            matches = titled
    if not matches:
        raise RuntimeError(f"Could not auto-detect workflow node for: {', '.join(sorted(class_types))}")
    return str(matches[0].get('id'))


def normalize_to_source_size(path: Path, source_path: Path, final_path: Path | None = None) -> None:
    """Scale (never crop) the Qwen output to exactly match the source image dimensions."""
    from PIL import Image

    with Image.open(source_path) as source_image:
        target_size = source_image.size
    with Image.open(path) as image:
        if image.size == target_size:
            return
        resampling = getattr(Image, 'Resampling', Image).LANCZOS
        # Use resize (scale), not ImageOps.fit (centre-crop), so no pixels are ever discarded.
        normalized = image.convert('RGB').resize(target_size, resampling)
        output_suffix = (final_path or path).suffix.lower()
        image_format = 'JPEG' if output_suffix in {'.jpg', '.jpeg'} else 'PNG'
        save_kwargs = {'quality': 95} if image_format == 'JPEG' else {}
        normalized.save(path, format=image_format, **save_kwargs)
    print(f'Normalized Qwen output to source size: {path} ({target_size[0]}x{target_size[1]})', flush=True)


def iter_workflow_nodes(workflow: dict[str, Any]):
    seen: set[int] = set()

    def visit(value: Any):
        if isinstance(value, dict):
            ident = id(value)
            if ident in seen:
                return
            seen.add(ident)
            if 'class_type' in value or 'type' in value:
                yield value
            for child in value.values():
                yield from visit(child)
        elif isinstance(value, list):
            for child in value:
                yield from visit(child)

    yield from visit(workflow)


def node_widget_values(node: dict[str, Any]) -> list[Any]:
    inputs = node.get('inputs', {})
    if isinstance(inputs, dict):
        input_values = list(inputs.values())
    else:
        input_values = []
    values = node.get('widgets_values', [])
    if isinstance(values, list):
        return input_values + values
    if isinstance(values, dict):
        return input_values + list(values.values())
    return input_values + [values]


def patch_qwen_model_backend(args, workflow: dict[str, Any]) -> None:
    if args.model_backend != 'gguf':
        return
    patched = 0
    for node in iter_workflow_nodes(workflow):
        class_type = node.get('class_type') or node.get('type')
        if class_type not in {'UNETLoader', 'UnetLoader'}:
            continue
        values = ' '.join(str(value).lower() for value in node_widget_values(node))
        title = str(node.get('title') or '').lower()
        if 'qwen' not in values and 'qwen' not in title:
            continue
        if 'class_type' in node:
            node['class_type'] = 'UnetLoaderGGUF'
            node.setdefault('inputs', {})['unet_name'] = args.gguf_model
        else:
            node['type'] = 'UnetLoaderGGUF'
            node['title'] = 'Unet Loader (GGUF)'
            node['inputs'] = [{'name': 'unet_name', 'type': 'COMBO', 'widget': {'name': 'unet_name'}}]
            node['widgets_values'] = [args.gguf_model]
        patched += 1
    if not patched:
        raise RuntimeError('Could not find a Qwen UNETLoader node to patch to UnetLoaderGGUF. Use a Qwen Image Edit workflow with a visible UNETLoader node, or run with --model-backend safetensors.')


def build_parser() -> argparse.ArgumentParser:
    config = load_local_config()
    parser = argparse.ArgumentParser(description='Colorize extracted reference stills with a single-image Qwen Image Edit ComfyUI workflow.')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--workflow', required=True, type=Path, help='ComfyUI Qwen Image Edit workflow JSON/API file.')
    parser.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    parser.add_argument('--comfy-dir', type=Path, default=Path(config.get('comfy_dir', ROOT / 'tools' / 'comfyui')), help='ComfyUI directory used for on-demand model downloads.')
    parser.add_argument('--comfy-output-root', type=Path, default=ROOT / 'tools' / 'comfyui' / 'output', help='ComfyUI output directory used to locate saved images.')
    parser.add_argument('--model-backend', choices=['gguf', 'safetensors'], default='gguf')
    parser.add_argument('--gguf-model', default=QWEN_IMAGE_EDIT_MODEL)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--prompt-suffix', default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument('--add-prompt', default='', help='Extra one-off guidance appended last, after generated reference descriptions.')
    parser.add_argument('--reference-description', default='', help='Static palette/continuity guidance appended to the image edit prompt.')
    parser.add_argument('--reference-description-file', type=Path, help='UTF-8 text file with static palette/continuity guidance.')
    parser.add_argument('--reference', action='append', default=[], help='Manual colour reference image to describe as text; can be repeated.')
    parser.add_argument('--no-normalize-to-source-size', action='store_true', help='Keep ComfyUI/Qwen output dimensions instead of cropping/resizing to the source image size.')
    parser.add_argument('--reference-description-provider', choices=['none', 'ollama'], default='none', help='Describe continuity/reference images as text instead of passing multiple images to Qwen.')
    parser.add_argument('--reference-description-prompt', default=REFERENCE_DESCRIPTION_PROMPT)
    parser.add_argument('--ollama-url', default=DEFAULT_OLLAMA_URL)
    parser.add_argument('--ollama-vision-model', default=DEFAULT_OLLAMA_VISION_MODEL)
    parser.add_argument('--force-reference-descriptions', action='store_true')
    parser.add_argument('--reference-description-max-chars', type=int, default=220, help='Clamp each local continuity description before appending it to the Qwen prompt.')
    parser.add_argument('--continuity-reference-count', type=int, default=0, help='Describe this many previous colour references and append them to the prompt. 0 disables automatic continuity descriptions.')
    parser.add_argument('--output-root', type=Path, help='Override manifest color_reference destinations.')
    parser.add_argument('--load-image-node-id', default='auto')
    parser.add_argument('--load-image-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--prompt-node-id')
    parser.add_argument('--prompt-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--save-node-id', default='auto')
    parser.add_argument('--save-prefix-widget', default='0', help='Widget name for API-format workflows, or widget index for normal exported workflows.')
    parser.add_argument('--reference-image', action='append', default=[], type=Path, help='A colour reference to show Qwen as image 2 or 3 (at most two); can be repeated.')
    parser.add_argument('--reference-count', type=int, default=0, help=f'Without --reference-image: a still always sees its own shot\'s other colour references; nearby shots\' fill up to this many (0-{MAX_REFERENCE_IMAGES}).')
    parser.add_argument('--describe-references', action=argparse.BooleanOptionalAction, default=True, help='Describe each colour reference\'s colours (Qwen3-VL) into the prompt; images alone carry scenery colours but not characters\'.')
    parser.add_argument('--seed', type=int, default=-1, help='Sampler seed; -1 picks a new random seed for every still, so each attempt differs.')
    parser.add_argument('--poll-seconds', type=float, default=2.0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--row-index', type=int, default=None, help='Process only this zero-based shot index in the manifest.')
    parser.add_argument('--reference-index', type=int, default=0, help='Reference index within --row-index when regenerating a single still.')
    parser.add_argument('--keep-comfy-loaded', action='store_true', help='Skip unloading ComfyUI models after finishing.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--print-final-prompt', action='store_true')
    parser.add_argument('--print-api-prompt', action='store_true')
    parser.add_argument(
        '--reference-runner',
        choices=['auto', 'guided_only', 'ddcolor_only'],
        default='guided_only',
        help='Reference colorization runner mode: guided_only (default), auto (DDColor draft without entities), or ddcolor_only.',
    )
    parser.add_argument(
        '--guided-backend',
        default='qwen2511',
        help='Default guided reference backend to use (default: qwen2511).',
    )
    parser.add_argument(
        '--ddcolor-model',
        default='piddnad/ddcolor_modelscope',
        help='Official piddnad/DDColor model variant used when runner routes a shot to DDColor.',
    )
    parser.add_argument(
        '--ddcolor-model-path',
        type=Path,
        default=None,
        help='Optional custom DDColor checkpoint path when runner routes a shot to DDColor.',
    )
    parser.add_argument(
        '--film-id',
        default=None,
        type=str,
        help='Film ID for Color Bible semantic reference matching.',
    )
    parser.add_argument(
        '--color-bible-db',
        default=None,
        type=Path,
        help='Path to Color Bible SQLite database.',
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.8,
        help='Confidence threshold for Color Bible assignments below which a shot is paused for review (default: 0.8).',
    )
    return parser


def main_with_args(args: argparse.Namespace) -> int:
    if args.continuity_reference_count < 0:
        raise RuntimeError('--continuity-reference-count must be zero or greater')
    manifest = resolve_path(args.manifest)
    workflow_path = resolve_path(args.workflow)
    comfy_dir = resolve_path(args.comfy_dir)
    _, shot_rows = read_manifest(manifest)
    rows = expanded_reference_rows(shot_rows)
    if getattr(args, 'row_index', None) is not None:
        target_shot = str(args.row_index)
        target_ref = str(getattr(args, 'reference_index', 0))
        filtered = [r for r in rows if r.get('_shot_index') == target_shot and r.get('_reference_index', '0') == target_ref]
        if not filtered:
            filtered = [r for r in rows if r.get('_shot_index') == target_shot]
        rows = filtered
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f'Manifest: {manifest}')
    print(f'Rows: {len(rows)}')
    reference_note = 'the given colour references' if args.reference_image else (
        "its shot's other colour references" + (f", topped up to {min(args.reference_count, MAX_REFERENCE_IMAGES)} from nearby shots" if args.reference_count else ''))
    print(f'Qwen mode: {args.model_backend}; each still with {reference_note}.')
    rows_without_source = [row for row in rows if not row_source(row).strip()]
    if rows_without_source:
        print(f'Skipping {len(rows_without_source)} shot(s) with no guide frame (empty source_reference).')
        rows = [row for row in rows if row_source(row).strip()]
    missing_sources = [resolve_path(row_source(row)) for row in rows if not resolve_path(row_source(row)).is_file()]
    if missing_sources:
        sample = ', '.join(str(path) for path in missing_sources[:3])
        more = f' and {len(missing_sources) - 3} more' if len(missing_sources) > 3 else ''
        raise FileNotFoundError(f'Reference source image not found: {sample}{more}')

    color_bible_service = None
    if getattr(args, 'film_id', None):
        try:
            from ai_remaster_gui.color_bible_service import ColorBibleService
            color_bible_service = ColorBibleService(db_path=getattr(args, 'color_bible_db', None))
        except Exception as e:
            print(f'Warning: Failed to initialize ColorBibleService: {e}')
            color_bible_service = None

    needs_comfy = not args.dry_run and rows and getattr(args, 'reference_runner', 'auto') != 'ddcolor_only'
    if needs_comfy:
        if args.model_backend == 'gguf' and not (comfy_dir / 'custom_nodes' / 'ComfyUI-GGUF').exists():
            raise FileNotFoundError(f'ComfyUI-GGUF is required for Qwen GGUF. Re-run install_windows.bat, then restart ComfyUI: {comfy_dir / "custom_nodes" / "ComfyUI-GGUF"}')
        ensure_qwen_image_edit_models(comfy_dir)
        print(f'Waiting for ComfyUI at {args.comfy_url}...')
        wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)

    try:
        from ai_remaster_gui.semantic_matching import (
            ColorBibleMatcher,
            PromptEnricher,
            TwoStageReferenceRouter,
            record_reference_compliance,
        )
        from ai_remaster_gui.commercial_policy import evaluate_component, CommercialPolicyError
        from ai_remaster_gui.manifests import update_manifest_row

        for index, row in enumerate(rows):
            src = resolve_path(row_source(row))
            dst = output_for_row(args, row)
            shot_idx = int(row.get('_shot_index', index))

            # 1. Resolve semantic references
            match_result = ColorBibleMatcher.resolve_references(
                rows=rows,
                index=index,
                count=MAX_REFERENCE_IMAGES,
                film_id=getattr(args, 'film_id', None),
                service=color_bible_service,
                confidence_threshold=getattr(args, 'confidence_threshold', 0.8),
            )

            # 2. Priority C Review Gate Check (MUST BE BEFORE CACHE HIT CHECK)
            if match_result.needs_review:
                row['qc_status'] = 'needs_review'
                try:
                    update_manifest_row(manifest, shot_idx, {'qc_status': 'needs_review'})
                except Exception as exc:
                    print(f'Warning: Failed to update manifest CSV for shot {shot_idx}: {exc}')

                if color_bible_service is not None and getattr(args, 'film_id', None):
                    try:
                        color_bible_service.update_shot_qc_status(args.film_id, shot_idx, 'needs_review')
                    except Exception as exc:
                        print(f'Warning: Failed to update Color Bible shot record for shot {shot_idx}: {exc}')

                print(f"[Review Gate] Shot {shot_idx:04d} paused for human review: {', '.join(match_result.review_reasons)}")
                continue

            # 3. Select Backend
            effective_runner = getattr(args, 'reference_runner', 'guided_only')

            backend = TwoStageReferenceRouter.select_backend(
                row=row,
                has_color_bible_entities=bool(match_result.entities),
                reference_runner=effective_runner,
                default_guided_backend=getattr(args, 'guided_backend', 'qwen2511'),
            )

            # 4. Enrich Prompt
            refs = continuity_reference_paths(args, rows, index)
            extra_description = continuity_description(args, refs)
            raw_prompt = build_prompt(args, extra_description, row.get('prompt') or row.get('custom_prompt') or '')
            prompt = PromptEnricher.enrich_prompt(raw_prompt, match_result.entities)

            # 5. Check Scoped Cache
            sig = signature(
                args,
                workflow_path,
                src,
                prompt,
                shot_index=shot_idx,
                retrieved_ref_hashes=match_result.retrieved_ref_hashes,
                retrieved_ref_ids=match_result.retrieved_ref_ids,
                color_bible_version=match_result.color_bible_version,
            )
            if args.print_final_prompt:
                print(f'Final prompt {index:04d}: {prompt}')
            if not args.force and resumable_output(dst, sig, image_like=src):
                print(f'Reuse {index:04d}: {dst}')
                continue

            # 6. Commercial Model Guard
            model_name = "ddcolor" if backend == "ddcolor" else "qwen-image-edit-2511"
            comm_settings = None
            try:
                from ai_remaster_gui import state
                if getattr(state, "APP", None) is not None and getattr(state.APP, "settings", None):
                    comm_settings = state.APP.settings
            except Exception:
                pass
            if comm_settings is None:
                local_cfg = load_local_config()
                comm_settings = {
                    "global": {
                        "commercial_model_guard": str(local_cfg.get("commercial_model_guard", "true")),
                        "non_commercial_benchmark_override": str(local_cfg.get("non_commercial_benchmark_override", "false")),
                        "qwen_2511_audited": str(local_cfg.get("qwen_2511_audited", "true")),
                    }
                }
            global_comm = comm_settings.get("global", {}) if isinstance(comm_settings.get("global"), dict) else comm_settings
            is_audited = str(global_comm.get("qwen_2511_audited", "true")).strip().lower() in ("true", "1", "yes")
            decision = evaluate_component(
                model_name,
                component_type="checkpoint",
                settings=comm_settings,
                conditional_attestations={"qwen_2511_audited": is_audited},
            )
            if not decision.allowed:
                raise CommercialPolicyError(f"Commercial policy blocked backend '{backend}' ({model_name}): {decision.reason}")

            args.active_seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)

            # 7. Execute Selected Backend
            if backend == "ddcolor":
                ddcolor_ckpt = getattr(args, 'ddcolor_model_path', None)
                ddcolor_variant = getattr(args, 'ddcolor_model', 'piddnad/ddcolor_modelscope')
                try:
                    if not args.dry_run:
                        ddcolor_ckpt = ensure_ddcolor_model(
                            variant=ddcolor_variant,
                            model_path=ddcolor_ckpt,
                            auto_download=ddcolor_ckpt is None,
                        )
                    print(f'DDColor colorize {index:04d}: {src} -> {dst}')
                    colorize_reference_shot(
                        src,
                        dst,
                        model_path=ddcolor_ckpt or DDCOLOR_DEFAULT_MODEL_PATH,
                        dry_run=args.dry_run,
                        seed=args.active_seed,
                        allow_builtin_arch=True,
                    )
                except Exception as exc:
                    if effective_runner == 'auto':
                        print(f'Warning: DDColor draft unavailable in auto mode ({exc}); falling back to guided Qwen backend.', flush=True)
                        backend = getattr(args, 'guided_backend', 'qwen2511')
                        model_name = "qwen-image-edit-2511"
                    else:
                        raise
            if backend != "ddcolor":
                ref_note = f', described refs={len(refs)}' if refs else ''
                explicit_refs = [resolve_path(path) for path in args.reference_image if resolve_path(path).is_file()][:MAX_REFERENCE_IMAGES]
                if getattr(args, 'film_id', None) and match_result.reference_paths:
                    shown = match_result.reference_paths[:MAX_REFERENCE_IMAGES]
                elif explicit_refs:
                    shown = explicit_refs
                else:
                    shown = reference_images_for(rows, index, args.reference_count)
                shown = [path for path in shown if path.resolve() != dst.resolve()]
                args.active_reference_names = [prepare_reference_image(path, comfy_dir) for path in shown] if not args.dry_run else [path.name for path in shown]
                colours = reference_colours(args, shown) if shown and args.describe_references and not args.dry_run else ''
                base_prompt = f'{prompt} {colours}'.strip()
                prompt = f'{reference_images_prompt(prompt, len(shown))} {colours}'.strip()
                if shown:
                    print(f'Qwen colour references {index:04d}: ' + ', '.join(path.name for path in shown), flush=True)
                print(f'Colorize {index:04d}: {src} -> {dst}{ref_note}')
                print(f'Qwen prompt {index:04d}: {prompt}')
                print(f'Qwen seed {index:04d}: {args.active_seed}', flush=True)
                workflow = load_workflow(workflow_path)
                patch_qwen_model_backend(args, workflow)
                if args.dry_run:
                    prompt_payload = patch_workflow(args, workflow, src, dst, prompt)
                    if args.print_api_prompt:
                        print(json.dumps(prompt_payload, indent=2))
                    continue
                if not src.is_file():
                    raise FileNotFoundError(f'Reference source not found: {src}')
                prompt_payload = patch_workflow(args, workflow, src, dst, prompt)
                if args.print_api_prompt:
                    print(json.dumps(prompt_payload, indent=2))
                produced = render(args, prompt_payload)
                if shown and composition_score(produced, src) < COPY_THRESHOLD:
                    print(f'Composition check {index:04d}: the colour references took over this still; colouring it again without them.', flush=True)
                    args.active_reference_names = []
                    workflow = load_workflow(workflow_path)
                    patch_qwen_model_backend(args, workflow)
                    produced = render(args, patch_workflow(args, workflow, src, dst, base_prompt))
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(dst.suffix + '.partial')
                shutil.copy2(produced, tmp)
                if not args.no_normalize_to_source_size:
                    normalize_to_source_size(tmp, src, dst)
                tmp.replace(dst)

            # 8. Write signature and record compliance
            write_signature(dst, sig)
            record_reference_compliance(
                output_path=dst,
                model_id=model_name,
                component_type="checkpoint",
                color_bible_version=match_result.color_bible_version,
                retrieved_ref_ids=match_result.retrieved_ref_ids,
                settings=comm_settings,
                seed=args.active_seed,
                conditional_attestations={"qwen_2511_audited": is_audited},
            )
            print(f'Wrote {dst}')
    finally:
        if color_bible_service is not None:
            color_bible_service.close()
        if needs_comfy and not getattr(args, 'keep_comfy_loaded', False):
            free_comfy_models(args.comfy_url)

    return 0


def render(args: argparse.Namespace, prompt_payload: dict[str, Any]) -> Path:
    prompt_id = queue_prompt(args.comfy_url, prompt_payload)
    print(f'Queued ComfyUI prompt: {prompt_id}', flush=True)
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    return newest_output(extract_output_files(history, resolve_path(args.comfy_output_root)))


def composition_score(result: Path, source: Path) -> float:
    """How well a still keeps its grey frame's composition (1.0 the same picture, a copy near 0)."""
    from PIL import Image
    import numpy as np
    from qwen21_generate_reference import structure_score

    with Image.open(source) as grey, Image.open(result) as coloured:
        grey_rgb = np.asarray(grey.convert('RGB'))
        coloured_rgb = np.asarray(coloured.convert('RGB').resize((grey_rgb.shape[1], grey_rgb.shape[0])))
    score = structure_score(coloured_rgb, grey_rgb)
    print(f'Composition check: {score:.2f} (copies score under {COPY_THRESHOLD})', flush=True)
    return score


def main() -> int:
    return main_with_args(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())



