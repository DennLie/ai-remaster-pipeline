"""Ordered single-video transforms; artifact-producing phases remain downstream."""
STAGES = ('denoise', 'deflicker', 'enhancement', 'cleanup', 'stabilize', 'dust', 'detail')


def configured(settings):
    return 'video_order' in settings.get('global', {})


def order(settings, legacy, *, strict=False):
    stored = settings.get('global', {}).get('video_order')
    if stored is None:
        if strict or configured(settings):
            raise ValueError('Choose a video processing order.')
        return [*legacy, *STAGES[3:]]
    keys = stored.split(',') if isinstance(stored, str) else []
    if len(keys) != len(STAGES) or set(keys) != set(STAGES):
        raise ValueError('Video order must contain each of the seven video stages exactly once.')
    return keys


def input_for(settings, stage, source, ordered, output):
    for key in ordered:
        if key == stage:
            break
        if source and settings.get('global', {}).get(key) == 'true':
            source = output(key, source)
    return source
