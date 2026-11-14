"""Explicit video and artifact bindings for the complete ordered pipeline."""
from .video_order import STAGES as VIDEO_STAGES

STAGES = (*VIDEO_STAGES,'outpaint','shots','references','colour','recomp','audio','upscale')


def configured(settings):
    return 'pipeline_order' in settings.get('global',{})


def order(settings, video, *, strict=False):
    if not configured(settings):
        if strict:
            raise ValueError('Choose a complete pipeline order.')
        return [*video,*STAGES[len(VIDEO_STAGES):]]
    value=settings['global']['pipeline_order']
    keys=value.split(',') if isinstance(value,str) else []
    if len(keys)!=len(STAGES) or set(keys)!=set(STAGES):
        raise ValueError('Pipeline order must contain every one of the fourteen phases exactly once.')
    return keys


def enabled(settings,key):
    flags=settings.get('global',{})
    yes=lambda flag:flags.get(flag)=='true'
    if key=='shots':return yes('colorize') or yes('upscale')
    if key in ('references','colour'):return yes('colorize')
    if key=='recomp':return yes('expand_outpaint') or yes('colorize')
    return yes({'outpaint':'expand_outpaint','audio':'add_soundtrack'}.get(key,key))


def build(settings, source, ordered, output):
    """No I/O or settings mutation. An invalid placement remains visible/editable."""
    rows={};current=source;shots=None;references=None;canvas=None;colour=None
    for key in ordered:
        if not enabled(settings,key):continue
        bindings={};problem='' if current else 'Choose a source video or resolve earlier missing video bindings.'
        if key=='shots':
            shots={'source':current,'manifest':output(key,current,{})}
            bindings=shots.copy();target=current
        elif key=='references':
            if not shots or shots['source']!=current:
                problem='References require earlier Shot Detection on this exact video input.'
            bindings=shots.copy() if shots else {};references=bindings.copy();target=current
        elif key=='colour':
            if not references or references.get('source')!=current:
                problem='Colorization requires earlier Shots and References for this exact video; move or regenerate them after intervening video transforms.'
            bindings=references.copy() if references else {}
            target=output(key,current,bindings)
            if not target and not problem:problem='Colorization has no unambiguous selected output. Choose the recomposition colour method.'
            colour={'source':current,'output':target,**bindings}
        elif key=='recomp':
            candidate=colour or canvas
            if not candidate or candidate['output']!=current:
                problem='Recomposition requires a preceding matching colour/outpaint result with no intervening video transform. Move Recomposition before that transform.'
            bindings={'source':canvas['source'] if canvas else colour['source'] if colour else current,
                'outpainted_video':canvas['output'] if canvas else '',
                'colorized_video':colour['output'] if colour else '',
                'manifest':colour.get('manifest','') if colour else ''}
            if canvas and colour and colour['source']!=canvas['output']:
                problem='Colour and outpaint branches do not share the same canvas. Regenerate Shots/References/Colorization on the outpainted video, or recompose colour before Outpainting.'
            target=output(key,current,bindings)
            canvas=None;colour=None
        else:
            if key=='upscale' and shots:bindings=shots.copy()
            target=output(key,current,{})
            if key=='outpaint':canvas={'source':current,'output':target}
        rows[key]={'input':current,'output':target,'bindings':bindings,'problem':problem}
        current=target
    return {'nodes':rows,'output':current,'problems':[f'{key}: {row["problem"]}' for key,row in rows.items() if row['problem']]}
