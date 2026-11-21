"""Compatibility shim: video order now lives in :mod:`ai_remaster_gui.stage_plan`."""
from .stage_plan import VIDEO_STAGES as STAGES
from .stage_plan import input_chain as input_for
from .stage_plan import ordered_video_stages as order
from .stage_plan import video_configured as configured

__all__ = ('STAGES', 'configured', 'order', 'input_for')
