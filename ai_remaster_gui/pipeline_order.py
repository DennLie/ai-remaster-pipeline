"""Compatibility shim: pipeline order and plan building now live in :mod:`ai_remaster_gui.stage_plan`."""
from .stage_plan import STAGES, VIDEO_STAGES, build, enabled
from .stage_plan import ordered_stages as order
from .stage_plan import pipeline_configured as configured

__all__ = ('STAGES', 'VIDEO_STAGES', 'build', 'enabled', 'configured', 'order')
