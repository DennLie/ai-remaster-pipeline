import os
from pathlib import Path

# Ensure bundled FFmpeg tools in .cache/tools/ffmpeg are on PATH for all test runs
ROOT = Path(__file__).resolve().parents[1]
ffmpeg_dir = ROOT / ".cache" / "tools" / "ffmpeg"
if ffmpeg_dir.is_dir() and str(ffmpeg_dir) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
