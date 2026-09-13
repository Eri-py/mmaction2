from pathlib import Path

from ..common import VIDEO_EXTENSIONS


def discover_videos(videos_dir):
    """List video files directly inside videos_dir (no recursion)."""
    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        raise SystemExit(f'Videos directory not found: {videos_dir}')
    return sorted(p for p in videos_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
