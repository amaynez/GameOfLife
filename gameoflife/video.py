"""
Video export for Game of Life - GIF and MP4 support.
"""
import os
from typing import Optional

import numpy as np

# Optional dependency
IMAGEIO_AVAILABLE = False
try:
    import imageio.v3 as imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    imageio = None


def is_available() -> bool:
    """Check if video export is available."""
    return IMAGEIO_AVAILABLE


def export_video(
    output_path: str,
    frames: list[np.ndarray],
    fps: int = 30,
    *,
    codec: str = "libx264",
    pixel_size: int = 1,
) -> bool:
    """
    Export frames to video (MP4/GIF).

    Args:
        output_path: Output file path (.mp4 or .gif)
        frames: List of 2D numpy arrays (0/1 values)
        fps: Frames per second
        codec: Video codec for MP4 (default: libx264)
        pixel_size: Scale each cell by this many pixels

    Returns:
        True if export succeeded, False otherwise
    """
    if not IMAGEIO_AVAILABLE:
        print("Video export requires imageio. Install with: pip install imageio scipy")
        return False

    ext = os.path.splitext(output_path)[1].lower()

    # Scale frames if needed
    if pixel_size > 1:
        try:
            from scipy.ndimage import zoom

            scaled_frames = []
            for frame in frames:
                if pixel_size != 1:
                    # Use kron for pixel art scaling
                    h, w = frame.shape
                    scaled = np.kron(frame, np.ones((pixel_size, pixel_size), dtype=frame.dtype))
                    scaled_frames.append(scaled)
            frames = scaled_frames
        except ImportError:
            print("Warning: scipy not available, exporting at original size")
            pixel_size = 1

    if ext == ".gif":
        return _export_gif(output_path, frames, fps)
    elif ext in (".mp4", ".mkv"):
        return _export_mp4(output_path, frames, fps, codec=codec)
    else:
        # Default to GIF
        return _export_gif(output_path, frames, fps)


def _export_gif(output_path: str, frames: list[np.ndarray], fps: int) -> bool:
    """Export to animated GIF."""
    try:
        # Convert frames to uint8 (black/white)
        frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]

        imageio.imwrite(output_path, frames_uint8, fps=fps, plugin="pyav")
        print(f"Exported GIF: {output_path} ({len(frames)} frames @ {fps}fps)")
        return True
    except Exception as e:
        print(f"Failed to export GIF: {e}")
        return False


def _export_mp4(output_path: str, frames: list[np.ndarray], fps: int, codec: str) -> bool:
    """Export to MP4."""
    try:
        # Convert frames to uint8 (black/white)
        frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]

        # MP4 export with pyav plugin
        imageio.imwrite(
            output_path,
            frames_uint8,
            fps=fps,
            plugin="pyav",
            codec=codec,
        )
        print(f"Exported MP4: {output_path} ({len(frames)} frames @ {fps}fps)")
        return True
    except Exception as e:
        print(f"Failed to export MP4: {e}")
        return False


def capture_simulation(
    engine,
    steps: int,
    skip: int = 1,
) -> list[np.ndarray]:
    """
    Capture simulation frames.

    Args:
        engine: Life engine to capture from
        steps: Number of generations to simulate
        skip: Capture every N frames (for performance)

    Returns:
        List of 2D numpy arrays representing each frame
    """
    frames = []

    for i in range(0, steps, skip):
        engine.step()
        if i % skip == 0:
            frame = engine.board_view()
            frames.append(frame.copy())

    return frames