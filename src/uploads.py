"""Validation for files uploaded to the public demo (kept free of Streamlit for testing)."""
import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, UnidentifiedImageError

from src import config

IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")
IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
VIDEO_TYPES = ("mp4", "mov", "avi")


@dataclass
class Checked:
    image: Optional[Image.Image] = None
    error: Optional[str] = None


def check_image(data: bytes, max_mb: float = config.MAX_UPLOAD_MB,
                max_pixels: int = config.MAX_IMAGE_PIXELS) -> Checked:
    """Return the decoded RGB image, or a message saying why it was refused."""
    if not data:
        return Checked(error="The file is empty.")
    if len(data) > max_mb * 2**20:
        return Checked(error=f"The file is {len(data) / 2**20:.1f} MB; the limit is {max_mb:g} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt, (w, h) = probe.format, probe.size
            if fmt not in IMAGE_FORMATS:
                return Checked(error=f"{fmt or 'This'} files are not supported. "
                                     "Use JPG, PNG or WebP.")
            if w * h > max_pixels:
                return Checked(error=f"The image is {w} x {h} pixels, too large to process.")
            probe.verify()            # catches truncated or corrupt files
        img = Image.open(io.BytesIO(data))
        img.load()
        return Checked(image=img.convert("RGB"))
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        return Checked(error="This file could not be read as an image.")


def check_video(name: str, size: int, max_mb: float = config.MAX_VIDEO_MB) -> Optional[str]:
    """Return an error message, or None if the video may be processed."""
    if not name.lower().endswith(tuple("." + t for t in VIDEO_TYPES)):
        return "Use an MP4, MOV or AVI video."
    if size > max_mb * 2**20:
        return f"The video is {size / 2**20:.1f} MB; the limit is {max_mb:g} MB."
    return None
