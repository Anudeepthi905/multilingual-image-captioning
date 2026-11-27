import io

from PIL import Image

from src.uploads import check_image, check_video


def _png(w=64, h=48, fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format=fmt)
    return buf.getvalue()


def test_valid_image_is_decoded_as_rgb():
    c = check_image(_png())
    assert c.error is None and c.image.mode == "RGB" and c.image.size == (64, 48)


def test_empty_file_refused():
    assert "empty" in check_image(b"").error


def test_oversize_file_refused():
    assert "limit is 1 MB" in check_image(b"x" * (2 * 2**20), max_mb=1).error


def test_too_many_pixels_refused():
    assert "too large" in check_image(_png(200, 200), max_pixels=10_000).error


def test_non_image_refused():
    assert "could not be read" in check_image(b"not an image at all").error


def test_unsupported_format_refused():
    assert "not supported" in check_image(_png(fmt="GIF")).error


def test_truncated_image_refused():
    data = _png(300, 300, fmt="JPEG")
    assert check_image(data[: len(data) // 3]).error


def test_video_checks():
    assert check_video("clip.mp4", 1000) is None
    assert "MP4" in check_video("clip.mkv", 1000)
    assert "limit" in check_video("clip.MOV", 30 * 2**20, max_mb=25)
