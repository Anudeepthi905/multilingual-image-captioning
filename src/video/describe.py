"""EXPERIMENTAL: timestamped descriptions of a video using the image captioner.

Samples one keyframe every `interval` seconds with OpenCV, captions each frame, and
drops consecutive duplicate captions. This is not a video model: it has no notion of
motion or events, it simply describes still frames.

Usage:
    python -m src.video.describe --video clip.mp4 --interval 2
"""
import argparse


def sample_keyframes(video_path: str, interval: float = 2.0, max_frames: int = 120):
    """Yield (timestamp_seconds, RGB uint8 frame) every `interval` seconds."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * interval)))
    count = 0
    try:
        for idx in range(0, total if total > 0 else 10**9, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            yield idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            count += 1
            if count >= max_frames:
                break
    finally:
        cap.release()


def dedupe(entries: list) -> list:
    """Drop entries whose caption equals the previous kept caption."""
    out = []
    for ts, text in entries:
        if not out or out[-1][1] != text:
            out.append((ts, text))
    return out


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def describe_video(video_path: str, captioner, interval: float = 2.0, beam: int = 1,
                   max_frames: int = 120) -> list:
    """Return [(timestamp_seconds, caption)] with consecutive duplicates removed."""
    entries = []
    for ts, frame in sample_keyframes(video_path, interval, max_frames):
        text, _, _ = captioner.caption(frame, beam=beam)
        entries.append((ts, text))
    return dedupe(entries)


def main():
    from src.caption import Captioner, list_runs

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--beam", type=int, default=1)
    a = p.parse_args()
    runs = list_runs()
    if not runs:
        raise SystemExit("No trained run found. Train one with: python -m src.train")
    for ts, text in describe_video(a.video, Captioner(runs[0]), a.interval, a.beam):
        print(f"[{fmt_time(ts)}] {text}")


if __name__ == "__main__":
    main()
