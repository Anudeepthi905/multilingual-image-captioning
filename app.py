"""Streamlit app: describe an image in the user's language, as text and speech.

    streamlit run app.py
"""
import base64
import html
import io
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from src import config
from src.accessibility import LANGUAGES, speak, translate_caption

st.set_page_config(page_title="Multilingual Image Captioning", page_icon="🗣️",
                   layout="wide")

# Large type, high contrast, always-visible keyboard focus.
st.markdown("""
<style>
html, body, [class*="css"], .stMarkdown p, label, .stSelectbox, .stRadio { font-size: 1.2rem; }
h1 { font-size: 2.3rem !important; }
.caption-box { font-size: 1.8rem; line-height: 1.5; font-weight: 600; color: #000;
               background: #fff8c4; border: 3px solid #000; border-radius: 8px;
               padding: 0.8rem 1rem; margin: 0.5rem 0 1rem 0; }
@media (prefers-color-scheme: dark) {
  .caption-box { color: #fff; background: #1d2a44; border-color: #fff; }
}
*:focus-visible { outline: 4px solid #ff9900 !important; outline-offset: 2px !important; }
.skip-hint { font-size: 1rem; opacity: 0.85; }
img.described { max-width: 100%; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model loading
@st.cache_resource(show_spinner="Loading the captioning model ...")
def load_captioner(run_dir: str):
    from src.caption import Captioner

    return Captioner(run_dir)


def available_runs():
    from src.caption import download_release_checkpoint, list_runs

    runs = list_runs()
    if not runs and config.RELEASE_CHECKPOINT_URL:
        try:
            with st.spinner("Downloading the trained checkpoint from the GitHub release ..."):
                runs = download_release_checkpoint()
        except Exception as exc:
            st.error(f"No local checkpoint and the release download failed: {exc}")
    return runs


@st.cache_data(show_spinner=False)
def cached_translate(text, lang):
    return translate_caption(text, lang)


@st.cache_data(show_spinner=False)
def cached_speech(text, lang):
    return speak(text, lang)


def show_image(img: Image.Image, alt: str):
    """Render an <img> with real alt text (st.image cannot set alt)."""
    buf = io.BytesIO()
    im = img.convert("RGB")
    im.thumbnail((900, 900))
    im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<img class="described" src="data:image/jpeg;base64,{b64}" '
                f'alt="{html.escape(alt, quote=True)}">', unsafe_allow_html=True)


def announce(label: str, text: str, lang: str):
    """Caption in a live region so screen readers read it when it appears."""
    st.markdown(f'<div class="caption-box" role="status" aria-live="polite" '
                f'lang="{lang}" aria-label="{html.escape(label)}">{html.escape(text)}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ layout
st.title("Multilingual Visual Description")
st.markdown('<p class="skip-hint">Upload a photo to hear a description in your language. '
            'Use <kbd>Tab</kbd> to move between controls.</p>', unsafe_allow_html=True)

runs = available_runs()
with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language", list(LANGUAGES), format_func=LANGUAGES.get, index=0)
    decoding = st.radio("Decoding", ["Beam search (3)", "Greedy"], index=0)
    show_attention = st.toggle("Show attention heatmap", value=False)
    autoplay = st.toggle("Play audio automatically", value=False)
    run_dir = None
    if runs:
        run_dir = st.selectbox("Model run", runs, format_func=lambda p: p.name)
beam = 3 if decoding.startswith("Beam") else 1

if not run_dir:
    st.warning("No trained model found. Train one (`python -m src.train`) or set "
               "MIC_CHECKPOINT_URL to a release asset.")
    st.stop()
captioner = load_captioner(str(run_dir))


def describe_and_present(img: Image.Image):
    from src.caption import plot_attention

    with st.spinner("Describing the image ..."):
        text, words, attention = captioner.caption(img, beam=beam)
    if not text:
        text = "no description could be generated"
    col_img, col_text = st.columns([1, 1])
    with col_img:
        show_image(img, alt=f"Image. Generated description: {text}")
    with col_text:
        st.subheader("English description")
        announce("English description", text, "en")
        if lang != "en":
            tr = cached_translate(text, lang)
            st.subheader(f"Description in {LANGUAGES[lang]}")
            if tr.warning:
                st.warning(tr.warning)
            announce(f"Description in {LANGUAGES[lang]}", tr.text, tr.lang)
        else:
            tr = translate_caption(text, "en")
        st.subheader("Listen")
        try:
            st.audio(cached_speech(tr.text, tr.lang), format="audio/mp3", autoplay=autoplay)
        except Exception as exc:
            st.warning(f"Speech is unavailable (gTTS needs internet): {type(exc).__name__}")
    if show_attention and words:
        st.subheader("Where the model looked for each word")
        fig = plot_attention(img, words, attention, backbone=captioner.backbone)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf.getvalue(), caption="Attention heatmap: brighter areas influenced "
                                         "each generated word more.")


def video_tab():
    from src.video.describe import describe_video, fmt_time

    st.info("Experimental: describes still keyframes; it does not understand motion.")
    vid = st.file_uploader("Upload a short video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
    interval = st.slider("Seconds between keyframes", 1, 10, 3)
    if vid and st.button("Describe video"):
        with tempfile.NamedTemporaryFile(suffix=Path(vid.name).suffix, delete=False) as tmp:
            tmp.write(vid.getvalue())
        with st.spinner("Captioning keyframes ..."):
            entries = describe_video(tmp.name, captioner, interval, beam=1, max_frames=40)
        Path(tmp.name).unlink(missing_ok=True)
        lines = [f"[{fmt_time(ts)}] {text}" for ts, text in entries]
        if lang != "en":
            lines = [f"[{fmt_time(ts)}] {cached_translate(text, lang).text}"
                     for ts, text in entries]
        st.markdown("\n".join(f"- {html.escape(line)}" for line in lines))
        try:
            st.audio(cached_speech(". ".join(t for _, t in entries), "en"), format="audio/mp3")
        except Exception:
            st.warning("Speech is unavailable (gTTS needs internet).")


tab_image, tab_video = st.tabs(["Image", "Video (experimental)"])
with tab_image:
    source = st.radio("Image source", ["Upload", "Sample image"], horizontal=True)
    image = None
    if source == "Upload":
        up = st.file_uploader("Drag and drop an image, or browse (JPG, PNG)",
                              type=["jpg", "jpeg", "png", "webp"])
        if up:
            image = Image.open(up)
    else:
        samples = sorted(p for p in config.SAMPLES_DIR.glob("*")
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if samples:
            pick = st.selectbox("Sample", samples, format_func=lambda p: p.stem.replace("_", " "))
            image = Image.open(pick)
        else:
            st.info("No sample images found in assets/samples/.")
    if image is not None:
        describe_and_present(image)
with tab_video:
    video_tab()
