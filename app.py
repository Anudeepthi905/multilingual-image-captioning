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
from src.caption import available_runs
from src.uploads import IMAGE_TYPES, VIDEO_TYPES, check_image, check_video

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
img.thumb { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model loading
@st.cache_resource(show_spinner=False)
def load_captioner(run_dir: str):
    from src.caption import Captioner

    cap = Captioner(run_dir)
    _ = cap.extractor          # load InceptionV3 now, not on the first request
    return cap


@st.cache_data(show_spinner=False, max_entries=500)
def cached_translate(text, lang):
    return translate_caption(text, lang)


@st.cache_data(show_spinner=False, max_entries=200)
def cached_speech(text, lang):
    return speak(text, lang)


def img_tag(img: Image.Image, alt: str, css: str, side: int = 900) -> str:
    """An <img> with real alt text (st.image cannot set alt)."""
    buf = io.BytesIO()
    im = img.convert("RGB")
    im.thumbnail((side, side))
    im.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (f'<img class="{css}" src="data:image/jpeg;base64,{b64}" '
            f'alt="{html.escape(alt, quote=True)}">')


def announce(label: str, text: str, lang: str):
    """Caption in a live region so screen readers read it when it appears."""
    st.markdown(f'<div class="caption-box" role="status" aria-live="polite" '
                f'lang="{lang}" aria-label="{html.escape(label)}">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def sample_files():
    return sorted(p for p in config.SAMPLES_DIR.glob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


# ------------------------------------------------------------------ layout
st.title("Multilingual Visual Description")
st.markdown('<p class="skip-hint">Upload a photo or try a sample to hear a description in '
            'your language. Use <kbd>Tab</kbd> to move between controls.</p>',
            unsafe_allow_html=True)

runs = available_runs()

with st.sidebar:
    st.header("Settings")
    lang = st.selectbox("Language", list(LANGUAGES), format_func=LANGUAGES.get, index=0)
    decoding = st.radio("Decoding", ["Beam search (3)", "Greedy"], index=0,
                        help="Beam search is slower and often more fluent.")
    show_attention = st.toggle("Show attention heatmap", value=False)
    autoplay = st.toggle("Play audio automatically", value=False)
    run_dir = runs[0] if runs else None
    if len(runs) > 1:
        run_dir = st.selectbox("Model run", runs, format_func=lambda p: p.name)
beam = 3 if decoding.startswith("Beam") else 1

if not run_dir:
    st.warning("No trained model is available. Train one (`python -m src.train` or the Colab "
               "notebook) so that `checkpoints/` or `model/` holds a run.")
    st.stop()
with st.spinner("Loading the captioning model ..."):
    captioner = load_captioner(str(run_dir))

with st.expander("About this demo"):
    cfg = captioner.cfg
    st.markdown(
        f"""
**Academic project.** This began as a B.Tech final-year project at G. Narayanamma Institute
of Technology & Science, Hyderabad (2024), and was re-implemented as open source.

**How it works.** A frozen InceptionV3 network turns the photo into 64 region features. An
attention decoder writes the English caption one word at a time, looking at the regions
that matter for each word. The caption is translated with Google Translate and read aloud
with Google Text-to-Speech.

**This model** was trained on {cfg.get('train_images', 'an unknown number of')} images from
MS-COCO 2014 ({cfg.get('epochs', '?')} epochs, {cfg.get('vocab_size', '?')}-word vocabulary,
{cfg.get('backbone', 'inceptionv3')} encoder).

**Limitations.**
- Captions can be wrong, vague or repetitive; do not rely on them for anything important.
- COCO photos are mostly from Western Flickr users, so other scenes are described less well.
- Translation and speech quality vary by language and need an internet connection.
- The video tab describes still frames only; it does not understand motion.

Sample photos: MS-COCO / Flickr, CC BY 2.0 (see `assets/samples/ATTRIBUTION.md`).
[Source code](https://github.com/Anudeepthi905/multilingual-image-captioning)
""")


def describe_and_present(img: Image.Image, label: str):
    from src.caption import plot_attention

    with st.spinner("Describing the image ..."):
        text, words, attention = captioner.caption(img, beam=beam)
    if not text:
        text = "no description could be generated"
    col_img, col_text = st.columns([1, 1])
    with col_img:
        st.markdown(img_tag(img, f"{label}. Generated description: {text}", "described"),
                    unsafe_allow_html=True)
    with col_text:
        st.subheader("English description")
        announce("English description", text, "en")
        tr = translate_caption(text, "en")
        if lang != "en":
            with st.spinner("Translating ..."):
                tr = cached_translate(text, lang)
            st.subheader(f"Description in {LANGUAGES[lang]}")
            if tr.warning:
                st.warning(tr.warning)
            announce(f"Description in {LANGUAGES[lang]}", tr.text, tr.lang)
        st.subheader("Listen")
        try:
            with st.spinner("Generating speech ..."):
                audio = cached_speech(tr.text, tr.lang)
            st.audio(audio, format="audio/mp3", autoplay=autoplay)
        except Exception as exc:
            st.warning(f"Speech is unavailable right now ({type(exc).__name__}).")
    if show_attention and words:
        st.subheader("Where the model looked for each word")
        fig = plot_attention(img, words, attention, backbone=captioner.backbone)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf.getvalue(), caption="Attention heatmap: brighter areas influenced "
                                         "each generated word more.")


def image_tab():
    st.subheader("Try a sample")
    samples = sample_files()
    if samples:
        cols = st.columns(len(samples))
        for col, path in zip(cols, samples):
            name = path.stem.replace("_", " ")
            with col:
                st.markdown(img_tag(Image.open(path), f"Sample photo: {name}", "thumb", 320),
                            unsafe_allow_html=True)
                if st.button(f"Try this sample: {name}", key=f"sample-{path.stem}",
                             use_container_width=True):
                    st.session_state["picked"] = str(path)
                    # a new key clears any earlier upload so the sample is shown
                    st.session_state["upload_n"] = st.session_state.get("upload_n", 0) + 1

    st.subheader("Or upload your own photo")
    up = st.file_uploader(f"JPG, PNG or WebP, up to {config.MAX_UPLOAD_MB} MB",
                          type=list(IMAGE_TYPES),
                          key=f"upload-{st.session_state.get('upload_n', 0)}")
    if up is not None:
        checked = check_image(up.getvalue())
        if checked.error:
            st.error(checked.error)
            return
        describe_and_present(checked.image, "Uploaded image")
    elif st.session_state.get("picked"):
        path = Path(st.session_state["picked"])
        describe_and_present(Image.open(path).convert("RGB"),
                             f"Sample photo: {path.stem.replace('_', ' ')}")


def video_tab():
    from src.video.describe import describe_video, fmt_time

    st.info("Experimental: describes still keyframes; it does not understand motion.")
    vid = st.file_uploader(f"Upload a short video (MP4, MOV or AVI, up to "
                           f"{config.MAX_VIDEO_MB} MB)", type=list(VIDEO_TYPES))
    interval = st.slider("Seconds between keyframes", 1, 10, 3)
    if vid and st.button("Describe video"):
        error = check_video(vid.name, vid.size)
        if error:
            st.error(error)
            return
        with tempfile.NamedTemporaryFile(suffix=Path(vid.name).suffix, delete=False) as tmp:
            tmp.write(vid.getvalue())
        try:
            with st.spinner("Captioning keyframes ..."):
                entries = describe_video(tmp.name, captioner, interval, beam=1, max_frames=40)
        except ValueError as exc:
            st.error(f"The video could not be read: {exc}")
            return
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        texts = [(ts, cached_translate(t, lang).text if lang != "en" else t)
                 for ts, t in entries]
        st.markdown("\n".join(f"- {html.escape(f'[{fmt_time(ts)}] {t}')}" for ts, t in texts))
        try:
            st.audio(cached_speech(". ".join(t for _, t in entries), "en"), format="audio/mp3")
        except Exception:
            st.warning("Speech is unavailable right now.")


tab_image, tab_video = st.tabs(["Image", "Video (experimental)"])
with tab_image:
    image_tab()
with tab_video:
    video_tab()
