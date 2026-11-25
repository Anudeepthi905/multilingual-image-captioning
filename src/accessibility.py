"""Caption translation (deep-translator / Google) and text-to-speech (gTTS)."""
import io
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Telugu and Hindi first: the primary audience of the original project.
LANGUAGES = {
    "te": "Telugu (తెలుగు)",
    "hi": "Hindi (हिन्दी)",
    "en": "English",
    "ta": "Tamil (தமிழ்)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "mr": "Marathi (मराठी)",
    "fr": "French (Français)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
}


@dataclass
class Translation:
    text: str
    lang: str            # language the text is actually in
    ok: bool = True
    warning: Optional[str] = None


def google_gtx_translate(text: str, lang: str, timeout: float = 10) -> str:
    """Backup path: Google's public `gtx` JSON endpoint (used when deep-translator's
    scraped endpoint is blocked or rate-limited for this IP)."""
    import requests

    r = requests.get("https://translate.googleapis.com/translate_a/single",
                     params={"client": "gtx", "sl": "en", "tl": lang, "dt": "t", "q": text},
                     timeout=timeout)
    r.raise_for_status()
    return "".join(part[0] for part in r.json()[0] if part and part[0])


def translate_caption(text: str, lang: str, translator_cls=None,
                      backup=google_gtx_translate) -> Translation:
    """Translate an English caption with deep-translator (GoogleTranslator).

    If that fails, try `backup(text, lang)`; if that fails too, return the English
    caption with a warning so the user still gets a description.
    """
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported language {lang!r}; choose from {list(LANGUAGES)}")
    if lang == "en" or not text.strip():
        return Translation(text, "en")
    try:
        if translator_cls is None:
            from deep_translator import GoogleTranslator as translator_cls
        out = translator_cls(source="en", target=lang).translate(text)
        if not out:
            raise RuntimeError("empty translation")
        return Translation(out, lang)
    except Exception as exc:  # network errors, rate limits, API changes
        log.warning("deep-translator to %s failed: %s", lang, exc)
        if backup is not None:
            try:
                out = backup(text, lang)
                if out:
                    return Translation(out, lang)
            except Exception as exc2:
                log.warning("Backup translation to %s failed: %s", lang, exc2)
        return Translation(text, "en", ok=False,
                           warning=f"Translation to {LANGUAGES[lang]} is unavailable right "
                                   f"now ({type(exc).__name__}). Showing the English caption.")


def speak(text: str, lang: str = "en", tts_cls=None) -> bytes:
    """Synthesize speech with gTTS and return MP3 bytes (kept in memory)."""
    if tts_cls is None:
        from gtts import gTTS as tts_cls
    buf = io.BytesIO()
    tts_cls(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()
