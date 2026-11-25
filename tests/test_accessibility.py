import pytest

from src.accessibility import LANGUAGES, speak, translate_caption


class _Boom:
    def __init__(self, **kwargs):
        pass

    def translate(self, text):
        raise ConnectionError("offline")


class _Upper:
    def __init__(self, source, target):
        self.target = target

    def translate(self, text):
        return f"[{self.target}] {text.upper()}"


def test_required_languages_with_telugu_and_hindi_first():
    codes = list(LANGUAGES)
    assert codes[:2] == ["te", "hi"]
    assert {"en", "hi", "te", "ta", "kn", "mr", "fr", "es", "de"} <= set(codes)


def _backup_boom(text, lang):
    raise TimeoutError("offline")


def test_translation_fallback_returns_english_with_warning():
    t = translate_caption("a dog on a beach", "te", translator_cls=_Boom, backup=_backup_boom)
    assert t.text == "a dog on a beach"
    assert t.lang == "en" and not t.ok
    assert "Telugu" in t.warning


def test_translation_uses_backup_when_primary_fails():
    t = translate_caption("a dog", "hi", translator_cls=_Boom,
                          backup=lambda text, lang: f"<{lang}>{text}")
    assert t.ok and t.lang == "hi" and t.text == "<hi>a dog"


def test_translation_success_uses_translator():
    t = translate_caption("a cat", "hi", translator_cls=_Upper)
    assert t.ok and t.lang == "hi" and t.text == "[hi] A CAT"


def test_english_is_passthrough():
    t = translate_caption("a cat", "en", translator_cls=_Boom, backup=_backup_boom)
    assert t.ok and t.text == "a cat"


def test_unsupported_language_raises():
    with pytest.raises(ValueError):
        translate_caption("a cat", "xx")


def test_speak_returns_in_memory_bytes():
    class FakeTTS:
        def __init__(self, text, lang):
            self.payload = f"{lang}:{text}".encode()

        def write_to_fp(self, fp):
            fp.write(self.payload)

    assert speak("hello", "te", tts_cls=FakeTTS) == b"te:hello"
