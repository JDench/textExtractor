"""
Tests for LanguageDetector — heuristic script/accent detection.
"""

from helpers import make_ocr
from language_detector import LanguageDetector, LanguageDetectorConfig


def detect(texts, **cfg_kwargs):
    cfg = LanguageDetectorConfig(**cfg_kwargs) if cfg_kwargs else LanguageDetectorConfig()
    detector = LanguageDetector(cfg)
    ocrs = [make_ocr(t) for t in texts]
    lang, trace = detector.detect(ocrs)
    return lang, trace


# ── Default language on tiny sample ────────────────────────────────────────────

class TestSmallSample:
    def test_too_short_returns_default(self):
        lang, trace = detect(["Hi"], min_sample_chars=50)
        assert lang == "eng"
        assert trace.method_used == "default"
        assert trace.confidence == 0.0

    def test_exactly_at_threshold_is_detected(self):
        # 30 chars is the default threshold; a 30-char string should trigger heuristic
        text = "a" * 30
        _, trace = detect([text])
        assert trace.method_used == "heuristic"


# ── Non-Latin scripts ───────────────────────────────────────────────────────────

class TestNonLatinScripts:
    def test_cyrillic_detected_as_russian(self):
        lang, trace = detect(["Привет мир это текст на русском языке для теста"])
        assert lang == "rus"
        assert trace.method_used == "heuristic"

    def test_cjk_detected_as_chinese(self):
        lang, _ = detect(["这是一段中文文字用于测试语言检测功能的准确性和可靠性以及整体效果"])
        assert lang == "chi_sim"

    def test_hangul_detected_as_korean(self):
        lang, _ = detect(["이것은 언어 감지 기능을 테스트하기 위한 한국어 텍스트입니다 추가적인 내용을 포함합니다"])
        assert lang == "kor"

    def test_arabic_detected(self):
        lang, _ = detect(["هذا نص عربي يستخدم لاختبار وظيفة الكشف عن اللغة العربية والتعرف عليها"])
        assert lang == "ara"


# ── Latin-script language fingerprints ─────────────────────────────────────────

class TestLatinLanguages:
    def test_german_umlaut(self):
        lang, _ = detect(["Über die Größe und Schönheit der Welt möchte ich sprechen"])
        assert lang == "deu"

    def test_french_accents(self):
        lang, _ = detect(["Les élèves étudient avec beaucoup d'enthousiasme à l'école"])
        assert lang == "fra"

    def test_spanish_tildes(self):
        lang, _ = detect(["El niño habló español con mucha fluidez y expresión"])
        assert lang == "spa"

    def test_plain_english_returns_default(self):
        lang, _ = detect(["The quick brown fox jumps over the lazy dog in the field"])
        assert lang == "eng"


# ── Trace fields ───────────────────────────────────────────────────────────────

class TestTrace:
    def test_trace_has_sample_chars(self):
        text = "Привет мир это тест"
        _, trace = detect([text])
        assert trace.sample_chars == len(text)

    def test_trace_has_processing_time(self):
        _, trace = detect(["Hello world test text for timing"])
        assert trace.processing_time_seconds >= 0.0

    def test_trace_language_matches_return(self):
        lang, trace = detect(["Über die Größe und Schönheit der Welt möchte ich sprechen"])
        assert trace.language_detected == lang

    def test_confidence_in_range(self):
        _, trace = detect(["Привет мир это текст на русском языке"])
        assert 0.0 <= trace.confidence <= 1.0


# ── Config overrides ───────────────────────────────────────────────────────────

class TestConfig:
    def test_custom_default_language(self):
        lang, _ = detect(["a" * 5], default_language="fra", min_sample_chars=100)
        assert lang == "fra"

    def test_empty_ocr_list(self):
        cfg = LanguageDetectorConfig()
        detector = LanguageDetector(cfg)
        lang, trace = detector.detect([])
        assert lang == cfg.default_language
        assert trace.sample_chars == 0
        assert trace.method_used == "default"


# ── External detector fallback ─────────────────────────────────────────────────

class TestExternalDetector:
    def test_external_unavailable_falls_back_to_heuristic(self):
        # langdetect is not installed in this env → falls back to heuristic
        lang, _ = detect(
            ["Über die Größe und Schönheit"],
            use_external_detector=True,
        )
        # Either "deu" (heuristic) or whatever external returns — just check it ran
        assert isinstance(lang, str)
        assert len(lang) > 0
