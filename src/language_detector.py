"""
Language Detection Module

Detects the primary language of a document from its OCR text.

Two detection strategies:
  1. Heuristic (always available): character-range analysis for major scripts +
     accent-frequency scoring for Latin-script languages.
  2. External (optional): delegates to the `langdetect` library when installed
     and `use_external_detector=True` in config.

The heuristic covers the eight most common document languages reliably
without any additional dependencies.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from data_models import OCRTextResult


# ── Script-level character ranges ─────────────────────────────────────────────

def _count_in_range(text: str, lo: int, hi: int) -> int:
    return sum(1 for ch in text if lo <= ord(ch) <= hi)


def _script_scores(text: str) -> Dict[str, int]:
    return {
        "cyrillic": _count_in_range(text, 0x0400, 0x04FF),
        "cjk":      _count_in_range(text, 0x4E00, 0x9FFF),
        "hangul":   _count_in_range(text, 0xAC00, 0xD7A3),
        "arabic":   _count_in_range(text, 0x0600, 0x06FF),
        "devanagari": _count_in_range(text, 0x0900, 0x097F),
        "thai":     _count_in_range(text, 0x0E00, 0x0E7F),
        "greek":    _count_in_range(text, 0x0370, 0x03FF),
        "hebrew":   _count_in_range(text, 0x0590, 0x05FF),
    }


# ── Latin-script language fingerprints ────────────────────────────────────────
# Maps Tesseract language code → regex patterns that are diagnostic for that
# language when found in Latin-script text.

_LATIN_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("deu", re.compile(r"[äöüÄÖÜß]")),
    ("fra", re.compile(r"[àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ]")),
    ("spa", re.compile(r"[áéíóúüñ¡¿ÁÉÍÓÚÜÑ]")),
    ("ita", re.compile(r"[àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]")),
    ("por", re.compile(r"[ãõâêôàáéíóúçÃÕÂÊÔÀÁÉÍÓÚÇ]")),
    ("nld", re.compile(r"(?<![a-z])ij(?![a-z])|IJ", re.IGNORECASE)),  # Dutch digraph
]


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class LanguageDetectorConfig:
    """
    Configuration for language detection.

    Attributes:
        use_external_detector: Try `langdetect` library first when True.
        min_sample_chars: Minimum character count before attempting detection.
        default_language: Returned when sample is too small or detection fails.
    """
    use_external_detector: bool = False
    min_sample_chars: int = 30
    default_language: str = "eng"


# ── Trace ──────────────────────────────────────────────────────────────────────

@dataclass
class LanguageDetectionTrace:
    """Diagnostic information from a single language detection call."""
    language_detected: str
    confidence: float
    method_used: str          # "heuristic" | "external" | "default"
    sample_chars: int
    processing_time_seconds: float = 0.0


# ── Detector ──────────────────────────────────────────────────────────────────

class LanguageDetector:
    """
    Detects the primary language from a list of OCR results.

    Usage::

        detector = LanguageDetector()
        lang, trace = detector.detect(ocr_results)
    """

    def __init__(self, config: Optional[LanguageDetectorConfig] = None) -> None:
        self.config = config or LanguageDetectorConfig()

    def detect(
        self,
        ocr_results: List[OCRTextResult],
    ) -> Tuple[str, LanguageDetectionTrace]:
        """
        Detect the primary language from OCR results.

        Args:
            ocr_results: List of OCR text results from the document.

        Returns:
            (language_code, trace)  where language_code is a Tesseract-style
            code (e.g. "eng", "fra", "chi_sim").
        """
        t0 = time.perf_counter()
        cfg = self.config

        text = " ".join(r.text for r in ocr_results if r.text)
        sample_chars = len(text)

        if sample_chars < cfg.min_sample_chars:
            elapsed = time.perf_counter() - t0
            return cfg.default_language, LanguageDetectionTrace(
                language_detected=cfg.default_language,
                confidence=0.0,
                method_used="default",
                sample_chars=sample_chars,
                processing_time_seconds=elapsed,
            )

        # External detector path
        if cfg.use_external_detector:
            result = self._try_external(text)
            if result is not None:
                lang, conf = result
                elapsed = time.perf_counter() - t0
                return lang, LanguageDetectionTrace(
                    language_detected=lang,
                    confidence=conf,
                    method_used="external",
                    sample_chars=sample_chars,
                    processing_time_seconds=elapsed,
                )

        # Heuristic path
        lang, conf = self._heuristic_detect(text)
        elapsed = time.perf_counter() - t0
        return lang, LanguageDetectionTrace(
            language_detected=lang,
            confidence=conf,
            method_used="heuristic",
            sample_chars=sample_chars,
            processing_time_seconds=elapsed,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _heuristic_detect(self, text: str) -> Tuple[str, float]:
        """Score text against known scripts and Latin-language patterns."""
        total = max(len(text), 1)

        # 1. Non-Latin scripts: winner-takes-all by character density
        scripts = _script_scores(text)
        best_script = max(scripts, key=scripts.__getitem__)
        best_count = scripts[best_script]
        if best_count / total > 0.15:
            lang = _SCRIPT_TO_LANG.get(best_script, self.config.default_language)
            confidence = min(1.0, best_count / total * 2)
            return lang, round(confidence, 3)

        # 2. Latin script: score diagnostic character patterns
        latin_scores: Dict[str, int] = {}
        for lang_code, pattern in _LATIN_PATTERNS:
            matches = len(pattern.findall(text))
            if matches:
                latin_scores[lang_code] = matches

        if latin_scores:
            winner = max(latin_scores, key=latin_scores.__getitem__)
            confidence = min(1.0, latin_scores[winner] / (total * 0.1))
            return winner, round(confidence, 3)

        return self.config.default_language, 0.5

    @staticmethod
    def _try_external(text: str) -> Optional[Tuple[str, float]]:
        """Attempt detection via `langdetect`; return None if unavailable."""
        try:
            from langdetect import detect_langs  # type: ignore
            results = detect_langs(text)
            if results:
                top = results[0]
                # langdetect uses ISO-639-1; map to Tesseract codes
                lang = _LANGDETECT_TO_TESSERACT.get(top.lang, top.lang)
                return lang, round(float(top.prob), 3)
        except Exception:
            pass
        return None


# ── Code maps ──────────────────────────────────────────────────────────────────

_SCRIPT_TO_LANG: Dict[str, str] = {
    "cyrillic":   "rus",
    "cjk":        "chi_sim",
    "hangul":     "kor",
    "arabic":     "ara",
    "devanagari": "hin",
    "thai":       "tha",
    "greek":      "ell",
    "hebrew":     "heb",
}

_LANGDETECT_TO_TESSERACT: Dict[str, str] = {
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "ru": "rus",
    "zh-cn": "chi_sim",
    "zh-tw": "chi_tra",
    "ja": "jpn",
    "ko": "kor",
    "ar": "ara",
    "hi": "hin",
    "th": "tha",
    "el": "ell",
    "he": "heb",
}
