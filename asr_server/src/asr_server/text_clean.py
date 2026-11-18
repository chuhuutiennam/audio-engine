"""Clean Qwen3-ASR output metadata WITHOUT editing its internal decode state.

In Qwen3-ASR forced-language mode, parse_asr_output(raw, user_language=...)
assumes text-only generation. Some model outputs nevertheless contain repeated
`language: Vietnamese<asr_text>` wrappers between decoded fragments.
"""
import re

# Qwen emits both `language Vietnamese<asr_text>` and, in the reported output,
# `language: Vietnamese<asr_text>`. Handle both and common chat token spellings.
# Match a marker only when the distinctive <asr_text> delimiter is present.
_WRAPPER = re.compile(
    r"(?i)language\s*:?\s*(?:[A-Za-z][A-Za-z _-]{0,50}|none)\s*"
    r"(?:<asr_text>|<\|asr_text\|>)"
)
_BARE_TAG = re.compile(r"(?i)<asr_text>|<\|asr_text\|>")


def strip_asr_markers(text: str) -> str:
    """Remove model protocol markers in a full utterance (also mid-string).

    Do not alter digits, punctuation, Vietnamese diacritics, or user words.
    Normalize whitespace created by marker deletion to a single space.
    """
    if not text:
        return ""
    cleaned = _WRAPPER.sub(" ", text)
    cleaned = _BARE_TAG.sub(" ", cleaned)
    return " ".join(cleaned.split())


_LANGUAGE = re.compile(
    r"(?i)language\s*:?\s*([A-Za-z][A-Za-z _-]{0,50}?)\s*(?:<asr_text>|<\|asr_text\|>)"
)


def split_asr_output(raw: str):
    """Split raw Qwen3-ASR output into (language, text).

    Qwen answers `language Vietnamese<asr_text>xin chao`. `language` is the name in
    the FIRST marker, or None when there is no marker or the model wrote `None`
    (nothing recognised). `text` has every marker removed.
    """
    raw = raw or ""
    match = _LANGUAGE.search(raw)
    language = match.group(1).strip() if match else None
    if language and language.lower() == "none":
        language = None
    return language, strip_asr_markers(raw)
