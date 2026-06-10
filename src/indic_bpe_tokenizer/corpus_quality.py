

import re
import unicodedata


DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F
MIN_DEVANAGARI_RATIO = 0.40
MAX_FOREIGN_SCRIPT_RATIO = 0.20

METADATA_PATTERNS = (
    re.compile(r"^https?://", re.IGNORECASE),
    re.compile(r"^www\.", re.IGNORECASE),
    re.compile(r"^\{\{.*\}\}$"),
    re.compile(r"^\[\[.*\]\]$"),
)


def is_devanagari_character(char: str) -> bool:
    """Return whether a character belongs to the Devanagari Unicode block."""
    return DEVANAGARI_START <= ord(char) <= DEVANAGARI_END


def calculate_script_ratios(text: str) -> tuple[float, float]:
    """
    Return Devanagari and foreign-script ratios across alphabetic characters.

    Latin characters are allowed so code-mixed Hindi text is retained.
    """
    alphabetic_characters = [char for char in text if char.isalpha()]

    if not alphabetic_characters:
        return 0.0, 0.0

    devanagari_count = sum(
        1 for char in alphabetic_characters if is_devanagari_character(char)
    )
    foreign_script_count = sum(
        1
        for char in alphabetic_characters
        if not is_devanagari_character(char)
        and "LATIN" not in unicodedata.name(char, "")
    )
    total_count = len(alphabetic_characters)

    return (
        devanagari_count / total_count,
        foreign_script_count / total_count,
    )


def contains_corrupted_characters(text: str) -> bool:
    """Return whether text contains common corruption markers."""
    return "\ufffd" in text or "\x00" in text


def is_metadata_like_line(text: str) -> bool:
    """Return whether a paragraph resembles markup, URLs, or metadata."""
    stripped_text = text.strip()
    return any(pattern.search(stripped_text) for pattern in METADATA_PATTERNS)


def get_corpus_rejection_reason(text: str) -> str | None:
    """
    Return the reason a paragraph should be rejected, or None when it is usable.
    """
    if contains_corrupted_characters(text):
        return "corrupted"

    if is_metadata_like_line(text):
        return "metadata_like"

    devanagari_ratio, foreign_script_ratio = calculate_script_ratios(text)

    if (
        devanagari_ratio < MIN_DEVANAGARI_RATIO
        or foreign_script_ratio > MAX_FOREIGN_SCRIPT_RATIO
    ):
        return "foreign_script_heavy"

    return None