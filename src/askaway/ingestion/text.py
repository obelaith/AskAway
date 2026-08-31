import re

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean extraction artifacts without changing the actual wording."""
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def detect_language(text: str) -> str:
    """
    Roughly label a chunk as Arabic, English, or mixed.

    This is deliberately lightweight. We only need a useful corpus label here,
    not full language identification.
    """
    letters = [char for char in text if char.isalpha()]

    if not letters:
        return "unknown"

    arabic_letters = sum(bool(ARABIC_RE.match(char)) for char in letters)
    arabic_ratio = arabic_letters / len(letters)

    if arabic_ratio >= 0.75:
        return "ar"

    if arabic_ratio <= 0.25:
        return "en"

    return "mixed"