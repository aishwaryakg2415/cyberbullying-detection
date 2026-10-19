"""Text normalisation for code-mixed social-media text.

Design choices (documented in RESEARCH_METHODOLOGY.md):
* Case, punctuation and code-switching are preserved; they carry signal for
  abuse and sarcasm (e.g. ALL-CAPS, "!!!", Hindi/English alternation).
* Unicode emoji are converted to their CLDR names (":face_with_tears_of_joy:")
  because MuRIL/XLM-R vocabularies map many emoji to [UNK]. This also matches
  the named-emoji convention already present in the user-supplied sarcasm files.
* URLs and @-mentions are replaced by placeholders (privacy + sparsity).
* Character elongations are capped at three repeats ("yaaaaar" -> "yaaar"),
  keeping the elongation signal while reducing sparsity.
* No transliteration and no lower-casing: Devanagari and Roman Hindi are kept
  as written.
"""
from __future__ import annotations

import re
import unicodedata

try:
    import emoji as _emoji
except ImportError:  # pragma: no cover
    _emoji = None

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
ELONG_RE = re.compile(r"(.)\1{3,}")
WS_RE = re.compile(r"\s+")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
# Named emoji tokens as they appear in the user-supplied sarcasm files.
NAMED_EMOJI_RE = re.compile(r"\b[a-z]+(?:_[a-z]+){1,}\b")


def clean_text(text: str, demojize: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = t.replace("​", " ").replace("﻿", "")
    t = URL_RE.sub(" URL ", t)
    t = MENTION_RE.sub(" @user ", t)
    if demojize and _emoji is not None:
        t = _emoji.demojize(t, delimiters=(" :", ": "))
    t = ELONG_RE.sub(r"\1\1\1", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def normalised_key(text: str) -> str:
    """Aggressive key for duplicate detection (not used as model input)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = re.sub(r"(.)\1+", r"\1", t)          # collapse all repeats
    t = re.sub(r"[^\wऀ-ॿ]+", " ", t)
    return WS_RE.sub(" ", t).strip()


def script_of(text: str) -> str:
    """'roman', 'devanagari', 'mixed' or 'other' based on letter scripts present."""
    has_dev = bool(DEVANAGARI_RE.search(text or ""))
    has_lat = bool(LATIN_RE.search(text or ""))
    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_lat:
        return "roman"
    return "other"


def has_emoji(text: str) -> bool:
    if _emoji is not None and _emoji.emoji_count(text or "") > 0:
        return True
    return ":" in (text or "") and bool(re.search(r":[a-z_]+:", text))
