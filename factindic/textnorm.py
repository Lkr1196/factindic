"""Unicode-aware text utilities for Hindi, Bengali, English and code-mixed text.

Everything here is pure Python so it runs on CPU in any notebook.

Why this module exists: standard English tooling breaks on Indic text.
* Python's ``\\w`` does not match Indic vowel signs (matras), so ``re.findall(r"\\w+")``
  cuts Hindi words in half. We split on whitespace and punctuation instead.
* Hindi and Bengali end sentences with the danda (।), not a full stop.
* Digits can be written in Devanagari (०-९), Bengali (০-৯) or ASCII (0-9).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# --------------------------------------------------------------------------------------
# Scripts
# --------------------------------------------------------------------------------------
DEVANAGARI = (0x0900, 0x097F)
BENGALI = (0x0980, 0x09FF)
SCRIPT_RANGES = {"deva": DEVANAGARI, "beng": BENGALI}
LANG_SCRIPT = {"hi": "deva", "bn": "beng"}

_DIGIT_MAP = {}
for _i in range(10):
    _DIGIT_MAP[0x0966 + _i] = ord("0") + _i  # Devanagari digits
    _DIGIT_MAP[0x09E6 + _i] = ord("0") + _i  # Bengali digits

# Characters that are invisible or formatting-only. ZWJ/ZWNJ (U+200D/U+200C) are kept
# because they change how Indic conjuncts are rendered.
_STRIP_CHARS = dict.fromkeys(map(ord, "﻿­​⁠‎‏"), None)

_QUOTES = {
    ord("‘"): "'", ord("’"): "'", ord("‚"): "'", ord("‛"): "'",
    ord("“"): '"', ord("”"): '"', ord("„"): '"', ord("«"): '"', ord("»"): '"',
    ord("–"): "-", ord("—"): " - ", ord("−"): "-", ord(" "): " ",
}

DANDA = "।"        # ।
DOUBLE_DANDA = "॥"  # ॥


def char_script(ch: str) -> str:
    """Return 'deva', 'beng', 'latn', 'digit', 'mark', 'space', 'punct' or 'other'."""
    cp = ord(ch)
    if ch.isspace():
        return "space"
    if ch in (DANDA, DOUBLE_DANDA):
        return "punct"
    if 0x0966 <= cp <= 0x096F or 0x09E6 <= cp <= 0x09EF or "0" <= ch <= "9":
        return "digit"
    if DEVANAGARI[0] <= cp <= DEVANAGARI[1]:
        return "deva"
    if BENGALI[0] <= cp <= BENGALI[1]:
        return "beng"
    if ("a" <= ch.lower() <= "z") or (0x00C0 <= cp <= 0x024F):
        return "latn"
    cat = unicodedata.category(ch)
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    if cat.startswith("M"):
        return "mark"
    return "other"


def normalize_text(text: str) -> str:
    """NFC-normalise, unify quotes/dashes, drop invisible characters, collapse spaces."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_STRIP_CHARS).translate(_QUOTES)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def normalize_digits(text: str) -> str:
    """Map Devanagari and Bengali digits to ASCII digits."""
    return text.translate(_DIGIT_MAP)


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------------------
_ABBREV = {
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "inc", "ltd", "jr", "sr",
    "govt", "dept", "gen", "col", "lt", "capt", "sh", "smt", "u.s", "u.k", "e.g", "i.e",
}
_SENT_END = re.compile(r"([।॥!?]+|\.(?!\d))(['\")\]]*)\s+")


def split_sentences(text: str) -> list[str]:
    """Split Hindi/Bengali/English/code-mixed text into sentences.

    Splits after danda, '?', '!' and after '.' unless it is a decimal point,
    a single-letter initial ("A. P. J.") or a common abbreviation. Newlines always split.
    """
    text = normalize_text(text)
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        start = 0
        for m in _SENT_END.finditer(para):
            end_punct = m.group(1)
            if end_punct.startswith("."):
                before = para[start:m.start()].split()
                last = before[-1].lower().strip("(\"'") if before else ""
                if len(last) == 1 or last in _ABBREV:
                    continue
            piece = para[start:m.end()].strip()
            if piece:
                out.append(piece)
            start = m.end()
        tail = para[start:].strip()
        if tail:
            out.append(tail)
    # merge fragments that are only punctuation / too short to be a sentence
    merged: list[str] = []
    for s in out:
        if merged and len(tokenize(s)) == 0:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


# --------------------------------------------------------------------------------------
# Word tokenisation (for ROUGE, METEOR, lengths, CMI)
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _is_punct_char(ch: str) -> bool:
    if ch in (DANDA, DOUBLE_DANDA):
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P") or (cat.startswith("S") and ch not in "%$₹৳")


def _strip_punct(s: str) -> str:
    return "".join(" " if _is_punct_char(c) else c for c in s)


def tokenize(text: str, lower: bool = True, keep_numbers_sep: bool = True) -> list[str]:
    """Whitespace/punctuation tokenizer that keeps Indic words (with matras) intact.

    Digits are normalised to ASCII. Thousands separators and decimal points inside
    numbers are preserved ("1,50,000", "6.2") when ``keep_numbers_sep`` is True.
    """
    if not text:
        return []
    text = normalize_digits(normalize_text(text))
    if lower:
        text = text.lower()
    if keep_numbers_sep:
        # protect separators inside numbers
        text = re.sub(r"(?<=\d)([.,])(?=\d)", lambda m: "\u0001" if m.group(1) == "." else "\u0002", text)
    text = _strip_punct(text)
    toks = text.split()
    if keep_numbers_sep:
        toks = [t.replace("\u0001", ".").replace("\u0002", ",") for t in toks]
    return toks


def word_count(text: str) -> int:
    return len(tokenize(text))


def token_script(tok: str) -> str:
    """Majority script of the letters in a token: 'deva', 'beng', 'latn', 'num' or 'other'."""
    counts: dict[str, int] = {}
    has_digit = False
    for ch in tok:
        s = char_script(ch)
        if s == "digit":
            has_digit = True
        elif s in ("deva", "beng", "latn"):
            counts[s] = counts.get(s, 0) + 1
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    if has_digit:
        return "num"
    return "other"


def script_profile(text: str) -> dict:
    """Share of letters in each script (ignores digits/punctuation)."""
    counts = {"deva": 0, "beng": 0, "latn": 0}
    for ch in text or "":
        s = char_script(ch)
        if s in counts:
            counts[s] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def dominant_script(text: str) -> str:
    prof = script_profile(text)
    return max(prof.items(), key=lambda kv: kv[1])[0]


def truncate_words(text: str, max_words: int) -> str:
    """Cut a summary to at most ``max_words`` whitespace words (keeps original spelling)."""
    words = (text or "").split()
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()


def lead_sentences(sents: list[str], max_words: int, min_sents: int = 1) -> list[str]:
    """Take sentences from the start until ``max_words`` would be exceeded."""
    out, n = [], 0
    for s in sents:
        w = word_count(s)
        if out and n + w > max_words and len(out) >= min_sents:
            break
        out.append(s)
        n += w
    return out


# --------------------------------------------------------------------------------------
# Indic script conversion (Devanagari <-> Bengali), used by IndicBART which reads
# every language in Devanagari. The two Unicode blocks are aligned (offset 0x80).
# --------------------------------------------------------------------------------------
_BENG_FALLBACK = {"व": "ব"}  # व -> ব (Bengali has no separate va)


def deva_to_beng(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if ch in _BENG_FALLBACK:
            out.append(_BENG_FALLBACK[ch])
        elif DEVANAGARI[0] <= cp <= DEVANAGARI[1] and ch not in (DANDA, DOUBLE_DANDA):
            target = chr(cp + 0x80)
            try:
                unicodedata.name(target)
                out.append(target)
            except ValueError:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def beng_to_deva(text: str) -> str:
    out = []
    for ch in text:
        cp = ord(ch)
        if BENGALI[0] <= cp <= BENGALI[1]:
            target = chr(cp - 0x80)
            try:
                unicodedata.name(target)
                out.append(target)
            except ValueError:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)
