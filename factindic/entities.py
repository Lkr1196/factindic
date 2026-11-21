"""Numbers, dates and named entities, with cross-script matching.

The factuality checks need to know whether a number, date or name in a summary also appears
in the source. Three things make this harder for Indic and code-mixed text:

1. Numbers come in three digit systems and with Indian scale words ("5 लाख" = 500000,
   "2.5 crore" = 25000000). We compare *values*, not strings.
2. The same name can be written in Devanagari, Bengali or Roman letters ("दिल्ली" / "Delhi").
   We map every name to a loose consonant skeleton ("dl") so the spellings meet.
3. Bengali glues case endings to names ("ভারতের" = of India). Prefix matching on the
   skeleton tolerates these endings.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from .textnorm import normalize_digits, normalize_text, token_script

# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------
SCALE_WORDS = {
    # English / Roman
    "hundred": 1e2, "thousand": 1e3, "hazar": 1e3, "hazaar": 1e3, "hajar": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "lakh-": 1e5,
    "crore": 1e7, "crores": 1e7, "karod": 1e7, "karor": 1e7, "koti": 1e7,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9,
    "trillion": 1e12,
    # Hindi
    "सौ": 1e2, "हज़ार": 1e3, "हजार": 1e3, "लाख": 1e5, "करोड़": 1e7, "करोड": 1e7,
    "मिलियन": 1e6, "बिलियन": 1e9, "अरब": 1e9, "खरब": 1e11, "ट्रिलियन": 1e12,
    # Bengali
    "শো": 1e2, "শত": 1e2, "হাজার": 1e3, "লাখ": 1e5, "লক্ষ": 1e5, "কোটি": 1e7,
    "মিলিয়ন": 1e6, "বিলিয়ন": 1e9, "ট্রিলিয়ন": 1e12,
}
PERCENT_WORDS = {"%", "percent", "per", "pc", "प्रतिशत", "फ़ीसदी", "फीसदी", "फ़ीसद", "শতাংশ", "শতকরা", "pratishat"}

_NUM_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class NumMention:
    value: float
    raw: str
    kind: str      # 'year' | 'percent' | 'number'
    decimals: int


def _scale_of(word: str) -> float | None:
    w = word.lower().strip(".,;:()[]'\"-।")
    if not w:
        return None
    if w in SCALE_WORDS:
        return SCALE_WORDS[w]
    # Bengali/Hindi suffixed forms: "হাজারের", "লাখের", "करोड़ों"
    for k, v in SCALE_WORDS.items():
        if len(k) >= 3 and w.startswith(k) and len(w) - len(k) <= 3:
            return v
    return None


def extract_numbers(text: str) -> list[NumMention]:
    """All numbers written with digits (any digit system), with Indian/English scale words applied."""
    if not text:
        return []
    t = normalize_digits(normalize_text(text))
    out: list[NumMention] = []
    for m in _NUM_RE.finditer(t):
        raw = m.group(1)
        clean = raw.replace(",", "")
        try:
            val = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        rest = t[m.end():m.end() + 24]
        nxt = rest.split()
        kind = "number"
        scale = None
        if nxt:
            first = nxt[0]
            if rest.startswith("%") or first.lower().strip(".,") in PERCENT_WORDS:
                kind = "percent"
            else:
                scale = _scale_of(first)
                if scale is None and first.startswith("-") and len(first) > 1:
                    scale = _scale_of(first[1:])
        if scale:
            val *= scale
        elif kind == "number" and decimals == 0 and "," not in raw and 1900 <= val <= 2100:
            kind = "year"
        out.append(NumMention(val, raw, kind, decimals))
    return out


def number_values(text: str) -> list[float]:
    return [n.value for n in extract_numbers(text)]


def same_number(a: NumMention, b: NumMention) -> bool:
    """True if summary number ``a`` is the source number ``b`` (allowing rounding of ``b``)."""
    if abs(a.value - b.value) <= 1e-9 * max(1.0, abs(b.value)):
        return True
    if a.decimals < b.decimals:
        scale = 10 ** a.decimals
        # compare after rounding the source number to the summary's precision
        if abs(round(b.value * scale) / scale - a.value) <= 1e-9 * max(1.0, abs(a.value)):
            return True
    # scale words: "1.5 million" vs "15 लाख" are the same value, already handled by value
    return False


def numbers_supported(summary: str, source: str) -> tuple[list[NumMention], list[NumMention]]:
    """Split the summary's numbers into (supported, unsupported) with respect to the source."""
    src = extract_numbers(source)
    sup, unsup = [], []
    for n in extract_numbers(summary):
        (sup if any(same_number(n, s) for s in src) else unsup).append(n)
    return sup, unsup


def numbers_multiset_equal(a: str, b: str) -> bool:
    ca = Counter(round(v, 6) for v in number_values(a))
    cb = Counter(round(v, 6) for v in number_values(b))
    return ca == cb


# --------------------------------------------------------------------------------------
# Dates (day + month). Years are handled as numbers of kind 'year'.
# --------------------------------------------------------------------------------------
_MONTHS = {
    1: ["january", "jan", "janvari", "जनवरी", "জানুয়ারি", "জানুয়ারী"],
    2: ["february", "feb", "farvari", "फ़रवरी", "फरवरी", "ফেব্রুয়ারি", "ফেব্রুয়ারী"],
    3: ["march", "mar", "मार्च", "মার্চ"],
    4: ["april", "apr", "अप्रैल", "अप्रेल", "এপ্রিল"],
    5: ["may", "मई", "মে"],
    6: ["june", "jun", "जून", "জুন"],
    7: ["july", "jul", "जुलाई", "জুলাই"],
    8: ["august", "aug", "agast", "अगस्त", "আগস্ট", "অগাস্ট", "অগস্ট"],
    9: ["september", "sep", "sept", "sitambar", "सितंबर", "सितम्बर", "সেপ্টেম্বর"],
    10: ["october", "oct", "aktubar", "अक्टूबर", "अक्तूबर", "অক্টোবর"],
    11: ["november", "nov", "navambar", "नवंबर", "नवम्बर", "নভেম্বর"],
    12: ["december", "dec", "disambar", "दिसंबर", "दिसम्बर", "ডিসেম্বর"],
}
_MONTH_LOOKUP = {w: m for m, ws in _MONTHS.items() for w in ws}
_AMBIGUOUS_MONTHS = {"may", "mar", "jan", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "मई", "মে"}


def _month_of(tok: str) -> int | None:
    w = tok.lower().strip(".,;:()[]'\"-।")
    if w in _MONTH_LOOKUP:
        return _MONTH_LOOKUP[w]
    for k, m in _MONTH_LOOKUP.items():  # Bengali suffixes: "মার্চের", "জুনে"
        if len(k) >= 3 and w.startswith(k) and len(w) - len(k) <= 3 and token_script(w) != "latn":
            return m
    return None


def _day_of(tok: str) -> int | None:
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th|ই|শে|লা|রা|ঠা|ই|तारीख|তারিখ)?[,.]?$", normalize_digits(tok))
    if m:
        d = int(m.group(1))
        if 1 <= d <= 31:
            return d
    return None


def extract_dates(text: str) -> list[tuple[int | None, int]]:
    """(day, month) pairs; day is None when only the month is mentioned."""
    toks = normalize_digits(normalize_text(text)).split()
    out = []
    for i, tok in enumerate(toks):
        m = _month_of(tok)
        if m is None:
            continue
        day = None
        if i > 0:
            day = _day_of(toks[i - 1])
        if day is None and i + 1 < len(toks):
            day = _day_of(toks[i + 1])
        bare = tok.lower().strip(".,;:()[]'\"-।")
        if day is None and bare in _AMBIGUOUS_MONTHS:
            continue  # "may"/"मई" without a day is too ambiguous
        out.append((day, m))
    return out


def dates_supported(summary: str, source: str) -> tuple[list, list]:
    src = extract_dates(source)
    src_full = set(src)
    src_months = {m for _, m in src}
    sup, unsup = [], []
    for d, m in extract_dates(summary):
        ok = ((d, m) in src_full) if d is not None else (m in src_months)
        (sup if ok else unsup).append((d, m))
    return sup, unsup


# --------------------------------------------------------------------------------------
# Cross-script name keys
# --------------------------------------------------------------------------------------
try:
    from indic_transliteration import sanscript as _sanscript
except Exception:  # pragma: no cover - optional dependency
    _sanscript = None

_HONORIFICS = {
    "shri", "shree", "sri", "smt", "mr", "mrs", "ms", "dr", "ji", "jee", "prof", "sir", "saheb", "sahib",
    "श्री", "श्रीमती", "जी", "डॉ", "डॉक्टर", "मिस्टर", "साहब", "শ্রী", "শ্রীমতী", "ডঃ", "ডা", "ডক্টর", "সাহেব", "বাবু",
}
_VOWELS = set("aeiou")


def _translit(tok: str, script: str) -> str:
    if _sanscript is None:
        return tok
    src = _sanscript.DEVANAGARI if script == "deva" else _sanscript.BENGALI
    try:
        return _sanscript.transliterate(tok, src, _sanscript.ITRANS)
    except Exception:
        return tok


def _loosen_itrans(r: str) -> str:
    # ITRANS is case-sensitive; resolve the special marks before lower-casing
    for a, b in ((".N", ""), ("M", ""), ("H", ""), (".a", ""), ("~N", "n"), ("~n", "n"),
                 ("R^i", "ri"), ("R^I", "ri"), ("L^i", "li"), (".D", "d"), ("JN", "gy"), ("GY", "gy"),
                 ("^", ""), ("x", "ks"), (".", "")):
        r = r.replace(a, b)
    return r


@lru_cache(maxsize=200_000)
def token_key(tok: str) -> str:
    """Loose consonant skeleton of one word, comparable across Devanagari/Bengali/Roman."""
    tok = normalize_digits(normalize_text(tok))
    tok = tok.strip(".,;:!?()[]{}'\"-।॥“”‘’")
    if not tok:
        return ""
    if re.fullmatch(r"[\d.,]+", tok):
        return tok.replace(",", "")
    script = token_script(tok)
    if script in ("deva", "beng"):
        r = _loosen_itrans(_translit(tok, script))
    else:
        r = tok.split("-")[0]
        r = re.sub(r"'s$", "", r)
    r = r.lower()
    r = re.sub(r"[^a-z]", "", r)
    if not r:
        return ""
    # digraphs -> single letters
    for a, b in (("ksh", "ks"), ("chh", "C"), ("ch", "C"), ("sh", "s"), ("kh", "k"), ("gh", "g"),
                 ("th", "t"), ("dh", "d"), ("ph", "f"), ("bh", "b"), ("jh", "j"), ("ck", "k")):
        r = r.replace(a, b)
    r = r.replace("c", "k").replace("C", "c").replace("q", "k").replace("x", "ks").replace("z", "j")
    r = r.replace("w", "b").replace("v", "b")
    # aspiration: drop 'h' after a consonant
    r = re.sub(r"(?<=[bcdfgjklmnpqrstvxyz])h", "", r)
    first = "V" if r[0] in _VOWELS else r[0]
    body = [c for c in r[1:] if c not in _VOWELS]
    sk = first + "".join(body)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    # nasal before a consonant (anusvara is dropped on the Indic side)
    sk = sk[0] + re.sub(r"[nm](?=[bcdfgjklpqrstvy])", "", sk[1:])
    return sk


def keys_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 2 and long_.startswith(short) and len(long_) - len(short) <= (1 if len(short) == 2 else 2):
        return True
    if len(short) >= 4 and short[0] == long_[0]:
        return SequenceMatcher(None, a, b).ratio() >= 0.8
    return False


def text_keys(text: str) -> list[str]:
    return [k for k in (token_key(t) for t in normalize_text(text).split()) if k]


class KeyIndex:
    """Fast lookup of word keys in a document (prefix-bucketed)."""

    def __init__(self, text: str):
        self.keys = set(text_keys(text))
        self.buckets: dict[str, list[str]] = {}
        for k in self.keys:
            self.buckets.setdefault(k[:1], []).append(k)

    def has(self, key: str) -> bool:
        if key in self.keys:
            return True
        return any(keys_match(key, k) for k in self.buckets.get(key[:1], ()))


def entity_tokens(entity: str) -> list[str]:
    toks = [t for t in normalize_text(entity).split() if t.lower().strip(".") not in _HONORIFICS]
    keys = [token_key(t) for t in toks]
    return [k for k in keys if len(k) >= 2 and not re.fullmatch(r"[\d.]+", k)]


def entity_in(entity: str, index: KeyIndex) -> bool:
    """An entity is supported if every content word of it occurs (fuzzily) in the indexed text."""
    keys = entity_tokens(entity)
    if not keys:
        return True
    return all(index.has(k) for k in keys)


def same_entity(a: str, b: str) -> bool:
    ka, kb = entity_tokens(a), entity_tokens(b)
    if not ka or not kb:
        return False
    small, big = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    return all(any(keys_match(x, y) for y in big) for x in small)


def dedupe_entities(ents: list[str]) -> list[str]:
    out: list[str] = []
    for e in ents:
        if not entity_tokens(e):
            continue
        if any(same_entity(e, o) for o in out):
            continue
        out.append(e)
    return out


# --------------------------------------------------------------------------------------
# Named-entity recognition
# --------------------------------------------------------------------------------------
_PUNCT_SPLIT = re.compile(r"([।॥.,;:!?()\[\]{}\"“”‘’'])")


def ner_words(text: str) -> list[str]:
    """Whitespace words with punctuation split off (the format IndicNER was trained on)."""
    words = []
    for chunk in normalize_text(text).split():
        for piece in _PUNCT_SPLIT.split(chunk):
            if piece:
                words.append(piece)
    return words


def latin_entities(text: str) -> list[str]:
    """Heuristic entities for Latin-script tokens: capitalised runs and acronyms."""
    try:
        from wordfreq import zipf_frequency
    except Exception:  # pragma: no cover
        zipf_frequency = None
    words = ner_words(text)
    ents, cur = [], []
    for i, w in enumerate(words):
        is_cap = token_script(w) == "latn" and (w[:1].isupper() or (w.isupper() and 2 <= len(w) <= 6))
        if is_cap:
            sentence_start = i == 0 or words[i - 1] in {".", "!", "?", "।"}
            common = zipf_frequency is not None and zipf_frequency(w.lower(), "en") >= 4.5 and not w.isupper()
            if sentence_start and common:
                is_cap = False
        if is_cap:
            cur.append(w)
        else:
            if cur:
                ents.append(" ".join(cur))
            cur = []
    if cur:
        ents.append(" ".join(cur))
    return ents


class NERTagger:
    """Wrapper around ai4bharat/IndicNER (word-level BIO tags) + Latin heuristics."""

    def __init__(self, model_name: str = "ai4bharat/IndicNER", device: str | None = None, batch_size: int = 32):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device.startswith("cuda"):
            self.model.half()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch_size = batch_size

    def _encode(self, batch_words: list[list[str]]):
        """Tokenise pre-split words; returns (tensors, word_ids per row). Works with slow tokenizers too."""
        if getattr(self.tok, "is_fast", False):
            enc = self.tok(batch_words, is_split_into_words=True, truncation=True, max_length=256,
                           padding=True, return_tensors="pt")
            return dict(enc), [enc.word_ids(b) for b in range(len(batch_words))]
        torch = self.torch
        rows, wids = [], []
        for words in batch_words:
            ids, wid = [self.tok.cls_token_id], [None]
            for i, w in enumerate(words):
                pieces = self.tok.convert_tokens_to_ids(self.tok.tokenize(w)) or [self.tok.unk_token_id]
                ids += pieces
                wid += [i] * len(pieces)
            ids, wid = ids[:255] + [self.tok.sep_token_id], wid[:255] + [None]
            rows.append(ids)
            wids.append(wid)
        L = max(len(r) for r in rows)
        pad = self.tok.pad_token_id or 0
        input_ids = torch.tensor([r + [pad] * (L - len(r)) for r in rows])
        attn = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows])
        return {"input_ids": input_ids, "attention_mask": attn}, wids

    def _tag_words(self, batch_words: list[list[str]]) -> list[list[str]]:
        enc, word_ids = self._encode(batch_words)
        with self.torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        pred = logits.argmax(-1).cpu().tolist()
        out = []
        for b, words in enumerate(batch_words):
            labels = ["O"] * len(words)
            seen = set()
            for pos, wid in enumerate(word_ids[b]):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                labels[wid] = self.id2label.get(pred[b][pos], "O")
            out.append(labels)
        return out

    def __call__(self, texts: list[str]) -> list[list[str]]:
        from .textnorm import split_sentences

        # tag sentence by sentence (short inputs), then regroup per text
        sent_index, sents = [], []
        for i, t in enumerate(texts):
            for s in split_sentences(t) or [t]:
                sent_index.append(i)
                sents.append(ner_words(s))
        tags: list[list[str]] = []
        order = sorted(range(len(sents)), key=lambda j: len(sents[j]))
        tagged: dict[int, list[str]] = {}
        for k in range(0, len(order), self.batch_size):
            idx = order[k:k + self.batch_size]
            batch = [sents[j] if sents[j] else ["."] for j in idx]
            for j, lab in zip(idx, self._tag_words(batch)):
                tagged[j] = lab
        results: list[list[str]] = [[] for _ in texts]
        for j, words in enumerate(sents):
            labs = tagged.get(j, ["O"] * len(words))
            results[sent_index[j]].extend(bio_spans(words, labs))
        for i, t in enumerate(texts):
            results[i] = dedupe_entities(results[i] + latin_entities(t))
        del tags
        return results


def bio_spans(words: list[str], labels: list[str]) -> list[str]:
    spans, cur, cur_type = [], [], None
    for w, lab in zip(words, labels):
        lab = lab or "O"
        if lab == "O" or "-" not in lab:
            if cur:
                spans.append(" ".join(cur))
            cur, cur_type = [], None
            continue
        bio, typ = lab.split("-", 1)
        if bio == "B" or typ != cur_type:
            if cur:
                spans.append(" ".join(cur))
            cur, cur_type = [w], typ
        else:
            cur.append(w)
    if cur:
        spans.append(" ".join(cur))
    return [s for s in spans if s.strip(".,;:!?()[]{}'\"।")]


class MockNER:
    """Stand-in NER for CPU dry runs: gazetteer lookup + Latin heuristics."""

    def __init__(self, gazetteer: list[str] | None = None):
        from .mock_data import GAZETTEER

        self.gaz = sorted(set(gazetteer or GAZETTEER), key=len, reverse=True)

    def __call__(self, texts: list[str]) -> list[list[str]]:
        out = []
        for t in texts:
            norm = normalize_text(t)
            found = [g for g in self.gaz if g in norm]
            out.append(dedupe_entities(found + latin_entities(t)))
        return out
