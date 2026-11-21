"""Code-mixing measurement and controlled code-mixed variants.

Measurement
-----------
Each token gets a language tag:
  * Devanagari/Bengali-script words -> the matrix language (hi/bn)
  * Latin-script words in a mostly-Indic-script text -> 'en'
  * Latin-script words in a romanised text -> 'en' if it is a common English word and not a
    common romanised Hindi/Bengali word, otherwise the matrix language
  * numbers, punctuation, symbols -> 'univ' (language-independent)

From the tags we compute the Code-Mixing Index (Das & Gambäck, 2014):
    CMI = 100 * (1 - max_lang_count / (n - u))   (0 if n == u)
where n is the number of tokens and u the number of language-independent tokens,
plus the switch rate (share of adjacent language-tagged tokens that switch language).

Variants
--------
For every test article we build two paired variants with the same facts:
  * ``cm``    – the article rewritten sentence by sentence as code-mixed text
               (native-script grammar, English content words in Latin letters);
  * ``roman`` – the ``cm`` text transliterated fully into Roman letters.
So mono -> cm isolates lexical mixing and cm -> roman isolates the script change.
Every rewritten sentence is validated (numbers unchanged, length, script) and retried once.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .entities import numbers_multiset_equal
from .textnorm import normalize_text, script_profile, token_script, tokenize, word_count

ROMAN_STOP = {
    "hi": set("""
hai hain ho hota hoti hote tha thi the thhe ka ki ke ko se me mein mai main mujhe par pe aur ya nahi nahin nhi na bhi
toh to kya kyu kyun kyon kaise kab kahan jo ye yeh wo woh vo voh is us iss uss isse usse inka unka unki inki unke
ek do teen char paanch hum ham tum aap apna apne apni mera meri mere tera teri tere uska uski uske kar karna karne
karke kiya kiye karte karta karti kare raha rahi rahe rha rhi gaya gayi gaye gya hua hui hue huye wala wali wale
liye lie liya diya dena lena jana jaana aana aaya aayi saath sath baad pehle phir abhi sab kuch bahut bohot zyada
jyada jab tab agar lekin magar kyunki jaise waise sirf bhai yaar ji haan han ne tak dwara sakta sakti sakte chahiye
aaj kal din saal log logon desh sarkar duniya bharat waqt samay baat bata batay bataya kehna kaha kahi kehte
hoga hogi honge rahega rahegi wahan yahan andar bahar upar niche taraf beech dauran mutabik anusar
""".split()),
    "bn": set("""
ami amra tumi tomra apni apnara se tara o ar ebong kintu na nei nai ache achhe achhen chilo chhilo chhilen hobe hoy
hocche hochhe hoyeche hoyechhe holo hoyechilo korche korchhe kore korte koreche korechhe korechen korlo korbe korben
theke jonno er ke te ta ti gulo guli khub onek ekta ekti ki keno kothay kemon din bochor bochhor sorkar shohor
bhalo ei oi sei je jodi tahole bole bolen bolechen gelo jabe jete ese esechhe esechen diye diyeche niye niyechhe
tar tader amar amader tomar tomader apnar okhane ekhane ekhon tokhon por age moddhe sathe shathe bhabe dike
""".split()),
}


@lru_cache(maxsize=100_000)
def _english_score(tok: str) -> float:
    try:
        from wordfreq import zipf_frequency

        return zipf_frequency(tok, "en")
    except Exception:  # pragma: no cover
        return 0.0


def token_langs(text: str, matrix: str, romanized: bool | None = None) -> list[tuple[str, str]]:
    """Tag every token with 'en', the matrix language, or 'univ'."""
    toks = tokenize(text, lower=False)
    if romanized is None:
        prof = script_profile(text)
        romanized = prof["latn"] > 0.8
    out = []
    stop = ROMAN_STOP.get(matrix, set())
    for t in toks:
        s = token_script(t)
        if s in ("deva", "beng"):
            out.append((t, matrix))
        elif s == "latn":
            if not romanized:
                out.append((t, "en"))
            else:
                low = t.lower()
                if low in stop:
                    out.append((t, matrix))
                elif len(low) >= 2 and _english_score(low) >= 3.3:
                    out.append((t, "en"))
                else:
                    out.append((t, matrix))
        else:
            out.append((t, "univ"))
    return out


def cmi_stats(text: str, matrix: str, romanized: bool | None = None) -> dict:
    tags = token_langs(text, matrix, romanized)
    n = len(tags)
    langs = [l for _, l in tags if l != "univ"]
    u = n - len(langs)
    counts = {}
    for l in langs:
        counts[l] = counts.get(l, 0) + 1
    cmi = 100.0 * (1 - max(counts.values()) / (n - u)) if n > u and counts else 0.0
    switches = sum(1 for a, b in zip(langs, langs[1:]) if a != b)
    prof = script_profile(text)
    return {
        "cmi": cmi,
        "en_token_share": counts.get("en", 0) / max(1, len(langs)),
        "switch_rate": switches / max(1, len(langs) - 1),
        "latin_char_share": prof["latn"],
        "n_tokens": n,
    }


# --------------------------------------------------------------------------------------
# Validation of rewritten sentences
# --------------------------------------------------------------------------------------
def validate_rewrite(src: str, out: str, mode: str, lang: str) -> dict:
    """Check one rewritten sentence. Returns {'ok': bool, 'hard': [...], 'soft': [...]}."""
    hard, soft = [], []
    out = normalize_text(out or "")
    if not out:
        return {"ok": False, "hard": ["empty"], "soft": []}
    if not numbers_multiset_equal(src, out):
        hard.append("numbers changed")
    ws, wo = word_count(src), word_count(out)
    ratio = wo / max(1, ws)
    if ws >= 4 and (ratio < 0.5 or ratio > 1.8):
        hard.append(f"length ratio {ratio:.2f}")
    prof = script_profile(out)
    native = "deva" if lang == "hi" else "beng"
    if mode == "cm":
        toks = [t for t in tokenize(out, lower=False) if token_script(t) in ("deva", "beng", "latn")]
        en_share = sum(token_script(t) == "latn" for t in toks) / max(1, len(toks))
        if prof[native] < 0.25:
            hard.append("matrix script lost")
        if ws >= 6 and en_share < 0.08:
            soft.append("too little English")
        if en_share > 0.75:
            hard.append("mostly English")
    elif mode == "roman":
        if prof["deva"] + prof["beng"] > 0.05:
            hard.append("Indic script left")
    return {"ok": not hard and not soft, "hard": hard, "soft": soft}


_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*[.):\-।]\s*(.*\S)\s*$")


def parse_numbered(text: str, n: int) -> dict[int, str]:
    """Parse "1. ...\n2. ..." model output into {index: line} (1-based)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            k = int(m.group(1))
            if 1 <= k <= n and k not in out:
                out[k] = m.group(2).strip().strip('"').strip()
    return out


def chunk_sentences(sents: list[str], size: int = 6) -> list[list[int]]:
    return [list(range(i, min(i + size, len(sents)))) for i in range(0, len(sents), size)]
