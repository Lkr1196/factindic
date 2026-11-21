"""Deterministic stand-ins for every model, used by profile="mock" and the tests.

They follow the real input/output formats so the whole pipeline (parsing, validation,
verification, statistics, figures) can be exercised on CPU in a minute. The mock summariser
deliberately corrupts numbers more often for code-mixed inputs so the analysis code has an
effect to find. None of this says anything about real models.
"""
from __future__ import annotations

import hashlib
import re

from .entities import extract_numbers, numbers_supported, KeyIndex, entity_tokens
from .mock_data import GAZETTEER
from .prompts import ART_CLOSE, ART_OPEN, SUM_CLOSE, SUM_OPEN
from .textnorm import normalize_text, script_profile, split_sentences, token_script, tokenize

CM_DICT = {
    "प्रदूषण": "pollution", "स्तर": "level", "सरकार": "government", "योजना": "scheme", "लोगों": "people",
    "घोषणा": "announcement", "अधिकारियों": "officials", "काम": "work", "बारिश": "rain", "स्कूल": "school",
    "गांव": "villages", "राहत": "relief", "टीमें": "teams", "अस्पतालों": "hospitals", "मामले": "cases",
    "मरीज़": "patients", "बिस्तर": "beds", "अभियान": "campaign", "आंकड़ों": "data", "फायदा": "benefit",
    "विपक्ष": "opposition", "आलोचना": "criticism", "बाढ़": "flood", "स्वास्थ्य": "health", "नई": "new",
    "শহরে": "city-তে", "প্রকল্পের": "project-এর", "প্রকল্পে": "project-এ", "প্রকল্প": "project", "মানুষ": "people",
    "দূষণের": "pollution-এর", "মাত্রা": "level", "সরকার": "government", "বৃষ্টির": "rain-এর", "স্কুল": "school",
    "গ্রাম": "village", "বন্যায়": "flood-এ", "হাসপাতালগুলিতে": "hospital-গুলিতে", "রোগী": "patient",
    "শয্যা": "bed", "অভিযান": "campaign", "সচেতনতা": "awareness", "বিরোধীরা": "opposition", "নতুন": "new",
}


def _h(s: str) -> float:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i + len(start): j if j >= 0 else None]


def _numbered(text: str, header: str | None = "SENTENCES:\n") -> list[str]:
    block = text.split(header, 1)[-1] if header else text
    return [re.sub(r"^\s*\d+\.\s*", "", l).strip() for l in block.splitlines() if re.match(r"^\s*\d+\.", l)]


def _romanize(s: str) -> str:
    from indic_transliteration import sanscript

    out = []
    for w in s.split():
        sc = token_script(w)
        if sc in ("deva", "beng"):
            src = sanscript.DEVANAGARI if sc == "deva" else sanscript.BENGALI
            r = sanscript.transliterate(w, src, sanscript.ITRANS)
            r = re.sub(r"[.~^]", "", r).replace("M", "n").replace("H", "").lower()
            out.append(r)
        else:
            out.append(w)
    return " ".join(out)


def _corrupt(summary: str) -> str:
    nums = extract_numbers(summary)
    if not nums:
        return summary + " 2019"
    raw = nums[0].raw
    try:
        new = str(int(float(raw.replace(",", ""))) + 1)
    except ValueError:
        new = "99"
    return summary.replace(raw, new, 1)


class MockLLM:
    name = "mock"

    def _one(self, it: dict) -> str:
        task = it.get("task", "")
        content = it["messages"][-1]["content"]
        if task in ("cm", "roman"):
            sents = _numbered(content)
            out = []
            for i, s in enumerate(sents, 1):
                if task == "cm":
                    words = s.split()
                    rep = [CM_DICT.get(w.strip("।,"), w) + ("।" if w.endswith("।") and w.strip("।,") in CM_DICT else "") for w in words]
                    out.append(f"{i}. " + " ".join(rep))
                else:
                    out.append(f"{i}. " + _romanize(s))
            return "\n".join(out)
        article = _between(content, ART_OPEN, ART_CLOSE)
        variant = "roman" if script_profile(article)["latn"] > 0.8 else ("cm" if script_profile(article)["latn"] > 0.05 else "mono")
        sents = split_sentences(article)
        lead = " ".join(sents[:1]) if sents else ""
        if task in ("sumA", "sumB", "revise"):
            p = {"mono": 0.15, "cm": 0.35, "roman": 0.5}[variant]
            if task == "sumB":
                p /= 2
            if task == "revise":
                p = 0.0
            summ = _corrupt(lead) if _h(article + task) < p else lead
            if task == "sumB":
                return f"FACTS:\n- {lead}\nENTITIES: \nSUMMARY: {summ}"
            if task == "revise":
                return f"SUMMARY: {summ}"
            return summ
        if task == "check":
            summ_sents = _numbered(_between(content, SUM_OPEN, SUM_CLOSE), header=None)
            lines = []
            for i, s in enumerate(summ_sents, 1):
                _, uns = numbers_supported(s, article)
                lines.append(f"{i}. UNSUPPORTED: number not in article" if uns else f"{i}. SUPPORTED")
            return "\n".join(lines)
        if task == "qg":
            summ = _between(content, SUM_OPEN, SUM_CLOSE)
            pairs = []
            for n in extract_numbers(summ)[:3]:
                pairs.append(f"Q: संख्या क्या है? || A: {n.raw}")
            for g in GAZETTEER:
                if g in summ:
                    pairs.append(f"Q: कौन? || A: {g}")
                    break
            return "\n".join(pairs)
        return lead

    def generate(self, items: list[dict], log=print) -> list[dict]:
        out = []
        for it in items:
            text = self._one(it)
            out.append({"id": it["id"], "text": text, "n_in": len(it["messages"][-1]["content"]) // 4,
                        "n_out": max(1, len(text) // 4), "time_s": 0.01 + 0.002 * len(text) / 10,
                        "batch_size": 1, "error": None})
        return out


class MockNLI:
    """P(entail) = share of hypothesis words found in the premise; unsupported numbers -> contradiction."""

    def probs(self, premises: list[str], hypotheses: list[str]):
        import numpy as np

        out = []
        for p, h in zip(premises, hypotheses):
            idx = KeyIndex(p)
            toks = [t for t in tokenize(h) if len(t) > 1]
            keys = [k for t in toks for k in entity_tokens(t)]
            cov = sum(idx.has(k) for k in keys) / max(1, len(keys))
            _, uns = numbers_supported(h, p)
            contra = 0.9 if uns else 0.05
            ent = max(0.0, min(1.0, cov)) * (1 - contra)
            neu = max(0.0, 1 - ent - contra)
            out.append([ent, neu, contra])
        return np.array(out)


class MockQA:
    def answer(self, questions: list[str], contexts: list[str]) -> list[dict]:
        res = []
        for q, c in zip(questions, contexts):
            nums = extract_numbers(c)
            gz = [g for g in GAZETTEER if g in c]
            if ("संख्या" in q or "कितन" in q or "কত" in q) and nums:
                res.append({"answer": nums[0].raw, "score": 0.9, "null": False})
            elif gz:
                res.append({"answer": gz[0], "score": 0.8, "null": False})
            else:
                res.append({"answer": "", "score": 0.0, "null": True})
        return res


class MockSeq2Seq:
    def summarize(self, texts: list[str], lang: str) -> list[str]:
        out = []
        for t in texts:
            s = split_sentences(t)
            out.append(normalize_text(s[0]) if s else "")
        return out
