"""Build the paired code-mixed variants of each article with an LLM (notebook 02).

mono  --(LLM: code-mix rewrite, sentence by sentence)-->  cm
cm    --(LLM: transliterate to Roman letters)-------->  roman

Sentences are rewritten in numbered chunks of up to 6, so sentence alignment is kept and every
rewritten sentence can be checked against its source sentence:
  * hard checks: numbers unchanged, length ratio, script (see codemix.validate_rewrite)
  * soft check (cm only): enough English words
Failing sentences are retried one at a time. If a retry still fails a hard check, the sentence
falls back (cm: original sentence; roman: rule-based transliteration) and is counted as invalid.
Documents with fewer than 80% valid sentences in either variant are excluded from the paired
comparison (``paired_ok`` = False) but kept in the files for inspection.
"""
from __future__ import annotations

import re

import pandas as pd

from . import prompts
from .codemix import chunk_sentences, cmi_stats, parse_numbered, validate_rewrite
from .llm import run_generation
from .textnorm import normalize_text, token_script

MIN_VALID = 0.8
VALID = ("ok", "retry_ok", "soft_fail")  # soft_fail = faithful but little English


def rule_romanize(text: str) -> str:
    """Rough fallback romanisation (ITRANS, lower-cased)."""
    try:
        from indic_transliteration import sanscript
    except ImportError:
        return text
    out = []
    for w in text.split():
        sc = token_script(w)
        if sc in ("deva", "beng"):
            src = sanscript.DEVANAGARI if sc == "deva" else sanscript.BENGALI
            r = sanscript.transliterate(w, src, sanscript.ITRANS)
            r = re.sub(r"[.~^]", "", r).replace("M", "n").replace("H", "h").lower()
            out.append(r)
        else:
            out.append(w)
    return " ".join(out)


def _max_new(sents: list[str]) -> int:
    chars = sum(len(s) for s in sents)
    return int(min(2000, chars * 1.1 + 64))


def rewrite_round(docs: pd.DataFrame, src_col: str, mode: str, model_id: str, cfg, log=print) -> dict:
    """Rewrite every document's sentences. Returns {doc_id: {"sents", "status", "time_s", "n_out"}}."""
    make = prompts.codemix_rewrite if mode == "cm" else prompts.romanize
    items, meta = [], {}
    for r in docs.itertuples(index=False):
        sents = list(getattr(r, src_col))
        for c, idx in enumerate(chunk_sentences(sents, 6)):
            chunk = [sents[i] for i in idx]
            iid = f"{r.doc_id}|{mode}|{c}"
            items.append({"id": iid, "messages": make(chunk, r.lang), "max_new_tokens": _max_new(chunk), "task": mode})
            meta[iid] = (r.doc_id, r.lang, idx)
    log(f"{mode}: {len(items)} prompts for {len(docs)} documents")
    res = run_generation(items, model_id, cfg, tag=f"rewrite_{mode}")
    out: dict = {}
    for r in docs.itertuples(index=False):
        n = len(getattr(r, src_col))
        out[r.doc_id] = {"sents": [None] * n, "status": [None] * n, "time_s": 0.0, "n_out": 0, "lang": r.lang,
                         "src": list(getattr(r, src_col))}
    retry = []
    for it, rr in zip(items, res):
        doc_id, lang, idx = meta[it["id"]]
        d = out[doc_id]
        d["time_s"] += rr["time_s"]
        d["n_out"] += rr["n_out"]
        parsed = parse_numbered(rr["text"], len(idx))
        for k, i in enumerate(idx, 1):
            cand = normalize_text(parsed.get(k, ""))
            v = validate_rewrite(d["src"][i], cand, mode, lang)
            if v["ok"]:
                d["sents"][i], d["status"][i] = cand, "ok"
            else:
                d["sents"][i] = cand if (cand and not v["hard"]) else None
                retry.append((doc_id, i))
    log(f"{mode}: {len(retry)} sentences need a retry")
    if retry:
        r_items = []
        for doc_id, i in retry:
            d = out[doc_id]
            r_items.append({"id": f"{doc_id}|{mode}|r{i}", "messages": make([d["src"][i]], d["lang"]),
                            "max_new_tokens": _max_new([d["src"][i]]), "task": mode})
        r_res = run_generation(r_items, model_id, cfg, tag=f"rewrite_{mode}_retry")
        for (doc_id, i), rr in zip(retry, r_res):
            d = out[doc_id]
            d["time_s"] += rr["time_s"]
            d["n_out"] += rr["n_out"]
            cand = normalize_text(parse_numbered(rr["text"], 1).get(1, "") or rr["text"].strip().split("\n")[0])
            cand = re.sub(r"^\s*1\s*[.)]\s*", "", cand)
            v = validate_rewrite(d["src"][i], cand, mode, d["lang"])
            if not v["hard"]:
                d["sents"][i], d["status"][i] = cand, "retry_ok"
            elif d["sents"][i] is not None:  # first attempt only failed the soft check
                d["status"][i] = "soft_fail"
            else:
                d["sents"][i] = d["src"][i] if mode == "cm" else rule_romanize(d["src"][i])
                d["status"][i] = "fallback"
    return out


def build_variants(docs: pd.DataFrame, cfg, log=print) -> pd.DataFrame:
    """Return a long table with one row per (doc, variant): mono, cm, roman."""
    model_id = cfg.codemix_model
    rows = []
    for r in docs.itertuples(index=False):
        rows.append({"doc_id": r.doc_id, "lang": r.lang, "role": r.role, "variant": "mono", "sents": list(r.sents),
                     "text": " ".join(r.sents), "valid_frac": 1.0, "n_fallback": 0, "gen_time_s": 0.0})
    want = set(cfg.variants)
    cm = rewrite_round(docs, "sents", "cm", model_id, cfg, log) if ({"cm", "roman"} & want) else {}
    if cm:
        for r in docs.itertuples(index=False):
            d = cm[r.doc_id]
            st = d["status"]
            rows.append({"doc_id": r.doc_id, "lang": r.lang, "role": r.role, "variant": "cm", "sents": d["sents"],
                         "text": " ".join(d["sents"]), "valid_frac": sum(s in VALID for s in st) / len(st),
                         "n_fallback": sum(s == "fallback" for s in st), "gen_time_s": d["time_s"], "status": st})
    if "roman" in want and cm:
        cm_df = pd.DataFrame([{"doc_id": k, "lang": v["lang"], "cm_sents": v["sents"]} for k, v in cm.items()])
        ro = rewrite_round(cm_df, "cm_sents", "roman", model_id, cfg, log)
        for r in docs.itertuples(index=False):
            d = ro[r.doc_id]
            st = d["status"]
            rows.append({"doc_id": r.doc_id, "lang": r.lang, "role": r.role, "variant": "roman", "sents": d["sents"],
                         "text": " ".join(d["sents"]), "valid_frac": sum(s in VALID for s in st) / len(st),
                         "n_fallback": sum(s == "fallback" for s in st), "gen_time_s": d["time_s"], "status": st})
    out = pd.DataFrame(rows)
    stats = [cmi_stats(t, l, romanized=(v == "roman")) for t, l, v in zip(out["text"], out["lang"], out["variant"])]
    for k in ("cmi", "en_token_share", "switch_rate", "latin_char_share"):
        out[k] = [s[k] for s in stats]
    # paired_ok: every requested variant of the document is valid enough
    ok = out.groupby("doc_id")["valid_frac"].min() >= MIN_VALID
    out["paired_ok"] = out["doc_id"].map(ok)
    return out


def add_natural_variants(natural: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in natural.itertuples(index=False):
        st = cmi_stats(r.text, r.lang)
        rows.append({"doc_id": r.doc_id, "lang": r.lang, "role": "natural", "variant": "natural",
                     "sents": list(r.sents), "text": r.text, "valid_frac": 1.0, "n_fallback": 0, "gen_time_s": 0.0,
                     "paired_ok": False, **{k: st[k] for k in ("cmi", "en_token_share", "switch_rate", "latin_char_share")}})
    return pd.DataFrame(rows)
