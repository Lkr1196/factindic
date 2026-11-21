"""LLM summarisation: Track A, Track B and Track BV (notebook 04), plus question generation
for the QA-based factuality check.

Every (document, variant) pair is summarised by every track with the same model, the same
length budget and greedy decoding, so the only controlled difference is the prompting strategy.
"""
from __future__ import annotations

import pandas as pd

from . import prompts
from .entities import dates_supported, numbers_supported
from .llm import run_generation
from .textnorm import split_sentences, truncate_words, word_count


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


_END_RE = __import__("re").compile(r"[।.!?…]['\"”’)\]]*$")


def _finalize(summary: str, max_words: int, hit_cap: bool = False) -> tuple[str, bool]:
    """Clean a generated summary. Returns (summary, cut).

    ``cut`` is True when the text was cut short: it ran past twice the word budget, or generation
    stopped at the token limit before the sentence ended. A dangling half sentence is dropped when
    at least one complete sentence remains, so that scores never depend on a broken fragment.
    """
    s = summary.strip()
    over = word_count(s) > max_words * 2
    if over:
        s = truncate_words(s, max_words * 2)
    if (over or hit_cap) and s and not _END_RE.search(s):
        sents = split_sentences(s)
        if len(sents) > 1:
            s = " ".join(sents[:-1]).strip()
        return s, True
    return s, over


def llm_summarize(variants: pd.DataFrame, cfg, log=print) -> pd.DataFrame:
    """Track A and B for every row of ``variants`` (columns doc_id, lang, variant, text, role)."""
    model_id = cfg.summarizer_model if cfg.backend != "openai" else (cfg.openai_model or cfg.summarizer_model)
    system = short_name(model_id)
    items, meta = [], {}
    for r in variants.itertuples(index=False):
        base = {"doc_id": r.doc_id, "lang": r.lang, "variant": r.variant, "role": r.role}
        if "A" in cfg.tracks:
            iid = f"{system}|A|{r.doc_id}|{r.variant}"
            items.append({"id": iid, "messages": prompts.summarize_A(r.text, r.lang, r.variant, cfg.summary_max_words),
                          "max_new_tokens": cfg.max_new_tokens_A, "task": "sumA"})
            meta[iid] = {**base, "_track": "A"}
        if "B" in cfg.tracks or "BV" in cfg.tracks:
            iid = f"{system}|B|{r.doc_id}|{r.variant}"
            items.append({"id": iid, "messages": prompts.summarize_B(r.text, r.lang, r.variant, cfg.summary_max_words),
                          "max_new_tokens": cfg.max_new_tokens_B, "task": "sumB"})
            meta[iid] = {**base, "_track": "B"}
    log(f"Summarising {len(variants)} inputs x {len({m['_track'] for m in meta.values()})} tracks = {len(items)} prompts")
    res = run_generation(items, model_id, cfg, tag="summarize")
    rows = []
    for it, r in zip(items, res):
        m = dict(meta[it["id"]])
        track = m.pop("_track")
        hit = bool(r["n_out"] and r["n_out"] >= it["max_new_tokens"])
        summ, trunc = _finalize(prompts.parse_summary(r["text"]), cfg.summary_max_words, hit)
        rows.append({"summary_id": it["id"], "system": system, "model_id": model_id, "track": track, **m,
                     "summary": summ, "truncated": trunc, "hit_cap": hit, "raw": r["text"], "n_in": r["n_in"],
                     "n_out": r["n_out"],
                     "time_s": r["time_s"], "batch_size": r["batch_size"], "error": r["error"]})
    df = pd.DataFrame(rows)
    if "BV" in cfg.tracks:
        bv = verify_and_revise(df[df["track"] == "B"], variants, cfg, model_id, system, log)
        df = pd.concat([df, bv], ignore_index=True)
    if "B" not in cfg.tracks:
        df = df[df["track"] != "B"]
    return df.reset_index(drop=True)


def rule_problems(summary: str, source: str) -> list[str]:
    """Numbers and dates in the summary that do not occur in the source."""
    probs = []
    _, bad_nums = numbers_supported(summary, source)
    for n in bad_nums:
        probs.append(f'The number "{n.raw}" does not appear in the article.')
    _, bad_dates = dates_supported(summary, source)
    for d, m in bad_dates:
        probs.append(f"The date (day {d}, month {m}) does not appear in the article." if d else
                     f"Month {m} is not mentioned in the article.")
    return probs


def verify_and_revise(b_rows: pd.DataFrame, variants: pd.DataFrame, cfg, model_id: str, system: str, log=print):
    """Track BV: rule-based number/date check + LLM self-check on Track B outputs, then one revision."""
    text_of = {(r.doc_id, r.variant): r.text for r in variants.itertuples(index=False)}
    check_items, sent_lists = [], {}
    for r in b_rows.itertuples(index=False):
        sents = split_sentences(r.summary) or [r.summary]
        sent_lists[r.summary_id] = sents
        check_items.append({"id": "chk|" + r.summary_id,
                            "messages": prompts.selfcheck(text_of[(r.doc_id, r.variant)], sents, r.lang),
                            "max_new_tokens": 48 * len(sents) + 32, "task": "check"})
    log(f"Track BV: self-checking {len(check_items)} Track B summaries")
    chk = run_generation(check_items, model_id, cfg, tag="selfcheck")
    rev_items, problems, chk_time = [], {}, {}
    for r, c in zip(b_rows.itertuples(index=False), chk):
        src = text_of[(r.doc_id, r.variant)]
        verdicts = prompts.parse_selfcheck(c["text"], len(sent_lists[r.summary_id]))
        probs = [f"Sentence {k}: {why}" for k, why in sorted(verdicts.items()) if why]
        probs += rule_problems(r.summary, src)
        problems[r.summary_id] = probs
        chk_time[r.summary_id] = c["time_s"]
        if probs:
            rev_items.append({"id": "rev|" + r.summary_id,
                              "messages": prompts.revise(src, r.lang, r.variant, r.summary, probs, cfg.summary_max_words),
                              "max_new_tokens": cfg.max_new_tokens_A, "task": "revise"})
    log(f"Track BV: {len(rev_items)} of {len(check_items)} summaries flagged; revising")
    rev = {x["id"][4:]: y for x, y in zip(rev_items, run_generation(rev_items, model_id, cfg, tag="revise"))}
    rows = []
    for r in b_rows.itertuples(index=False):
        rr = rev.get(r.summary_id)
        new = prompts.parse_summary(rr["text"]) if rr else ""
        revised = bool(rr) and bool(new.strip())
        hit = bool(revised and rr["n_out"] and rr["n_out"] >= cfg.max_new_tokens_A)
        summ, trunc = _finalize(new if revised else r.summary, cfg.summary_max_words, hit)
        trunc = trunc or (not revised and bool(r.truncated))
        hit = hit or (not revised and bool(getattr(r, "hit_cap", False)))
        rows.append({"summary_id": r.summary_id.replace("|B|", "|BV|", 1), "system": system, "model_id": model_id,
                     "track": "BV", "doc_id": r.doc_id, "lang": r.lang, "variant": r.variant, "role": r.role,
                     "summary": summ, "truncated": trunc, "hit_cap": hit, "raw": rr["text"] if rr else r.raw,
                     "n_in": r.n_in, "n_out": r.n_out + (rr["n_out"] if rr else 0),
                     "time_s": r.time_s + chk_time[r.summary_id] + (rr["time_s"] if rr else 0.0),
                     "batch_size": r.batch_size, "error": r.error, "bv_flagged": bool(problems[r.summary_id]),
                     "bv_problems": problems[r.summary_id], "bv_revised": revised})
    return pd.DataFrame(rows)


def generate_questions(summaries: pd.DataFrame, cfg, log=print) -> pd.DataFrame:
    """QG for the QA-based check: up to k (question, answer-span) pairs per summary."""
    from .textnorm import normalize_text, tokenize

    model_id = cfg.summarizer_model if cfg.backend != "openai" else (cfg.openai_model or cfg.summarizer_model)
    items = []
    for r in summaries.itertuples(index=False):
        if not isinstance(r.summary, str) or not r.summary.strip():
            continue
        items.append({"id": "qg|" + r.summary_id, "messages": prompts.question_generation(r.summary, r.lang, cfg.qg_max_questions),
                      "max_new_tokens": 60 * cfg.qg_max_questions, "task": "qg"})
    log(f"Question generation for {len(items)} summaries")
    res = run_generation(items, model_id, cfg, tag="qg")
    summ_of = dict(zip(summaries["summary_id"], summaries["summary"]))
    rows = []
    for it, r in zip(items, res):
        sid = it["id"][3:]
        summ = normalize_text(summ_of[sid])
        pairs = []
        summ_toks = set(tokenize(summ))
        for q, a in prompts.parse_qg(r["text"]):
            a_toks = tokenize(a)
            in_summary = normalize_text(a) in summ or (a_toks and all(t in summ_toks for t in a_toks))
            if a_toks and in_summary and (q, a) not in pairs:
                pairs.append((q, a))
        rows.append({"summary_id": sid, "qa_pairs": pairs[:cfg.qg_max_questions], "qg_raw": r["text"]})
    return pd.DataFrame(rows)
