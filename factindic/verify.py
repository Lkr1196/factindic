"""Factuality verification layer (notebook 05).

Three independent automatic checks, each computed against
  * the ORIGINAL monolingual article (primary): the facts are identical across mono/cm/roman by
    construction, so this measures the summariser, not how well the checker reads code-mixed text;
  * the actual input the model saw (secondary; the only option for natural code-mixed documents).

1. NLI entailment (SummaC-style, zero-shot): each summary sentence is checked against single
   source sentences and 3-sentence windows with a multilingual NLI model; its support is the
   maximum entailment probability. A sentence is *unsupported* if support < threshold (0.5).
2. QA-based check (QAGS/QAFactEval-style, multilingual): questions generated from the summary
   (answers are spans of the summary) are answered from the source by an extractive QA model;
   the score is the token-F1 between the two answers (0 if the source has no answer).
3. Entity and numeral alignment (Nan et al., 2021 style):
     precision-source = share of summary entities/numbers/dates found in the source (hallucination),
     precision/recall/F1-target against the reference summary (content preservation).
Each unsupported summary sentence also gets an error type: number, date, entity or other.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .entities import (KeyIndex, dates_supported, entity_in, extract_dates, extract_numbers, keys_match,
                       numbers_supported, token_key)
from .textnorm import normalize_text, split_sentences, tokenize


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class NLIScorer:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64, max_length: int = 320):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.fp16 = self.device.startswith("cuda")
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        find = lambda name: next((i for i, l in labels.items() if l.startswith(name)), None)
        self.order = [find("entail"), find("neutral"), find("contra")]
        if None in self.order:  # MoritzLaurer models: entailment, neutral, contradiction
            self.order = [0, 1, 2]
        self.batch_size = batch_size
        self.max_length = max_length

    def probs(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(premises), 3), dtype=np.float32)
        order = sorted(range(len(premises)), key=lambda i: -(len(premises[i]) + len(hypotheses[i])))
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([premises[i] for i in idx], [hypotheses[i] for i in idx], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                if self.fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = self.model(**enc).logits.float()
                    if not torch.isfinite(logits).all():
                        logits = self.model(**enc).logits.float()
                else:
                    logits = self.model(**enc).logits.float()
            p = torch.softmax(logits, -1).cpu().numpy()[:, self.order]
            out[idx] = p
        return out


class QAScorer:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32, max_length: int = 384,
                 stride: int = 128):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        if self.device.startswith("cuda"):
            self.model.half()
        self.batch_size, self.max_length, self.stride = batch_size, max_length, stride

    def answer(self, questions: list[str], contexts: list[str], max_answer_tokens: int = 30) -> list[dict]:
        torch = self.torch
        enc = self.tok(questions, contexts, truncation="only_second", max_length=self.max_length, stride=self.stride,
                       return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
        sample_map = enc.pop("overflow_to_sample_mapping")
        offsets = enc.pop("offset_mapping")
        n_feat = len(enc["input_ids"])
        starts, ends = [], []
        for s in range(0, n_feat, self.batch_size):
            batch = {k: torch.tensor(v[s:s + self.batch_size], device=self.device) for k, v in enc.items()}
            with torch.inference_mode():
                o = self.model(**batch)
            starts.append(o.start_logits.float().cpu().numpy())
            ends.append(o.end_logits.float().cpu().numpy())
        starts = np.concatenate(starts) if starts else np.zeros((0, self.max_length))
        ends = np.concatenate(ends) if ends else np.zeros((0, self.max_length))
        best: dict[int, dict] = {}
        for f in range(n_feat):
            ex = sample_map[f]
            seq_ids = enc.sequence_ids(f)
            ctx = [i for i, sid in enumerate(seq_ids) if sid == 1]
            if not ctx:
                continue
            st, en = starts[f], ends[f]
            null = float(st[0] + en[0])
            cand = None
            top_s = [i for i in np.argsort(st)[::-1][:20] if seq_ids[i] == 1]
            top_e = [i for i in np.argsort(en)[::-1][:20] if seq_ids[i] == 1]
            for i in top_s:
                for j in top_e:
                    if j < i or j - i + 1 > max_answer_tokens:
                        continue
                    sc = float(st[i] + en[j])
                    if cand is None or sc > cand[0]:
                        cand = (sc, i, j)
            b = best.setdefault(ex, {"span": None, "span_score": -1e9, "null": 1e9})
            b["null"] = min(b["null"], null)
            if cand and cand[0] > b["span_score"]:
                a0, a1 = offsets[f][cand[1]][0], offsets[f][cand[2]][1]
                b["span"], b["span_score"] = contexts[ex][a0:a1], cand[0]
        out = []
        for ex in range(len(questions)):
            b = best.get(ex)
            if not b or b["span"] is None or b["null"] > b["span_score"]:
                out.append({"answer": "", "score": 0.0, "null": True})
            else:
                out.append({"answer": b["span"].strip(), "score": b["span_score"], "null": False})
        return out


# --------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------
def premises_for(src_sents: list[str]) -> list[str]:
    """Single sentences plus 3-sentence sliding windows."""
    src_sents = [s for s in src_sents if s.strip()]
    prem = list(src_sents)
    if len(src_sents) > 3:
        prem += [" ".join(src_sents[i:i + 3]) for i in range(len(src_sents) - 2)]
    elif len(src_sents) > 1:
        prem.append(" ".join(src_sents))
    return prem or [""]


def nli_consistency(jobs: list[tuple[str, str, list[str]]], nli, threshold: float = 0.5, log=print) -> dict:
    """jobs = [(key, summary, source_sentences)] -> {key: metrics}."""
    pair_index: dict[tuple[str, str], int] = {}
    prem_list, hyp_list = [], []
    plan = {}
    for key, summary, src in jobs:
        hyps = split_sentences(summary) if isinstance(summary, str) else []
        hyps = [h for h in hyps if tokenize(h)]
        prems = premises_for(src)
        plan[key] = (hyps, prems)
        for h in hyps:
            for p in prems:
                k = (p, h)
                if k not in pair_index:
                    pair_index[k] = len(prem_list)
                    prem_list.append(p)
                    hyp_list.append(h)
    log(f"  NLI: {len(prem_list)} premise-hypothesis pairs")
    probs = nli.probs(prem_list, hyp_list) if prem_list else np.zeros((0, 3))
    out = {}
    for key, (hyps, prems) in plan.items():
        if not hyps:
            out[key] = {"nli_score": np.nan, "nli_min": np.nan, "nli_support_rate": np.nan, "nli_halluc": np.nan,
                        "nli_sents": []}
            continue
        sent = []
        for h in hyps:
            rows = probs[[pair_index[(p, h)] for p in prems]]
            b = int(rows[:, 0].argmax())
            sent.append({"sent": h, "support": float(rows[b, 0]), "contra": float(rows[b, 2]),
                         "max_contra": float(rows[:, 2].max())})
        sup = [s["support"] for s in sent]
        out[key] = {"nli_score": float(np.mean(sup)), "nli_min": float(np.min(sup)),
                    "nli_support_rate": float(np.mean([x >= threshold for x in sup])),
                    "nli_halluc": float(any(x < threshold for x in sup)), "nli_sents": sent}
    return out


def answer_f1(pred: str, gold: str) -> float:
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return float(p == g)
    pv = {round(n.value, 6) for n in extract_numbers(pred)}
    gv = {round(n.value, 6) for n in extract_numbers(gold)}
    if pv and gv:
        if pv == gv:
            return 1.0
        if not (pv & gv):
            return 0.0
    pk = [token_key(t) or t for t in p]
    gk = [token_key(t) or t for t in g]
    used, matched = set(), 0
    for x in pk:
        for j, y in enumerate(gk):
            if j not in used and (x == y or keys_match(x, y)):
                used.add(j)
                matched += 1
                break
    if not matched:
        return 0.0
    prec, rec = matched / len(p), matched / len(g)
    return 2 * prec * rec / (prec + rec)


def qa_consistency(jobs: list[tuple[str, list, str]], qa, log=print) -> dict:
    """jobs = [(key, [(question, answer_in_summary)], source_text)] -> {key: metrics}."""
    qs, cs, owners = [], [], []
    for key, pairs, src in jobs:
        for q, a in pairs or []:
            qs.append(q)
            cs.append(src)
            owners.append((key, a))
    log(f"  QA: {len(qs)} questions")
    answers = qa.answer(qs, cs) if qs else []
    per: dict[str, list] = {}
    for (key, a_sum), ans in zip(owners, answers):
        f1 = 0.0 if ans["null"] else answer_f1(ans["answer"], a_sum)
        per.setdefault(key, []).append({"answer_summary": a_sum, "answer_source": ans["answer"], "f1": f1,
                                        "unanswerable": ans["null"]})
    out = {}
    for key, pairs, _ in jobs:
        items = per.get(key, [])
        if not items:
            out[key] = {"qa_f1": np.nan, "qa_unanswerable": np.nan, "qa_n": 0, "qa_items": []}
        else:
            out[key] = {"qa_f1": float(np.mean([x["f1"] for x in items])),
                        "qa_unanswerable": float(np.mean([x["unanswerable"] for x in items])),
                        "qa_n": len(items), "qa_items": items}
    return out


def _prf(n_hit_p, n_p, n_hit_r, n_r):
    p = n_hit_p / n_p if n_p else np.nan
    r = n_hit_r / n_r if n_r else np.nan
    f = 2 * p * r / (p + r) if (n_p and n_r and (p + r) > 0) else (0.0 if (n_p and n_r) else np.nan)
    return p, r, f


def alignment(summary: str, source: str, reference: str | None, ents_sum: list[str], ents_ref: list[str]) -> dict:
    src_idx = KeyIndex(source)
    sum_idx = KeyIndex(summary)
    res = {}
    # entities
    ent_ok = [entity_in(e, src_idx) for e in ents_sum]
    res["n_ents"] = len(ents_sum)
    res["ent_prec_src"] = float(np.mean(ent_ok)) if ents_sum else np.nan
    res["unsupported_ents"] = [e for e, ok in zip(ents_sum, ent_ok) if not ok]
    # numbers and dates
    sup, uns = numbers_supported(summary, source)
    res["n_nums"] = len(sup) + len(uns)
    res["num_prec_src"] = len(sup) / res["n_nums"] if res["n_nums"] else np.nan
    res["unsupported_nums"] = [n.raw for n in uns]
    dsup, duns = dates_supported(summary, source)
    res["n_dates"] = len(dsup) + len(duns)
    res["date_prec_src"] = len(dsup) / res["n_dates"] if res["n_dates"] else np.nan
    res["unsupported_dates"] = [f"{d}/{m}" for d, m in duns]
    res["fact_halluc"] = float(bool(res["unsupported_ents"] or res["unsupported_nums"] or res["unsupported_dates"]))
    if reference:
        ref_idx = KeyIndex(reference)
        hit_p = sum(entity_in(e, ref_idx) for e in ents_sum)
        hit_r = sum(entity_in(e, sum_idx) for e in ents_ref)
        res["ent_prec_ref"], res["ent_rec_ref"], res["ent_f1_ref"] = _prf(hit_p, len(ents_sum), hit_r, len(ents_ref))
        s_ref, u_ref = numbers_supported(summary, reference)
        r_in, r_out = numbers_supported(reference, summary)
        res["num_prec_ref"], res["num_rec_ref"], res["num_f1_ref"] = _prf(len(s_ref), len(s_ref) + len(u_ref),
                                                                          len(r_in), len(r_in) + len(r_out))
    else:
        for k in ("ent_prec_ref", "ent_rec_ref", "ent_f1_ref", "num_prec_ref", "num_rec_ref", "num_f1_ref"):
            res[k] = np.nan
    return res


def error_types(nli_sents: list[dict], align: dict, threshold: float) -> dict:
    """Label each unsupported summary sentence: number > date > entity > other."""
    counts = {"err_number": 0, "err_date": 0, "err_entity": 0, "err_other": 0}
    for s in nli_sents:
        text = s["sent"]
        has_num = any(n in normalize_text(text) for n in align.get("unsupported_nums", []))
        bad_dates = set(align.get("unsupported_dates", []))
        has_date = any(f"{d}/{m}" in bad_dates for d, m in extract_dates(text)) if bad_dates else False
        has_ent = any(e in text for e in align.get("unsupported_ents", []))
        if has_num:
            counts["err_number"] += 1
        elif has_date:
            counts["err_date"] += 1
        elif has_ent:
            counts["err_entity"] += 1
        elif s["support"] < threshold:
            counts["err_other"] += 1
    return counts


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def verify_summaries(summ: pd.DataFrame, docs: pd.DataFrame, variants: pd.DataFrame, nli, ner, qa=None,
                     qg: pd.DataFrame | None = None, threshold: float = 0.5, log=print) -> pd.DataFrame:
    doc = docs.set_index("doc_id")
    var_text = {(r.doc_id, r.variant): (r.text, list(r.sents)) for r in variants.itertuples(index=False)}
    summ = summ.copy()
    summ["summary"] = summ["summary"].fillna("")
    # --- NER on summaries and references
    uniq_sum = sorted(set(summ["summary"]))
    refs = sorted({r for r in doc["summary_ref"].dropna() if isinstance(r, str) and r})
    log(f"NER on {len(uniq_sum)} summaries and {len(refs)} references")
    ents = dict(zip(uniq_sum + refs, ner(uniq_sum + refs)))
    # --- NLI vs mono source (primary) and vs actual input (secondary)
    jobs_mono, jobs_in = [], []
    for r in summ.itertuples(index=False):
        mono_sents = list(doc.at[r.doc_id, "sents"])
        jobs_mono.append((r.summary_id, r.summary, mono_sents))
        in_text, in_sents = var_text.get((r.doc_id, r.variant), (doc.at[r.doc_id, "text"], mono_sents))
        if r.variant not in ("mono",):
            jobs_in.append((r.summary_id, r.summary, in_sents))
    log("NLI against the original article")
    nli_mono = nli_consistency(jobs_mono, nli, threshold, log)
    log("NLI against the model's actual input")
    nli_in = nli_consistency(jobs_in, nli, threshold, log)
    # --- QA
    qa_res = {}
    if qa is not None and qg is not None and len(qg):
        qa_pairs = dict(zip(qg["summary_id"], qg["qa_pairs"]))
        jobs = [(r.summary_id, qa_pairs[r.summary_id], doc.at[r.doc_id, "text"])
                for r in summ.itertuples(index=False) if r.summary_id in qa_pairs]
        log("QA-based check against the original article")
        qa_res = qa_consistency(jobs, qa, log)
    # --- assemble
    rows = []
    for r in summ.itertuples(index=False):
        mono_text = doc.at[r.doc_id, "text"]
        ref = doc.at[r.doc_id, "summary_ref"]
        ref = ref if isinstance(ref, str) and ref else None
        in_text = var_text.get((r.doc_id, r.variant), (mono_text, None))[0]
        al = alignment(r.summary, mono_text, ref, ents.get(r.summary, []), ents.get(ref, []) if ref else [])
        al_in = alignment(r.summary, in_text, None, ents.get(r.summary, []), [])
        nm = nli_mono[r.summary_id]
        ni = nli_in.get(r.summary_id, nm)
        q = qa_res.get(r.summary_id, {"qa_f1": np.nan, "qa_unanswerable": np.nan, "qa_n": 0, "qa_items": []})
        et = error_types(nm["nli_sents"], al, threshold)
        rows.append({
            "summary_id": r.summary_id,
            **{k: nm[k] for k in ("nli_score", "nli_min", "nli_support_rate", "nli_halluc")},
            "nli_score_input": ni["nli_score"], "nli_halluc_input": ni["nli_halluc"],
            "qa_f1": q["qa_f1"], "qa_unanswerable": q["qa_unanswerable"], "qa_n": q["qa_n"],
            **{k: v for k, v in al.items() if not isinstance(v, list)},
            "ent_prec_input": al_in["ent_prec_src"], "num_prec_input": al_in["num_prec_src"],
            "fact_halluc_input": al_in["fact_halluc"],
            "any_halluc": float(bool(nm["nli_halluc"] == 1.0 or al["fact_halluc"] == 1.0)) if not (
                isinstance(nm["nli_halluc"], float) and math.isnan(nm["nli_halluc"])) else np.nan,
            **et,
            "summary_entities": json.dumps(ents.get(r.summary, []), ensure_ascii=False),
            "unsupported": json.dumps({"ents": al["unsupported_ents"], "nums": al["unsupported_nums"],
                                       "dates": al["unsupported_dates"]}, ensure_ascii=False),
            "nli_sents": json.dumps(nm["nli_sents"], ensure_ascii=False),
            "qa_items": json.dumps(q["qa_items"], ensure_ascii=False),
        })
    return pd.DataFrame(rows)
