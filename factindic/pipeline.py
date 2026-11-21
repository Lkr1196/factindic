"""The seven stages of FactIndic. Each Kaggle notebook calls one ``run_stageXX`` function.

01 data       – XL-Sum Hindi/Bengali (+ optional natural code-mixed docs): clean, filter, dedupe, sample
02 codemix    – LLM builds the paired code-mixed (cm) and romanised (roman) variants, validated
03 seq2seq    – mT5-XLSum and IndicBART-XLSum summaries; LoRA adaptation of mT5 to code-mixed input
04 llm        – LLM summaries with Track A / B / BV; question generation for the QA check
05 verify     – factuality verification (NLI, QA, entity/numeral/date alignment, error types)
06 evaluate   – ROUGE/METEOR/BERTScore/LaBSE, length/compression/time, tests, analysis, figures,
                human-evaluation sheets
07 human      – agreement (Cohen's kappa) and verifier-vs-human P/R/F1 once sheets are filled in
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis as A
from .config import Config
from .data import (add_mix_stats, load_jsonl, near_duplicate_ids, prepare_docs, prepare_natural, sample_df,
                   save_jsonl, split_articles)
from .handoff import bring_forward, cleanup_jobs, record_env, require
from .textnorm import script_profile


def _log_df(df: pd.DataFrame, log=print, n=20):
    try:
        from IPython.display import display

        display(df.head(n))
    except Exception:
        log(df.head(n).to_string())


def _md(df: pd.DataFrame, digits=2) -> str:
    if df is None or not len(df):
        return "(empty)\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in d.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines) + "\n"


# ======================================================================================
# 01 data
# ======================================================================================
def run_stage01(cfg: Config, log=print) -> pd.DataFrame:
    from .data import load_user_natural

    st = cfg.path("01_data", "x").parent
    stats: dict = {}
    all_docs = []
    for lang in cfg.langs:
        log(f"== {lang}: loading XL-Sum")
        if cfg.data_source == "mock":
            from .mock_data import make_docs

            raw = {s: pd.DataFrame(make_docs(lang, n, s, cfg.seed)) for s, n in
                   (("test", cfg.n_test * 3), ("validation", cfg.n_dev * 3), ("train", cfg.n_train_cm * 3))}
        else:
            from .data import load_xlsum

            raw = {s: load_xlsum(lang, s) for s in ("test", "validation", "train")}
        prep = {s: prepare_docs(df, lang, s, cfg, stats) for s, df in raw.items()}
        for s, df in prep.items():
            log(f"   {s}: {len(raw[s])} raw -> {len(df)} after filters")
        del raw  # the full articles are no longer needed; frees a few GB for Hindi
        import gc

        gc.collect()
        # test: sample candidates, drop near-duplicates of train/validation articles
        cand = sample_df(prep["test"], min(len(prep["test"]), cfg.n_test * 2), cfg.seed)
        pool = pd.concat([prep["train"], prep["validation"]], ignore_index=True)
        dups = near_duplicate_ids(cand, pool, cfg.near_dup_threshold) if len(pool) and len(cand) else set()
        dups |= near_duplicate_ids(cand, cand, cfg.near_dup_threshold) if len(cand) > 1 else set()
        stats[(lang, "test")]["near_dup_removed"] = len(dups)
        test = sample_df(cand[~cand["doc_id"].isin(dups)], cfg.n_test, cfg.seed)
        test["role"] = "test"
        dev = sample_df(prep["validation"], cfg.n_dev, cfg.seed)
        dev["role"] = "dev"
        parts = [test, dev]
        if cfg.n_train_cm > 0 and cfg.lora:
            tr_cand = sample_df(prep["train"], min(len(prep["train"]), cfg.n_train_cm * 2), cfg.seed)
            tr_d = near_duplicate_ids(tr_cand, test, cfg.near_dup_threshold) if len(tr_cand) and len(test) else set()
            train = sample_df(tr_cand[~tr_cand["doc_id"].isin(tr_d)], cfg.n_train_cm, cfg.seed)
            train["role"] = "train"
            parts.append(train)
        all_docs += parts
        log(f"   selected: {len(test)} test, {len(dev)} dev, {sum(len(p) for p in parts[2:])} train")
    # natural code-mixed documents (optional)
    nat_raw = load_user_natural()
    if cfg.scrape and not cfg.mock:
        from .scrape import scrape

        log("== scraping recent news (best effort)")
        nat_raw = pd.concat([nat_raw, scrape(cfg.langs, cfg.n_natural_per_lang, log=log)], ignore_index=True)
    nat = prepare_natural(nat_raw, cfg) if len(nat_raw) else pd.DataFrame()
    if cfg.data_source == "mock":  # a few naturally mixed toy docs so the code path runs
        from .mock_data import make_docs
        from .mock_models import CM_DICT

        toy = []
        for lang in cfg.langs:
            for d in make_docs(lang, cfg.n_natural_per_lang, "natural", cfg.seed):
                txt = " ".join(CM_DICT.get(w.strip("।,"), w) for w in d["text"].split())
                toy.append({"lang": lang, "text": txt, "source": "mock-natural"})
        nat = prepare_natural(pd.DataFrame(toy), cfg)
    if len(nat):
        nat = nat.groupby("lang").head(cfg.n_natural_per_lang).reset_index(drop=True)
        nat["natural_split"] = split_articles(nat, cfg.natural_split, cfg.seed)
        nat["role"] = "natural"
        all_docs.append(nat)
        log(f"== natural code-mixed docs: {nat.groupby('lang').size().to_dict()}")
    docs = pd.concat(all_docs, ignore_index=True)
    docs = add_mix_stats(docs)
    save_jsonl(docs, st / "docs.jsonl")
    srows = [{"lang": l, "split": s, **c} for (l, s), c in stats.items()]
    sdf = pd.DataFrame(srows)
    sdf.to_csv(st / "data_stats.csv", index=False)
    summary = docs.groupby(["lang", "role"]).agg(docs=("doc_id", "count"), words=("n_words", "mean"),
                                                 sents=("n_sents", "mean"), cmi=("cmi", "mean"),
                                                 ref_words=("summary_ref", lambda s: np.mean([len(x.split()) for x in s if isinstance(x, str)]) if s.notna().any() else np.nan)).reset_index()
    (st / "data_report.md").write_text("# Data\n\n## Filtering\n\n" + _md(sdf, 0) + "\n## Selected documents\n\n"
                                       + _md(summary) + "\n", encoding="utf-8")
    cfg.save("01_data")
    _log_df(summary, log)
    return docs


# ======================================================================================
# 02 code-mixed variants
# ======================================================================================
def run_stage02(cfg: Config, log=print) -> pd.DataFrame:
    from .variants import add_natural_variants, build_variants

    bring_forward(cfg, log)
    require(cfg, "01_data")
    st = cfg.path("02_codemix", "x").parent
    record_env(cfg, "02_codemix")
    docs = load_jsonl(Path(cfg.out_root) / "01_data" / "docs.jsonl")
    test = docs[docs["role"] == "test"]
    train = docs[docs["role"] == "train"].groupby("lang").head(cfg.n_train_cm if cfg.lora else 0)
    todo = pd.concat([test, train], ignore_index=True)
    log(f"Rewriting {len(test)} test and {len(train)} training articles")
    t0 = time.time()
    var = build_variants(todo, cfg, log)
    nat = docs[docs["role"] == "natural"]
    if len(nat):
        var = pd.concat([var, add_natural_variants(nat)], ignore_index=True)
    save_jsonl(var, st / "variants.jsonl")
    rep = var.groupby(["lang", "role", "variant"]).agg(
        docs=("doc_id", "count"), valid=("valid_frac", "mean"), fallback_sents=("n_fallback", "sum"),
        cmi=("cmi", "mean"), en_share=("en_token_share", "mean"), switch_rate=("switch_rate", "mean"),
        latin_chars=("latin_char_share", "mean"), gen_min=("gen_time_s", lambda s: s.sum() / 60)).reset_index()
    paired = var[var["role"] == "test"].groupby("lang")["paired_ok"].mean().rename("paired_ok_share").reset_index()
    (st / "codemix_report.md").write_text(
        "# Code-mixed variants\n\n" + _md(rep) + "\n## Share of test articles usable for paired comparison\n\n"
        + _md(paired) + f"\nTotal time: {(time.time() - t0) / 60:.1f} min\n", encoding="utf-8")
    _write_codemix_check_sheet(var, st / "codemix_quality_check.xlsx", cfg)
    cfg.save("02_codemix")
    cleanup_jobs(cfg)
    _log_df(rep, log)
    return var


def _write_codemix_check_sheet(var: pd.DataFrame, path: Path, cfg, n_per_lang: int = 20) -> None:
    """A small sheet for a human check of the synthetic code-mixed text (meaning kept? natural?)."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "Check"
    ws.append(["id", "lang", "original", "code-mixed", "romanised", "meaning_same (yes/no)",
               "cm_natural (1-5)", "roman_natural (1-5)", "comment"])
    for c in ws[1]:
        c.font = Font(bold=True)
    dv_yn = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    dv_15 = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True)
    ws.add_data_validation(dv_yn)
    ws.add_data_validation(dv_15)
    rng = np.random.default_rng(cfg.seed)
    test = var[var["role"] == "test"]
    row = 2
    for lang in cfg.langs:
        ids = sorted(set(test[test["lang"] == lang]["doc_id"]))
        for d in rng.choice(ids, size=min(n_per_lang, len(ids)), replace=False) if ids else []:
            g = test[test["doc_id"] == d].set_index("variant")
            if not {"mono", "cm", "roman"} <= set(g.index):
                continue
            k = int(rng.integers(0, len(g.at["mono", "sents"])))
            ws.append([f"{d}#{k}", lang, g.at["mono", "sents"][k], g.at["cm", "sents"][k], g.at["roman", "sents"][k]])
            for col in (3, 4, 5):
                ws.cell(row=row, column=col).alignment = Alignment(wrap_text=True, vertical="top")
            dv_yn.add(ws.cell(row=row, column=6))
            dv_15.add(ws.cell(row=row, column=7))
            dv_15.add(ws.cell(row=row, column=8))
            row += 1
    for col, w in zip("ABCDEFGHI", (24, 5, 60, 60, 60, 14, 14, 14, 30)):
        ws.column_dimensions[col].width = w
    wb.save(path)


# ======================================================================================
# 03 seq2seq baselines + LoRA
# ======================================================================================
def run_stage03(cfg: Config, log=print) -> pd.DataFrame:
    bring_forward(cfg, log)
    require(cfg, "01_data", "02_codemix")
    st = cfg.path("03_seq2seq", "x").parent
    docs = load_jsonl(Path(cfg.out_root) / "01_data" / "docs.jsonl")
    var = load_jsonl(Path(cfg.out_root) / "02_codemix" / "variants.jsonl")
    evalset = var[var["role"].isin(["test", "natural"])]
    rows = []

    def summarise(name, summarizer):
        for lang, g in evalset.groupby("lang"):
            t0 = time.time()
            if cfg.mock:
                outs = [{"summary": s, "time_s": 0.05, "batch_size": 1, "n_out": len(s) // 4}
                        for s in summarizer.summarize(list(g["text"]), lang)]
            else:
                outs = summarizer.summarize(list(g["text"]), lang)
            for r, o in zip(g.itertuples(index=False), outs):
                rows.append({"summary_id": f"{name}|-|{r.doc_id}|{r.variant}", "system": name, "model_id": name,
                             "track": "-", "doc_id": r.doc_id, "lang": lang, "variant": r.variant, "role": r.role,
                             "summary": o["summary"], "truncated": False, "raw": o["summary"], "n_in": 0,
                             "n_out": o["n_out"], "time_s": o["time_s"], "batch_size": o["batch_size"], "error": None})
            log(f"  {name} {lang}: {len(g)} summaries in {(time.time() - t0) / 60:.1f} min")

    def load(kind, model_id, adapter=None):
        if cfg.mock:
            from .mock_models import MockSeq2Seq

            return MockSeq2Seq()
        from .seq2seq import Seq2SeqSummarizer

        return Seq2SeqSummarizer(kind, model_id, adapter_dir=adapter)

    for kind in cfg.seq2seq_models:
        model_id = cfg.mt5_model if kind == "mt5" else cfg.indicbart_model
        try:
            log(f"== {kind}: {model_id}")
            m = load(kind, model_id)
            summarise(kind, m)
            del m
        except Exception as e:  # keep going: IndicBART's tokenizer is fragile across library versions
            log(f"!! {kind} failed: {type(e).__name__}: {e}. Continuing without it.")
        _free()
    lora_info = None
    train = var[(var["role"] == "train") & (var["valid_frac"] >= 0.8)]  # every valid variant of each training article
    if cfg.lora and len(train):
        ref = docs.set_index("doc_id")["summary_ref"]
        pairs = [(r.text, ref[r.doc_id]) for r in train.itertuples(index=False)]
        dev_ids = sorted(set(train["doc_id"]))[: max(2, len(set(train["doc_id"])) // 10)]
        dev_pairs = [p for p, r in zip(pairs, train.itertuples(index=False)) if r.doc_id in dev_ids]
        tr_pairs = [p for p, r in zip(pairs, train.itertuples(index=False)) if r.doc_id not in dev_ids]
        log(f"== LoRA on {cfg.mt5_model}: {len(tr_pairs)} train / {len(dev_pairs)} dev pairs (mono + cm + roman)")
        save_jsonl(pd.DataFrame(rows), st / "summaries_seq2seq.jsonl")  # keep the baselines if LoRA fails
        if cfg.mock:
            lora_info = {"mock": True}
            from .mock_models import MockSeq2Seq

            summarise("mt5-lora-cm", MockSeq2Seq())
        else:
            from .seq2seq import train_lora

            adapter = str(st / "mt5_lora_cm")
            n_before = len(rows)
            try:
                lora_info = train_lora(tr_pairs, dev_pairs, cfg, adapter, log)
                _free()
                summarise("mt5-lora-cm", load("mt5", cfg.mt5_model, adapter))
            except Exception as e:  # the two baselines are still worth keeping
                import traceback

                del rows[n_before:]
                lora_info = {"failed": f"{type(e).__name__}: {e}"}
                log("WARNING: LoRA fine-tuning failed, so the results have no mt5-lora-cm system. The mT5 and "
                    "IndicBART baselines are saved. Error:\n" + traceback.format_exc()[-1500:])
            _free()
        (st / "lora_training.json").write_text(json.dumps(lora_info, indent=2, default=str), encoding="utf-8")
    out = pd.DataFrame(rows)
    save_jsonl(out, st / "summaries_seq2seq.jsonl")
    cfg.save("03_seq2seq")
    return out


def _free():
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


# ======================================================================================
# 04 LLM tracks + question generation
# ======================================================================================
def run_stage04(cfg: Config, log=print) -> pd.DataFrame:
    from .summarize import generate_questions, llm_summarize

    bring_forward(cfg, log)
    require(cfg, "01_data", "02_codemix")
    if not (Path(cfg.out_root) / "03_seq2seq" / "summaries_seq2seq.jsonl").exists():
        log("WARNING: notebook 03's output (mT5 / IndicBART / LoRA baselines) is not attached. Stop now and attach it "
            "if the baselines should be in the results; otherwise this run covers the LLM only.")
    st = cfg.path("04_llm", "x").parent
    record_env(cfg, "04_llm")
    var = load_jsonl(Path(cfg.out_root) / "02_codemix" / "variants.jsonl")
    evalset = var[var["role"].isin(["test", "natural"])]
    t0 = time.time()
    llm = llm_summarize(evalset, cfg, log)
    save_jsonl(llm, st / "summaries_llm.jsonl")
    log(f"LLM summaries done in {(time.time() - t0) / 60:.1f} min")
    # QA subsample: the first k paired test docs per language + all natural docs
    s2s_path = Path(cfg.out_root) / "03_seq2seq" / "summaries_seq2seq.jsonl"
    if not s2s_path.exists():
        log("WARNING: no output of notebook 03 (mT5 / IndicBART / LoRA baselines) was found. Attach notebook 03's "
            "output and run again if the baselines should be in the results; continuing with the LLM only.")
    allsum = pd.concat([llm, load_jsonl(s2s_path)] if s2s_path.exists() else [llm], ignore_index=True)
    qa_docs = set()
    test = var[(var["role"] == "test") & (var["paired_ok"] == True)]  # noqa: E712
    for lang, g in test.groupby("lang"):
        qa_docs |= set(sorted(set(g["doc_id"]))[: cfg.qa_docs_per_lang])
    qa_docs |= set(var[var["role"] == "natural"]["doc_id"])
    qg = generate_questions(allsum[allsum["doc_id"].isin(qa_docs)], cfg, log)
    save_jsonl(qg, st / "qg.jsonl")
    cfg.save("04_llm")
    cleanup_jobs(cfg)
    rep = llm.groupby(["lang", "track", "variant"]).agg(n=("summary_id", "count"), time_s=("time_s", "mean"),
                                                        words=("summary", lambda s: np.mean([len(x.split()) for x in s])),
                                                        errors=("error", lambda s: int(s.notna().sum())),
                                                        cut_pct=("truncated", lambda s: 100 * float(np.mean(s.astype(bool)))),
                                                        empty_pct=("summary", lambda s: 100 * float(np.mean([not x.strip() for x in s])))
                                                        ).reset_index()
    bad = rep[(rep["cut_pct"] > 5) | (rep["empty_pct"] > 2)]
    if len(bad):
        log(f"WARNING: in {len(bad)} of {len(rep)} conditions more than 5% of the summaries were cut off at the token "
            f"limit or more than 2% are empty. Raise cfg.max_new_tokens_A / cfg.max_new_tokens_B and run this stage again; "
            f"otherwise conditions are not comparable.")
    if "bv_flagged" in llm.columns:
        fl = llm[llm["track"] == "BV"].groupby(["lang", "variant"])["bv_flagged"].mean().rename("BV flagged share").reset_index()
    else:
        fl = pd.DataFrame()
    (st / "llm_report.md").write_text("# LLM summaries\n\n" + _md(rep) + "\n## Track BV: share of B summaries flagged\n\n"
                                      + _md(fl) + f"\nQuestions generated for {len(qg)} summaries.\n", encoding="utf-8")
    _log_df(rep, log)
    return llm


# ======================================================================================
# 05 verification
# ======================================================================================
def _all_summaries(cfg) -> pd.DataFrame:
    parts = []
    for p in (Path(cfg.out_root) / "03_seq2seq" / "summaries_seq2seq.jsonl", Path(cfg.out_root) / "04_llm" / "summaries_llm.jsonl"):
        if p.exists():
            parts.append(load_jsonl(p))
    return pd.concat(parts, ignore_index=True)


def run_stage05(cfg: Config, log=print) -> pd.DataFrame:
    from .verify import verify_summaries

    bring_forward(cfg, log)
    require(cfg, "01_data", "02_codemix", "04_llm")
    st = cfg.path("05_verify", "x").parent
    docs = load_jsonl(Path(cfg.out_root) / "01_data" / "docs.jsonl")
    var = load_jsonl(Path(cfg.out_root) / "02_codemix" / "variants.jsonl")
    summ = _all_summaries(cfg)
    qg_path = Path(cfg.out_root) / "04_llm" / "qg.jsonl"
    qg = load_jsonl(qg_path) if qg_path.exists() else None
    if cfg.mock:
        from .entities import MockNER
        from .mock_models import MockNLI, MockQA

        nli, ner, qa = MockNLI(), MockNER(), MockQA()
    else:
        from .entities import NERTagger
        from .verify import NLIScorer, QAScorer

        log(f"Loading {cfg.ner_model}, {cfg.nli_model}, {cfg.qa_model}")
        ner, nli, qa = NERTagger(cfg.ner_model), NLIScorer(cfg.nli_model), QAScorer(cfg.qa_model)
    t0 = time.time()
    ver = verify_summaries(summ, docs, var, nli, ner, qa, qg, cfg.nli_threshold, log)
    save_jsonl(ver, st / "verification.jsonl")
    log(f"Verified {len(ver)} summaries in {(time.time() - t0) / 60:.1f} min")
    cfg.save("05_verify")
    m = summ.merge(ver, on="summary_id")
    rep = m.groupby(["lang", "system", "track", "variant"])[["nli_score", "any_halluc", "ent_prec_src", "num_prec_src",
                                                            "qa_f1"]].mean().reset_index()
    (st / "verify_report.md").write_text("# Verification (means, 0-1)\n\n" + _md(rep, 3), encoding="utf-8")
    _log_df(rep, log)
    return ver


# ======================================================================================
# 06 evaluation and analysis
# ======================================================================================
def run_stage06(cfg: Config, log=print) -> pd.DataFrame:
    from .annotate import build_sheets
    from .codemix import cmi_stats
    from .metrics import LaBSE, bertscore, length_stats, meteor, rouge

    bring_forward(cfg, log)
    require(cfg, "01_data", "02_codemix", "04_llm", "05_verify")
    st = cfg.path("06_evaluate", "x").parent
    (st / "figures").mkdir(exist_ok=True)
    (st / "tables").mkdir(exist_ok=True)
    docs = load_jsonl(Path(cfg.out_root) / "01_data" / "docs.jsonl")
    var = load_jsonl(Path(cfg.out_root) / "02_codemix" / "variants.jsonl")
    summ = _all_summaries(cfg)
    ver = load_jsonl(Path(cfg.out_root) / "05_verify" / "verification.jsonl")
    res = summ.merge(ver, on="summary_id", how="left")
    ref = docs.set_index("doc_id")["summary_ref"]
    vt = var.set_index(["doc_id", "variant"])
    res["reference"] = res["doc_id"].map(ref)
    res["input_text"] = [vt.at[(d, v), "text"] for d, v in zip(res["doc_id"], res["variant"])]
    res["paired_ok"] = [bool(vt.at[(d, v), "paired_ok"]) for d, v in zip(res["doc_id"], res["variant"])]
    res["input_cmi"] = [vt.at[(d, v), "cmi"] for d, v in zip(res["doc_id"], res["variant"])]
    res["input_latin_share"] = [vt.at[(d, v), "latin_char_share"] for d, v in zip(res["doc_id"], res["variant"])]
    res["summary"] = res["summary"].fillna("")
    res["empty"] = (res["summary"].str.strip() == "").astype(float)
    res["summary_latin_share"] = [script_profile(s)["latn"] for s in res["summary"]]
    log("ROUGE / METEOR / lengths")
    has_ref = res["reference"].notna()
    rg = [rouge(s, r) if isinstance(r, str) else {} for s, r in zip(res["summary"], res["reference"])]
    res = pd.concat([res, pd.DataFrame(rg, index=res.index)], axis=1)
    res["meteor"] = [meteor(s, r) if isinstance(r, str) else np.nan for s, r in zip(res["summary"], res["reference"])]
    from .metrics import official_xlsum_rouge

    off = np.full((len(res), 3), np.nan)
    for lang in res["lang"].unique():
        ii = np.where((res["lang"] == lang) & has_ref)[0]
        if len(ii):
            sc = official_xlsum_rouge(list(res["summary"].iloc[ii]), list(res["reference"].iloc[ii]), lang)
            if sc is not None:
                off[ii] = sc
    res["xlsum_rouge1"], res["xlsum_rouge2"], res["xlsum_rougeL"] = off[:, 0], off[:, 1], off[:, 2]
    if np.isnan(off).all():
        log("  (official XL-Sum ROUGE scorer not installed; only the Unicode ROUGE is reported)")
    ls = pd.DataFrame([length_stats(s, t) for s, t in zip(res["summary"], res["input_text"])], index=res.index)
    res = pd.concat([res, ls], axis=1)
    log("BERTScore / LaBSE")
    bs = np.full((len(res), 3), np.nan)
    idx = np.where(has_ref)[0]
    if len(idx):
        bs[idx] = bertscore(list(res["summary"].iloc[idx]), list(res["reference"].iloc[idx]), cfg.bertscore_model,
                            cfg.bertscore_layers, mock=cfg.mock)
    res["bertscore_p"], res["bertscore_r"], res["bertscore_f"] = bs[:, 0], bs[:, 1], bs[:, 2]
    lab = LaBSE(cfg.labse_model, mock=cfg.mock)
    sims = np.full(len(res), np.nan)
    if len(idx):
        sims[idx] = lab.similarity(list(res["summary"].iloc[idx]), list(res["reference"].iloc[idx]))
    res["labse_ref"] = sims
    del lab
    _free()
    res.to_csv(st / "results_all.csv", index=False)
    save_jsonl(res, st / "results_all.jsonl")

    log("Tables")
    test = res[res["role"] == "test"]
    pt = test[test["paired_ok"] == True]  # noqa: E712
    if not len(pt):
        log("!! No test article passed the code-mix validation in all variants (see 02_codemix/codemix_report.md). "
            "Paired tables and tests will be empty.")
    t = A.main_table(pt, n_boot=cfg.n_bootstrap)
    t.to_csv(st / "tables" / "main_results.csv", index=False)
    groups = {
        "content_meaning": ["rouge1_f", "rouge2_f", "rougeL_f", "xlsum_rougeL", "meteor", "bertscore_f", "labse_ref", "ent_f1_ref", "num_f1_ref"],
        "factuality": ["nli_score", "nli_halluc", "qa_f1", "ent_prec_src", "num_prec_src", "date_prec_src", "any_halluc"],
        "efficiency": ["summary_words", "compression_ratio", "time_s", "empty"],
    }
    md = ["# FactIndic results", "", f"Paired test articles only (n per cell in the `n` column). Mean ± half-width of the 95% bootstrap CI.", ""]
    for name, ms in groups.items():
        ms = [m for m in ms if m in t.columns]
        md += [f"## {name.replace('_', ' / ').title()}", "", A.table_markdown(t, ms), ""]
        (st / "tables" / f"{name}.tex").write_text(A.table_latex(t, ms, f"FactIndic {name.replace('_', ' ')} results.",
                                                                 f"tab:{name}"), encoding="utf-8")
    prf_cols = ["rouge1_p", "rouge1_r", "rouge1_f", "rouge2_p", "rouge2_r", "rouge2_f", "rougeL_p", "rougeL_r", "rougeL_f",
                "bertscore_p", "bertscore_r", "bertscore_f", "ent_prec_ref", "ent_rec_ref", "ent_f1_ref",
                "num_prec_ref", "num_rec_ref", "num_f1_ref"]
    prf = pt.groupby(A.GROUP)[[c for c in prf_cols if c in pt.columns]].mean().reset_index()
    for c in ("ent_prec_ref", "ent_rec_ref", "ent_f1_ref", "num_prec_ref", "num_rec_ref", "num_f1_ref"):
        if c in prf:
            prf[c] = prf[c] * 100
    prf.to_csv(st / "tables" / "precision_recall_f1.csv", index=False)
    md += ["## Precision / recall / F1", "", _md(prf, 1)]
    nat = res[res["role"] == "natural"]
    if len(nat):
        tn = A.main_table(nat, ["nli_score", "nli_halluc", "ent_prec_src", "num_prec_src", "any_halluc", "summary_words",
                                "compression_ratio", "time_s"], n_boot=cfg.n_bootstrap)
        tn.to_csv(st / "tables" / "natural_results.csv", index=False)
        md += ["## Naturally code-mixed documents (reference-free)", "",
               A.table_markdown(tn, ["nli_score", "any_halluc", "ent_prec_src", "num_prec_src", "summary_words"]), ""]
    (st / "results.md").write_text("\n".join(md), encoding="utf-8")

    log("Hypothesis tests")
    h1 = A.h1_tests(res, cfg.n_bootstrap)
    h2, did = A.h2_tests(res, cfg.n_bootstrap)
    h1.to_csv(st / "tables" / "h1_codemix_gap.csv", index=False)
    h2.to_csv(st / "tables" / "h2_tracks.csv", index=False)
    did.to_csv(st / "tables" / "h2_gap_reduction_did.csv", index=False)
    (st / "summary.md").write_text(A.key_findings(h1, h2, did), encoding="utf-8")

    log("Code-mixing analysis")
    cb = A.cmi_bins(res[res["role"].isin(["test", "natural"])], "nli_score")
    cb.to_csv(st / "tables" / "cmi_bins.csv", index=False)
    om = A.output_mix(res)
    om.to_csv(st / "tables" / "output_mix.csv", index=False)
    ep = A.error_profile(pt)
    ep.to_csv(st / "tables" / "error_types.csv", index=False)
    ft = pd.DataFrame()
    if not cfg.mock:
        toks = {}
        for name in cfg.tokenizers_for_fertility:
            try:
                from transformers import AutoTokenizer

                toks[name] = AutoTokenizer.from_pretrained(name)
            except Exception as e:
                log(f"  tokenizer {name} unavailable: {e}")
        ft = A.fertility_table(var[var["role"].isin(["test", "natural"])], toks)
    else:
        class _T:
            def __call__(self, t, add_special_tokens=False):
                return {"input_ids": list(t.encode("utf-8"))[::3]}
        ft = A.fertility_table(var[var["role"].isin(["test", "natural"])], {"mock/bytes": _T()})
    ft.to_csv(st / "tables" / "tokenizer_fertility.csv", index=False)
    ex = A.pick_examples(pt, docs, var)
    (st / "examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    log("Figures")
    fd = st / "figures"
    figs = [
        (A.fig_by_variant, (t, "nli_score", str(fd / "nli_support_by_input.png"), "NLI support of summary sentences by input type")),
        (A.fig_by_variant, (t, "any_halluc", str(fd / "hallucination_rate_by_input.png"), "Summaries with any detected hallucination (%)")),
        (A.fig_by_variant, (t, "rougeL_f", str(fd / "rougeL_by_input.png"), "ROUGE-L F1 against the XL-Sum reference")),
        (A.fig_gap, (h1, "nli_score", str(fd / "codemix_gap_nli.png"), "Code-mixing gap in NLI support (paired, 95% CI)")),
        (A.fig_gap, (h1, "any_halluc", str(fd / "codemix_gap_halluc.png"), "Code-mixing gap in hallucination rate (paired, 95% CI)")),
        (A.fig_cmi, (cb, "nli_score", str(fd / "nli_by_cmi.png"), "NLI support vs. degree of code-mixing")),
        (A.fig_fertility, (ft, str(fd / "tokenizer_fertility.png"))),
        (A.fig_errors, (ep, str(fd / "error_types.png"))),
        (A.fig_tradeoff, (t, str(fd / "time_vs_factuality.png"))),
    ]
    for fn, args in figs:
        try:
            if args[0] is not None and len(args[0]):
                fn(*args)
        except Exception as e:  # a figure must never stop the run
            log(f"  figure {Path(args[2] if isinstance(args[2], str) else args[1]).name} skipped: {type(e).__name__}: {e}")

    log("Human-evaluation sheets")
    llm_sys = [s for s in res["system"].unique() if s not in ("mt5", "indicbart", "mt5-lora-cm")]
    key = build_sheets(res, docs, str(st / "human_eval"), cfg, system=llm_sys[0] if llm_sys else None)
    log(f"  {len(key)} sentences to annotate in {st / 'human_eval'}")
    cfg.save("06_evaluate")
    print((st / "summary.md").read_text(encoding="utf-8"))
    return res


# ======================================================================================
# 07 human evaluation analysis
# ======================================================================================
def run_stage07(cfg: Config, annotation_files: list[str] | None = None, log=print, judge: str | None = None) -> dict:
    """Analyse the filled-in annotation sheets.

    If the folder with the sheets contains ``judge.json`` (written by ``annotate.apply_judge_labels``) the labels
    come from an LLM judge: results go to ``07_llm_judge`` and are reported as LLM-judge evaluation, without
    inter-annotator agreement. Otherwise they are human annotations and go to ``07_human``.
    """
    import glob
    import json

    from .annotate import analyse, load_annotations

    bring_forward(cfg, log)
    require(cfg, "06_evaluate")
    files = annotation_files or sorted(set(glob.glob("/kaggle/input/**/human_eval_*_annotator*.xlsx", recursive=True)))
    files = [f for f in files if "/factindic_out/" not in f]
    if not files:
        raise FileNotFoundError("No filled-in annotation files found. Upload the completed human_eval_*.xlsx files "
                                "as a Kaggle dataset and attach it.")
    info = {}
    for d in {str(Path(f).parent) for f in files}:
        if (Path(d) / "judge.json").exists():
            info = json.loads((Path(d) / "judge.json").read_text(encoding="utf-8"))
    judge = judge or info.get("judge", "human")
    llm_judge = judge == "llm"
    stage = "07_llm_judge" if llm_judge else "07_human"
    who = "LLM judge" if llm_judge else "human"
    st = cfg.path(stage, "x").parent
    key_path = Path(cfg.out_root) / "06_evaluate" / "human_eval" / "human_eval_KEY_do_not_share.csv"
    key = pd.read_csv(key_path)
    res = load_jsonl(Path(cfg.out_root) / "06_evaluate" / "results_all.jsonl")
    ann = load_annotations(files)
    log(f"{len(ann)} annotated rows from {len(files)} files ({who} labels)")
    out = analyse(ann, key, res, cfg.nli_threshold, judge=judge)
    if llm_judge:
        md = ["# LLM-judge evaluation", "",
              "The sentence labels below were produced by an LLM judge"
              + (f" ({info['name']})" if info.get("name") else "") + ", not by "
              "human annotators. The judge saw each article and its summaries without system, track or input labels.",
              "", f"Labelled sentences: {out['n_labels']}", ""]
        out["human_rates"] = out["human_rates"].rename(columns={"human_halluc_rate": "judge_halluc_rate"})
        (st / "judge.json").write_text(json.dumps(info or {"judge": "llm"}, indent=1), encoding="utf-8")
        tables = (("detection", "detection"), ("correlation", "correlation"), ("human_rates", "judge_rates"),
                  ("fluency_coverage", "fluency_coverage"), ("error_types", "error_types"))
    else:
        md = ["# Human evaluation", "", f"Labelled sentences: {out['n_labels']}", ""]
        tables = tuple((k, k) for k in ("agreement", "detection", "correlation", "human_rates", "fluency_coverage",
                                        "error_types"))
    for k, fname in tables:
        df = out[k]
        df.to_csv(st / f"{fname}.csv", index=False)
        md += [f"## {fname.replace('_', ' ').title()}", "", _md(df, 3), ""]
    cm_files = sorted(set(glob.glob("/kaggle/input/**/codemix_quality_check*.xlsx", recursive=True)))
    for d in {str(Path(f).parent) for f in files}:  # e.g. the Google Drive folder with the filled sheets
        cm_files += glob.glob(f"{d}/codemix_quality_check*.xlsx")
    cm_files = sorted(set(cm_files))
    cm_files = [f for f in cm_files if "/factindic_out/" not in f]
    if cm_files:
        cq = pd.concat([pd.read_excel(f, sheet_name="Check") for f in cm_files], ignore_index=True)
        cq = cq[cq.iloc[:, 5].notna()]
        if len(cq):
            agg = cq.groupby("lang").agg(n=("id", "count"),
                                         meaning_kept=(cq.columns[5], lambda s: 100 * (s.astype(str).str.lower() == "yes").mean()),
                                         cm_naturalness=(cq.columns[6], lambda s: pd.to_numeric(s, errors="coerce").mean()),
                                         roman_naturalness=(cq.columns[7], lambda s: pd.to_numeric(s, errors="coerce").mean())).reset_index()
            agg.to_csv(st / "codemix_quality.csv", index=False)
            md += [f"## Synthetic code-mixed text: {who} check", "", _md(agg, 2), ""]
    (st / ("llm_judge_report.md" if llm_judge else "human_eval_report.md")).write_text("\n".join(md), encoding="utf-8")
    cfg.save(stage)
    print("\n".join(md))
    return out
