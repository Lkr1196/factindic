"""Human evaluation: annotation sheets, agreement, and verifier-vs-human scores (notebooks 06 and 07).

Protocol (slide 9 of the deck)
  * 2-3 bilingual annotators per language label every summary *sentence* against the original
    article as SUPPORTED / PARTIAL / HALLUCINATED, with an error type for the last two.
  * Each summary also gets fluency (1-5) and coverage (1-5) ratings.
  * Annotators are blind: summaries of the same article from different conditions are shuffled
    and carry no system/track/input labels. The key file stays with the organiser.
  * A share of the articles (default 25%) is labelled by two annotators to compute Cohen's kappa
    (target > 0.6).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, cohen_kappa, fleiss_kappa, prf_binary
from .textnorm import split_sentences

LABELS = ["Supported", "Partial", "Hallucinated"]
ERROR_TYPES = ["Entity", "Number", "Date", "Event/relation", "Extrinsic (not in article)",
               "Code-mix misreading", "Other"]
INSTRUCTIONS = [
    "FactIndic human evaluation - instructions",
    "",
    "You will see a news article followed by several short summaries of it (S1, S2, ...).",
    "Each summary is split into sentences. For every sentence, compare it with the ARTICLE only (not your own knowledge):",
    "  Supported    - everything in the sentence is stated in, or directly follows from, the article.",
    "  Partial      - mostly right, but one detail is unsupported, vague or slightly wrong.",
    "  Hallucinated - the main claim is not in the article or contradicts it (wrong name, number, date, event...).",
    "For Partial or Hallucinated, choose the error type:",
    "  Entity, Number, Date, Event/relation (who did what), Extrinsic (new information), Code-mix misreading",
    "  (an English or romanised word in the input was misunderstood), Other.",
    "On the FIRST sentence row of each summary also rate the whole summary:",
    "  Fluency 1-5  (5 = natural, grammatical Hindi/Bengali; 1 = unreadable)",
    "  Coverage 1-5 (5 = contains the main point of the article; 1 = misses it)",
    "Summaries come from different systems in random order. Do not try to guess which is which.",
    "Work at your own pace; save the file with the same name when you finish.",
]


def build_sheets(res: pd.DataFrame, docs: pd.DataFrame, out_dir: str, cfg, system: str | None = None,
                 tracks=("A", "B"), variants=("mono", "cm", "roman")) -> pd.DataFrame:
    """Write one .xlsx per annotator and language plus the hidden key. Returns the key."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    llm = res[res["track"].isin(tracks) & res["variant"].isin(variants) & (res["role"] == "test")]
    if system:
        llm = llm[llm["system"] == system]
    elif len(llm):
        system = llm["system"].iloc[0]
        llm = llm[llm["system"] == system]
    doc_text = docs.set_index("doc_id")["text"]
    key_rows = []
    for lang in cfg.langs:
        g = llm[(llm["lang"] == lang) & (llm["paired_ok"] == True)]  # noqa: E712
        cand = sorted(set(g["doc_id"]))
        rng.shuffle(cand)
        chosen = cand[: cfg.human_docs_per_lang]
        n_ann = max(1, cfg.n_annotators)
        n_overlap = int(round(cfg.human_overlap * len(chosen))) if n_ann > 1 else 0
        assign: dict[int, list[tuple[str, bool]]] = {k: [] for k in range(n_ann)}
        for i, d in enumerate(chosen):
            prim = i % n_ann
            assign[prim].append((d, True))
            if i < n_overlap:
                assign[(prim + 1) % n_ann].append((d, False))
        for k, docs_k in assign.items():
            if not docs_k:
                continue
            wb = Workbook()
            ws0 = wb.active
            ws0.title = "Instructions"
            for i, line in enumerate(INSTRUCTIONS, 1):
                ws0.cell(row=i, column=1, value=line)
            ws0.column_dimensions["A"].width = 120
            ws0["A1"].font = Font(bold=True, size=13)
            ws = wb.create_sheet("Annotate")
            head = ["item_id", "doc", "summary", "sent", "text", "label", "error_type", "fluency_1to5",
                    "coverage_1to5", "comment"]
            ws.append(head)
            for c in ws[1]:
                c.font = Font(bold=True)
            dv_lab = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
            dv_err = DataValidation(type="list", formula1='"' + ",".join(ERROR_TYPES) + '"', allow_blank=True)
            dv_15 = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True)
            for dv in (dv_lab, dv_err, dv_15):
                ws.add_data_validation(dv)
            art_fill = PatternFill("solid", fgColor="EEF3FB")
            grey = PatternFill("solid", fgColor="F2F2F0")
            row = 2
            rng.shuffle(docs_k)
            for dn, (d, primary) in enumerate(docs_k, 1):
                ws.cell(row=row, column=1, value=f"ARTICLE D{dn}")
                ws.cell(row=row, column=5, value=doc_text[d])
                ws.merge_cells(start_row=row, start_column=5, end_row=row, end_column=10)
                for c in range(1, 11):
                    ws.cell(row=row, column=c).fill = art_fill
                ws.cell(row=row, column=5).alignment = Alignment(wrap_text=True, vertical="top")
                ws.row_dimensions[row].height = min(400, 15 * (len(doc_text[d]) // 110 + 2))
                row += 1
                summ = g[g["doc_id"] == d].sample(frac=1, random_state=rng.randint(0, 10 ** 6))
                for sn, r in enumerate(summ.itertuples(index=False), 1):
                    sents = split_sentences(r.summary) or [r.summary or "(empty)"]
                    for si, s in enumerate(sents, 1):
                        iid = f"{lang}-a{k + 1}-D{dn}-S{sn}-{si}"
                        ws.append([iid, f"D{dn}", f"S{sn}", si, s, None, None, None, None, None])
                        ws.cell(row=row, column=5).alignment = Alignment(wrap_text=True, vertical="top")
                        dv_lab.add(ws.cell(row=row, column=6))
                        dv_err.add(ws.cell(row=row, column=7))
                        if si == 1:
                            dv_15.add(ws.cell(row=row, column=8))
                            dv_15.add(ws.cell(row=row, column=9))
                        else:
                            ws.cell(row=row, column=8).fill = grey
                            ws.cell(row=row, column=9).fill = grey
                        key_rows.append({"item_id": iid, "lang": lang, "annotator": f"{lang}-a{k + 1}", "primary": primary,
                                         "doc_id": d, "summary_id": r.summary_id, "system": r.system, "track": r.track,
                                         "variant": r.variant, "sent_idx": si - 1, "n_sents": len(sents), "sentence": s})
                        row += 1
            for col, wdt in zip("ABCDEFGHIJ", (20, 6, 8, 5, 90, 14, 22, 12, 12, 30)):
                ws.column_dimensions[col].width = wdt
            ws.freeze_panes = "A2"
            wb.save(out / f"human_eval_{lang}_annotator{k + 1}.xlsx")
    key = pd.DataFrame(key_rows)
    key.to_csv(out / "human_eval_KEY_do_not_share.csv", index=False)
    return key


def load_annotations(files: list[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        try:
            df = pd.read_excel(f, sheet_name="Annotate")
        except Exception as e:
            print(f"  skipping {f}: {e}")
            continue
        df = df[df["item_id"].astype(str).str.match(r"^\w\w-a\d+-D\d+-S\d+-\d+$")]
        m = re.search(r"human_eval_(\w\w)_annotator(\d+)", Path(f).name)
        df["annotator"] = f"{m.group(1)}-a{m.group(2)}" if m else Path(f).stem
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    ann = pd.concat(parts, ignore_index=True)
    ann["label"] = ann["label"].astype(str).str.strip().str.capitalize().where(ann["label"].notna())
    return ann


def _norm_item(iid: str) -> str:
    """Overlapping items have different ids per annotator; map to (summary_id, sentence)."""
    return iid


def analyse(ann: pd.DataFrame, key: pd.DataFrame, res: pd.DataFrame, threshold: float = 0.5,
            judge: str = "human") -> dict:
    """Agreement, verifier-vs-judge P/R/F1, metric correlations and judged hallucination rates.

    ``judge="human"``: the sheets were filled in by annotators (agreement is computed on the overlap).
    ``judge="llm"``: one LLM judge labelled every sentence once; the copies of a sentence that sit in two
    sheets carry the same label, so agreement is not defined and is left out.
    """
    from scipy.stats import spearmanr

    m = ann.merge(key, on="item_id", how="inner", suffixes=("", "_key"))
    m = m[m["label"].isin(LABELS)]
    m["unit"] = m["summary_id"] + "#" + m["sent_idx"].astype(str)
    if judge == "llm":
        m = m.sort_values("primary", ascending=False).drop_duplicates("unit").assign(primary=True)
    out: dict = {"n_labels": int(len(m)), "judge": judge}
    # ---- agreement on doubly-annotated sentences
    agree_rows = []
    for lang, g in (m.groupby("lang") if judge != "llm" else []):
        wide = g.pivot_table(index="unit", columns="annotator", values="label", aggfunc="first")
        pairs = []
        cols = list(wide.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                both = wide[[cols[i], cols[j]]].dropna()
                if len(both) >= 5:
                    pairs.append((cols[i], cols[j], both))
        for a, b, both in pairs:
            agree_rows.append({"lang": lang, "pair": f"{a} vs {b}", "n": len(both),
                               "kappa_3class": cohen_kappa(both[a], both[b]),
                               "kappa_binary": cohen_kappa(both[a] == "Hallucinated", both[b] == "Hallucinated"),
                               "raw_agreement": float((both[a] == both[b]).mean())})
    out["agreement"] = pd.DataFrame(agree_rows)
    # ---- gold label per sentence: primary annotator
    gold = m[m["primary"] == True].drop_duplicates("unit")  # noqa: E712
    gold = gold.merge(res[["summary_id", "nli_sents", "unsupported", "any_halluc", "nli_score", "qa_f1",
                           "ent_prec_src", "num_prec_src"]], on="summary_id", how="left")
    auto_nli, auto_fact = [], []
    for r in gold.itertuples(index=False):
        sents = json.loads(r.nli_sents) if isinstance(r.nli_sents, str) else []
        sup = sents[r.sent_idx]["support"] if r.sent_idx < len(sents) else np.nan
        auto_nli.append(np.nan if np.isnan(sup) else float(sup < threshold))
        un = json.loads(r.unsupported) if isinstance(r.unsupported, str) else {}
        flagged = any(x and x in r.sentence for x in un.get("nums", []) + un.get("ents", []))
        auto_fact.append(float(flagged))
    gold["auto_nli"] = auto_nli
    gold["auto_fact"] = auto_fact
    gold["auto_any"] = ((gold["auto_nli"] == 1) | (gold["auto_fact"] == 1)).astype(float)
    det = []
    for strict, name in ((["Hallucinated"], "Hallucinated"), (["Hallucinated", "Partial"], "Hallucinated+Partial")):
        y = gold["label"].isin(strict).astype(int)
        for col, lab in (("auto_nli", "NLI < threshold"), ("auto_fact", "entity/number check"), ("auto_any", "NLI or entity/number")):
            ok = gold[col].notna()
            det.append({"gold": name, "detector": lab, **prf_binary(y[ok], gold.loc[ok, col].astype(int))})
    out["detection"] = pd.DataFrame(det)
    # ---- summary level: human support rate vs automatic scores
    summ = gold.groupby("summary_id").agg(human_support=("label", lambda s: float((s == "Supported").mean())),
                                          human_halluc=("label", lambda s: float((s == "Hallucinated").any())),
                                          lang=("lang", "first"), track=("track", "first"), variant=("variant", "first"),
                                          nli_score=("nli_score", "first"), qa_f1=("qa_f1", "first"),
                                          ent_prec_src=("ent_prec_src", "first"), num_prec_src=("num_prec_src", "first"),
                                          any_halluc=("any_halluc", "first"))
    corr = []
    for col in ("nli_score", "qa_f1", "ent_prec_src", "num_prec_src"):
        d = summ[["human_support", col]].dropna()
        if len(d) >= 5:
            rho, p = spearmanr(d["human_support"], d[col])
            corr.append({"metric": col, "n": len(d), "spearman_rho": rho, "p": p})
    out["correlation"] = pd.DataFrame(corr)
    rates = []
    for (lang, track, variant), g in summ.groupby(["lang", "track", "variant"]):
        mean, lo, hi = bootstrap_ci(g["human_halluc"].values)
        rates.append({"lang": lang, "track": track, "variant": variant, "n": len(g),
                      "human_halluc_rate": 100 * mean, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                      "auto_any_halluc_rate": 100 * g["any_halluc"].mean()})
    out["human_rates"] = pd.DataFrame(rates)
    # ---- fluency / coverage
    first = m[(m["sent_idx"] == 0) & (m["primary"] == True)]  # noqa: E712
    fc = []
    for (lang, track, variant), g in first.groupby(["lang", "track", "variant"]):
        fc.append({"lang": lang, "track": track, "variant": variant,
                   "fluency": pd.to_numeric(g["fluency_1to5"], errors="coerce").mean(),
                   "coverage": pd.to_numeric(g["coverage_1to5"], errors="coerce").mean(), "n": len(g)})
    out["fluency_coverage"] = pd.DataFrame(fc)
    err = m[m["label"] != "Supported"]["error_type"].value_counts().rename_axis("error_type").reset_index(name="count")
    out["error_types"] = err
    return out


# ======================================================================================
# LLM judge: write one set of labels into the annotation sheets
# ======================================================================================
JUDGE_NOTE = ("NOTE: the labels in this file were produced by an LLM judge{name}, not by human annotators. "
              "The judge read each article and labelled the summaries blind (no system, track or input labels).")


def apply_judge_labels(sheet_dir: str, labels: pd.DataFrame, out_dir: str, judge_name: str = "") -> list[str]:
    """Copy the annotation sheets to ``out_dir`` and fill them from ``labels``.

    ``labels`` has one row per summary sentence: summary_id, sent_idx, label, error_type, and on the first
    sentence of a summary fluency, coverage and comment. Also writes ``judge.json`` so that later stages
    report the result as LLM-judge evaluation and never as human evaluation. ``judge_name`` is optional; when it
    is empty, no model name appears in the sheets, the report or the paper.
    """
    import glob

    from openpyxl import load_workbook

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(Path(sheet_dir) / "human_eval_KEY_do_not_share.csv")
    lab = labels.set_index(["summary_id", "sent_idx"])
    item = key.set_index("item_id")[["summary_id", "sent_idx"]]
    written = []
    for f in sorted(glob.glob(str(Path(sheet_dir) / "human_eval_*_annotator*.xlsx"))):
        wb = load_workbook(f)
        wb["Instructions"].insert_rows(1, 2)
        wb["Instructions"].cell(row=1, column=1, value=JUDGE_NOTE.format(name=f" ({judge_name})" if judge_name else ""))
        ws = wb["Annotate"]
        for row in ws.iter_rows(min_row=2):
            iid = row[0].value
            if iid not in item.index:
                continue
            k = (item.at[iid, "summary_id"], int(item.at[iid, "sent_idx"]))
            if k not in lab.index:
                continue
            r = lab.loc[k]
            for col, name in ((5, "label"), (6, "error_type"), (7, "fluency"), (8, "coverage"), (9, "comment")):
                v = r.get(name)
                if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
                    continue
                row[col].value = int(v) if name in ("fluency", "coverage") else str(v)
        wb.save(out / Path(f).name)
        written.append(str(out / Path(f).name))
    (out / "judge.json").write_text(json.dumps({"judge": "llm", **({"name": judge_name} if judge_name else {}), "blind": True,
                                                 "n_sentences": int(labels["label"].notna().sum())}, indent=1),
                                    encoding="utf-8")
    return written
