"""Tables, hypothesis tests, code-mixing analysis and figures (notebook 06).

Inputs are one merged results table with one row per summary:
    summary_id, system, track, lang, variant, doc_id, role, paired_ok, ...metrics...
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, diff_in_diff, holm, paired_test

# metric -> (label, objective, higher_is_better, binary)
METRICS = {
    "rouge1_f": ("ROUGE-1 F1", "content", True, False),
    "rouge2_f": ("ROUGE-2 F1", "content", True, False),
    "rougeL_f": ("ROUGE-L F1", "content", True, False),
    "meteor": ("METEOR", "content", True, False),
    "xlsum_rougeL": ("ROUGE-L (XL-Sum scorer)", "content", True, False),
    "bertscore_f": ("BERTScore F1", "meaning", True, False),
    "labse_ref": ("LaBSE sim. (ref)", "meaning", True, False),
    "ent_rec_ref": ("Entity recall (ref)", "content", True, False),
    "ent_f1_ref": ("Entity F1 (ref)", "content", True, False),
    "num_rec_ref": ("Numeral recall (ref)", "content", True, False),
    "num_f1_ref": ("Numeral F1 (ref)", "content", True, False),
    "nli_score": ("NLI support", "factual", True, False),
    "nli_support_rate": ("NLI supported sents", "factual", True, False),
    "nli_halluc": ("NLI hallucination rate", "factual", False, True),
    "qa_f1": ("QA consistency F1", "factual", True, False),
    "ent_prec_src": ("Entity precision (src)", "factual", True, False),
    "num_prec_src": ("Numeral precision (src)", "factual", True, False),
    "date_prec_src": ("Date precision (src)", "factual", True, False),
    "fact_halluc": ("Entity/number/date halluc. rate", "factual", False, True),
    "any_halluc": ("Any hallucination rate", "factual", False, True),
    "summary_words": ("Avg. summary length (words)", "efficiency", None, False),
    "compression_ratio": ("Compression ratio", "efficiency", None, False),
    "time_s": ("Inference time (s/summary)", "efficiency", False, False),
    "empty": ("Empty outputs (%)", "efficiency", False, True),
}
H1_METRICS = ["nli_score", "nli_halluc", "qa_f1", "ent_prec_src", "num_prec_src", "fact_halluc", "any_halluc"]
H2_QUALITY = ["rougeL_f", "bertscore_f", "labse_ref"]
GROUP = ["lang", "system", "track", "variant"]
VARIANT_ORDER = {"mono": 0, "cm": 1, "roman": 2, "natural": 3}
VARIANT_LABEL = {"mono": "Monolingual", "cm": "Code-mixed", "roman": "Romanised", "natural": "Natural CM"}
COLORS = {"mono": "#2a78d6", "cm": "#eb6834", "roman": "#1baf7a", "natural": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
FIG_SCALE = 1.0  # < 1 shrinks the canvas so the (fixed-size) text is relatively larger, e.g. for the paper


def _scale(metric: str, v):
    """Rates and [0,1] scores are shown as percentages."""
    if metric in ("nli_score", "nli_support_rate", "nli_halluc", "qa_f1", "ent_prec_src", "num_prec_src",
                  "date_prec_src", "fact_halluc", "any_halluc", "ent_rec_ref", "ent_f1_ref", "num_rec_ref",
                  "num_f1_ref", "ent_prec_ref", "num_prec_ref", "empty"):
        return v * 100
    return v


PRETTY = {"mt5": "mT5-XLSum", "indicbart": "IndicBART-XLSum", "mt5-lora-cm": "mT5 + LoRA (CM)"}
LANG_ORDER = {"hi": 0, "bn": 1}


def pretty_system(system: str) -> str:
    if system in PRETTY:
        return PRETTY[system]
    import re as _re

    s = _re.sub(r"-(instruct|chat|it)(-\d+)?$", "", system, flags=_re.I)
    return _re.sub(r"(\d+(?:\.\d+)?)b\b", lambda m: m.group(1) + "B", s).replace("qwen", "Qwen").replace("llama", "Llama")


def system_label(system: str, track: str) -> str:
    return f"{pretty_system(system)} · {track}" if track not in ("-", "", None) else pretty_system(system)


def _langs(df) -> list:
    return sorted(dict.fromkeys(df["lang"]), key=lambda l: LANG_ORDER.get(l, 9))


# --------------------------------------------------------------------------------------
# Main table
# --------------------------------------------------------------------------------------
def main_table(res: pd.DataFrame, metrics=None, n_boot=2000) -> pd.DataFrame:
    metrics = [m for m in (metrics or METRICS) if m in res.columns]
    rows = []
    for key, g in res.groupby(GROUP, sort=False):
        row = dict(zip(GROUP, key))
        row["n"] = len(g)
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].astype(float).values, n_boot)
            row[m] = _scale(m, mean)
            row[m + "_lo"] = _scale(m, lo)
            row[m + "_hi"] = _scale(m, hi)
        rows.append(row)
    t = pd.DataFrame(rows)
    if len(t):
        t["_v"] = t["variant"].map(VARIANT_ORDER)
        t = t.sort_values(["lang", "system", "track", "_v"]).drop(columns="_v").reset_index(drop=True)
    return t


def table_markdown(t: pd.DataFrame, metrics: list[str], ci: bool = True, digits: int = 1) -> str:
    cols = ["lang", "system", "track", "variant", "n"] + metrics
    head = "| " + " | ".join(METRICS.get(c, (c,))[0] if c in METRICS else c for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in t.itertuples(index=False):
        d = r._asdict()
        cells = []
        for c in cols:
            v = d.get(c)
            if c in metrics:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("–")
                elif ci and not math.isnan(d.get(c + "_lo", np.nan)):
                    half = (d[c + "_hi"] - d[c + "_lo"]) / 2
                    cells.append(f"{v:.{digits}f} ± {half:.{digits}f}")
                else:
                    cells.append(f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table_latex(t: pd.DataFrame, metrics: list[str], caption: str, label: str, digits: int = 1) -> str:
    cols = ["lang", "system", "track", "variant"] + metrics
    out = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{tabular}{" + "llll" + "r" * len(metrics) + "}", "\\toprule",
           " & ".join(["Lang", "System", "Track", "Input"] + [METRICS[m][0] for m in metrics]) + " \\\\", "\\midrule"]
    for r in t.itertuples(index=False):
        d = r._asdict()
        cells = [str(d["lang"]), str(d["system"]).replace("_", "\\_"), str(d["track"]), VARIANT_LABEL.get(d["variant"], d["variant"])]
        for m in metrics:
            v = d.get(m)
            cells.append("--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}")
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# Hypothesis tests
# --------------------------------------------------------------------------------------
def _pivot(res, metric, index="doc_id", column="variant"):
    return res.pivot_table(index=index, columns=column, values=metric, aggfunc="first")


def h1_tests(res: pd.DataFrame, n_boot=2000) -> pd.DataFrame:
    """H1: code-mixed input degrades factual consistency (paired, same documents)."""
    paired = res[(res["role"] == "test") & (res["paired_ok"] == True)]  # noqa: E712
    rows = []
    for (lang, system, track), g in paired.groupby(["lang", "system", "track"]):
        for m in H1_METRICS:
            if m not in g.columns:
                continue
            pv = _pivot(g, m)
            if "mono" not in pv.columns:
                continue
            for v in ("cm", "roman"):
                if v not in pv.columns:
                    continue
                t = paired_test(pv["mono"].values, pv[v].values, n_boot, binary=METRICS[m][3])
                rows.append({"lang": lang, "system": system, "track": track, "metric": m, "contrast": f"{v} - mono",
                             **{k: (_scale(m, t[k]) if k in ("mean_a", "mean_b", "diff", "ci_lo", "ci_hi") else t[k])
                                for k in t}})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].tolist())
        out["worse_with_cm"] = [(_worse(m, d)) for m, d in zip(out["metric"], out["diff"])]
    return out


def _worse(metric, diff):
    hib = METRICS[metric][2]
    if diff is None or (isinstance(diff, float) and math.isnan(diff)) or hib is None:
        return None
    return diff < 0 if hib else diff > 0


def h2_tests(res: pd.DataFrame, n_boot=2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H2: factuality-aware prompting reduces hallucination without hurting quality; plus DiD on the gap."""
    test = res[(res["role"] == "test") & (res["paired_ok"] == True)]  # noqa: E712
    rows, did_rows = [], []
    llm = test[test["track"].isin(["A", "B", "BV"])]
    for (lang, system, variant), g in llm.groupby(["lang", "system", "variant"]):
        for m in H1_METRICS + H2_QUALITY:
            if m not in g.columns:
                continue
            pv = _pivot(g, m, column="track")
            for a, b in (("A", "B"), ("A", "BV"), ("B", "BV")):
                if a in pv.columns and b in pv.columns:
                    t = paired_test(pv[a].values, pv[b].values, n_boot, binary=METRICS[m][3])
                    rows.append({"lang": lang, "system": system, "variant": variant, "metric": m,
                                 "contrast": f"{b} - {a}",
                                 **{k: (_scale(m, t[k]) if k in ("mean_a", "mean_b", "diff", "ci_lo", "ci_hi") else t[k])
                                    for k in t}})
    for (lang, system), g in llm.groupby(["lang", "system"]):
        for m in H1_METRICS:
            if m not in g.columns:
                continue
            pv = g.pivot_table(index="doc_id", columns=["track", "variant"], values=m, aggfunc="first")
            for tb in ("B", "BV"):
                for v in ("cm", "roman"):
                    need = [("A", "mono"), ("A", v), (tb, "mono"), (tb, v)]
                    if all(c in pv.columns for c in need):
                        d = diff_in_diff(*(pv[c].values for c in need), n_boot=n_boot)
                        did_rows.append({"lang": lang, "system": system, "metric": m, "track": tb, "variant": v,
                                         **{k: (_scale(m, d[k]) if k in ("gap_A", "gap_B", "did", "ci_lo", "ci_hi") else d[k])
                                            for k in d}})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].tolist())
    return out, pd.DataFrame(did_rows)


# --------------------------------------------------------------------------------------
# Code-mixing analysis
# --------------------------------------------------------------------------------------
def cmi_bins(res: pd.DataFrame, metric="nli_score", n_bins=4) -> pd.DataFrame:
    """Factuality by quantile bin of the input's Code-Mixing Index (cm/roman/natural inputs)."""
    from scipy.stats import spearmanr

    d = res[res["variant"].isin(["cm", "roman", "natural"])].dropna(subset=["input_cmi", metric])
    rows = []
    for (lang, system, track), g in d.groupby(["lang", "system", "track"]):
        if len(g) < 2 * n_bins:
            continue
        try:
            bins = pd.qcut(g["input_cmi"], n_bins, duplicates="drop")
        except ValueError:
            continue
        rho, p = spearmanr(g["input_cmi"], g[metric])
        for b, gg in g.groupby(bins, observed=True):
            rows.append({"lang": lang, "system": system, "track": track, "cmi_bin": str(b),
                         "cmi_mid": float(gg["input_cmi"].mean()), "n": len(gg),
                         metric: _scale(metric, float(gg[metric].mean())), "spearman_rho": rho, "spearman_p": p})
    return pd.DataFrame(rows)


def fertility_table(variants: pd.DataFrame, tokenizers: dict) -> pd.DataFrame:
    """Subword tokens per word for each input variant and tokenizer."""
    from .textnorm import tokenize

    rows = []
    for name, tok in tokenizers.items():
        for (lang, variant), g in variants.groupby(["lang", "variant"]):
            words = sum(len(tokenize(t)) for t in g["text"])
            pieces = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in g["text"])
            rows.append({"tokenizer": name, "lang": lang, "variant": variant, "docs": len(g),
                         "tokens_per_word": pieces / max(1, words)})
    return pd.DataFrame(rows)


def output_mix(res: pd.DataFrame) -> pd.DataFrame:
    """Does the model keep the mix? Share of Latin-script tokens in summaries vs. in the inputs."""
    rows = []
    for key, g in res.groupby(GROUP):
        rows.append({**dict(zip(GROUP, key)), "input_latin_share": 100 * g["input_latin_share"].mean(),
                     "summary_latin_share": 100 * g["summary_latin_share"].mean(), "n": len(g)})
    return pd.DataFrame(rows)


def error_profile(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["err_number", "err_date", "err_entity", "err_other"]
    rows = []
    for key, g in res.groupby(GROUP):
        rows.append({**dict(zip(GROUP, key)), "n": len(g), **{c: 100 * g[c].sum() / len(g) for c in cols if c in g}})
    return pd.DataFrame(rows)


def pick_examples(res: pd.DataFrame, docs: pd.DataFrame, variants: pd.DataFrame, n: int = 8) -> list[dict]:
    """Articles where the mono summary is clean but the code-mixed one hallucinates (Track A), with Track B."""
    a = res[(res["track"] == "A")]
    ex = []
    piv = a.pivot_table(index=["lang", "system", "doc_id"], columns="variant", values="any_halluc", aggfunc="first")
    if "mono" not in piv.columns:
        return ex
    for v in ("cm", "roman"):
        if v not in piv.columns:
            continue
        hits = piv[(piv["mono"] == 0) & (piv[v] == 1)].index.tolist()
        for lang, system, doc_id in hits[: n // 2]:
            get = lambda tr, vv: res[(res["doc_id"] == doc_id) & (res["system"] == system) & (res["track"] == tr)
                                     & (res["variant"] == vv)]
            rec = {"lang": lang, "system": system, "doc_id": doc_id, "variant": v,
                   "source_mono": docs.set_index("doc_id").at[doc_id, "text"],
                   "input": variants[(variants["doc_id"] == doc_id) & (variants["variant"] == v)]["text"].iloc[0],
                   "reference": docs.set_index("doc_id").at[doc_id, "summary_ref"]}
            for tr in ("A", "B", "BV"):
                for vv in ("mono", v):
                    r = get(tr, vv)
                    if len(r):
                        rec[f"{tr}_{vv}"] = r["summary"].iloc[0]
                        rec[f"{tr}_{vv}_unsupported"] = r["unsupported"].iloc[0]
            ex.append(rec)
    return ex


# --------------------------------------------------------------------------------------
# Figures (static PNG for the paper)
# --------------------------------------------------------------------------------------
def _fs(w, h):
    return (w * FIG_SCALE, h * FIG_SCALE)


def _style(ax, ylabel):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def fig_by_variant(t: pd.DataFrame, metric: str, path: str, title: str) -> None:
    """Grouped bars: x = system/track, colour = input variant, one panel per language."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = _langs(t)
    fig, axes = plt.subplots(1, len(langs), figsize=_fs(5.2 * len(langs), 3.4), squeeze=False, sharey=True)
    for ax, lang in zip(axes[0], langs):
        g = t[t["lang"] == lang]
        sys_keys = list(dict.fromkeys(zip(g["system"], g["track"])))
        variants = [v for v in VARIANT_ORDER if v in set(g["variant"])]
        w = 0.8 / max(1, len(variants))
        for k, v in enumerate(variants):
            xs, ys, lo, hi = [], [], [], []
            for i, (s, tr) in enumerate(sys_keys):
                r = g[(g["system"] == s) & (g["track"] == tr) & (g["variant"] == v)]
                if len(r) and not math.isnan(r[metric].iloc[0]):
                    xs.append(i + (k - (len(variants) - 1) / 2) * w)
                    ys.append(r[metric].iloc[0])
                    lo.append(r[metric].iloc[0] - r[metric + "_lo"].iloc[0])
                    hi.append(r[metric + "_hi"].iloc[0] - r[metric].iloc[0])
            ax.bar(xs, ys, width=w * 0.92, color=COLORS[v], label=VARIANT_LABEL[v], edgecolor="white", linewidth=1)
            err = np.array([lo, hi])
            err = np.where(np.isnan(err), 0, err)
            ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        ax.set_xticks(range(len(sys_keys)))
        ax.set_xticklabels([system_label(s, tr) for s, tr in sys_keys], rotation=25, ha="right", fontsize=8)
        ax.set_title({"hi": "Hindi", "bn": "Bengali"}.get(lang, lang), fontsize=10, color=INK)
        _style(ax, METRICS[metric][0] if ax is axes[0][0] else "")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gap(h1: pd.DataFrame, metric: str, path: str, title: str) -> None:
    """Paired difference (code-mixed minus mono) with 95% CI, per system/track."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = h1[h1["metric"] == metric]
    if not len(d):
        return
    langs = _langs(d)
    fig, axes = plt.subplots(1, len(langs), figsize=_fs(5.2 * len(langs), 3.2), squeeze=False, sharey=True)
    for ax, lang in zip(axes[0], langs):
        g = d[d["lang"] == lang]
        keys = list(dict.fromkeys(zip(g["system"], g["track"])))
        for k, contrast in enumerate(["cm - mono", "roman - mono"]):
            v = contrast.split(" ")[0]
            gg = g[g["contrast"] == contrast]
            xs, ys, lo, hi = [], [], [], []
            for i, (s, tr) in enumerate(keys):
                r = gg[(gg["system"] == s) & (gg["track"] == tr)]
                if len(r):
                    xs.append(i + (k - 0.5) * 0.3)
                    ys.append(r["diff"].iloc[0])
                    lo.append(r["diff"].iloc[0] - r["ci_lo"].iloc[0])
                    hi.append(r["ci_hi"].iloc[0] - r["diff"].iloc[0])
            err = np.where(np.isnan(np.array([lo, hi], dtype=float)), 0, np.array([lo, hi], dtype=float))
            ax.errorbar(xs, ys, yerr=err, fmt="o", color=COLORS[v], ms=6, elinewidth=1.4, capsize=2,
                        label=f"{VARIANT_LABEL[v]} - Monolingual")
        ax.axhline(0, color="#c3c2b7", linewidth=1)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([system_label(s, tr) for s, tr in keys], rotation=25, ha="right", fontsize=8)
        ax.set_title({"hi": "Hindi", "bn": "Bengali"}.get(lang, lang), fontsize=10, color=INK)
        _style(ax, f"Δ {METRICS[metric][0]}" if ax is axes[0][0] else "")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cmi(cb: pd.DataFrame, metric: str, path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not len(cb):
        return
    langs = _langs(cb)
    fig, axes = plt.subplots(1, len(langs), figsize=_fs(4.6 * len(langs), 3.2), squeeze=False, sharey=True)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for ax, lang in zip(axes[0], langs):
        g = cb[cb["lang"] == lang]
        for k, ((s, tr), gg) in enumerate(g.groupby(["system", "track"], sort=False)):
            gg = gg.sort_values("cmi_mid")
            ax.plot(gg["cmi_mid"], gg[metric], marker="o", ms=4, lw=2, color=palette[k % len(palette)],
                    label=system_label(s, tr))
        ax.set_xlabel("Input CMI (bin mean)", fontsize=9, color=INK2)
        ax.set_title({"hi": "Hindi", "bn": "Bengali"}.get(lang, lang), fontsize=10, color=INK)
        _style(ax, METRICS[metric][0] if ax is axes[0][0] else "")
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_fertility(ft: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not len(ft):
        return
    langs = _langs(ft)
    fig, axes = plt.subplots(1, len(langs), figsize=_fs(4.6 * len(langs), 3.0), squeeze=False, sharey=True)
    for ax, lang in zip(axes[0], langs):
        g = ft[ft["lang"] == lang]
        toks = list(dict.fromkeys(g["tokenizer"]))
        variants = [v for v in VARIANT_ORDER if v in set(g["variant"])]
        w = 0.8 / max(1, len(variants))
        for k, v in enumerate(variants):
            ys = [g[(g["tokenizer"] == t) & (g["variant"] == v)]["tokens_per_word"].mean() for t in toks]
            ax.bar([i + (k - (len(variants) - 1) / 2) * w for i in range(len(toks))], ys, width=w * 0.92,
                   color=COLORS[v], label=VARIANT_LABEL[v], edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels([t.split("/")[-1] for t in toks], rotation=20, ha="right", fontsize=8)
        ax.set_title({"hi": "Hindi", "bn": "Bengali"}.get(lang, lang), fontsize=10, color=INK)
        _style(ax, "Subword tokens per word" if ax is axes[0][0] else "")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Tokenizer fertility by input type", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_errors(ep: pd.DataFrame, path: str) -> None:
    """Small multiples: one panel per error type, bars by system/track, colour by variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [("err_number", "Number errors"), ("err_date", "Date errors"), ("err_entity", "Entity errors"),
            ("err_other", "Other unsupported")]
    cols = [c for c in cols if c[0] in ep.columns]
    if not len(ep) or not cols:
        return
    langs = _langs(ep)
    fig, axes = plt.subplots(len(langs), len(cols), figsize=_fs(3.4 * len(cols), 2.8 * len(langs)), squeeze=False,
                             sharey="row")
    for i, lang in enumerate(langs):
        g = ep[ep["lang"] == lang]
        keys = list(dict.fromkeys(zip(g["system"], g["track"])))
        variants = [v for v in VARIANT_ORDER if v in set(g["variant"])]
        w = 0.8 / max(1, len(variants))
        for j, (c, lab) in enumerate(cols):
            ax = axes[i][j]
            for k, v in enumerate(variants):
                ys = []
                for s, tr in keys:
                    r = g[(g["system"] == s) & (g["track"] == tr) & (g["variant"] == v)]
                    ys.append(r[c].iloc[0] if len(r) else 0)
                ax.bar([x + (k - (len(variants) - 1) / 2) * w for x in range(len(keys))], ys, width=w * 0.92,
                       color=COLORS[v], label=VARIANT_LABEL[v], edgecolor="white", linewidth=1)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels([system_label(s, tr) for s, tr in keys], rotation=35, ha="right", fontsize=7)
            ax.set_title(f"{lab} ({ {'hi': 'Hindi', 'bn': 'Bengali'}.get(lang, lang)})", fontsize=9, color=INK)
            _style(ax, "per 100 summaries" if j == 0 else "")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(t: pd.DataFrame, path: str, metric="nli_score") -> None:
    """Inference time vs. factuality, one point per system/track/variant (mono and cm only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = t[t["variant"].isin(["mono", "cm", "roman"])].dropna(subset=["time_s", metric])
    if not len(d):
        return
    langs = _langs(d)
    fig, axes = plt.subplots(1, len(langs), figsize=_fs(4.8 * len(langs), 3.4), squeeze=False, sharey=True)
    for ax, lang in zip(axes[0], langs):
        g = d[d["lang"] == lang]
        for v, gg in g.groupby("variant"):
            ax.scatter(gg["time_s"], gg[metric], s=40, color=COLORS[v], edgecolor="white", linewidth=1.5,
                       label=VARIANT_LABEL[v], zorder=3)
        for r in g[g["variant"] == "mono"].itertuples(index=False):
            ax.annotate(system_label(r.system, r.track), (r.time_s, getattr(r, metric)), fontsize=6.5, color=INK2,
                        xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Inference time per summary (s, log scale)", fontsize=9, color=INK2)
        ax.set_title({"hi": "Hindi", "bn": "Bengali"}.get(lang, lang), fontsize=10, color=INK)
        _style(ax, METRICS[metric][0] if ax is axes[0][0] else "")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Cost of factuality: time vs. NLI support", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Plain-language summary of key numbers
# --------------------------------------------------------------------------------------
def key_findings(h1: pd.DataFrame, h2: pd.DataFrame, did: pd.DataFrame) -> str:
    lines = ["# Key findings (auto-generated)", "",
             "All differences are paired over the same test articles; 95% bootstrap CIs; p-values Holm-corrected.", ""]
    if len(h1):
        lines.append("## H1 - does code-mixed input reduce factual consistency?")
        for r in h1[h1["metric"].isin(["nli_score", "any_halluc"])].itertuples(index=False):
            sig = "significant" if r.p_holm < 0.05 else "not significant"
            lines.append(f"- {r.lang} {system_label(r.system, r.track)}, {METRICS[r.metric][0]}: "
                         f"{r.mean_a:.1f} (mono) -> {r.mean_b:.1f} ({r.contrast.split(' ')[0]}), "
                         f"Δ = {r.diff:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}], p_holm = {r.p_holm:.3g} ({sig}).")
        lines.append("")
    if len(h2):
        lines.append("## H2 - does factuality-aware prompting help without hurting quality?")
        for r in h2[h2["metric"].isin(["nli_score", "any_halluc", "rougeL_f"])].itertuples(index=False):
            sig = "significant" if r.p_holm < 0.05 else "not significant"
            lines.append(f"- {r.lang} {r.system}, {r.variant} input, {METRICS[r.metric][0]}: {r.contrast} = "
                         f"{r.diff:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}], p_holm = {r.p_holm:.3g} ({sig}).")
        lines.append("")
    if len(did):
        lines.append("## Does Track B/BV shrink the code-mixing gap? (difference-in-differences)")
        for r in did[did["metric"].isin(["nli_score", "any_halluc"])].itertuples(index=False):
            lines.append(f"- {r.lang} {r.system}, {METRICS[r.metric][0]}, {r.variant}: gap A = {r.gap_A:+.1f}, "
                         f"gap {r.track} = {r.gap_B:+.1f}, DiD = {r.did:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}], p = {r.p:.3g}.")
    return "\n".join(lines) + "\n"
