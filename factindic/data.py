"""Data collection and preprocessing.

Pipeline (slide 7 of the deck): SCRAPE/LOAD -> NORMALISE -> SCRIPT-MIX DETECT -> DEDUPLICATE -> SPLIT.

XL-Sum (BBC article/summary pairs) is the monolingual backbone. We keep XL-Sum's *official*
train/validation/test splits: the public mT5 and IndicBART summarisers were fine-tuned on the
official training split, so re-splitting would leak their training articles into our test set.
Collected naturally code-mixed documents (scraped news, your own posts) are split 70/10/20 at
article level.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import shutil
from pathlib import Path

import pandas as pd

from .codemix import cmi_stats
from .config import LANGS
from .textnorm import (lead_sentences, normalize_text, script_profile, split_sentences, tokenize,
                       word_count)

# Boilerplate sentences in BBC Hindi/Bengali articles (app adverts, captions, "read more")
BOILERPLATE = [
    "यहां क्लिक", "यहाँ क्लिक", "एंड्रॉएड ऐप", "एंड्रॉइड ऐप", "फ़ेसबुक", "ट्विटर", "इंस्टाग्राम", "यूट्यूब",
    "इमेज कैप्शन", "इमेज स्रोत", "वीडियो कैप्शन", "ये भी पढ़ें", "यह भी पढ़ें", "बीबीसी हिन्दी", "बीबीसी हिंदी",
    "Getty Images", "getty images", "BBC", "ছবির ক্যাপশান", "ছবির ক্যাপশন", "ছবির উৎস", "ভিডিও ক্যাপশান",
    "আরো পড়ুন", "আরও পড়ুন", "আরো খবর", "আরও খবর", "বিবিসি বাংলা", "ফেসবুক", "টুইটার", "ইউটিউব",
]


def clean_sentences(text: str) -> list[str]:
    sents = split_sentences(text)
    out = []
    for s in sents:
        if any(b in s for b in BOILERPLATE):
            continue
        if word_count(s) < 3:
            continue
        s = re.sub(r"^\(|\)$", "", s).strip()
        out.append(s)
    return out


def text_hash(text: str) -> str:
    return hashlib.md5(" ".join(tokenize(text)).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# XL-Sum loading
# --------------------------------------------------------------------------------------
def _local_xlsum_files(cfgname: str, split: str) -> list[str]:
    pats = [f"/kaggle/input/**/{cfgname}*{split}*.parquet", f"/kaggle/input/**/{cfgname}/*{split}*.parquet",
            f"/kaggle/input/**/{cfgname}_{split}.jsonl", f"/kaggle/input/**/{cfgname}/{split}.jsonl"]
    files = []
    for p in pats:
        files += glob.glob(p, recursive=True)
    return sorted(set(files))


XLSUM_REPO = "csebuetnlp/xlsum"
PARQUET_REV = "refs/convert/parquet"   # Hugging Face's automatic Parquet copy of the dataset
_TAR_NAMES = {"train": "_train.jsonl", "validation": "_val.jsonl", "test": "_test.jsonl"}


def _xlsum_from_parquet_branch(cfgname: str, split: str) -> list[str]:
    """Download the split from the dataset's Parquet branch (no loading script involved)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    all_files = list_repo_files(XLSUM_REPO, repo_type="dataset", revision=PARQUET_REV)
    nested = [f for f in all_files if f.startswith(f"{cfgname}/{split}/") and f.endswith(".parquet")]
    top = [f for f in all_files if re.fullmatch(rf"{cfgname}/xlsum-{split}(-\d+-of-\d+)?\.parquet", f)]
    chosen = nested or top
    return [hf_hub_download(XLSUM_REPO, f, repo_type="dataset", revision=PARQUET_REV) for f in sorted(chosen)]


def _xlsum_from_archive(cfgname: str, split: str) -> list[str]:
    """Download the original archive (data/<language>_XLSum_v*.tar.bz2) and extract the split's jsonl once."""
    import tarfile

    from huggingface_hub import hf_hub_download, list_repo_files

    cache = Path(os.environ.get("FACTINDIC_CACHE", Path.home() / ".cache" / "factindic")) / "xlsum" / cfgname
    target = cache / f"{cfgname}{_TAR_NAMES[split]}"
    if not target.exists():
        files = list_repo_files(XLSUM_REPO, repo_type="dataset")
        arch = sorted(f for f in files if f.startswith("data/") and Path(f).name.lower().startswith(cfgname + "_")
                      and f.endswith((".tar.bz2", ".tar.gz", ".tgz", ".tar")))
        if not arch:
            return []
        path = hf_hub_download(XLSUM_REPO, arch[-1], repo_type="dataset")
        cache.mkdir(parents=True, exist_ok=True)
        wanted = {f"{cfgname}{suffix}" for suffix in _TAR_NAMES.values()}
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                name = Path(m.name).name
                if m.isfile() and name in wanted:
                    with tf.extractfile(m) as src, open(cache / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    return [str(target)] if target.exists() else []


def load_xlsum(lang: str, split: str) -> pd.DataFrame:
    """Load one XL-Sum split ('train' | 'validation' | 'test') as a DataFrame (id, url, title, summary, text).

    The Hub repository still ships an old loading script (xlsum.py), which recent versions of the
    ``datasets`` library refuse to run, so we read the data directly, trying in order:
    files attached on Kaggle -> the Parquet branch -> the original .tar.bz2 archive -> ``datasets``.
    """
    cfgname = LANGS[lang]["xlsum"]
    files, source = _local_xlsum_files(cfgname, split), "attached files"
    for name, fn in (("Parquet branch", _xlsum_from_parquet_branch), ("original archive", _xlsum_from_archive)):
        if files:
            break
        try:
            files, source = fn(cfgname, split), name
        except Exception as e:  # pragma: no cover - network dependent
            print(f"  XL-Sum {cfgname}/{split}: {name} failed ({type(e).__name__}: {str(e)[:150]})")
            files = []
    if files:
        parts = [pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f, lines=True) for f in files]
        df = pd.concat(parts, ignore_index=True)
    else:
        from datasets import load_dataset

        source = "datasets (Parquet revision)"
        df = load_dataset(XLSUM_REPO, cfgname, split=split, revision=PARQUET_REV).to_pandas()
    print(f"  XL-Sum {cfgname}/{split}: {len(df):,} articles from the {source}")
    df = df.drop_duplicates(subset=["id"]) if "id" in df.columns else df
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Filtering and preparation
# --------------------------------------------------------------------------------------
def prepare_docs(df: pd.DataFrame, lang: str, split: str, cfg, stats: dict) -> pd.DataFrame:
    native = LANGS[lang]["script_code"]
    rows, seen = [], set()
    c = {"raw": len(df), "length_filter": 0, "summary_filter": 0, "script_filter": 0, "exact_dup": 0}
    for r in df.itertuples(index=False):
        text = normalize_text(getattr(r, "text", ""))
        n_raw = word_count(text)
        if n_raw < cfg.min_words or n_raw > cfg.max_words_raw:
            c["length_filter"] += 1
            continue
        summ = normalize_text(getattr(r, "summary", ""))
        sw = word_count(summ)
        if sw < cfg.min_summary_words or sw > cfg.max_summary_words:
            c["summary_filter"] += 1
            continue
        sents = clean_sentences(text)
        sents = lead_sentences(sents, cfg.lead_words)[:cfg.max_sents]
        lead = " ".join(sents)
        if len(sents) < 3 or script_profile(lead)[native] < 0.7:
            c["script_filter"] += 1
            continue
        h = text_hash(lead)
        if h in seen:
            c["exact_dup"] += 1
            continue
        seen.add(h)
        rows.append({
            "doc_id": f"{lang}-{split}-{getattr(r, 'id', len(rows))}", "lang": lang, "split": split,
            "source": "xlsum", "title": normalize_text(getattr(r, "title", "")), "url": getattr(r, "url", ""),
            "sents": sents, "text": lead, "summary_ref": summ, "n_words": word_count(lead),
            "n_sents": len(sents), "n_words_full": n_raw, "hash": h,
        })
    c["kept"] = len(rows)
    stats[(lang, split)] = c
    cols = ["doc_id", "lang", "split", "source", "title", "url", "sents", "text", "summary_ref", "n_words",
            "n_sents", "n_words_full", "hash"]
    return pd.DataFrame(rows, columns=cols)


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = tokenize(text)
    return {" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1))}


def near_duplicate_ids(queries: pd.DataFrame, pool: pd.DataFrame, threshold: float = 0.8) -> set[str]:
    """doc_ids in ``queries`` whose 5-gram Jaccard with any ``pool`` doc is >= threshold (MinHash LSH)."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:  # small exact fallback
        pool_sh = [_shingles(t) for t in pool["text"]]
        out = set()
        for did, t in zip(queries["doc_id"], queries["text"]):
            s = _shingles(t)
            if any(len(s & p) / max(1, len(s | p)) >= threshold for p in pool_sh):
                out.add(did)
        return out
    num_perm = 64

    def signature(text):  # one document at a time: memory stays flat even for ~80k Hindi articles
        m = MinHash(num_perm=num_perm)
        m.update_batch([sh.encode("utf-8") for sh in _shingles(text)])
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for did, text in zip(pool["doc_id"], pool["text"]):
        lsh.insert(did, signature(text), check_duplication=False)
    out = set()
    for did, text in zip(queries["doc_id"], queries["text"]):
        if any(h != did for h in lsh.query(signature(text))):
            out.add(did)
    return out


def add_mix_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = [cmi_stats(t, l) for t, l in zip(df["text"], df["lang"])]
    for k in ("cmi", "en_token_share", "switch_rate", "latin_char_share"):
        df[k] = [s[k] for s in stats]
    return df


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0]
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_values("doc_id").reset_index(drop=True)


def split_articles(df: pd.DataFrame, ratios=(0.7, 0.1, 0.2), seed: int = 13) -> pd.Series:
    """Article-level random split -> Series of 'train'/'dev'/'test' aligned to df.index."""
    ids = list(df.index)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr = int(round(ratios[0] * n))
    n_dev = int(round(ratios[1] * n))
    lab = {}
    for i, idx in enumerate(ids):
        lab[idx] = "train" if i < n_tr else ("dev" if i < n_tr + n_dev else "test")
    return pd.Series(lab).reindex(df.index)


# --------------------------------------------------------------------------------------
# Naturally code-mixed documents (optional)
# --------------------------------------------------------------------------------------
def load_user_natural() -> pd.DataFrame:
    """Your own collected code-mixed documents.

    Put a CSV or JSONL with columns ``lang`` (hi/bn) and ``text`` (optional ``summary``, ``url``)
    in a Kaggle dataset whose file name contains ``natural``; attach it to notebook 01.
    """
    files = [f for f in glob.glob("/kaggle/input/**/*natural*.*", recursive=True)
             if f.endswith((".csv", ".jsonl", ".json"))]
    files += [f for f in glob.glob(os.environ.get("FACTINDIC_NATURAL_GLOB", "/nonexistent"), recursive=True)]
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f) if f.endswith(".csv") else pd.read_json(f, lines=f.endswith(".jsonl"))
            d["source"] = "user:" + Path(f).name
            parts.append(d)
        except Exception as e:
            print(f"  could not read {f}: {e}")
    if not parts:
        return pd.DataFrame(columns=["lang", "text", "summary", "url", "source"])
    return pd.concat(parts, ignore_index=True)


def prepare_natural(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(raw.itertuples(index=False)):
        lang = str(getattr(r, "lang", "")).strip()
        if lang not in cfg.langs:
            continue
        sents = clean_sentences(normalize_text(getattr(r, "text", "")))
        sents = lead_sentences(sents, cfg.lead_words)[:cfg.max_sents]
        text = " ".join(sents)
        if word_count(text) < 40:
            continue
        summ = getattr(r, "summary", None)
        summ = normalize_text(summ) if isinstance(summ, str) and summ.strip() else None
        rows.append({"doc_id": f"{lang}-natural-{i:05d}", "lang": lang, "split": "natural",
                     "source": getattr(r, "source", "natural"), "title": str(getattr(r, "title", "") or ""),
                     "url": str(getattr(r, "url", "") or ""), "sents": sents, "text": text,
                     "summary_ref": summ, "n_words": word_count(text), "n_sents": len(sents),
                     "n_words_full": word_count(text), "hash": text_hash(text)})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(subset=["hash"]).reset_index(drop=True)
    return df


def save_jsonl(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


def load_jsonl(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, float) and o != o:
        return None
    return str(o)
