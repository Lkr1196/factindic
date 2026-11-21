"""Fluency/overlap, semantic and efficiency metrics (notebook 06).

* ROUGE-1/2/L (precision, recall, F1) with a Unicode-aware tokenizer. The common ``rouge_score``
  default tokenizer deletes every non-ASCII character, so it would score Hindi/Bengali as empty.
* METEOR (Banerjee & Lavie, 2005), exact-match module. The WordNet synonym and Porter-stem modules
  are English-only, so they are switched off for Hindi/Bengali.
* BERTScore P/R/F1 with multilingual BERT (layer 9) – content/meaning preservation.
* LaBSE cosine similarity between summary and reference – script-agnostic meaning preservation.
* Average summary length (words), compression ratio (input words / summary words), inference time.
"""
from __future__ import annotations

import zlib
from collections import Counter

import numpy as np

from .textnorm import tokenize


# --------------------------------------------------------------------------------------
# ROUGE
# --------------------------------------------------------------------------------------
def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _lcs(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _prf(overlap, n_pred, n_ref):
    p = overlap / n_pred if n_pred else 0.0
    r = overlap / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def rouge(pred: str, ref: str) -> dict:
    pt, rt = tokenize(pred), tokenize(ref)
    out = {}
    for n in (1, 2):
        pc, rc = _ngrams(pt, n), _ngrams(rt, n)
        ov = sum((pc & rc).values())
        p, r, f = _prf(ov, sum(pc.values()), sum(rc.values()))
        out.update({f"rouge{n}_p": p, f"rouge{n}_r": r, f"rouge{n}_f": f})
    p, r, f = _prf(_lcs(pt, rt), len(pt), len(rt))
    out.update({"rougeL_p": p, "rougeL_r": r, "rougeL_f": f})
    return {k: 100 * v for k, v in out.items()}


# --------------------------------------------------------------------------------------
# METEOR (exact-match module)
# --------------------------------------------------------------------------------------
class _NoWordNet:
    @staticmethod
    def synsets(*_a, **_k):
        return []


class _NoStem:
    @staticmethod
    def stem(w):
        return w


def meteor(pred: str, ref: str) -> float:
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt or not rt:
        return 0.0
    try:
        from nltk.translate.meteor_score import single_meteor_score

        return 100 * single_meteor_score(rt, pt, preprocess=str.lower, stemmer=_NoStem(), wordnet=_NoWordNet())
    except Exception:
        return 100 * _meteor_exact(pt, rt)


def _meteor_exact(pt, rt, alpha=0.9, beta=3.0, gamma=0.5):
    used, align = set(), []
    for i, w in enumerate(pt):
        for j, v in enumerate(rt):
            if j not in used and w == v:
                used.add(j)
                align.append((i, j))
                break
    m = len(align)
    if m == 0:
        return 0.0
    p, r = m / len(pt), m / len(rt)
    fmean = p * r / (alpha * p + (1 - alpha) * r)
    chunks = 1
    for (i0, j0), (i1, j1) in zip(align, align[1:]):
        if not (i1 == i0 + 1 and j1 == j0 + 1):
            chunks += 1
    pen = gamma * (chunks / m) ** beta
    return fmean * (1 - pen)


# --------------------------------------------------------------------------------------
# Model-based similarity
# --------------------------------------------------------------------------------------
def bertscore(preds: list[str], refs: list[str], model_type: str, num_layers: int, batch_size: int = 64,
              mock: bool = False) -> np.ndarray:
    """Returns (N, 3) array of BERTScore P, R, F1 (x100)."""
    if mock:
        out = []
        for p, r in zip(preds, refs):
            pt, rt = set(tokenize(p)), set(tokenize(r))
            ov = len(pt & rt)
            out.append(_prf(ov, len(pt), len(rt)))
        return 100 * np.array(out)
    from bert_score import score

    pairs = [(p if p.strip() else ".", r if r.strip() else ".") for p, r in zip(preds, refs)]
    P, R, F = score([p for p, _ in pairs], [r for _, r in pairs], model_type=model_type, num_layers=num_layers,
                    batch_size=batch_size, verbose=False)
    return 100 * np.stack([P.numpy(), R.numpy(), F.numpy()], axis=1)


class LaBSE:
    def __init__(self, model_name: str, mock: bool = False):
        self.mock = mock
        if not mock:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mock:
            vecs = []
            for t in texts:
                v = np.zeros(512)
                for tok in tokenize(t):
                    v[zlib.crc32(tok.encode("utf-8")) % 512] += 1
                vecs.append(v / (np.linalg.norm(v) or 1))
            return np.array(vecs)
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        ea, eb = self.encode(a), self.encode(b)
        return 100 * (ea * eb).sum(1)


def official_xlsum_rouge(preds: list[str], refs: list[str], lang: str):
    """ROUGE as computed in the XL-Sum paper (language-specific tokenisation and stemming).

    Needs the scorer from the XL-Sum repository, which replaces the ``rouge_score`` module:
        pip install "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
    Returns an (N, 3) array of F1 x100, or None if that scorer is not installed. Our own
    Unicode ROUGE (``rouge``) has no stemming, so its numbers are lower than published XL-Sum numbers.
    """
    try:
        from rouge_score import rouge_scorer

        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True,
                                      lang={"hi": "hindi", "bn": "bengali"}[lang])
    except Exception:
        return None
    out = []
    for p, r in zip(preds, refs):
        s = sc.score(r, p)
        out.append([100 * s["rouge1"].fmeasure, 100 * s["rouge2"].fmeasure, 100 * s["rougeL"].fmeasure])
    return np.array(out)


def length_stats(summary: str, input_text: str) -> dict:
    sw = len(tokenize(summary))
    iw = len(tokenize(input_text))
    return {"summary_words": sw, "input_words": iw, "compression_ratio": iw / sw if sw else np.nan}
