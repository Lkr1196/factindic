"""Statistics: bootstrap confidence intervals, paired tests, agreement.

All comparisons are *paired*: the same article appears in every condition, so we test the
per-article differences (Wilcoxon signed-rank for scores, McNemar for binary hallucination
flags) and report the mean difference with a 95% bootstrap confidence interval.
Multiple tests are corrected with Holm's method.
"""
from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return (float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_test(a, b, n_boot: int = 2000, seed: int = 0, binary: bool = False) -> dict:
    """Compare paired samples b - a (e.g. a = mono, b = code-mixed)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a
    res = {"n": int(len(d)), "mean_a": float(a.mean()) if len(a) else np.nan,
           "mean_b": float(b.mean()) if len(b) else np.nan}
    m, lo, hi = bootstrap_ci(d, n_boot, seed)
    res.update({"diff": m, "ci_lo": lo, "ci_hi": hi})
    if len(d) < 2 or np.allclose(d, 0):
        res.update({"p": 1.0 if len(d) else np.nan, "test": "none", "effect": 0.0})
        return res
    if binary:
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        res.update({"p": mcnemar_exact(n01, n10), "test": "McNemar", "effect": (n01 - n10) / max(1, n01 + n10),
                    "n01": n01, "n10": n10})
    else:
        from scipy.stats import wilcoxon

        try:
            p = float(wilcoxon(b, a, zero_method="wilcox").pvalue)
        except ValueError:
            p = 1.0
        nz = d[d != 0]
        ranks = _rank(np.abs(nz))
        rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
        res.update({"p": p, "test": "Wilcoxon", "effect": float(rb)})  # rank-biserial correlation
    return res


def _rank(x):
    from scipy.stats import rankdata

    return rankdata(x)


def mcnemar_exact(n01: int, n10: int) -> float:
    from scipy.stats import binomtest

    n = n01 + n10
    if n == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n, 0.5).pvalue)


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = [i for i in range(len(p)) if not np.isnan(p[i])]
    order = sorted(idx, key=lambda i: p[i])
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        adj = min(1.0, (m - k) * p[i])
        running = max(running, adj)
        out[i] = running
    return out.tolist()


def diff_in_diff(a_mono, a_cm, b_mono, b_cm, n_boot: int = 2000, seed: int = 0) -> dict:
    """Does Track B shrink the code-mixing gap? DiD = (cm_B - mono_B) - (cm_A - mono_A)."""
    arr = np.array([a_mono, a_cm, b_mono, b_cm], dtype=float)
    ok = ~np.isnan(arr).any(0)
    arr = arr[:, ok]
    gap_a = arr[1] - arr[0]
    gap_b = arr[3] - arr[2]
    did = gap_b - gap_a
    m, lo, hi = bootstrap_ci(did, n_boot, seed)
    p = np.nan
    if len(did) > 1 and not np.allclose(did, 0):
        from scipy.stats import wilcoxon

        try:
            p = float(wilcoxon(did).pvalue)
        except ValueError:
            p = 1.0
    return {"n": int(len(did)), "gap_A": float(gap_a.mean()) if len(did) else np.nan,
            "gap_B": float(gap_b.mean()) if len(did) else np.nan, "did": m, "ci_lo": lo, "ci_hi": hi, "p": p}


def cohen_kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score

    a, b = list(a), list(b)
    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return np.nan
    return float(cohen_kappa_score(a, b))


def fleiss_kappa(table: np.ndarray) -> float:
    """table: items x categories counts (each row sums to the number of raters)."""
    table = np.asarray(table, dtype=float)
    n = table.sum(1)
    if len(table) == 0 or not np.allclose(n, n[0]) or n[0] < 2:
        return np.nan
    n = n[0]
    p_j = table.sum(0) / table.sum()
    P_i = ((table ** 2).sum(1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else np.nan


def prf_binary(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if (tp and (p + r)) else (0.0 if tp + fp and tp + fn else np.nan)
    acc = float((y_true == y_pred).mean()) if len(y_true) else np.nan
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "n": int(len(y_true))}
