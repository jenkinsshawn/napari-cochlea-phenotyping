"""Intervals and comparisons that resample the independent units, the mice.

Cells from one mouse are more alike than cells from different mice, and neighboring cells along a row are alike, so
intervals that resample single cells are far too narrow (they covered the true value about 60% of the time against
95% intended in simulation). The functions here:
  * cluster_bootstrap: resample whole mice, keeping each mouse's cells as they are, with a small-sample correction;
  * compare_groups_by_mouse: one value per mouse; exact relabeling P, leave-one-mouse-out, Welch t interval;
  * field_summary: count, median and middle half for one field, without an interval.
The simulations behind these choices are summarized in the README.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

MIN_MICE_FOR_INTERVAL = 3   # the small-sample correction was checked by simulation from 3 mice upward (see README)


def distinct_resamples(n: int) -> int:
    """Number of different resamples (with replacement, order ignored) of n units: C(2n-1, n)."""
    return comb(2 * n - 1, n) if n > 0 else 0


def _interval(draws: np.ndarray, level: float):
    a = (1 - level) / 2
    return float(np.nanquantile(draws, a)), float(np.nanquantile(draws, 1 - a))


def small_sample_factor(k: int, level: float = 0.95) -> float:
    """How much to stretch a percentile interval from resampling k units: sqrt(k/(k-1)) * t(k-1) / z.
    Resampling k units understates the spread by about sqrt((k-1)/k), and with few units the t distribution is wider
    than the normal one. In simulation (README) uncorrected intervals from resampling 3, 5 and 8 mice covered the true
    value 75%, 84% and 88% of the time against 95% intended; corrected ones covered 93-95% (means and medians,
    2,000 simulated studies per row, interval_coverage_correction_2026-11-12.csv)."""
    from scipy import stats
    if k < 2:
        return float("nan")
    q = 1 - (1 - level) / 2
    return float(np.sqrt(k / (k - 1)) * stats.t.ppf(q, k - 1) / stats.norm.ppf(q))


def _stretch(est, lo, hi, f):
    return est - (est - lo) * f, est + (hi - est) * f


def cluster_bootstrap(values, clusters, stat: Callable = np.mean, n_boot: int = 10000, level: float = 0.95,
                      seed: int = 0, per_cluster: bool = False, min_clusters: int = MIN_MICE_FOR_INTERVAL,
                      small_sample_correction: bool = True) -> dict:
    """Interval for stat(values) resampling whole clusters (e.g. mice) with replacement.
    per_cluster=False pools the cells of the drawn clusters; True applies stat to the per-cluster values
    (stat of each cluster first, then the mean of those), giving every mouse equal weight."""
    v = np.asarray(values, float); c = np.asarray(clusters)
    keep = ~np.isnan(v); v, c = v[keep], c[keep]
    units = pd.unique(c); k = len(units)
    groups = [v[c == u] for u in units]
    rng = np.random.default_rng(seed)
    if per_cluster:
        per = np.array([stat(g) for g in groups])
        est = float(np.mean(per))
        draws = np.array([np.mean(per[rng.integers(0, k, k)]) for _ in range(n_boot)])
    else:
        est = float(stat(v))
        draws = np.array([stat(np.concatenate([groups[i] for i in rng.integers(0, k, k)])) for _ in range(n_boot)])
    lo, hi = _interval(draws, level)
    f = small_sample_factor(k, level) if small_sample_correction else 1.0
    lo, hi = _stretch(est, lo, hi, f)
    return dict(estimate=est, low=lo, high=hi, level=level, n_values=int(v.size), n_clusters=k,
                distinct_resamples=distinct_resamples(k), correction_factor=f, interval_reliable=k >= min_clusters)


def field_summary(values) -> dict:
    """Description of one field's cells: count, median and the middle half (25th-75th percentile). No interval is
    given for a single field: neighboring cells along a row are alike (next-door correlation about 0.2-0.5 in the
    P1 data), and with 10-25 cells per field no resampling scheme tested gave intervals that held their stated
    coverage (README). Intervals come from resampling mice."""
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if v.size == 0:
        return dict(cells=0, median=np.nan, q25=np.nan, q75=np.nan)
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return dict(cells=int(v.size), median=float(med), q25=float(q25), q75=float(q75))


def compare_groups_by_mouse(df: pd.DataFrame, value: str, group: str, mouse: str, control, other,
                            cell_stat: Callable = np.median, level: float = 0.95) -> dict:
    """Compare two groups with one value per mouse (cell_stat of the mouse's cells).
    Returns the mouse table, the difference of group means (other minus control), an exact permutation P over every
    way of relabeling the mice (two-sided, |difference| at least as large), the smallest P the design allows, the
    difference with each mouse left out, and a Welch t interval for the difference from the mouse values (each group
    needs at least 2 mice). Resampling mice is not used here: with 2-3 mice per group it gives intervals that are far
    too narrow (README)."""
    d = df[df[group].isin([control, other])][[group, mouse, value]].dropna()
    m = d.groupby(mouse).agg(group=(group, "first"), cells=(value, "size"), value=(value, lambda s: cell_stat(s.to_numpy(float)))).reset_index()
    if (d.groupby(mouse)[group].nunique() > 1).any():
        raise ValueError("a mouse appears in both groups")
    a = m.loc[m.group == control, "value"].to_numpy(float); b = m.loc[m.group == other, "value"].to_numpy(float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("each group needs at least one mouse")
    diff = float(b.mean() - a.mean())
    allv = m["value"].to_numpy(float); n, kb = len(allv), len(b)
    perm = np.array([allv[list(ix)].mean() - np.delete(allv, list(ix)).mean() for ix in combinations(range(n), kb)])
    p = float(np.mean(np.abs(perm) >= abs(diff) - 1e-12))
    loo = []
    for i, r in m.iterrows():
        rest = m.drop(i)
        ra, rb = rest.loc[rest.group == control, "value"], rest.loc[rest.group == other, "value"]
        loo.append(dict(left_out=r[mouse], group=r["group"],
                        difference=float(rb.mean() - ra.mean()) if len(ra) and len(rb) else np.nan))
    out = dict(mice=m.rename(columns={mouse: "mouse"}), difference=diff, p_permutation=p,
               relabelings=len(perm), smallest_possible_p=float(np.min(np.mean(np.abs(perm)[None, :] >= np.abs(perm)[:, None] - 1e-12, axis=1))),
               leave_one_mouse_out=pd.DataFrame(loo), low=None, high=None, interval_note="")
    if len(a) >= 2 and len(b) >= 2:
        from scipy import stats
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = np.sqrt(va + vb)
        df_w = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else np.inf
        q = stats.t.ppf(1 - (1 - level) / 2, df_w)
        out["low"], out["high"] = diff - q * se, diff + q * se
        out["interval_note"] = f"Welch t interval from {len(a)} and {len(b)} mouse values ({df_w:.1f} degrees of freedom)"
    else:
        out["interval_note"] = (f"no interval: a group has one mouse ({len(a)} and {len(b)}); see each mouse's value and "
                                "the leave-one-mouse-out differences")
    return out
