"""Cell numbers for the clicking tools (click_bundle_points.py, click_directions.py).

The saved clicks name cells by their numbers in the first run's computer outlines, kept unchanged in
S.CLICK_OUTLINES. Measurement runs number cells in their own outlines (S.cell_labels_path: hand-corrected outlines
where they exist, otherwise the run's v1 outlines). This module matches each click-outline cell to the run cell
whose outline center is nearest, within MATCH_UM, and gives the run's tables keyed by click-outline numbers, so
the tools show the right cell and the right plate depth whichever run the settings point at.
"""
import os
import numpy as np
import pandas as pd
import pipeline_settings as S

MATCH_UM = 2.5
CUTOUTS = os.path.join(S.ANNOTATIONS, "click_reference_cutouts")   # first-run cut-out positions the samples name
_cache = {}


def _centers(lab):
    lab = np.asarray(lab); ys, xs = np.nonzero(lab); v = lab[ys, xs]; n = np.bincount(v)
    keep = np.nonzero(n)[0]; keep = keep[keep > 0]
    return keep, np.bincount(v, xs)[keep] / n[keep], np.bincount(v, ys)[keep] / n[keep]


def click_labels(field):
    """Click-outline label image of a field."""
    return np.load(os.path.join(S.CLICK_OUTLINES, field + ".npy"))


def run_cell_map(field):
    """{click-outline cell: run cell} for cells whose centers lie within MATCH_UM; unmatched cells are left out."""
    if field in _cache: return _cache[field]
    from scipy.spatial import cKDTree
    ca, xa, ya = _centers(click_labels(field)); cb, xb, yb = _centers(np.load(S.cell_labels_path(field)))
    d, k = cKDTree(np.c_[xb, yb]).query(np.c_[xa, ya], distance_upper_bound=MATCH_UM / S.PX_UM)
    m = {}; used = {}
    for a, dd, kk in zip(ca, d, k):              # one run cell per click cell: the nearer claim wins
        if np.isfinite(dd) and (kk not in used or dd < used[kk][1]):
            if kk in used: m.pop(used[kk][0], None)
            m[int(a)] = int(cb[kk]); used[kk] = (int(a), dd)
    _cache[field] = m
    return m


def rekey(df, field, col="cell"):
    """Rows of a run table for one field, with cell numbers changed to click-outline numbers (unmatched rows dropped)."""
    inv = {v: k for k, v in run_cell_map(field).items()}
    d = df[df[col].isin(inv)].copy(); d[col] = d[col].map(inv).astype(int)
    return d


def types(field):
    return rekey(pd.read_csv(os.path.join(S.WORK, "cells", "types", field + "_types.csv")), field)


def plate_depths(field):
    b = rekey(pd.read_csv(os.path.join(S.WORK, "bundles_detector", field + "_bundles.csv")), field)
    return dict(zip(b.cell, b.plate_depth_um))


def spatial(path):
    c = pd.read_csv(path)
    return pd.concat([rekey(g, f) for f, g in c.groupby("field")], ignore_index=True)


def cutout_number(kind, field, cell, tol_um=2.0, xy=None):
    """Number (1-based) of the current run's cut-out in S.WORK/<kind> showing the cell that a saved sample names as cut-out
    `cell` of the first run's stacks (or the cell at position xy, for samples that store positions); None when no current
    cut-out lies within tol_um."""
    cur = pd.read_csv(os.path.join(S.WORK, kind, field + "_cells.csv"))
    if xy is None:
        ref = pd.read_csv(os.path.join(CUTOUTS, kind, field + "_cells.csv")); xy = ref.x.iloc[int(cell) - 1], ref.y.iloc[int(cell) - 1]
    x, y = xy; d = np.hypot(cur.x.values - x, cur.y.values - y) * S.PX_UM
    k = int(np.argmin(d)) if len(d) else -1
    return k + 1 if k >= 0 and d[k] <= tol_um else None
