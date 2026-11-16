"""Cell type and row of every outlined cell, from stains and position.

Uses the corrected outlines (E:\\Webber_confocal\\annotations\\cells) where they exist, otherwise
the computer outlines of --tag. For each field:
  1. Inner hair cells are the cells outlined from Calb2. A line is traced through them
     (sorted along the row, smoothed, extended at both ends).
  2. Every cell gets its distance along that line and across it (um). The outer side is the
     side with more bright, compact cells.
  3. Outer hair cells: on the outer side, 12-90 um2, not elongated (long/short axis < 2.2),
     phalloidin above the field's 25th percentile, and dark around the cell: the mean
     phalloidin in a ring 0.5-2 um outside the cell is less than 0.6 of the cell's own mean.
     Pillar cells in the bright band are bright all around (ratio 0.7-0.85).
  4. Band cells: on the outer side within 35 um of the line, bright around (ratio >= 0.6).
  5. Outer hair cell rows: distances across are taken from the rows' own center line (median of
     the outer hair cells within 20-25 um along), split into three groups, then re-centered on
     row 2 twice; row = offset / row spacing rounded to -1, 0, +1. If the three groups are not
     7-14 um apart, rows are taken as 10 um apart around the median offset.
  6. Inner hair cells: surface and body outlines of one cell are merged into one inner hair
     cell when their positions along the row are closer than half the cell width.
Output: E:\\Webber_confocal\\work\\cells\\types\\<field>_types.csv and figures\\cell_types\\<field>.png

Run from the scripts folder:
    python assign_cell_types.py            (all fields, about 2 minutes)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, glob, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from skimage.morphology import disk, binary_dilation
from skimage.segmentation import find_boundaries
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
FIELDS = os.path.join(S.WORK, 'fields')
CELLS = os.path.join(S.WORK, 'cells')
ANN = os.path.join(S.ANNOTATIONS, 'cells')
OUT = os.path.join(CELLS, "types")
RING_CUT = 0.6
COLORS = {"IHC": [1, 0.2, 1], "OHC1": [0.1, 1, 0.1], "OHC2": [0.1, 0.7, 1], "OHC3": [1, 1, 0.1],
          "band": [1, 0.5, 0], "other": [0.6, 0.6, 0.6]}


def stretch(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(float) - lo) / max(hi - lo, 1), 0, 1)


def load(field, tag):
    z = np.load(os.path.join(FIELDS, field + ".npz")); st = list(z["stains"]); px = float(z["voxel_um"][2])
    corr = os.path.join(ANN, field + "_cells.npy")
    if os.path.exists(corr):
        a = np.load(corr).astype(np.int32)
        given = pd.read_csv(os.path.join(ANN, field + "_cell_types.csv")).fillna("").rename(columns={"type": "given"})
        source = "corrected"
    else:
        a = np.load(os.path.join(CELLS, tag, field + ".npy")).astype(np.int32)
        g = pd.read_csv(os.path.join(CELLS, tag, field + "_cells.csv"))
        given = pd.DataFrame({"cell": g.cell, "given": np.where(g.get("source", "") == "calb2", "inner", "")})
        source = tag
    ph = stretch(z["surface_layers"][st.index("Phalloidin")])
    cb = stretch(z["maxproj"][st.index("Calb2")])
    eg = [i for i, s in enumerate(st) if s in ("Ocm", "PMCA2")]
    eg = stretch(z["surface_layers"][eg[0]]) if eg else np.zeros_like(ph)
    p = pd.DataFrame(regionprops_table(a, properties=("label", "centroid", "area", "major_axis_length", "minor_axis_length")))
    p.columns = ["cell", "y", "x", "area_px", "major", "minor"]
    p["area_um2"] = p.area_px * px * px
    ids = p.cell.values
    p["phal"] = ndi.mean(ph, a, ids); p["calb2"] = ndi.mean(cb, a, ids); p["egfp"] = ndi.mean(eg, a, ids)
    p = p.merge(given, on="cell", how="left").fillna({"given": ""})
    r = max(1, int(round(1.0 / px)))
    ring = []
    for c, y, x in zip(p.cell, p.y.astype(int), p.x.astype(int)):
        y0, x0 = max(0, y - 80), max(0, x - 80)
        m = a[y0:y + 80, x0:x + 80] == c
        rr = binary_dilation(m, disk(2 * r)) & ~binary_dilation(m, disk(max(1, r // 2)))
        win = ph[y0:y + 80, x0:x + 80]
        ring.append(win[rr].mean() / max(win[m].mean(), 1e-3))
    p["ring"] = ring
    return p, a, ph, px, st, source


def position(p, px):
    ih = p[p.given == "inner"]
    P = ih[["x", "y"]].values.astype(float)
    if len(P) < 3:
        p["along_um"] = np.nan; p["across_um"] = np.nan; return p
    mu = P.mean(0); vt = np.linalg.svd(P - mu)[2]; P = P[np.argsort((P - mu) @ vt[0])]
    if len(P) >= 4:
        k = np.ones(3) / 3
        P = np.c_[np.convolve(np.pad(P[:, 0], 1, mode="edge"), k, "valid"), np.convolve(np.pad(P[:, 1], 1, mode="edge"), k, "valid")]
    t0 = P[1] - P[0]; t1 = P[-1] - P[-2]
    P = np.vstack([P[0] - t0 / np.linalg.norm(t0) * 800, P, P[-1] + t1 / np.linalg.norm(t1) * 800])
    seg = np.vstack([np.linspace(P[i], P[i + 1], 50) for i in range(len(P) - 1)])
    tang = np.gradient(seg, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    X = p[["x", "y"]].values.astype(float)
    dist, idx = cKDTree(seg).query(X)
    v = X - seg[idx]; sgn = np.sign(tang[idx, 0] * v[:, 1] - tang[idx, 1] * v[:, 0])
    d = dist * sgn * px
    along = np.cumsum(np.r_[0, np.linalg.norm(np.diff(seg, axis=0), axis=1)])[idx] * px
    cand = ((p.phal > p.phal.median()) & (p.major / p.minor.clip(1) < 2.2)).values
    side = 1 if (d[cand] > 5).sum() >= (d[cand] < -5).sum() else -1
    p["along_um"] = along - np.nanmin(along[p.given == "inner"]) if (p.given == "inner").any() else along
    p["across_um"] = d * side
    return p


def classify(p):
    asp = p.major / p.minor.clip(1)
    p["type"] = "other"; p.loc[p.given == "inner", "type"] = "inner hair cell"
    side = (p.across_um > 5) & (p.across_um < 80) & (p.type == "other")
    ohc = side & (p.ring < RING_CUT) & (p.area_um2 > 12) & (p.area_um2 < 90) & (asp < 2.2) & (p.phal > np.percentile(p.phal, 25))
    p.loc[ohc, "type"] = "outer hair cell"
    p.loc[side & ~ohc & (p.ring >= RING_CUT) & (p.across_um < 35), "type"] = "band"
    p["row"] = np.where(p.type == "inner hair cell", "IHC", np.where(p.type == "band", "band", np.where(p.type == "other", "other", "")))
    o = (p.type == "outer hair cell").values
    if o.sum() < 6:
        p.loc[o, "row"] = "OHC"; return p
    al = p.along_um.values; ac = p.across_um.values; oi = np.where(o)[0]
    center = np.array([np.median(ac[oi][np.abs(al[oi] - al[i]) < 25]) if (np.abs(al[oi] - al[i]) < 25).any() else np.nan for i in range(len(p))])
    center = pd.Series(center).interpolate(limit_direction="both").values
    r = ac - center
    km = KMeans(3, n_init=10, random_state=0).fit(r[oi].reshape(-1, 1))
    cen = np.sort(km.cluster_centers_.ravel()); spacing = np.diff(cen).mean()
    if not 7 <= spacing <= 14:                                               # three groups not found; use 10 um rows around the median
        spacing = 10.0; cen = np.median(r[oi]) + np.array([-10.0, 0.0, 10.0])
    rows = np.clip(np.round((r - cen[1]) / spacing), -1, 1).astype(int) + 2
    p.loc[o, "row"] = [f"OHC{k}" for k in rows[o]]
    for _ in range(2):                                                       # re-center on row 2
        mid = np.where(p.row.values == "OHC2")[0]
        if len(mid) < 3: break
        center = np.array([np.median(ac[mid][np.abs(al[mid] - al[i]) < 20]) if (np.abs(al[mid] - al[i]) < 20).any() else np.nan for i in range(len(p))])
        center = pd.Series(center).interpolate(limit_direction="both").values
        rows = np.clip(np.round((ac - center) / spacing), -1, 1).astype(int) + 2
        p.loc[o, "row"] = [f"OHC{k}" for k in rows[o]]
    p["row_offset_um"] = ac - center
    p.attrs["row_spacing_um"] = round(float(spacing), 1)
    return p


def inner_cells(p):
    i = np.where(p.type.values == "inner hair cell")[0]; p["inner_cell_id"] = 0
    if len(i) == 0: return p
    big = i[p.area_um2.values[i] >= np.median(p.area_um2.values[i])]
    gap = 0.5 * np.median(np.sqrt(p.area_um2.values[big]))
    order = i[np.argsort(p.along_um.values[i])]; gid, start = 1, None
    for k in order:
        a = p.along_um.values[k]
        if start is not None and a - start > gap: gid += 1; start = a
        if start is None: start = a
        p.iloc[k, p.columns.get_loc("inner_cell_id")] = gid
    p.attrs["merge_gap_um"] = round(float(gap), 1)
    return p


def picture(p, a, ph, field, path):
    rgb = np.dstack([ph * 0.75] * 3); b = find_boundaries(a, mode="inner")
    lut = np.zeros((a.max() + 1, 3)); lut[:] = [1, 1, 1]
    for c, rw in zip(p.cell, p.row):
        lut[c] = COLORS.get(rw, [1, 1, 1])
    rgb[b] = lut[a[b]]
    fig, ax = plt.subplots(figsize=(11, 11)); ax.imshow(rgb); ax.axis("off")
    for rw, y, x in zip(p.row, p.y, p.x):
        if rw.startswith("OHC") and len(rw) == 4: ax.text(x, y, rw[-1], color="w", fontsize=7, ha="center", va="center")
    n = p.row.value_counts().to_dict()
    ax.set_title(f"{field}\ninner hair cells {p.inner_cell_id.max()} (magenta, {n.get('IHC', 0)} outlines); outer rows 1/2/3 "
                 f"{n.get('OHC1', 0)}/{n.get('OHC2', 0)}/{n.get('OHC3', 0)} (green/blue/yellow); band {n.get('band', 0)} (orange)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v1"); ap.add_argument("--only", default="")
    a_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); fig_dir = os.path.join(S.FIGURES, "cell_types"); os.makedirs(fig_dir, exist_ok=True)
    fields = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(CELLS, a_.tag, "*.npy")))]
    fields = [f for f in fields if a_.only in f]
    rows = []
    for f in fields:
        p, a, ph, px, st, source = load(f, a_.tag)
        p = position(p, px)
        if p.across_um.isna().all():
            print(f"  {f}: fewer than 3 inner hair cells, skipped", flush=True); continue
        p = inner_cells(classify(p))
        p.insert(0, "field", f); p["outlines"] = source
        keep = ["field", "cell", "type", "row", "inner_cell_id", "along_um", "across_um", "row_offset_um", "area_um2",
                "phal", "calb2", "egfp", "ring", "y", "x", "outlines"]
        p.reindex(columns=keep).round(3).to_csv(os.path.join(OUT, f + "_types.csv"), index=False)
        picture(p, a, ph, f, os.path.join(fig_dir, f + ".png"))
        n = p.row.value_counts().to_dict()
        rows.append(dict(field=f, outlines=source, inner_hair_cells=int(p.inner_cell_id.max()), inner_outlines=n.get("IHC", 0),
                         OHC1=n.get("OHC1", 0), OHC2=n.get("OHC2", 0), OHC3=n.get("OHC3", 0), band=n.get("band", 0), other=n.get("other", 0),
                         row_spacing_um=p.attrs.get("row_spacing_um"), stains=";".join(st)))
        print(f"  {f}: {rows[-1]}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(S.RESULTS, "cell_type_counts.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
