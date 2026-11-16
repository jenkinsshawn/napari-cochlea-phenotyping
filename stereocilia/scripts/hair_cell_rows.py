"""Hair cells and their rows, from phalloidin plates and the Calb2 band (settings frozen 10 November 2026).

Outer hair cells: one peak per cuticular plate. Phalloidin is averaged within 0.6 um of each
cell's plate depth, scaled by the field (median to 99th percentile), and filtered with a
Laplacian of Gaussian (sigma 2.3 um); peaks at least 4 um apart with score >= 0.05.
Across-row position: distance to the edge of the Calb2 band (Calb2 surface layer smoothed by
2 um, above 0.25 of the field's scale, largest region), positive on the side with the peaks.
Peaks past a band end that stops inside the image are not assessed.
Rows: stacks of peaks across the row (Gaussian kernel 1.2 um) holding at least 0.35 x
(row length / 7 um) peaks; each peak goes to the nearest stack within 4 um, each stack may drift
linearly along the field. Stacks are named by the nearest usual position from the band edge
(pillar 4, OHC1 17, OHC2 26.5, OHC3 35, OHC4 44 um; within 4.5 um).
Field row check: OHC1, OHC2 and OHC3 present, 7-12 um apart, at least half the peaks assessed.
Inner hair cells: Cellpose outlines of the log-scaled Calb2 projection (ihc_cellpose.py, <field>_ihc_log.npy;
since 13 Nov 2026, before that <field>_ihc.npy, which still builds the band), in the band's outer
20 um (-20 to +3 um), at least 20 um2 and Calb2 >= 0.2; outlines grouped along the band axis with a
new cell after a gap above 4 um, at most two outlines (top and body) per cell.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
from skimage.measure import label as cclabel, regionprops
from skimage.morphology import closing, opening, disk

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
W = S.WORK
IHC_DIR = os.path.join(W, "cells", "ihc_calb2")
IHC_CELL_OUTLINES = "_ihc_log.npy"   # inner hair cells from the log-scaled outlines (ihc_cellpose.py, 13 Nov 2026); the band uses _ihc.npy
PX = S.PX_UM
CANON = {"pillar": 4.0, "OHC1": 17.0, "OHC2": 26.5, "OHC3": 35.0, "OHC4": 44.0}


def _field(fl):
    z = np.load(os.path.join(W, "fields", fl + ".npz")); return z, [str(s) for s in z["stains"]]


def _scale(c, hi=99.5):
    c = c.astype(float); return (c - np.median(c)) / (np.percentile(c, hi) - np.median(c) + 1e-9)


def plate_image(F, half=0.6):
    ph, off, pmap = F["ph"].astype(float), F["off"], np.nan_to_num(F["pmap"], nan=0.6)
    w = (np.abs(off[:, None, None] - pmap[None]) <= half).astype(float)
    return (ph * w).sum(0) / np.maximum(w.sum(0), 1)


def plate_peaks(img, sig_um=2.3, min_um=4.0, thr=0.05):
    img = (img - np.median(img)) / (np.percentile(img, 99) - np.median(img))
    g = -ndi.gaussian_laplace(img, sig_um / PX) * (sig_um / PX) ** 2
    pk = peak_local_max(g, min_distance=int(min_um / PX), exclude_border=False); v = g[pk[:, 0], pk[:, 1]]
    p = pd.DataFrame(dict(x=pk[:, 1], y=pk[:, 0], v=v)); return p[p.v >= thr].reset_index(drop=True)


def calb2_surface(fl):
    z, st = _field(fl); return _scale(ndi.gaussian_filter(z["surface_layers"][st.index("Calb2")].astype(float), 1.0 / PX))


def calb2_projection(fl):
    z, st = _field(fl); return _scale(z["maxproj"][st.index("Calb2")])


def calb2_band(fl, thr=0.25):
    m = ndi.gaussian_filter(calb2_surface(fl), 2.0 / PX) > thr
    m = closing(m, disk(int(3 / PX))).astype(bool); m = opening(m, disk(int(2 / PX))).astype(bool)
    lab = cclabel(m)
    if lab.max() == 0: return None
    s = np.bincount(lab.ravel()); s[0] = 0; return ndi.binary_fill_holes(lab == s.argmax())


def band_frame(fl, peaks, end_margin_um=5.0, band=None):
    """Returns coords(xs, ys) -> (distance from band edge e in um, position along band axis u in um, assessed)."""
    m = calb2_band(fl) if band is None else band   # band: a Calb2 band mask given by the caller (frame_v2)
    if m is None: return None
    dout = ndi.distance_transform_edt(~m) * PX; din = ndi.distance_transform_edt(m) * PX; comp = cclabel(~m)
    H, Wd = m.shape
    px = peaks[["x", "y"]].values.astype(int); cnt = np.bincount(comp[px[:, 1], px[:, 0]], minlength=comp.max() + 1); cnt[0] = 0; outer = cnt.argmax()
    B = np.c_[np.where(m)[1], np.where(m)[0]].astype(float); mu = B.mean(0); vt = np.linalg.svd(B[::50] - mu, full_matrices=False)[2]
    ub = (B - mu) @ vt[0]; umin, umax = ub.min(), ub.max()
    def is_open(sel):
        return (B[sel, 0] <= 2).any() or (B[sel, 0] >= Wd - 3).any() or (B[sel, 1] <= 2).any() or (B[sel, 1] >= H - 3).any()
    lo_open = is_open(ub <= umin + 10 / PX); hi_open = is_open(ub >= umax - 10 / PX)
    def coords(xs, ys):
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        xi, yi = np.clip(xs.astype(int), 0, Wd - 1), np.clip(ys.astype(int), 0, H - 1)
        e = np.where(m[yi, xi], -din[yi, xi], np.where(comp[yi, xi] == outer, dout[yi, xi], np.nan))
        u = (np.c_[xs, ys] - mu) @ vt[0]; ok = np.ones(len(xs), bool)
        if not lo_open: ok &= u >= umin + end_margin_um / PX
        if not hi_open: ok &= u <= umax - end_margin_um / PX
        return e, u * PX, ok
    return coords


def stacks(a, r, bw=1.2, spacing=7.0, tol=4.0, lo=0.0, hi=70.0, need_frac=0.35, deg=1):
    sel = (r >= lo) & (r <= hi); L = np.ptp(a[sel]) if sel.any() else 0.0; need = max(3, need_frac * L / spacing)
    grid = np.arange(lo, hi, 0.1); h = np.exp(-0.5 * ((grid[:, None] - r[sel][None]) / bw) ** 2).sum(1)
    pk, _ = find_peaks(h, distance=int(6 / 0.1)); cents = sorted(grid[p] for p in pk if (np.abs(r[sel] - grid[p]) <= 3).sum() >= need)
    coef = [np.array([0.0, c]) for c in cents]; lab = np.zeros(len(r), int)
    for _ in range(3):
        if not coef: break
        D = np.stack([np.abs(r - np.polyval(cf, a)) for cf in coef], 1); k = D.argmin(1); lab = np.where(D.min(1) <= tol, k + 1, 0)
        for i in range(len(coef)):
            s = lab == i + 1
            if s.sum() >= 6:
                d = deg if (deg > 1 and s.sum() >= 10 and np.ptp(a[s]) > 50) else 1
                coef[i] = np.polyfit(a[s], r[s], d)
    return lab, coef


def outer_rows(fl, F=None, peak_kw=None, deg=1, naming="position", band=None):
    """Plate peaks with row names. Columns x, y (pixels), v, e, u, ok, row."""
    if F is None:
        import bundle_trace as bt; F = bt.field_maps_bowl(fl)
    z = plate_peaks(plate_image(F), **(peak_kw or {})); z["row"] = "not assessed"
    co = band_frame(fl, z, band=band)
    if co is None: return z, pd.DataFrame(), None
    z["e"], z["u"], z["ok"] = co(z.x.values, z.y.values)
    w = z[z.ok & (z.e > 0)]; lab, coef = stacks(w.u.values, w.e.values, deg=deg)
    st = pd.DataFrame([dict(c=k, e=np.median(w.e.values[lab == k]), n=int((lab == k).sum())) for k in range(1, len(coef) + 1)])
    names = {}
    for _, s in st.iterrows():
        nm, dv = min(((n, abs(s.e - v)) for n, v in CANON.items()), key=lambda t: t[1])
        if dv <= 4.5 and nm not in names.values(): names[s.c] = nm
    if not st.empty: st["row"] = st.c.map(names).fillna("unnamed stack")
    z.loc[z.ok & (z.e <= 0), "row"] = "inside Calb2 band"
    z.loc[w.index, "row"] = [names.get(l, "outside rows") if l > 0 else "outside rows" for l in lab]
    z.loc[z.ok & z.e.isna(), "row"] = "inner side"
    return z, st, co


def row_check(z, st):
    e = dict(zip(st.row, st.e)) if not st.empty else {}
    ok3 = all(k in e for k in ("OHC1", "OHC2", "OHC3")) and all(7 <= e[b] - e[a] <= 12 for a, b in (("OHC1", "OHC2"), ("OHC2", "OHC3")))
    return bool(ok3 and (z.row != "not assessed").mean() >= 0.5)


def inner_cells(fl, co, zone=(-20, 3), min_area=20.0, cb_min=0.2, gap=4.0):
    q = os.path.join(IHC_DIR, fl + IHC_CELL_OUTLINES)
    if not os.path.exists(q):
        raise FileNotFoundError(f"{q} is missing: run ihc_cellpose.py, which now also writes the log-scaled outlines")
    mk = np.load(q); c = calb2_projection(fl)
    R = pd.DataFrame([dict(label=r.label, y=r.centroid[0], x=r.centroid[1], area=r.area * PX ** 2, cb=r.intensity_mean) for r in regionprops(mk, intensity_image=c)])
    if R.empty: return pd.DataFrame(columns=["x", "y", "u", "n_outlines"])
    R["e"], R["u"], R["ok"] = co(R.x.values, R.y.values)
    R = R[R.ok & R.e.between(*zone) & (R.area >= min_area) & (R.cb >= cb_min)].sort_values("u").reset_index(drop=True)
    groups, cur = [], []
    for i, u in enumerate(R.u.values):
        if cur and (u - R.u.values[cur[-1]] > gap or len(cur) == 2): groups.append(cur); cur = []
        cur.append(i)
    if cur: groups.append(cur)
    return pd.DataFrame([dict(x=R.x.values[g].mean(), y=R.y.values[g].mean(), u=R.u.values[g].mean(), n_outlines=len(g)) for g in groups])
