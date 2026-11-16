"""Outer hair cells and their rows with the two learned classifiers (frozen 10 November 2026), and scoring
against Shawn's row clicks.

Steps for each field:
  1. Candidates: peaks of the plate image (hair_cell_rows.plate_image; phalloidin within 0.6 um of each cell's
     plate depth) filtered with a Laplacian of Gaussian (sigma 2.0 um), at least 3 um apart, score >= -0.05.
  2. Features of each candidate: filter response at sigma 1.5, 2.0 and 2.8 um; plate brightness within 2 um;
     contrast with the ring 3.5-5 um out, and the ring's spread; bundle phalloidin (95th percentile and mean
     within 3 um); Calb2 within 2 um; distance from the edge of the Calb2 band (e).
  3. Hair cell classifier (models\\hair_cell_classifier.joblib), cutoff 0.3.
  4. Row classifier (models\\hair_cell_row_classifier.joblib) on neighbor geometry of the hair cells: e; offsets
     from the innermost and outermost hair cells within 15 um along the row; offset from the 10th percentile of e
     within 25 um; counts of hair cells more than 3 um inward and outward within 6 um; share inward within 15 um;
     number within 15 um.
Both classifiers were trained on the 12 tuning fields (mice AR524, AR454, JW5436 (2), AR1383-2) with labels
cleaned by Shawn's blind re-look; test once on 14 fields of AR1064, JW5436, AR170 and AR47: 80.5% of clicked
outer hair cells in the right row.

Across-row frame, second version (10 November 2026, models unchanged): the Calb2 surface band picked out-of-focus
Calb2 haze in JW5436 Apex, and apex outer hair cells carry some Calb2, so the band is now the Calb2 surface band pieces
that touch Cellpose inner hair cell outlines at least half as bright in Calb2 as the field's brightest (90th
percentile), joined with those outlines, and continued past any end that stops inside the image along the line of
its last 40 um. Against every row click (26 fields): 86.8% right (tuning fields 92.7%, test fields 81.8%), against
77.4% with the first frame (88.2% and 68.1%). frame=None gives the first frame.

Run from the scripts folder:
    python hair_cell_rows_classify.py           (all fields; writes results\\hair_cell_rows_cells.csv)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, sys
import numpy as np
import pandas as pd
import joblib
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hair_cell_rows as hr

PX = hr.PX
MOD = S.MODELS
CP = os.path.join(S.ANNOTATIONS, 'cell_points')
RL = os.path.join(S.ANNOTATIONS, 'row_relook')
RESULTS = S.RESULTS
MR = 4.0                       # um, match radius between a click and a hair cell
FT = ["log15", "log20", "log28", "plate", "contrast", "ring_sd", "bund95", "bundmean", "calb2"]
RF = ["e", "above_inner15", "below_outer15", "above_p10_25", "n_inner6", "n_outer6", "frac_inner15", "n15"]
_yy, _xx = np.mgrid[-60:61, -60:61] * PX; RR = np.hypot(_yy, _xx)
_models = {}


WORKERS = 4    # fields classified at once; identical to one at a time once frame_v2 passed the band explicitly (12 Nov 2026)
MODEL_VERSION = "v2"   # "v1": the models frozen on 10 Nov 2026 (12 tuning fields)


def models():
    if not _models:
        v = "_v2" if MODEL_VERSION == "v2" else ""   # v2 (11 Nov 2026): retrained on all 26 clicked fields
        _models["hc"] = joblib.load(os.path.join(MOD, f"hair_cell_classifier{v}.joblib"))
        _models["row"] = joblib.load(os.path.join(MOD, f"hair_cell_row_classifier{v}.joblib"))
    return _models["hc"], _models["row"]


def _crop(a, x, y):
    H, Wd = a.shape; ys = np.clip(np.arange(y - 60, y + 61), 0, H - 1); xs = np.clip(np.arange(x - 60, x + 61), 0, Wd - 1)
    return a[np.ix_(ys, xs)]


def cand_features(fl, F, co):
    img = hr.plate_image(F); s = (img - np.median(img)) / (np.percentile(img, 99) - np.median(img))
    logs = {sg: -ndi.gaussian_laplace(s, sg / PX) * (sg / PX) ** 2 for sg in (1.5, 2.0, 2.8)}
    pk = peak_local_max(logs[2.0], min_distance=int(3.0 / PX), exclude_border=False); pk = pk[logs[2.0][pk[:, 0], pk[:, 1]] >= -0.05]
    z = pd.DataFrame(dict(x=pk[:, 1], y=pk[:, 0]))
    b = F["I_bowl"].astype(float); b = (b - np.median(b)) / (np.percentile(b, 99.5) - np.median(b)); cb = hr.calb2_surface(fl)
    z["e"], z["u"], z["ok"] = co(z.x.values, z.y.values)
    f = []
    for x, y in pk[:, ::-1]:
        S = _crop(s, x, y); B = _crop(b, x, y); C = _crop(cb, x, y)
        f.append([logs[1.5][y, x], logs[2.0][y, x], logs[2.8][y, x], S[RR <= 2].mean(), S[RR <= 2].mean() - S[(RR >= 3.5) & (RR <= 5)].mean(),
                  S[(RR >= 3.5) & (RR <= 5)].std(), np.percentile(B[RR <= 3], 95), B[RR <= 3].mean(), C[RR <= 2].mean()])
    z[FT] = np.array(f) if len(f) else np.zeros((0, len(FT))); return z


def row_feats(z, thr=0.3):
    hc = z[z.ok & (z.e > 0) & (z.p >= thr)].copy(); a, e = hc.u.values, hc.e.values; f = []
    for i in range(len(hc)):
        n15 = np.abs(a - a[i]) <= 15; n6 = np.abs(a - a[i]) <= 6; n25 = np.abs(a - a[i]) <= 25
        lo15 = np.min(e[n15]); hi15 = np.max(e[n15]); lo25 = np.percentile(e[n25], 10)
        f.append([e[i], e[i] - lo15, hi15 - e[i], e[i] - lo25, ((e < e[i] - 3) & n6).sum(), ((e > e[i] + 3) & n6).sum(),
                  ((e < e[i] - 3) & n15).sum() / max(n15.sum(), 1), n15.sum()])
    hc[RF] = np.array(f) if len(f) else np.zeros((0, len(RF))); return hc


def classify(fl, F=None, frame="v2"):
    """Returns (candidates with p and row, frame function co). row: OHC1/2/3, 'not hair cell' or 'not assessed'."""
    import bundle_trace as bt
    F = F if F is not None else bt.field_maps_bowl(fl)
    if frame == "v2": frame = frame_v2
    co = frame(fl, F) if frame else hr.outer_rows(fl, F)[2]          # frame=None: the band of the 10 November test
    if co is None: return None, None
    hc_m, row_m = models(); z = cand_features(fl, F, co).reset_index(drop=True)
    sel = z.e.notna(); z["p"] = np.nan
    if sel.any(): z.loc[sel, "p"] = hc_m["model"].predict_proba(z.loc[sel, hc_m["features"]])[:, 1]
    hc = row_feats(z, hc_m["cutoff"]); z["row"] = "not assessed"; z.loc[z.ok & z.e.notna(), "row"] = "not hair cell"
    if len(hc): z.loc[hc.index, "row"] = row_m["model"].predict(hc[row_m["features"]])
    return z, co


# ---------------- band used for the across-row distance (10 November 2026, second version) ----------------
def bright_band(fl, rel=0.5, min_area=20.0, close_um=4.0):
    """Cellpose inner hair cell outlines at least rel x the field's 90th percentile of outline Calb2, joined."""
    from skimage.measure import regionprops, label as cclabel
    from skimage.morphology import closing, disk
    mk = np.load(os.path.join(hr.IHC_DIR, fl + "_ihc.npy")); c = hr.calb2_projection(fl)
    R = [(r.label, r.area * PX ** 2, r.intensity_mean) for r in regionprops(mk, intensity_image=c)]
    R = [r for r in R if r[1] >= min_area]
    if not R: return None
    cut = rel * np.percentile([r[2] for r in R], 90); keep = np.isin(mk, [r[0] for r in R if r[2] >= cut])
    m = ndi.binary_fill_holes(closing(keep, disk(int(close_um / PX))).astype(bool)); lab = cclabel(m); s = np.bincount(lab.ravel()); s[0] = 0
    return lab == s.argmax()


def combined_band(fl, touch_um=5.0):
    """Pieces of the Calb2 surface band that touch the bright inner hair cell outlines, plus those outlines."""
    from skimage.measure import label as cclabel
    b0 = hr.calb2_band(fl); b2 = bright_band(fl)
    if b2 is None: return b0
    if b0 is None: return b2
    near = ndi.binary_dilation(b2, iterations=int(touch_um / PX)); lab = cclabel(b0); keep = np.zeros_like(b0)
    for k in range(1, lab.max() + 1):
        if (near & (lab == k)).any(): keep |= lab == k
    return keep | b2


def extend_mask(m, near_um=40):
    """Continue the band past each end that stops inside the image, along the line of its last near_um."""
    if m is None: return None
    H, Wd = m.shape; ys, xs = np.where(m); B = np.c_[xs, ys].astype(float); mu = B.mean(0); vt = np.linalg.svd(B[::50] - mu, full_matrices=False)[2]
    ub = (B - mu) @ vt[0]; vb = (B - mu) @ vt[1]
    gg = pd.DataFrame(dict(b=np.round(ub / (2 / PX)).astype(int), v=vb)).groupby("b").v.agg(["min", "max", "size"]); gg = gg[gg["size"] >= 5]
    uu = gg.index.values * (2 / PX); yy, xx = np.mgrid[0:H, 0:Wd]; P = np.stack([xx.ravel(), yy.ravel()], 1).astype(float) - mu
    U = P @ vt[0]; V = P @ vt[1]; lo, hi = uu.min(), uu.max(); ext = np.zeros(H * Wd, bool)
    for side, selu in [("lo", U < lo), ("hi", U > hi)]:
        near = (uu <= lo + near_um / PX) if side == "lo" else (uu >= hi - near_um / PX)
        if not selu.any() or near.sum() < 4: continue
        a = np.polyfit(uu[near], gg["min"].values[near], 1); b = np.polyfit(uu[near], gg["max"].values[near], 1)
        ext |= selu & (V >= np.polyval(a, U)) & (V <= np.polyval(b, U))
    return m | ext.reshape(H, Wd)


def frame_v2(fl, F):
    """Across-row frame from the combined band, extended past its ends."""
    m = extend_mask(combined_band(fl))
    return hr.outer_rows(fl, F, band=m)[2]   # band passed in (12 Nov 2026; was a temporary swap of hr.calb2_band, unsafe with several fields at once)


# ---------------- scoring against the row clicks ----------------
def mouse_of(fl):
    return S.mouse_of(fl)


def hmatch(A, B, r):
    ma = np.zeros(len(A), bool); mb = np.zeros(len(B), bool); jj = -np.ones(len(A), int)
    if len(A) and len(B):
        D = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1]); D[D > r] = 1e6
        i, j = linear_sum_assignment(D); k = D[i, j] < 1e6; ma[i[k]] = True; mb[j[k]] = True; jj[i[k]] = j[k]
    return ma, mb, jj


def load_clicks(fl, co):
    """Shawn's clicks with e, u, ok; fields clicked on 5 November get the carry-over relabel used in tuning."""
    done = pd.read_csv(os.path.join(CP, "fields_done.csv"))
    d = pd.read_csv(os.path.join(CP, fl + "_points.csv")); d["e"], d["u"], d["ok"] = co(d.x.values, d.y.values)
    d["row"] = d["class"].replace({"inner": "IHC"}); d["field"] = fl
    if fl in done[done.saved.astype(str).str.startswith("2026-11-05")].field.tolist():
        d.loc[d.row.isin(["IHC", "extra"]) & d.e.between(10, 24), "row"] = "OHC1"
        d.loc[(d.row == "extra") & (d.e < 8), "row"] = "IHC body"
    return d


def score_field(z, d, mr=MR):
    """Each clicked outer hair cell is right, missed, not in a row, or wrong row."""
    d = d[d.row.isin(["OHC1", "OHC2", "OHC3"]) & d.ok].reset_index(drop=True)
    ma, mb, jj = hmatch(d[["x", "y"]].values, z[["x", "y"]].values, mr / PX); got = np.where(jj >= 0, z.row.values[np.maximum(jj, 0)], "no peak")
    kinds = np.where(got == d.row.values, "right", np.where(got == "no peak", "missed",
                     np.where(pd.Series(got).str.startswith("OHC").values, "wrong row", "not in a row")))
    return pd.Series(kinds).value_counts()


def main():
    import glob
    fields = [os.path.basename(f)[:-10] for f in sorted(glob.glob(os.path.join(S.WORK, 'ihc_stacks', '*_cells.csv')))]
    out = []
    models()                                     # load the classifiers once, before the threads start
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(WORKERS) as ex:       # fields are independent; results kept in field order
        results = list(ex.map(classify, fields))
    for fl, (z, co) in zip(fields, results):
        if z is None: continue
        z["field"] = fl; out.append(z[["field", "x", "y", "e", "u", "ok", "p", "row"]])
    R = pd.concat(out, ignore_index=True); R.to_csv(os.path.join(RESULTS, "hair_cell_rows_cells.csv"), index=False); field_summary(R)


ROW_END_GAP = 2.0   # an end cell is dropped when its gap to the next cell exceeds this many median spacings


def row_stretch(u, gap=ROW_END_GAP):
    """Sorted positions along the row with isolated end cells trimmed (16 Nov 2026). Each row is measured over its
    own first-to-last stretch, as the hand counts are; one shared span for all three rows let a stray cell at
    either end lower every row's density."""
    u = np.sort(np.asarray(u, float)); u = u[np.isfinite(u)]
    if len(u) < 4: return u
    sp = np.diff(u); med = np.median(sp[sp > 1.0]) if (sp > 1.0).any() else np.median(sp)
    while len(u) >= 4 and (u[1] - u[0]) > gap * med: u = u[1:]
    while len(u) >= 4 and (u[-1] - u[-2]) > gap * med: u = u[:-1]
    return u


def field_summary(R):
    """Per-field outer hair cell row measures from the cell table (writes hair_cell_rows_fields.csv).
    Density per row = 100 x (cells - 1) / (first-to-last length of that row after trimming isolated end cells);
    outer hair cells per 100 um = sum of the three rows."""
    rows = []
    for f, d in R.groupby("field"):
        hc = d[d.row.isin(["OHC1", "OHC2", "OHC3"])]
        if len(hc) < 10: rows.append(dict(field=f, note="fewer than 10 outer hair cells")); continue
        r = dict(field=f, outer_hair_cells=len(hc), assessed_share=(d.row != "not assessed").mean()); spans = []
        for rw in ["OHC1", "OHC2", "OHC3"]:
            s = hc[hc.row == rw].sort_values("u"); u = row_stretch(s.u.values)
            L = (u[-1] - u[0]) if len(u) >= 4 else np.nan; spans.append(L)
            r[f"{rw}_length_um"] = L
            r[f"{rw}_per_100um"] = 100 * (len(u) - 1) / L if L and L > 0 else np.nan
            sp = np.diff(s.u.values); r[f"{rw}_spacing_um"] = np.median(sp[sp < 15]) if len(sp) else np.nan; r[f"{rw}_e_um"] = s.e.median()
        r["row_length_um"] = float(np.nanmedian(spans))
        r["row1_to_2_um"] = r["OHC2_e_um"] - r["OHC1_e_um"]; r["row2_to_3_um"] = r["OHC3_e_um"] - r["OHC2_e_um"]
        r["outer_hair_cells_per_100um"] = r["OHC1_per_100um"] + r["OHC2_per_100um"] + r["OHC3_per_100um"]; rows.append(r)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RESULTS, "hair_cell_rows_fields.csv"), index=False); return out


if __name__ == "__main__":
    main()
