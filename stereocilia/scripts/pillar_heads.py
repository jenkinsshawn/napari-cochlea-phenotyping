r"""Pillar heads: the dividing lines between neighboring pillar heads along the band between the inner hair cell row
and the first outer row, and from them head width, heads per 100 um and the spread of head widths.

Method (built and scored in the session of 10 Nov 2026; this script reproduces it):
  1. Geometry per field: distance across the row from the Calb2 band edge (e) and position along the row (u), from
     hair_cell_rows.outer_rows; the band runs from the Calb2 edge to the first outer row (e1, first row at e > 8 um).
     Position along the row is taken at the nearest Calb2 band-edge pixel and smoothed over 3 um, so it follows the
     curve of the band.
  2. Along-row profiles (0.1 um steps) of thin dark-line, bright-line, brightness and edge filters of the phalloidin
     surface layer, in five depth bands (0.2-0.7 of e1), with contrasts to neighbors 0.3, 0.6 and 1.2 um away.
  3. Gradient boosting (scikit-learn HistGradientBoostingClassifier, fixed settings) trained on Shawn's clicked lines
     (annotations\pillar_points and pillar_points_more, 18 fields of 7 mice): positions within 0.3 um of a click are
     lines, 1 um or more away are not, in between unused.
  4. Lines are peaks of the line probability; the peak height and the minimum spacing are chosen leaving one mouse
     out (grid 0.3-0.8 and 2.5-4.5 um), and the final setting is the one chosen most often.
  5. Per field: spacings between consecutive lines, without gaps (more than 1.6 x the median spacing).
Scores leaving one mouse out (results\tuning\pillar_lines_vs_clicks_leave_one_mouse_out.csv): middle and base lines
found 83%, extra 20%, spacing error 0.24 um; apex is not assessable with this method (found 66%, extra 38%).

Outputs: RESULTS\pillar_heads_fields.csv, RESULTS\tuning\pillar_lines_vs_clicks_leave_one_mouse_out.csv,
MODELS\pillar_detector.joblib (classifier, feature names, peak settings).

Run from the scripts folder:  python pillar_heads.py   (applies the saved detector; --train retrains it on the clicks)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
from collections import Counter
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from skimage.filters import sato
from sklearn.ensemble import HistGradientBoostingClassifier

import bundle_trace as bt
import hair_cell_rows as hcr

PX = S.PX_UM
FIELDS = os.path.join(S.WORK, "fields")
CLICKS = os.path.join(S.ANNOTATIONS, "pillar_points")
CLICKS_MORE = os.path.join(S.ANNOTATIONS, "pillar_points_more")
RECLICKED = ("AR524 Ocm, Calb2 Phall P1 60X MIDDLE", "JW5436 Calb2 PMCA2 Phall P1 60X middle",
             "AR1064 Ocm Calb2 Phall P1 60X APEX", "AR1383-2 Calb2 PMCA2 Phall P1 60X APEX (3)")   # left out of the first set
BANDS_E = [(0.25, 0.35), (0.35, 0.45), (0.45, 0.55), (0.2, 0.6), (0.55, 0.7)]
GRID = [(t, mu) for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] for mu in [2.5, 3.0, 3.5, 4.0, 4.5]]
CLF = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=0)
GEOM, FMAP, CLK = {}, {}, {}


def first_outer_e(stk):
    if stk.empty: return np.nan
    s = stk[stk.e > 8].sort_values("e"); return float(s.e.iloc[0]) if len(s) else np.nan


def field_geom(fl):
    if fl in GEOM: return GEOM[fl]
    z = np.load(os.path.join(FIELDS, fl + ".npz")); st = [str(s) for s in z["stains"]]
    ph = z["surface_layers"][st.index("Phalloidin")].astype(float)
    if fl not in FMAP: FMAP[fl] = bt.field_maps_bowl(fl)
    pk, stk, co = hcr.outer_rows(fl, FMAP[fl]); e1 = first_outer_e(stk)
    H, Wd = ph.shape; yy, xx = np.mgrid[0:H, 0:Wd]; e, u, ok = co(xx.ravel(), yy.ravel())
    g = dict(ph=ph, co=co, e1=e1, stk=stk, e=e.reshape(H, Wd), u=u.reshape(H, Wd), ok=ok.reshape(H, Wd))
    if np.isfinite(e1):   # position along the curve of the band: u of the nearest Calb2 band-edge pixel, smoothed 3 um
        m = hcr.calb2_band(fl); dist, (iy, ix) = ndi.distance_transform_edt(~m, return_indices=True)
        uc = g["u"][iy, ix]; okm = g["ok"] & np.isfinite(g["e"])
        w = ndi.gaussian_filter(okm.astype(float), 3 / PX)
        g["ucs"] = ndi.gaussian_filter(np.where(okm, uc, 0), 3 / PX) / np.maximum(w, 1e-6)
    GEOM[fl] = g; return g


def field_feats(fl, ustep=0.1):
    """Along-row profiles at positions u (0.1 um), with labels from the clicks in CLK[fl]."""
    g = field_geom(fl); ph = g["ph"]; phs = (ph - np.median(ph)) / (np.percentile(ph, 99.5) - np.median(ph) + 1e-9)
    maps = {"dark07": sato(phs, sigmas=[0.7], black_ridges=True), "dark12": sato(phs, sigmas=[1.2], black_ridges=True),
            "dark20": sato(phs, sigmas=[2.0], black_ridges=True), "bright15": sato(phs, sigmas=[1.5], black_ridges=False),
            "bright30": sato(phs, sigmas=[3.0], black_ridges=False), "int": ndi.gaussian_filter(phs, 1.0),
            "dx": np.abs(ndi.gaussian_filter(phs, 1.5, order=(0, 1))) + np.abs(ndi.gaussian_filter(phs, 1.5, order=(1, 0)))}
    U = g["ucs"]; e1 = g["e1"]; ok = g["ok"] & np.isfinite(g["e"]); ub_all = np.round(U / ustep).astype(int)
    b0 = ub_all[ok].min(); nb = ub_all[ok].max() - b0 + 1; cols = {}; covs = []
    for (a, b) in BANDS_E:
        sel = ok & (g["e"] >= a * e1) & (g["e"] <= b * e1); ub = ub_all[sel] - b0; c_ = np.bincount(ub, minlength=nb); covs.append(c_)
        for k, M in maps.items():
            s_ = np.bincount(ub, weights=M[sel], minlength=nb); prof = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
            prof = pd.Series(prof).interpolate(limit=5).values; prof = ndi.gaussian_filter1d(np.nan_to_num(prof), 0.7)
            sc = np.nanpercentile(prof, 95) - np.nanpercentile(prof, 5) + 1e-9; prof = (prof - np.nanmedian(prof)) / sc
            nm = f"{k}_{a}"; cols[nm] = prof
            for sh in (3, 6, 12):
                cols[f"{nm}_d{sh}"] = prof - 0.5 * (np.roll(prof, sh) + np.roll(prof, -sh))
    X = pd.DataFrame(cols); X["u"] = (np.arange(nb) + b0) * ustep; X["cov"] = covs[1] >= 5; X["field"] = fl; X["mouse"] = S.mouse_of(fl)
    c = CLK.get(fl, np.array([-1e9])); d = np.min(np.abs(X.u.values[:, None] - c[None]), 1); inext = (X.u >= c.min() - 1) & (X.u <= c.max() + 1)
    X["label"] = np.where(d <= 0.3, 1, np.where(d >= 1.0, 0, -1)); X.loc[~inext, "label"] = -1
    return X


def clicks_u(fl, path):
    d = pd.read_csv(path); g = field_geom(fl); yi, xi = d.y.values.astype(int).clip(0, 1023), d.x.values.astype(int).clip(0, 1023)
    ok = g["ok"][yi, xi] & np.isfinite(g["e"][yi, xi]); return np.sort(g["ucs"][yi, xi][ok])


def m1d(a, b, r):
    ma = np.zeros(len(a), bool); mb = np.zeros(len(b), bool)
    if len(a) and len(b):
        D = np.abs(a[:, None] - b[None]); D[D > r] = 1e6; i, j = linear_sum_assignment(D); k = D[i, j] < 1e6; ma[i[k]] = True; mb[j[k]] = True
    return ma, mb


def score(f, det, r=1.0, near=6.0):
    c = CLK[f]; elig = det[(np.min(np.abs(det[:, None] - c[None]), 1) <= near)] if len(c) and len(det) else det
    mc, md = m1d(c, elig, r); lo, hi = c.min(), c.max(); n_c = len(c); n_d = ((det >= lo - 0.5) & (det <= hi + 0.5)).sum()
    return dict(found=mc.mean(), extra=1 - md.mean() if len(md) else np.nan, count_ratio=n_d / n_c, n_click=n_c, n_det=int(n_d))


def peaks(x, p, thr, mu):
    pk, _ = find_peaks(np.where(x["cov"], p, 0), height=thr, distance=int(mu / 0.1)); return x.u.values[pk]


def clicked_fields():
    first = [f for f in pd.read_csv(os.path.join(CLICKS, "fields_done.csv")).field if f not in RECLICKED]
    more = list(pd.read_csv(os.path.join(CLICKS_MORE, "fields_done.csv")).field)
    src = {f: os.path.join(CLICKS, f + "_points.csv") for f in first}; src.update({f: os.path.join(CLICKS_MORE, f + "_points.csv") for f in more})
    return first + more, src


def train():
    PF, SRC = clicked_fields()
    for f in PF: CLK[f] = clicks_u(f, SRC[f])
    XA = pd.concat([field_feats(f) for f in PF], ignore_index=True)
    FEATS = [c for c in XA.columns if c not in ("u", "cov", "field", "mouse", "label")]
    XA["p"] = np.nan
    for m in XA.mouse.unique():
        tr = (XA.mouse != m) & (XA.label >= 0) & XA["cov"]
        clf = HistGradientBoostingClassifier(**CLF).fit(XA.loc[tr, FEATS], XA.loc[tr, "label"])
        XA.loc[XA.mouse == m, "p"] = clf.predict_proba(XA.loc[XA.mouse == m, FEATS])[:, 1]
    SP = pd.DataFrame([dict(field=f, mouse=S.mouse_of(f), gp=gp, **score(f, peaks(XA[XA.field == f], XA.p[XA.field == f].values, *gp)))
                       for f in PF for gp in GRID]); SP["crit"] = SP.found - SP.extra - np.abs(SP.count_ratio - 1)
    LA = []
    for m in SP.mouse.unique():
        b_ = SP[SP.mouse != m].groupby("gp")["crit"].mean().idxmax(); LA.append(SP[(SP.mouse == m) & (SP.gp == b_)].assign(chosen=str(b_)))
    LA = pd.concat(LA); LA["turn"] = LA.field.str.lower().str.extract(r"(apex|middle|base)")[0]
    gp_final = eval(Counter(LA.chosen).most_common(1)[0][0])
    tr = (XA.label >= 0) & XA["cov"]; clf_final = HistGradientBoostingClassifier(**CLF).fit(XA.loc[tr, FEATS], XA.loc[tr, "label"])
    return clf_final, FEATS, gp_final, LA


def measure(f, clf, FEATS, gp):
    g = field_geom(f)
    if not np.isfinite(g["e1"]): return dict(field=f, pillar_note="no first outer row")
    X = field_feats(f); U = peaks(X, clf.predict_proba(X[FEATS])[:, 1], *gp)
    sp = np.diff(U); med = np.median(sp) if len(sp) else np.nan; sp = sp[sp < 1.6 * med] if len(sp) else sp
    return dict(field=f, pillar_lines=len(U), pillar_head_width_um=np.median(sp) if len(sp) else np.nan,
                pillar_heads_per_100um=100 / np.median(sp) if len(sp) else np.nan,
                pillar_width_sd_um=np.std(sp) if len(sp) > 2 else np.nan, pillar_band_um=float(g["e1"]))


def study_fields():
    fs = sorted(f[:-4] for f in os.listdir(FIELDS) if f.endswith(".npz"))
    out = []
    for f in fs:
        z = np.load(os.path.join(FIELDS, f + ".npz")); st = [str(s) for s in z["stains"]]
        if S.group_of(f) is not None and "Phalloidin" in st and "Calb2" in st: out.append(f)
    return out


def main():
    import joblib
    model = os.path.join(S.MODELS, "pillar_detector.joblib")
    if "--train" in sys.argv or not os.path.exists(model):   # training is explicit; a run applies the saved detector
        if not os.path.isdir(CLICKS):
            sys.exit(f"no saved pillar detector ({model}) and no pillar clicks to train one ({CLICKS})")
        clf, FEATS, gp, LA = train()
        print(f"peak settings {gp}; left-out mouse scores, middle and base: found "
              f"{(LA[LA.turn != 'apex'].found * LA[LA.turn != 'apex'].n_click).sum() / LA[LA.turn != 'apex'].n_click.sum():.2f}, "
              f"extra {LA[LA.turn != 'apex'].extra.mean():.2f}", flush=True)
        os.makedirs(os.path.join(S.RESULTS, "tuning"), exist_ok=True); os.makedirs(S.MODELS, exist_ok=True)
        LA.to_csv(os.path.join(S.RESULTS, "tuning", "pillar_lines_vs_clicks_leave_one_mouse_out.csv"), index=False)
        joblib.dump(dict(classifier=clf, features=FEATS, peak_height=gp[0], min_spacing_um=gp[1]), model)
    else:
        d = joblib.load(model); clf, FEATS, gp = d["classifier"], d["features"], (d["peak_height"], d["min_spacing_um"])
        print(f"using the saved pillar detector {model} (peak settings {gp}; pillar_heads.py --train retrains it)", flush=True)
    rows = []
    for i, f in enumerate(study_fields(), 1):
        try:
            r = measure(f, clf, FEATS, gp); msg = r.get("pillar_note") or f"{r['pillar_lines']} lines"
        except Exception as e:
            r = dict(field=f, pillar_note=f"failed: {e}"); msg = r["pillar_note"]
        rows.append(r); print(f"{i} {f}: {msg}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(S.RESULTS, "pillar_heads_fields.csv"), index=False)


if __name__ == "__main__":
    main()
