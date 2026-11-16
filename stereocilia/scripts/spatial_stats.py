"""Spatial statistics of hair cells and bundles in each field (plan: section Spatial statistics).

Inputs per field: cell types and positions in the row frame (E:\\Webber_confocal\\work\\cells\\types,
from assign_cell_types.py) and bundle measures (E:\\Webber_confocal\\work\\bundles_detector, from
measure_bundles.py --detector). The surface and body outlines of one inner hair cell are joined
into one cell. Only inner and outer hair cells are used (rows IHC, OHC1, OHC2, OHC3).

Per field it computes
  Spacing and packing
    spacing to the next cell along each row (um), its coefficient of variation, the regularity
    index (mean spacing / its standard deviation), gaps (spacing > 1.5 x local median: likely
    missing cells, with the number missing) and doubles (spacing < 0.5 x local median), distance
    between rows, stagger of second-row and third-row outer hair cells against the row next to
    them (fraction of the spacing), number of neighbors on the neighbor network (Delaunay
    triangulation of cell centers, edges longer than 2.5 x the median removed), and the share
    of interior outer hair cells with 6 neighbors.
  Orientation
    direction of each outer hair cell: from the cell center to the dark notch in its flat top
    (the fonticulus, where the kinocilium sits; see notches). Per row: circular mean, mean
    resultant length (0-1, 1 = all the same), circular standard deviation, Rayleigh P value,
    share pointing inward, drift along the row (degrees per 100 um, circular-linear fit).
    Per cell: deviation from the row mean and local misorientation (deviation against the
    circular mean of neighbors' deviations). Orientation correlation with distance (correlation
    of deviations between pairs of cells in 4 um distance bins) and the correlation length (first
    distance where it falls below 0). Inner hair cells: direction not assessed.
    Bundle axis (all rows): angle of the line joining the bundle's two arm ends, as an axis
    (axial statistics). It depends on the bundle outline, which in several fields covers the
    flat top of the cell rather than the V, so it is kept as a shape measure and not used for flags.
  Clustering of abnormal cells
    flags: misoriented (notch direction more than 45 degrees from the row mean), doubtful bundle shape, no bundle
    found, next to a gap; join counts (pairs of flagged neighbors) against 10,000 shuffles of
    each flag among cells of the same row in the field, given as an observed/expected ratio and
    a one-sided P value; runs of flagged cells along each row against the same shuffles.
  Hot spots
    local Moran's I of bundle span, V angle, height and local misorientation on the neighbor
    network (row-standardized), with 999 conditional shuffles per cell; P values adjusted by
    Benjamini-Hochberg within each field and measure. Cells with adjusted P < 0.05 and both the
    cell and its neighbors above the mean are "high clusters", both below are "low clusters".
Outputs
    results\\spatial\\cells_spatial.csv   one row per hair cell
    results\\spatial\\fields_spatial.csv  one row per field
    results\\spatial\\orientation_correlogram.csv
    figures\\spatial\\<field>.png          map: bundle directions, misorientation, flags, hot spots
These describe pattern within a field. Groups are compared with the mouse as the unit, from
per-field values, not by pooling cells.

Run from the scripts folder:   python spatial_stats.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob, os, sys, warnings
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJ = os.path.dirname(HERE)
TYPES = os.path.join(S.WORK, 'cells', 'types')
BUNDLES = os.path.join(S.WORK, 'bundles_detector')
OUT = os.path.join(S.RESULTS, "spatial")
FIG = os.path.join(S.FIGURES, "spatial")
PX = S.PX_UM
ROWS = ["IHC", "OHC1", "OHC2", "OHC3"]
DAMAGED = dict(S.DAMAGED)   # field: reason (pipeline_settings.py)
MISORIENT_DEG = 45       # notch direction more than this from the row mean (the measure disagrees with clear V shapes by 24 degrees, median)
NOTCH_MARGIN_PX = 5      # notch search stays 0.5 um inside the cell outline
PAINT = os.path.join(S.WORK, 'paint_stacks')
CLEAR_V_UM = 0.5          # bundles bulging at least this far to one side of their chord have a readable direction
N_SHUFFLE = 10000
N_LOCAL = 999
RNG = np.random.default_rng(20261106)


# ---------------- loading ----------------
def smooth_row_frame(t):
    """Distance along and across the row from a smooth curve: a quadratic fitted to the inner
    hair cell centers (in their principal-axis frame), sampled every 0.1 pixel. The types files
    measure along_um on a line of straight pieces through the inner hair cells; cells 25-45 um
    away on the outer side of a joint all land on the joint, so outer hair cells got identical
    positions along the row. Signs and origin follow the types files (along increases the same
    way, across is positive toward the outer side, 0 at the first inner hair cell)."""
    ih = t[t.row == "IHC"]
    if len(ih) < 4:
        return t
    P = ih[["x", "y"]].values.astype(float); mu = P.mean(0); vt = np.linalg.svd(P - mu)[2]
    u_, v_ = (P - mu) @ vt[0], (P - mu) @ vt[1]
    coef = np.polyfit(u_, v_, 2)
    X = t[["x", "y"]].values.astype(float); U, V = (X - mu) @ vt[0], (X - mu) @ vt[1]
    lo, hi = min(U.min(), u_.min()) - 200, max(U.max(), u_.max()) + 200
    s = np.arange(lo, hi, 0.1); cv = np.polyval(coef, s)
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(s), np.diff(cv)))]
    from scipy.spatial import cKDTree
    d, k = cKDTree(np.c_[s, cv]).query(np.c_[U, V])
    sgn = np.sign(V - cv[k])
    along = arc[k] * PX; across = d * sgn * PX
    if np.corrcoef(along, t.along_um.fillna(0))[0, 1] < 0: along = -along
    if np.corrcoef(across, t.across_um.fillna(0))[0, 1] < 0: across = -across
    t = t.copy(); t["along_um"] = along - along[(t.row == "IHC").values].min(); t["across_um"] = across
    # unit vectors along and across the row at each cell, in image x, y: tangent of the fitted curve at the
    # cell's nearest curve point (along increases with along_um; across points to increasing across_um)
    slope = np.polyval(np.polyder(coef), s[k]); tang = np.c_[np.ones_like(slope), slope]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    tx = tang[:, 0] * vt[0][0] + tang[:, 1] * vt[1][0]; ty = tang[:, 0] * vt[0][1] + tang[:, 1] * vt[1][1]
    flip = np.sign(np.corrcoef(along, U)[0, 1]) if np.std(along) > 0 else 1.0
    tx, ty = tx * flip, ty * flip
    nx, ny = -ty, tx
    side = np.sign(np.corrcoef(across, X[:, 0] * nx + X[:, 1] * ny)[0, 1])     # across points toward increasing across_um
    t["along_dx"], t["along_dy"] = tx, ty
    t["across_dx"], t["across_dy"] = nx * side, ny * side
    return t


def row_frame(t, cell):
    """(along, across) unit vectors in image x, y at a cell, from the curve fitted through the inner hair
    cells (smooth_row_frame). Replaces measure_bundles.local_frame, whose main direction of all hair cells
    within 25 um could point across the rows."""
    r = t[t.cell == cell].iloc[0]
    return np.array([r.along_dx, r.along_dy]), np.array([r.across_dx, r.across_dy])


def load_field(field):
    t = pd.read_csv(os.path.join(TYPES, field + "_types.csv"))
    t = t[t.row.isin(ROWS)].copy()
    t = smooth_row_frame(t)
    t["unit"] = np.where(t.row == "IHC", "I" + t.inner_cell_id.astype(str), "C" + t.cell.astype(str))
    g = t.groupby("unit")
    cells = g.agg(row=("row", "first"), cell=("cell", "min"), x=("x", "mean"), y=("y", "mean"),
                  along_um=("along_um", "mean"), across_um=("across_um", "mean"), area_um2=("area_um2", "sum")).reset_index()
    bpath = os.path.join(BUNDLES, field + "_bundles.csv")
    if os.path.exists(bpath):
        b = pd.read_csv(bpath)
        keep = ["cell", "found", "span_um", "v_angle_deg", "height_median_um", "orientation_deg", "shape_check", "area_um2"]
        b = b[[c for c in keep if c in b]].rename(columns={"area_um2": "bundle_area_um2", "orientation_deg": "center_to_vertex_deg"})
        b = b.merge(bundle_axes(field, b, t), on="cell", how="left")
        cells = cells.merge(b, on="cell", how="left")
        cells = cells.merge(notches(field, t, b), on="cell", how="left")
    cells.insert(0, "field", field)
    return cells.reset_index(drop=True)


def bundle_axes(field, b, types):
    """Axis and direction of each bundle from its mask.
    axis_deg: angle of the perpendicular to the line joining the two arm ends, in the row frame
      (0 = straight across the row, positive toward increasing distance along the row), folded to
      -90..90 because a line has no head.
    bulge_um: mean distance of the bundle's pixels from that line (how strongly it bends to one side).
    direction_deg: the same perpendicular pointed toward the bulge (the V vertex side), -180..180,
      given only for outer hair cell bundles with bulge_um >= CLEAR_V_UM."""
    from measure_bundles import shape          # imported here: measure_bundles needs the nd2 reader
    lab = np.load(os.path.join(BUNDLES, field + "_bundles.npy"))
    rows = []
    for _, r in b[b.found == True].iterrows():
        m = lab == r.cell
        tc = types[types.cell == r.cell]
        if m.sum() < 5 or tc.empty: continue
        sh = shape(m, PX)
        if sh is None: continue
        along, across = row_frame(types, r.cell)
        ch = sh["end2"] - sh["end1"]; L = np.linalg.norm(ch)
        if L == 0: continue
        n = np.array([-ch[1], ch[0]]) / L
        pts = np.argwhere(m)[:, ::-1].astype(float)
        bulge = float(((pts - (sh["end1"] + sh["end2"]) / 2) @ n).mean() * PX)
        nd = n if bulge >= 0 else -n
        na = n if n @ across >= 0 else -n
        dirn = np.degrees(np.arctan2(nd @ along, nd @ across))
        rows.append(dict(cell=r.cell, axis_deg=float(np.degrees(np.arctan2(na @ along, na @ across))), bulge_um=abs(bulge),
                         direction_deg=float(dirn) if abs(bulge) >= CLEAR_V_UM and tc.row.iloc[0] != "IHC" else np.nan))
    return pd.DataFrame(rows, columns=["cell", "axis_deg", "bulge_um", "direction_deg"])


def notches(field, types, b):
    """Direction of the dark notch in each outer hair cell's flat top (the fonticulus, where the
    kinocilium sits). Image: mean of the images within 0.3 um of the cell's own plate depth
    (plate_depth_um from measure_bundles.py), smoothed (Gaussian, 1.5 pixels); the notch is the
    darkest point at least 0.5 um inside the cell outline. notch_deg: direction from the cell
    center to the notch in the row frame (0 = straight across toward the outer side).
    Inner hair cells are left out: their darkest point is the outline edge, not a notch.
    Checked on 7 fields: notch within 45 degrees of the direction of clear V-shaped bundles in
    72% of 114 cells; rows in the two control fields aligned at mean resultant length 0.6-0.88."""
    from scipy import ndimage as ndi
    from skimage.morphology import binary_erosion, disk
    z = np.load(os.path.join(PAINT, field + ".npz")); ph = z["phal"].astype(np.float32); off = z["offsets_um"]
    cl = np.load(S.cell_labels_path(field))   # corrected outlines where they exist
    plate = dict(zip(b.cell, b.plate_depth_um)) if "plate_depth_um" in b else {}
    rows = []
    for _, t in types[types.row.str.startswith("OHC")].iterrows():
        m = cl == t.cell
        if m.sum() < 50: continue
        pdp = plate.get(t.cell, np.nan); pdp = 0.6 if not np.isfinite(pdp) else pdp
        ks = np.where(np.abs(off - pdp) <= 0.31)[0]
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = max(ys.min() - 3, 0), ys.max() + 4, max(xs.min() - 3, 0), xs.max() + 4
        raw = ph[ks][:, y0:y1, x0:x1].mean(0); mm = binary_erosion(m[y0:y1, x0:x1], disk(NOTCH_MARGIN_PX))
        if mm.sum() < 30 or (raw[binary_erosion(m[y0:y1, x0:x1], disk(2))] > 0).mean() < 0.9: continue
        im = ndi.gaussian_filter(raw, 1.5); med = np.median(im[mm])
        v = np.where(mm, im, np.inf); iy, ix = np.unravel_index(np.argmin(v), v.shape)
        along, across = row_frame(types, t.cell); vec = np.array([ix - (xs.mean() - x0), iy - (ys.mean() - y0)])
        rows.append(dict(cell=t.cell, notch_deg=float(np.degrees(np.arctan2(vec @ along, vec @ across))),
                         notch_contrast=float((med - im[iy, ix]) / med), notch_rel_dist=float(np.linalg.norm(vec) / np.sqrt(m.sum() / np.pi)),
                         notch_x=float(ix + x0), notch_y=float(iy + y0)))
    return pd.DataFrame(rows, columns=["cell", "notch_deg", "notch_contrast", "notch_rel_dist", "notch_x", "notch_y"])


# ---------------- neighbors ----------------
def neighbor_graph(c):
    P = np.c_[c.along_um.values, c.across_um.values]
    if len(P) < 4:
        return [set() for _ in range(len(P))], np.zeros(len(P), bool)
    tri = Delaunay(P)
    edges = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = sorted((s[i], s[(i + 1) % 3])); edges.add((a, b))
    edges = np.array(sorted(edges)); L = np.hypot(*(P[edges[:, 0]] - P[edges[:, 1]]).T)
    edges = edges[L <= 2.5 * np.median(L)]
    nb = [set() for _ in range(len(P))]
    for a, b in edges:
        nb[a].add(b); nb[b].add(a)
    hull = np.zeros(len(P), bool); hull[np.unique(tri.convex_hull)] = True
    return nb, hull


# ---------------- circular helpers ----------------
def wrap(d):
    return (d + 180) % 360 - 180


def wrap90(d):
    return (d + 90) % 180 - 90


def axial_summary(deg):
    """Statistics of axes (lines without heads): angles are doubled, summarized, and halved."""
    s = circ_summary(2 * np.asarray(deg, float))
    return dict(n=s["n"], mean_deg=s["mean_deg"] / 2 if np.isfinite(s["mean_deg"]) else np.nan, R=s["R"],
                circ_sd_deg=s["circ_sd_deg"] / 2 if np.isfinite(s["circ_sd_deg"]) else np.nan, rayleigh_p=s["rayleigh_p"])


def circ_summary(deg):
    t = np.radians(deg[np.isfinite(deg)]); n = len(t)
    if n < 3:
        return dict(n=n, mean_deg=np.nan, R=np.nan, circ_sd_deg=np.nan, rayleigh_p=np.nan)
    C, S = np.cos(t).mean(), np.sin(t).mean(); R = np.hypot(C, S)
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))
    return dict(n=n, mean_deg=np.degrees(np.arctan2(S, C)), R=R, circ_sd_deg=np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))), rayleigh_p=min(1, p))


def circ_linear_slope(x, deg):
    """Slope (degrees per um) of angle against position, by grid search maximizing the mean resultant
    length of the residuals (Fisher's circular-linear regression without intercept bias)."""
    ok = np.isfinite(deg) & np.isfinite(x)
    if ok.sum() < 5: return np.nan
    x, t = x[ok] - x[ok].mean(), np.radians(deg[ok])
    slopes = np.linspace(-np.radians(3), np.radians(3), 601)          # up to 3 degrees per um
    R = [np.abs(np.exp(1j * (t - s * x)).mean()) for s in slopes]
    return np.degrees(slopes[int(np.argmax(R))])


# ---------------- spacing ----------------
def spacing(c):
    rows = []
    c["spacing_next_um"] = np.nan; c["gap_next"] = False; c["missing_next"] = 0; c["double_next"] = False
    for r in ROWS:
        s = c[c.row == r].sort_values("along_um")
        if len(s) < 4: continue
        d = np.diff(s.along_um.values)
        loc = pd.Series(d).rolling(5, center=True, min_periods=2).median().values
        loc = np.where(loc > 1.0, loc, np.nan)                               # under 1 um is not a usable spacing
        gap = d > 1.5 * loc; dbl = d < 0.5 * loc
        idx = s.index[:-1]
        c.loc[idx, "spacing_next_um"] = d; c.loc[idx, "gap_next"] = gap; c.loc[idx, "double_next"] = dbl
        c.loc[idx, "missing_next"] = np.where(gap, np.round(np.nan_to_num(d / loc)) - 1, 0)
        dd = d[~gap & ~dbl]
        rows.append(dict(row=r, n=len(s), spacing_median_um=np.median(dd), spacing_cv=dd.std() / dd.mean() if len(dd) > 2 else np.nan,
                         regularity_index=dd.mean() / dd.std() if len(dd) > 2 and dd.std() > 0 else np.nan,
                         gaps=int(gap.sum()), missing_estimate=int(np.where(gap, np.round(np.nan_to_num(d / loc)) - 1, 0).sum()), doubles=int(dbl.sum()),
                         across_median_um=s.across_um.median()))
    return pd.DataFrame(rows)


def stagger(c, sp):
    out = {}
    for r, ref in (("OHC2", "OHC1"), ("OHC3", "OHC2")):
        a = c[c.row == r].along_um.values; b = np.sort(c[c.row == ref].along_um.values)
        row = sp[sp.row == ref]
        if len(a) < 3 or len(b) < 3 or row.empty: continue
        s = float(row.spacing_median_um.iloc[0])
        k = np.searchsorted(b, a); k = np.clip(k, 1, len(b) - 1)
        prev = b[k - 1]; off = (a - prev) / s
        ok = (off >= 0) & (off <= 1.5)
        out[f"stagger_{r}_vs_{ref}"] = float(np.median(off[ok] % 1)) if ok.any() else np.nan
    return out


# ---------------- orientation ----------------
def orientation(c, nb):
    """Per row: direction of the notch (outer hair cells; circular statistics) and the bundle
    axis (all rows; axial statistics). Per cell: deviation of the notch direction from its row
    mean (dev_deg), local misorientation (dev_deg against the circular mean of neighbors' dev_deg),
    and the bundle axis deviation from its row mean (axis_dev_deg)."""
    c["dev_deg"] = np.nan; c["local_misorient_deg"] = np.nan; c["axis_dev_deg"] = np.nan
    stats = []
    usable = c.axis_deg.notna() & (c.shape_check != "not found")
    for r in ROWS:
        s = {"row": r}
        m = (c.row == r) & c.notch_deg.notna()
        n = circ_summary(c.loc[m, "notch_deg"].values)
        if np.isfinite(n["mean_deg"]):
            c.loc[m, "dev_deg"] = wrap(c.loc[m, "notch_deg"] - n["mean_deg"])
        n["drift_deg_per_100um"] = 100 * circ_linear_slope(c.loc[m, "along_um"].values, c.loc[m, "notch_deg"].values)
        n["share_pointing_inward"] = float((c.loc[m, "notch_deg"].abs() > 90).mean()) if m.any() else np.nan
        s.update({f"notch_{k}": v for k, v in n.items()})
        m = (c.row == r) & usable
        a = axial_summary(c.loc[m, "axis_deg"].values)
        if np.isfinite(a["mean_deg"]):
            c.loc[m, "axis_dev_deg"] = wrap90(c.loc[m, "axis_deg"] - a["mean_deg"])
        s.update({f"axis_{k}": v for k, v in a.items()})
        stats.append(s)
    th = np.radians(c.dev_deg.values.astype(float))
    for i in range(len(c)):
        if not np.isfinite(th[i]): continue
        j = [k for k in nb[i] if np.isfinite(th[k])]
        if len(j) >= 2:
            mu = np.angle(np.exp(1j * th[j]).mean())
            c.iloc[i, c.columns.get_loc("local_misorient_deg")] = abs(wrap(np.degrees(th[i] - mu)))
    return pd.DataFrame(stats)


def correlogram(c, bins=np.arange(0, 44, 4)):
    m = c.dev_deg.notna().values
    P = np.c_[c.along_um.values, c.across_um.values][m]; d = c.dev_deg.values[m]
    if len(d) < 10: return pd.DataFrame(), np.nan
    D = np.hypot(*(P[:, None, :] - P[None, :, :]).transpose(2, 0, 1)); iu = np.triu_indices(len(d), 1)
    dist = D[iu]; z = (d - d.mean()) / (d.std() + 1e-9); prod = (z[:, None] * z[None, :])[iu]
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (dist > a) & (dist <= b)
        rows.append(dict(dist_from_um=a, dist_to_um=b, pairs=int(sel.sum()), correlation=float(prod[sel].mean()) if sel.sum() >= 10 else np.nan))
    cg = pd.DataFrame(rows)
    below = cg[(cg.correlation < 0)]
    length = float(below.dist_from_um.iloc[0]) if len(below) else np.nan
    return cg, length


# ---------------- clustering of flags ----------------
def edges_of(nb):
    return np.array([(i, j) for i in range(len(nb)) for j in nb[i] if i < j])


def shuffles(flag, groups, n):
    out = np.empty((n, len(flag)), bool)
    for k in range(n):
        f = flag.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]; f[idx] = flag[idx][RNG.permutation(len(idx))]
        out[k] = f
    return out


def join_count(flag, groups, E, valid):
    if len(E) == 0 or flag[valid].sum() < 2:
        return dict(flagged=int(flag[valid].sum()), joins=np.nan, expected=np.nan, ratio=np.nan, p=np.nan)
    sh = shuffles(flag & valid, np.where(valid, groups, "x"), N_SHUFFLE)
    obs = int((flag[E[:, 0]] & flag[E[:, 1]] & valid[E[:, 0]] & valid[E[:, 1]]).sum())
    null = (sh[:, E[:, 0]] & sh[:, E[:, 1]]).sum(1)
    return dict(flagged=int(flag[valid].sum()), joins=obs, expected=float(null.mean()),
                ratio=obs / null.mean() if null.mean() > 0 else np.nan, p=float(((null >= obs).sum() + 1) / (N_SHUFFLE + 1)))


def runs_along(c, flag, valid):
    """Number of runs of flagged cells along each row (fewer runs than expected = flags bunched)."""
    orders = [c[(c.row == r) & valid].sort_values("along_um").index.values for r in ROWS]
    def count(f):
        n = 0
        for s in orders:
            v = f[s]
            if len(v): n += int((v[1:] & ~v[:-1]).sum() + v[0])
        return n
    if flag[valid].sum() < 2: return dict(runs=np.nan, expected_runs=np.nan, p=np.nan)
    obs = count(flag)
    sh = shuffles(flag & valid, np.where(valid, c.row.values, "x"), 2000)
    null = np.array([count(s) for s in sh])
    return dict(runs=obs, expected_runs=float(null.mean()), p=float(((null <= obs).sum() + 1) / (len(null) + 1)))


# ---------------- hot spots ----------------
def local_moran(x, nb):
    ok = np.isfinite(x); out = np.full((len(x), 3), np.nan)
    if ok.sum() < 10: return out
    z = np.where(ok, (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12), np.nan)
    pool = np.where(ok)[0]
    for i in pool:
        j = [k for k in nb[i] if ok[k]]
        if len(j) < 2: continue
        lag = z[j].mean(); I = z[i] * lag
        others = pool[pool != i]
        sims = z[others[RNG.integers(0, len(others), (N_LOCAL, len(j)))]].mean(1) * z[i]   # neighbors drawn at random
        p = ((np.abs(sims) >= abs(I)).sum() + 1) / (N_LOCAL + 1)
        out[i] = (I, p, lag)
    return out


def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full_like(p, np.nan)
    if ok.sum():
        pv = p[ok]; o = np.argsort(pv); r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        q_ok = np.minimum.accumulate(r[::-1])[::-1]; tmp = np.empty_like(q_ok); tmp[o] = np.minimum(q_ok, 1); q[ok] = tmp
    return q


# ---------------- per field ----------------
def run_field(field):
    c = load_field(field)
    if len(c) < 10:
        return None, None, None
    nb, hull = neighbor_graph(c)
    c["neighbors"] = [len(s) for s in nb]; c["edge_of_field"] = hull
    sp = spacing(c)
    ori = orientation(c, nb)
    cg, corr_len = correlogram(c)
    if "shape_check" not in c: c["shape_check"] = np.nan
    c["flag_misoriented"] = c.dev_deg.abs() > MISORIENT_DEG
    c["flag_shape"] = c.shape_check == "doubtful"
    c["flag_no_bundle"] = c.shape_check == "not found"
    prev_gap = c.groupby("row", group_keys=False).apply(lambda s: s.sort_values("along_um").gap_next.shift(1).fillna(False))
    c["flag_next_to_gap"] = c.gap_next | prev_gap.reindex(c.index).fillna(False).astype(bool)
    c["flag_any"] = c.flag_misoriented | c.flag_shape | c.flag_no_bundle
    E = edges_of(nb); valid = np.ones(len(c), bool)
    rec = dict(field=field, damaged=DAMAGED.get(field, ""), hair_cells=len(c), **{f"cells_{r}": int((c.row == r).sum()) for r in ROWS})
    for _, r in sp.iterrows():
        for k in ("spacing_median_um", "spacing_cv", "regularity_index", "gaps", "missing_estimate", "doubles"):
            rec[f"{r.row}_{k}"] = r[k]
    acr = sp.set_index("row").across_median_um
    for a, b in (("IHC", "OHC1"), ("OHC1", "OHC2"), ("OHC2", "OHC3")):
        if a in acr and b in acr: rec[f"row_distance_{a}_{b}_um"] = abs(acr[b] - acr[a])
    rec.update(stagger(c, sp))
    interior = (~c.edge_of_field) & c.row.str.startswith("OHC")
    rec["ohc_interior"] = int(interior.sum())
    rec["ohc_interior_share_6_neighbors"] = float((c.loc[interior, "neighbors"] == 6).mean()) if interior.any() else np.nan
    for _, r in ori.iterrows():
        for k in r.index:
            if k != "row": rec[f"{r.row}_{k}"] = r[k]
    rec["local_misorient_median_deg"] = c.local_misorient_deg.median()
    rec["orientation_corr_length_um"] = corr_len
    for flag in ("flag_misoriented", "flag_shape", "flag_no_bundle", "flag_any"):
        jc = join_count(c[flag].values.astype(bool), c.row.values, E, valid)
        ru = runs_along(c, c[flag].values.astype(bool), valid)
        for k, v in jc.items(): rec[f"{flag}_{k}"] = v
        rec[f"{flag}_runs"] = ru["runs"]; rec[f"{flag}_expected_runs"] = ru["expected_runs"]; rec[f"{flag}_runs_p"] = ru["p"]
    for meas in ("span_um", "v_angle_deg", "height_median_um", "local_misorient_deg"):
        if meas not in c: continue
        lm = local_moran(c[meas].values.astype(float), nb)
        q = bh(lm[:, 1])
        mean = np.nanmean(c[meas].values)
        hi = (q < 0.05) & (c[meas].values > mean) & (lm[:, 2] > 0)
        lo = (q < 0.05) & (c[meas].values < mean) & (lm[:, 2] < 0)
        c[f"hotspot_{meas}"] = np.where(hi, "high cluster", np.where(lo, "low cluster", ""))
        rec[f"hotspot_{meas}_high_cells"] = int(hi.sum()); rec[f"hotspot_{meas}_low_cells"] = int(lo.sum())
    cg.insert(0, "field", field)
    picture(c, nb, field)
    return c, rec, cg


def picture(c, nb, field):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(14, 9))
    col = {"IHC": "magenta", "OHC1": "limegreen", "OHC2": "deepskyblue", "OHC3": "gold"}
    for ax, what in zip(axs, ("orientation", "flags")):
        for i in range(len(c)):
            for j in nb[i]:
                if i < j: ax.plot(c.along_um.iloc[[i, j]], c.across_um.iloc[[i, j]], color="0.85", lw=0.5, zorder=1)
        if what == "orientation":
            m = c.local_misorient_deg.notna()
            sc = ax.scatter(c.along_um[m], c.across_um[m], c=c.local_misorient_deg[m], cmap="viridis_r", vmin=0, vmax=45, s=60, zorder=2)
            nv = np.radians(c.notch_deg.values); hasn = c.notch_deg.notna()
            ax.quiver(c.along_um[hasn], c.across_um[hasn], np.sin(nv[hasn]), np.cos(nv[hasn]), angles="xy", scale_units="xy",
                      scale=0.4, width=0.003, zorder=3)
            fig.colorbar(sc, ax=ax, label="Difference from neighbors (degrees)")
            ax.set_title("Direction of the dark notch in each outer hair cell top (arrows) and difference from neighbors (color)")
        else:
            ax.scatter(c.along_um, c.across_um, c=[col[r] for r in c.row], s=40, zorder=2)
            for flag, mk, lab in (("flag_misoriented", "x", "notch > 45 degrees from row"), ("flag_no_bundle", "s", "no bundle found"),
                                  ("flag_shape", "^", "doubtful bundle shape"), ("flag_next_to_gap", "D", "next to a gap")):
                m = c[flag].astype(bool)
                ax.scatter(c.along_um[m], c.across_um[m], marker=mk, facecolors="none" if mk != "x" else "red",
                           edgecolors="red", s=140, lw=1.5, zorder=3, label=lab)
            if "hotspot_span_um" in c:
                h = c.hotspot_span_um == "high cluster"; l = c.hotspot_span_um == "low cluster"
                ax.scatter(c.along_um[h], c.across_um[h], s=300, facecolors="none", edgecolors="black", lw=2, zorder=4, label="cluster of wide bundles")
                ax.scatter(c.along_um[l], c.across_um[l], s=300, facecolors="none", edgecolors="black", lw=2, ls="--", zorder=4, label="cluster of narrow bundles")
            ax.legend(loc="upper right", fontsize=8, ncol=3)
            ax.set_title("Rows (color) and flagged cells")
        ax.set_xlabel("Distance along the row (um)"); ax.set_ylabel("Distance across (um)"); ax.set_aspect("equal")
    fig.suptitle(field); fig.tight_layout()
    fig.savefig(os.path.join(FIG, field + ".png"), dpi=80); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    fields = [os.path.basename(f)[:-12] for f in sorted(glob.glob(os.path.join(BUNDLES, "*_bundles.csv")))]
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    cells, recs, cgs = [], [], []
    for f in [f for f in fields if only in f]:
        c, rec, cg = run_field(f)
        if c is None:
            print(f"  {f}: too few hair cells, skipped", flush=True); continue
        cells.append(c); recs.append(rec); cgs.append(cg)
        print(f"  {f}: {rec['hair_cells']} hair cells, flagged {int(c.flag_any.sum())}, "
              f"misoriented {rec['flag_misoriented_flagged']}, their neighbor pairs {rec['flag_misoriented_joins']} vs {rec['flag_misoriented_expected']:.1f} expected (P {rec['flag_misoriented_p']:.3f})", flush=True)
    pd.concat(cells).to_csv(os.path.join(OUT, "cells_spatial.csv"), index=False)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "fields_spatial.csv"), index=False)
    pd.concat(cgs).to_csv(os.path.join(OUT, "orientation_correlogram.csv"), index=False)


if __name__ == "__main__":
    main()
