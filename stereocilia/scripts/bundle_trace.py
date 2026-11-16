"""Trace each outer hair cell bundle as a centerline and measure its bend (shape) from it.

Image: for every pixel, the brightest phalloidin between LO_UM and HI_UM above the estimated
flat top (plate) of the hair cell that owns the pixel (the paint stacks hold images at fixed
heights from the fitted surface; plate depth per cell is plate_depth_um from measure_bundles.py),
smoothed (Gaussian, 1 pixel), with the broad glow of the flat top removed (white top-hat,
radius TOPHAT_PX).
Bundle: within the cell's search area (pixels within 2.5 um of a hair cell outline, given to the
nearest hair cell), the brightest PCT percent of pixels, small gaps closed (1 pixel), largest piece.
Centerline: the longest path through the skeleton of that piece.

Measures from the centerline (pixel 0.1083 um; frame from the row: 0 degrees = straight across
the row toward the outer side, positive toward increasing distance along the row):
    span_um           distance between the two ends
    length_um         length along the centerline
    bend_depth_um     largest distance of the centerline from the end-to-end line
    rel_bend_depth    bend_depth_um / span_um (0 = straight)
    bend_angle_deg    angle at the bend point between the lines to the two ends
    arm_ratio         shorter / longer centerline length on either side of the bend point (1 = symmetric)
    bend_position     position of the bend point along the centerline (0.5 = middle)
    sharpness         share of the total turning of the centerline that happens within the middle
                      30% of its length around the bend point (V-like = high, U-like = lower)
    axis_deg          perpendicular to the end-to-end line, as an axis (-90..90)
    direction_deg     from the middle of the ends to the bend point (-180..180); blank when
                      bend_depth_um < 0.3
Settings were chosen against Shawn's strokes and PMCA2 (results/tuning) and are scored against
the hand reference from click_bundle_points.py before any measure is used.

The centerline (trace) follows the brighter arm only in many cells and is kept for comparison.
The measure in use is the bowl fit (fit_bowl), chosen on 10 November 2026 against Shawn's
three-click reference (96 readable outer hair cells in 6 fields; settings chosen on AR454 middle,
AR1064 middle and AR1383-2 middle, scored on JW5436 apex (3), AR524 base and AR47 middle):
    image      brightest phalloidin 1.5-3.6 um above the cell's flat top, top-hat as above
    curve      apex point, opening direction, half-width and depth on each side, and an exponent
               p for the shape of the bottom (1 = V, 2 = parabola-like bowl, higher = flatter U):
               point(t) = apex + w(t) t u - d(t) |t|^p v,  t in [-1, 1]
    fit        maximizes the sum over the curve of (normalized brightness - 0.35) per unit
               length, inside the cell outline grown by 0.5 um, excluding 0.2 um along every
               cell outline (bright cell borders); 40 bright start points x 12 directions x 5
               sizes x 2 shapes, best 16 refined (Powell); each side at least 0.87 um wide and
               0.43 um deep
    quality    curve_q20: 20th percentile of normalized brightness along the curve; cells below
               QUALITY_CUT are 'not measured' (keeps 86% of hand-readable cells, removes 71% of
               cells Shawn could not read)
Scores against all 96 hand-read cells (optimizer settings chosen on simulations only): span
median error 0.26 um (r 0.74), bend depth 0.24 um (r 0.65), depth/span r 0.42, direction within
30 degrees 84% (straight-out guess 72%), axis within 15 degrees 73%. Test fields alone: span
0.16 um (r 0.92), bend depth 0.21 um (r 0.81), direction 91%. AR454 middle and AR1064 middle are
the weak fields. The exponent does not agree between phalloidin and PMCA2 fits (rank r 0.31) and is
not used as a phenotype.
Hand repeatability (15 cells read twice): span 0.16 um, bend depth 0.15 um, direction 3.5 degrees.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os
from collections import deque
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from skimage.morphology import white_tophat, disk, closing, skeletonize, dilation
from skimage.segmentation import find_boundaries

PAINT = os.path.join(S.WORK, 'paint_stacks')
CELLS = os.path.join(S.WORK, 'cells', 'v1')
TYPES = os.path.join(S.WORK, 'cells', 'types')
BUNDLES = os.path.join(S.WORK, 'bundles_detector')
PX = S.PX_UM
LO_UM, HI_UM, TOPHAT_PX, PCT, REACH_UM = 0.9, 3.3, 15, 88, 2.5      # centerline (trace), kept for comparison
BOWL_LO_UM, BOWL_HI_UM, BOWL_TAU, BOWL_GROW_PX, BOWL_BORDER_PX, QUALITY_CUT = 1.5, 3.6, 0.35, 5, 2, 0.55
HAIR = ("IHC", "OHC1", "OHC2", "OHC3")


def field_maps(field):
    """Phalloidin stack, heights, per-pixel owner cell and that cell's plate depth."""
    z = np.load(os.path.join(PAINT, field + ".npz")); ph = z["phal"].astype(np.float32); off = z["offsets_um"]
    lab = np.load(S.cell_labels_path(field))   # corrected outlines where they exist, as for cell types and bundles
    t = pd.read_csv(os.path.join(TYPES, field + "_types.csv")); hair = t[t.row.isin(HAIR)]
    hm = np.isin(lab, hair.cell.values); d, (iy, ix) = ndi.distance_transform_edt(~hm, return_indices=True)
    owner = np.where(d * PX <= REACH_UM, lab[iy, ix], 0)
    b = pd.read_csv(os.path.join(BUNDLES, field + "_bundles.csv"))
    missing = sorted(set(b.cell.astype(int)) - set(np.unique(lab).astype(int)))
    if missing:   # bundle table and outlines from different outline versions
        raise ValueError(f"bundle table names {len(missing)} cells not in the outlines (e.g. {missing[:3]}); "
                         "rerun cell types and bundles on the outlines in use (pipeline_settings.cell_labels_path)")
    pl = np.full(max(lab.max(), owner.max()) + 1, np.nan)
    for k, v in zip(b.cell, b.plate_depth_um):
        pl[k] = v
    pmap = np.where(owner > 0, pl[owner], np.nan)
    return dict(ph=ph, off=off, owner=owner, pmap=pmap, lab=lab, types=t, z=z)


def window_max(stack, off, pmap, lo, hi):
    out = np.zeros(stack.shape[1:], np.float32)
    for k, h in enumerate(off):
        use = (h >= pmap + lo) & (h <= pmap + hi)
        out = np.where(use, np.maximum(out, stack[k]), out)
    return out


def bundle_image(F, lo=LO_UM, hi=HI_UM, tophat=TOPHAT_PX, stack=None):
    W = window_max(F["ph"] if stack is None else stack, F["off"], F["pmap"], lo, hi)
    I = gaussian_filter(W, 1.0)
    return (white_tophat(I, disk(tophat)) if tophat else I), W


def longest_path(skel):
    pts = np.argwhere(skel)
    if len(pts) < 3:
        return None
    index = {tuple(p): i for i, p in enumerate(pts)}
    nb = [[index[(y + dy, x + dx)] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
           if (dy or dx) and (y + dy, x + dx) in index] for y, x in pts]

    def bfs(s):
        dist = np.full(len(pts), -1.0); prev = np.full(len(pts), -1); dist[s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in nb[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1; prev[v] = u; q.append(v)
        return dist, prev
    d, _ = bfs(0); a = int(np.argmax(d)); d, prev = bfs(a); b = int(np.argmax(d))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return pts[path][:, ::-1].astype(float)          # x, y


def trace(I, owner, cell, pct=PCT):
    terr = owner == cell; ys, xs = np.nonzero(terr)
    if len(ys) < 100:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tr = terr[y0:y1, x0:x1]; sub = np.where(tr, I[y0:y1, x0:x1], 0)
    m = closing(tr & (sub > np.percentile(sub[tr], pct)), disk(1)); lb, n = ndi.label(m)
    if n == 0:
        return None
    m = lb == (np.bincount(lb.ravel())[1:].argmax() + 1)
    p = longest_path(skeletonize(m))
    if p is None:
        return None
    p[:, 0] += x0; p[:, 1] += y0
    return p


def shape_measures(path, frame, smooth=3):
    """Bend measures of a centerline (x, y pixels), in the row frame (along, across unit vectors)."""
    along, across = frame
    e1, e2 = path[0], path[-1]; ch = e2 - e1; L = np.linalg.norm(ch)
    if L < 3:
        return {}
    n = np.array([-ch[1], ch[0]]) / L; sd = (path - e1) @ n; k = int(np.argmax(np.abs(sd))); apex = path[k]
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]; total = seg[-1]
    s1, s2 = seg[k], total - seg[k]
    u1, u2 = e1 - apex, e2 - apex
    ang = np.degrees(np.arccos(np.clip(u1 @ u2 / max(np.linalg.norm(u1) * np.linalg.norm(u2), 1e-9), -1, 1)))
    # sharpness: share of total turning near the bend point, on a lightly smoothed centerline
    sharp = np.nan
    if len(path) >= 2 * smooth + 5:
        P = np.c_[np.convolve(path[:, 0], np.ones(smooth) / smooth, "valid"), np.convolve(path[:, 1], np.ones(smooth) / smooth, "valid")]
        d = np.diff(P, axis=0); ds = np.hypot(*d.T); ok = ds > 0; d = d[ok] / ds[ok, None]
        turn = np.abs(np.arctan2(d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0], (d[:-1] * d[1:]).sum(1)))
        ss = np.r_[0, np.cumsum(ds[ok])][1:-1]; ss = ss / max(ss[-1], 1e-9)
        pos = s1 / max(total, 1e-9)
        if turn.sum() > 0:
            sharp = float(turn[np.abs(ss - pos) <= 0.15].sum() / turn.sum())
    bis = apex - (e1 + e2) / 2; depth = abs(sd[k]) * PX
    return dict(span_um=L * PX, length_um=total * PX, bend_depth_um=depth, rel_bend_depth=depth / (L * PX),
                bend_angle_deg=ang, arm_ratio=min(s1, s2) / max(s1, s2) if max(s1, s2) > 0 else np.nan,
                bend_position=s1 / total if total > 0 else np.nan, sharpness=sharp,
                axis_deg=(np.degrees(np.arctan2(n @ along, n @ across)) + 90) % 180 - 90,
                direction_deg=np.degrees(np.arctan2(bis @ along, bis @ across)) if depth >= 0.3 else np.nan,
                apex_x=apex[0], apex_y=apex[1], end1_x=e1[0], end1_y=e1[1], end2_x=e2[0], end2_y=e2[1])


# ---------------- bowl fit ----------------
_T = np.linspace(-1, 1, 41)


def field_maps_bowl(field):
    """field_maps with a missing plate depth set to 0.6 um (as in the click tool) and the bowl image."""
    F = field_maps(field); F["pmap"] = np.where(np.isfinite(F["pmap"]), F["pmap"], 0.6)
    F["I_bowl"], _ = bundle_image(F, lo=BOWL_LO_UM, hi=BOWL_HI_UM)
    return F


def curve_points(q):
    ax_, ay_, th, w1, w2, d1, d2, p = q
    v = np.array([np.cos(th), np.sin(th)]); u = np.array([-v[1], v[0]]); p = np.clip(p, 0.8, 6)
    w = np.where(_T < 0, w1, w2); d = np.where(_T < 0, d1, d2)
    return np.array([ax_, ay_])[None, :] + (w * _T)[:, None] * u[None, :] - (d * np.abs(_T) ** p)[:, None] * v[None, :]


def _cell_image(I, lab, cell, grow, border_px):
    cm = lab == cell; ys, xs = np.nonzero(cm)
    y0, y1, x0, x1 = max(ys.min() - grow - 8, 0), ys.max() + grow + 9, max(xs.min() - grow - 8, 0), xs.max() + grow + 9
    reg = dilation(cm[y0:y1, x0:x1], disk(grow))
    if border_px:
        reg &= ~dilation(find_boundaries(lab[y0:y1, x0:x1], mode="thick"), disk(border_px))
    sub = I[y0:y1, x0:x1]; med = np.median(sub[reg]); hi = np.percentile(sub[reg], 99)
    return (sub - med) / max(hi - med, 1e-6), reg, (y0, x0)


def _score(q, Z, tau, min_w=3, min_d=0):
    if q[3] < min_w or q[4] < min_w or q[5] < min_d or q[6] < min_d:
        return 1e3
    P = curve_points(q); seg = np.hypot(*np.diff(P, axis=0).T); ds = np.r_[seg, 0] * 0.5 + np.r_[0, seg] * 0.5
    return -np.sum((map_coordinates(Z, [P[:, 1], P[:, 0]], order=1, cval=-2) - tau) * ds)


def fit_bowl(I, lab, cell, tau=BOWL_TAU, grow=BOWL_GROW_PX, border_px=BOWL_BORDER_PX, n_starts=16, min_w=8, min_d=4,
             sizes=((14, 12), (20, 16), (18, 8), (24, 20), (12, 6))):
    """Defaults chosen on simulated bowls of known shape (results/tuning/bowl_fit_optimizer_synthetic_2026-11-10.csv):
    16 refined starts, a wider range of start sizes, and lower limits on each side's half-width (8 px, 0.87 um)
    and depth (4 px, 0.43 um), below the smallest values in Shawn's readings (shortest arm 1.44 um, shallowest
    bend 0.86 um). These stop the fit from settling on one arm."""
    z, reg, (y0, x0) = _cell_image(I, lab, cell, grow, border_px); Z = np.where(reg, z, -2.0)
    cand = np.argwhere(reg & (z > np.percentile(z[reg], 80)))
    if len(cand) == 0:
        return None
    if len(cand) > 40:
        cand = cand[np.linspace(0, len(cand) - 1, 40).astype(int)]
    inits = []
    for cy, cx in cand:
        for th in np.radians(np.arange(0, 360, 30)):
            for w, d in (sizes or ((14, 12), (20, 16), (18, 8))):
                for p in (1.3, 2.2):
                    q = [cx, cy, th, w, w, d, d, p]; inits.append((_score(q, Z, tau, min_w, min_d), q))
    inits.sort(key=lambda x: x[0]); best = None
    for _, q0 in inits[:n_starts]:
        res = minimize(_score, q0, args=(Z, tau, min_w, min_d), method="Powell", options={"maxiter": 3000, "xtol": 0.05, "ftol": 1e-3})
        if best is None or res.fun < best.fun:
            best = res
    q = best.x.copy(); P = curve_points(q)
    vals = map_coordinates(Z, [P[:, 1], P[:, 0]], order=1, cval=-2)
    P[:, 0] += x0; P[:, 1] += y0
    return dict(end1=P[0], apex=np.array([q[0] + x0, q[1] + y0]), end2=P[-1], exponent=float(np.clip(q[7], 0.8, 6)),
                points=P, curve_q20=float(np.percentile(vals, 20)), measured=bool(np.percentile(vals, 20) >= QUALITY_CUT))


def bowl_measures(fit, frame):
    """Same definitions as the hand reference (click_bundle_points.measures), plus the exponent."""
    along, across = frame; p1, pb, p2 = fit["end1"], fit["apex"], fit["end2"]
    ch = p2 - p1; L = np.linalg.norm(ch)
    if L == 0:
        return {}
    n = np.array([-ch[1], ch[0]]) / L; bis = pb - (p1 + p2) / 2; depth = abs(n @ (pb - p1))
    a1, a2 = np.linalg.norm(p1 - pb), np.linalg.norm(p2 - pb)
    ang = np.degrees(np.arccos(np.clip(((p1 - pb) @ (p2 - pb)) / max(a1 * a2, 1e-9), -1, 1)))
    seg = np.hypot(*np.diff(fit["points"], axis=0).T)
    return dict(span_um=L * PX, bend_depth_um=depth * PX, rel_bend_depth=depth / L, bend_angle_deg=ang,
                arm_ratio=min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else np.nan, length_um=seg.sum() * PX,
                bowl_exponent=fit["exponent"], axis_deg=(np.degrees(np.arctan2(n @ along, n @ across)) + 90) % 180 - 90,
                direction_deg=np.degrees(np.arctan2(bis @ along, bis @ across)) if depth * PX >= 0.3 else np.nan,
                curve_q20=fit["curve_q20"], measured=fit["measured"])
