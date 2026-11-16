"""Inner hair cell stereocilia bundles: automatic 3D outline and measures (settings frozen 10 November 2026).

Input: cut-outs from ihc_stacks.py (E:\\Webber_confocal\\work\\ihc_stacks\\<field>.npz; planes 3 um below to
15 um above the field surface at 0.3 um; 24 x 24 um, x along the row, y toward the outer hair cells).

Outline (checked against Shawn's paint of 6 cells from above: median overlap 0.76, length along the row
within -0.1 to +1.1 um, depth across within 0.3-1.2 um):
  1. Phalloidin smoothed (0.2 um across planes, 0.09 um in plane); thin bright structures kept by a white
     top-hat of 1 um radius in each plane (removes the broad cuticular plate and cell bodies).
  2. Threshold: Otsu within the cell's window along the row; pieces under 64 voxels dropped; pieces at
     least 15% of the largest kept.
  3. The bundle is one leaning sheet: a line is fitted (weighted principal axis, across x height) through
     the brightest 15% of voxels; voxels within 1.2 um of it, and at most 1 um beyond the bright core at
     either end, are kept; then the largest connected piece.
  4. Window along the row: +-4.3 um around the cell centre to find the bundle, then +-4.5 um around the
     bundle's own brightness centre. Bundles touching the window ends are flagged (length capped near 9 um).
Measures (um unless stated; heights in stage steps of 0.3 um, no correction for refractive index):
  length_along, depth_across, footprint_area_um2, bow_um (largest distance of the bundle's ridge from the
  straight line between its ends, seen from above), base_um (lowest 2% of bundle heights, from the field
  surface), tip_um (highest 2%), height_um (tip - base), axis_length_um (extent along the fitted lean
  axis), lean_deg (angle of the axis from vertical; positive = tips toward the outer hair cells),
  row_height_median / row_height_iqr / end_minus_middle (tallest height at each position along the row:
  median, spread, and mean of the outer quarters minus the middle half), n_tall_stereocilia and
  tall_spacing_um (brightness peaks along the row at 70% of the bundle height, at least 0.35 um apart),
  phal_sum (background-subtracted) and phal_mean, offset_across_um (footprint centre from the cell centre),
  touches_window_end, pmca2_mean (fields with PMCA2 only; reference, not a measure input).

Run from the scripts folder:
    python ihc_bundles.py          (all fields, about 15-20 minutes; writes results\\ihc_bundles_all.csv)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob, os, time
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from skimage.measure import label as cclab
from skimage.morphology import white_tophat, disk, remove_small_objects

STACKS = os.path.join(S.WORK, 'ihc_stacks')
HERE = os.path.dirname(os.path.abspath(__file__)); RESULTS = S.RESULTS
PX, DZ = 0.1083, 0.3


def _threshold_mask(V, T, a0, a1):
    sub = T[:, :, a0:a1]; pos = sub[sub > 0]
    if pos.size < 100: return np.zeros(T.shape, bool)
    m = np.zeros(T.shape, bool); m[:, :, a0:a1] = sub > threshold_otsu(pos)
    m = remove_small_objects(m, 64); L = cclab(m)
    if L.max() == 0: return m
    s = np.bincount(L.ravel()); s[0] = 0; return np.isin(L, np.where(s >= 0.15 * s.max())[0])


def _sheet(m, V, band_um=1.2, ext_um=1.0):
    zz, yy, xx = np.where(m); vals = V[m]; core = vals >= np.percentile(vals, 85)
    P = np.c_[yy[core] * PX, zz[core] * DZ]; w = vals[core]; mu = (P * w[:, None]).sum(0) / w.sum()
    C = ((P - mu) * w[:, None]).T @ (P - mu) / w.sum(); d = np.linalg.eigh(C)[1][:, -1]
    if d[1] < 0: d = -d                                                    # axis points up (increasing plane = higher)
    Q = np.c_[yy * PX, zz * DZ] - mu; t = Q @ d; n = np.abs(Q @ np.array([-d[1], d[0]])); tc = (P - mu) @ d
    keep = (n <= band_um) & (t >= tc.min() - ext_um) & (t <= tc.max() + ext_um)
    out = np.zeros_like(m); out[zz[keep], yy[keep], xx[keep]] = True
    L = cclab(out)
    if L.max() == 0: return out, d
    s = np.bincount(L.ravel()); s[0] = 0; return L == s.argmax(), d


def outline(vol):
    """vol (planes low->high, across, along) -> bundle mask, lean axis (across, height), shift of the window (px)."""
    nz, ny, nx = vol.shape; c = nx // 2
    V = ndi.gaussian_filter(vol.astype(np.float32), (0.7, 0.8, 0.8))
    T = np.stack([white_tophat(p, disk(int(1.0 / PX))) for p in V])
    m = _threshold_mask(V, T, int(c - 4.3 / PX), int(c + 4.3 / PX))
    if not m.any(): return m, None, 0
    m, d = _sheet(m, V)
    w = np.where(m, V, 0).sum((0, 1)); sh = int(round((np.arange(nx) * w).sum() / max(w.sum(), 1e-9))) - c
    m = _threshold_mask(V, T, int(c + sh - 4.5 / PX), int(c + sh + 4.5 / PX))
    if not m.any(): return m, None, sh
    m, d = _sheet(m, V)
    return m, d, sh


def measures(vol, m, d, sh, offsets_um, pm=None):
    nz, ny, nx = vol.shape; c = nx // 2
    if m is None or not m.any(): return dict(found=False)
    zz, yy, xx = np.where(m); hz = offsets_um[zz]
    V = ndi.gaussian_filter(vol.astype(np.float32), (0.7, 0.8, 0.8)); bg = np.percentile(V, 20)
    top = m.any(0); ys, xs = np.where(top)
    r = dict(found=True, length_along=float(np.ptp(xs) * PX), depth_across=float(np.ptp(ys) * PX), footprint_area_um2=float(top.sum() * PX * PX))
    # ridge seen from above: for each x, the across position of the brightest projected voxel inside the footprint
    proj = np.where(m, V, 0).max(0); cols = np.unique(xs); ridge = np.array([np.argmax(proj[:, x]) for x in cols]) * PX; xu = cols * PX
    if len(cols) >= 5:
        line = np.interp(xu, [xu[0], xu[-1]], [ridge[0], ridge[-1]]); r["bow_um"] = float(np.max(np.abs(ndi.median_filter(ridge, 5) - line)))
    base, tip = np.percentile(hz, 2), np.percentile(hz, 98)
    r.update(base_um=float(base), tip_um=float(tip), height_um=float(tip - base))
    if d is not None:
        t = np.c_[yy * PX, zz * DZ] @ d; r["axis_length_um"] = float(np.percentile(t, 98) - np.percentile(t, 2))
        r["lean_deg"] = float(np.degrees(np.arctan2(d[0], d[1])))
    colh = np.array([offsets_um[np.where(m[:, :, x].any(1))[0]].max() for x in cols]) - base
    q = len(colh) // 4
    r.update(row_height_median=float(np.median(colh)), row_height_iqr=float(np.subtract(*np.percentile(colh, [75, 25]))),
             end_minus_middle=float(np.mean(np.r_[colh[:q], colh[-q:]]) - np.mean(colh[q:-q])) if q >= 2 else np.nan)
    h70 = base + 0.7 * (tip - base); k70 = np.argmin(np.abs(offsets_um - h70)); band = slice(max(k70 - 2, 0), k70 + 3)
    prof = np.where(m[band], V[band], 0).max((0, 1)); prof = ndi.gaussian_filter1d(prof, 1.0)
    pk, _ = find_peaks(prof, distance=max(1, int(0.35 / PX)), prominence=0.1 * prof.max() if prof.max() > 0 else 1)
    r.update(n_tall_stereocilia=int(len(pk)), tall_spacing_um=float(np.median(np.diff(pk)) * PX) if len(pk) >= 2 else np.nan)
    r.update(phal_sum=float(np.clip(V[m] - bg, 0, None).sum()), phal_mean=float(V[m].mean()),
             offset_across_um=float((ys.mean() - ny / 2) * PX),
             touches_window_end=bool(xs.min() <= int(c + sh - 4.5 / PX) + 1 or xs.max() >= int(c + sh + 4.5 / PX) - 2),
             window_shift_um=float(sh * PX))
    if pm is not None:
        r["pmca2_mean"] = float(ndi.gaussian_filter(pm.astype(np.float32), (0.7, 0.8, 0.8))[m].mean())
    return r


def run_field(field):
    z = np.load(os.path.join(STACKS, field + ".npz")); cells = pd.read_csv(os.path.join(STACKS, field + "_cells.csv"))
    ph = z["phal"]; offs = z["offsets_um"]; pm = z["pmca2"] if "pmca2" in z.files else None
    rows = []
    for i, c in cells.iterrows():
        vol = ph[i].astype(np.float32); m, d, sh = outline(vol)
        rows.append(dict(field=field, cell=int(c.cell), u=c.u, **measures(vol, m, d, sh, offs, None if pm is None else pm[i])))
    return pd.DataFrame(rows)


def main():
    fields = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(STACKS, "*.npz")))
    out = []
    for i, f in enumerate(fields, 1):
        t = time.time(); r = run_field(f); out.append(r)
        print(f"{i}/{len(fields)} {f}: {int(r.found.sum())} of {len(r)} bundles ({time.time() - t:.0f} s)", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    pd.concat(out, ignore_index=True).to_csv(os.path.join(RESULTS, "ihc_bundles_all.csv"), index=False); print("done", flush=True)


if __name__ == "__main__":
    main()
