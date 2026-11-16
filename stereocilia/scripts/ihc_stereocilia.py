"""Inner hair cell stereocilia: learned outline, base and tip, and measures for every inner hair cell.

Input: cut-outs from ihc_stacks.py (E:\\Webber_confocal\\work\\ihc_stacks; planes 3 um below to 15 um above the
field surface, 24 x 24 um, x along the row, y toward the outer hair cells).
Classifier: E:\\Webber_confocal\\models\\ihc_stereocilia_classifier.joblib (random forest on multiscale
brightness, edge and second-derivative features, trained on Shawn's stereocilia strokes in 12 cells with
counterexamples outside his paint from above).

Steps for each cell (volumes resampled to 0.2166 um in all directions, scaled to the cell's 99.9th percentile):
  1. Classifier probability P and a rod filter (Sato tubeness, scales 0.22-0.43 um) for every voxel.
  2. Bundle core: P >= 0.85, rod response in the cell's top 2% and brightness in its top 10%, cut to the cell's own
     stretch of the row (half the distance to each neighbor, 3.5-5 um); the largest piece plus pieces within 1.5 um of it.
  3. Lean axis: weighted principal axis of the core (across x height); the base is the core's low end along it.
  4. Extension toward the tip: voxels with P >= 0.5, rod response in the top 7% and brightness in the top 25%,
     connected to the core, beyond the core's base along the lean axis and within 2 um of it.
Heights use the scale along the optical axis measured from DAPI nuclei (nuclei 1.19 times taller than wide in
the two upper layers of three fields): z distances are multiplied by Z_SCALE = 0.84.
Measures (um, degrees): base_um and tip_um (from the field surface), height_um (tip - base), length_um (along
the lean axis, base to tip), lean_deg (0 = upright; negative = tips away from the outer hair cells),
depth_across_um, width_along_um (capped by the window near 9-10 um), volume_um3, phal_mean, contrast (core
mean / median of the shell around the outline), tip_may_be_cut (tip within 1 um of the cut-out edge),
category: standing; lying flat (core axis within 25 degrees of horizontal: the outline grows at most 0.5 um past the core's
  end away from the outer hair cells and not past its end toward them; length is the extent along the axis, and tip, height and lean are left empty because the direction is not readable); no bundle
  (core under 50 voxels). contrast uses a shell 0.65-1.7 um around the outline (padding outside the image ignored).
Output: results\\ihc_stereocilia_cells.csv and E:\\Webber_confocal\\work\\ihc_masks\\<field>.npz (outline of each cell).

Run from the scripts folder:
    python ihc_stereocilia.py          (all fields; about 6 s per cell)
    python ihc_stereocilia.py --v2     (cut-outs from ihc_stacks_v2; writes results\\ihc_stereocilia_cells_v2.csv)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import joblib
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage.feature import multiscale_basic_features
from skimage.filters import sato
from skimage.measure import label as cclab

STACKS = os.path.join(S.WORK, 'ihc_stacks')
MASKS = os.path.join(S.WORK, 'ihc_masks')
MODEL = os.path.join(S.MODELS, 'ihc_stereocilia_classifier.joblib')
HERE = os.path.dirname(os.path.abspath(__file__)); RESULTS = S.RESULTS
PX, DZ = S.PX_UM, S.DZ_UM; ISO = 2 * PX
ZF = (DZ / ISO, PX / ISO, PX / ISO)
Z_SCALE = S.Z_SCALE
Z0 = -3.0          # height of the cut-out's first plane above the field surface (nominal um)
_rf = None


def model():
    global _rf
    if _rf is None:
        _rf = joblib.load(MODEL)["model"]; _rf.n_jobs = -1
    return _rf


def iso(vol):
    # Brightness with both levels over real voxels (12 Nov 2026): the 1st percentile maps to S.IHC_BACKGROUND_RATIO and the
    # 99.9th to 1, zero fill stays 0. The rule lives in the plugin package (napari_cochlea_phenotyping._standardize).
    from napari_cochlea_phenotyping._standardize import scale_real_voxels
    v = ndi.zoom(vol.astype(np.float32), ZF, order=1)
    return scale_real_voxels(v, S.IHC_BACKGROUND_RATIO).astype(np.float32)



def fill_mask(vol):
    """Voxels of the resampled cut-out that lie outside the image (zero fill)."""
    return ndi.zoom(vol.astype(np.float32), ZF, order=1) <= 0


def features(v):
    return multiscale_basic_features(v, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=4, num_sigma=4,
                                     channel_axis=None).astype(np.float32)


def territory(shape, half=(4.3, 4.3)):
    """Voxels within the cell's own stretch of the row: half the distance to each neighbor along the row."""
    c = shape[2] // 2; x = np.arange(shape[2]); keep = (x >= c - half[0] / ISO) & (x <= c + half[1] / ISO)
    return np.broadcast_to(keep, shape)


def in_window(m, half=(4.3, 4.3)):
    m = m & territory(m.shape, half); L = cclab(m); out = np.zeros_like(m)
    s = np.bincount(L.ravel()); s[0] = 0
    out[np.isin(L, np.where(s >= 30)[0])] = True
    return out


def main_piece(m, near_um=1.5):
    L = cclab(m)
    if L.max() == 0: return m
    s = np.bincount(L.ravel()); s[0] = 0; big = L == s.argmax(); d = distance_transform_edt(~big) * ISO; keep = big.copy()
    for k in range(1, L.max() + 1):
        if k != s.argmax() and s[k] >= 30 and d[L == k].min() <= near_um: keep |= L == k
    return keep


def heights(zz):
    return (Z0 + zz * ISO) * Z_SCALE


def lean_axis(v, m):
    zz, yy, xx = np.where(m); h = heights(zz); y = yy * ISO; w = v[m]; P = np.c_[y, h]
    mu = (P * w[:, None]).sum(0) / w.sum(); C = ((P - mu) * w[:, None]).T @ (P - mu) / w.sum(); d = np.linalg.eigh(C)[1][:, -1]
    t = (P - mu) @ d
    if np.mean(h[t >= np.percentile(t, 98) - 0.3]) < np.mean(h[t <= np.percentile(t, 2) + 0.3]): d = -d
    return mu, d


def outline(v, P, Tb, half=(4.3, 4.3), flat_deg=65, real=None):
    """Returns the outline, the core and whether the bundle lies flat (core axis within 25 degrees of horizontal)."""
    # real: voxels inside the image; percentile thresholds are taken over them only (12 Nov 2026), so zero fill
    # where the cut-out reaches past the image does not loosen the thresholds
    real = np.ones(v.shape, bool) if real is None else real; pc = lambda a, q: np.percentile(a[real], q)
    core = in_window((P >= 0.85) & (Tb >= pc(Tb, 98)) & (v >= pc(v, 90)) & real, half); core = main_piece(core)
    if core.sum() < 50: return core, core, False
    mu, d = lean_axis(v, core); flat = abs(np.degrees(np.arctan2(d[0], d[1]))) > flat_deg
    zz, yy, xx = np.indices(v.shape); Q = np.stack([yy * ISO, heights(zz)], -1) - mu; t = Q @ d; n = np.abs(Q @ np.array([-d[1], d[0]]))
    tc = t[core]; lo, hi = np.percentile(tc, [2, 98])
    if flat:   # folded bundles point away from the outer hair cells (Shawn's paint, 10 of 10 cells): no growth past the
        ohc_hi = d[0] > 0   # base end (toward the outer hair cells), at most 0.5 um past the tip end
        allowed = (n <= 2.0) & (t >= lo - (0.0 if not ohc_hi else 0.5)) & (t <= hi + (0.0 if ohc_hi else 0.5))
    else:
        allowed = (t >= lo) & (n <= 2.0)
    weak = (P >= 0.5) & (Tb >= pc(Tb, 93)) & (v >= pc(v, 75)) & real
    L = cclab((weak & allowed & territory(v.shape, half)) | core); keep = np.isin(L, np.unique(L[core])); keep[L == 0] = False
    return keep, core, flat


def measures(v, m, core, flat=False, fill=None):
    shell = ndi.binary_dilation(m, iterations=8) & ~ndi.binary_dilation(m, iterations=3) & (v > 0)
    r = dict(core_voxels=int(core.sum()),
             contrast=float(v[core].mean() / max(np.median(v[shell]), 1e-6)) if core.any() and shell.any() else np.nan)
    if core.sum() < 50:
        r["category"] = "no bundle"; return r
    zz, yy, xx = np.where(m); h = heights(zz); y = yy * ISO; w = v[m]
    mu, d = lean_axis(v, m); t = (np.c_[y, h] - mu) @ d
    zc, yc, _ = np.where(core); hc = heights(zc); tcore = (np.c_[yc * ISO, hc] - mu) @ d
    tb = np.percentile(tcore, 2); tt = np.percentile(t, 98)
    bsel = tcore <= tb + 0.3
    q_top = np.percentile(h, 99.5); tsel = (h >= q_top - 0.2) if not flat else (t >= tt - 0.3)   # standing: top of the outline
    base_h, base_y = hc[bsel].mean(), (yc * ISO)[bsel].mean(); tip_y = y[tsel].mean()
    tip_h = q_top - TIP_BLUR_UM if not flat else h[tsel].mean()
    nz, ny, nx = m.shape; e = int(1 / ISO)
    end = tsel | (t <= np.percentile(t, 2) + 0.3) if flat else tsel
    cut = bool(zz[end].max() >= nz - 1 - e or yy[end].min() <= e or yy[end].max() >= ny - 1 - e)
    r.update(base_um=float(base_h), volume_um3=float(m.sum() * ISO ** 3 * Z_SCALE), phal_mean=float(w.mean()),
             depth_across_um=float(np.ptp(yy) * ISO), width_along_um=float(np.ptp(xx) * ISO), tip_may_be_cut=cut,
             bundle_at_image_edge=bool(fill is not None and (m & ndi.binary_dilation(fill, iterations=2)).any()))   # outline within 2 voxels of the image edge
    if flat:   # direction of a bundle lying flat is not readable; length is the extent along its axis
        r.update(length_um=float(np.percentile(t, 98) - np.percentile(t, 2)), tip_um=np.nan, height_um=np.nan, lean_deg=np.nan,
                 category="lying flat")
    else:
        r.update(tip_um=float(tip_h), height_um=float(tip_h - base_h), length_um=float(np.hypot(tip_y - base_y, tip_h - base_h)),
                 lean_deg=float(np.degrees(np.arctan2(tip_y - base_y, tip_h - base_h))), category="standing")
    return r


TIP_BLUR_UM = 0.26   # top of the outline (99.5th percentile of height) reads this much above the true tip on the synthetic set


def run_cell(vol, half=(4.3, 4.3)):
    v = iso(vol); F = features(v); P = model().predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(v.shape)
    Tb = sato(v, sigmas=[1.0, 1.5, 2.0], black_ridges=False)
    fill = fill_mask(vol); m, core, flat = outline(v, P, Tb, half, real=~fill)
    return m, core, measures(v, m, core, flat, fill=fill)


def halves(cells, lo=3.5, hi=5.0, edge=4.3, min_um=2.0):
    """Half the distance to the nearest inner hair cell on each side, measured along each cell's own cut-out
    axis (um), limited to lo-hi; cells closer than min_um (top and body of one cell) are ignored."""
    xy = cells[["x", "y"]].values * PX; hl = np.full(len(cells), edge); hr = np.full(len(cells), edge)
    for i, a in enumerate(np.radians(cells.angle_deg.values)):
        d = xy - xy[i]; p = d @ np.array([np.cos(a), np.sin(a)]); q = np.abs(d @ np.array([-np.sin(a), np.cos(a)]))
        ok = (np.hypot(p, q) >= min_um) & (q <= 6)
        if (ok & (p < 0)).any(): hl[i] = np.clip(-p[ok & (p < 0)].max() / 2, lo, hi)
        if (ok & (p > 0)).any(): hr[i] = np.clip(p[ok & (p > 0)].min() / 2, lo, hi)
    return hl, hr


def main(fields=None, v2=False, workers=16):
    stacks, mask_dir = (STACKS + "_v2", MASKS + "_v2") if v2 else (STACKS, MASKS)   # module settings stay unchanged
    fields = fields or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(stacks, "*.npz")))
    os.makedirs(mask_dir, exist_ok=True); rows = []
    for i, f in enumerate(fields, 1):
        t0 = time.time(); z = np.load(os.path.join(stacks, f + ".npz")); cells = pd.read_csv(os.path.join(stacks, f + "_cells.csv"))
        masks, cores = [], []; hl, hr = halves(cells)
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda j: run_cell(z["phal"][j], (hl[j], hr[j])), range(len(cells))))
        for j, c in cells.iterrows():
            m, core, r = res[j]; masks.append(m); cores.append(core)
            rows.append(dict(field=f, cell=int(c.cell), u=c.u, x=c.x, y=c.y, half_left_um=hl[j], half_right_um=hr[j], **r))
        np.savez_compressed(os.path.join(mask_dir, f + ".npz"), outline=np.stack(masks), core=np.stack(cores))
        print(f"{i}/{len(fields)} {f}: {len(cells)} cells ({time.time() - t0:.0f} s)", flush=True)
    out = pd.DataFrame(rows); os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "ihc_stereocilia_cells_v2.csv" if v2 else "ihc_stereocilia_cells.csv"), index=False)
    print("done", flush=True); return out


if __name__ == "__main__":
    main(v2="--v2" in sys.argv)
