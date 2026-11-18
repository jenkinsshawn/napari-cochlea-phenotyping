"""Stereocilia bundle phenotype panel (inner hair cells and outer rows 1-3) with robustness conditions.

Input (official run folders from pipeline_settings):
  WORK\\paint_stacks\\<field>.npz       phalloidin flattened to the cell surface (planes -1.8 to +6 um)
  WORK\\bundles_detector\\<field>_bundles.npy / .csv   bundle outline per cell (label = cell) and its row

Each bundle is read from the top-down maximum of the planes TOP_LOW_UM to TOP_HIGH_UM above the surface. Texture
uses values rescaled per bundle to its own 1st and 99.5th percentiles (gain and offset drop out). Measures:

  texture inside the outline   glcm_<p>_d<k>: contrast, dissimilarity, homogeneity, energy, correlation, entropy at
                               k = 1, 2, 3 pixels (0.11-0.33 um), 16 levels, 4 directions averaged
  texture along the skeleton   strip_glcm_<p>_d2: same, in a strip 0.33 um either side of the outline's skeleton
  local binary patterns        lbp_uniform_share, lbp_entropy (8 neighbours, radius 1)
  orientation coherence        coherence (structure tensor, 0.11 um smoothing, 0.33 um window; 1 = one direction)
  brightness distribution      intensity_cv, intensity_skew, intensity_kurtosis, histogram_entropy, bundle_to_surface
  shape                        area_um2, perimeter_um, solidity, eccentricity, major_um, minor_um, skeleton_um,
                               branch_points, thickness_um, thickness_cv, pieces, largest_piece_share
  height                       tip_height_um, tip_height_sd_um, top_p90_um, volume_um3
  position (not phenotypes)    local_depth_um, local_slope_deg: depth and slope of the cell surface at the bundle
                               (field_quality.local_surface), for noise adjustment in phenotype_statistics.py
  outline ring                 outline_ring_um2: area the outline nearly encloses (it runs onto the rim below the bundle);
                               used by the reliability ranking, not tested as a phenotype
  detector confidence          confidence: mean detector probability in the bundle (0-1), used to rank bundles by
                               reliability in phenotype_statistics.py, not tested as a phenotype

Outlines are smoothed first when SMOOTH_BUNDLE_OUTLINE is True in the settings (or the environment variable is 1).

Conditions (robustness): base; outline eroded 1 px; outline grown 1 px; planes 0.6-5.4 um; added noise
(Gaussian, SD equal to the field's own high-frequency noise).

Output: RESULTS\\bundle_phenotypes_cells.csv (one row per bundle and condition)
Run (imaging env, scripts folder):  python bundle_phenotypes.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import field_quality as FQ
import glob
import numpy as np, pandas as pd
from scipy import ndimage as ndi, stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, structure_tensor
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize

PAINT = os.path.join(S.WORK, "paint_stacks")
BUND = os.path.join(S.WORK, "bundles_detector")
PROB = os.path.join(S.WORK, "bundle_prob")   # detector voxel probabilities (optional; gives each bundle a confidence)
LEVELS = 16
DISTANCES = (1, 2, 3)
ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
TOP_LOW_UM, TOP_HIGH_UM = 0.3, 6.0
GROW_UM = 0.5
STRIP_PX = 3
MIN_PIECE_UM2 = 0.05
SMOOTH_OUTLINE = bool(getattr(S, "SMOOTH_BUNDLE_OUTLINE", False)) or os.environ.get("SMOOTH_BUNDLE_OUTLINE", "") == "1"
SMOOTH_SIGMA_PX = 1.3


def smooth_outline(m):
    """Detector outline with holes filled, only its largest piece kept and its edge smoothed (Gaussian, SMOOTH_SIGMA_PX)."""
    m = ndi.binary_fill_holes(m); lab_, n = ndi.label(m)
    if n > 1: m = lab_ == (1 + int(np.argmax(ndi.sum(m, lab_, range(1, n + 1)))))
    return ndi.binary_fill_holes(ndi.gaussian_filter(m.astype(np.float32), SMOOTH_SIGMA_PX) > 0.5)
PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
CONDITIONS = ("base", "eroded", "grown", "planes", "noise")


def glcm(q, inside, distances=DISTANCES, prefix="glcm"):
    g = graycomatrix(np.where(inside, q, 0).astype(np.uint8), distances, ANGLES, levels=LEVELS + 1, symmetric=True, normed=False)
    g = g[1:, 1:].astype(float); out = {}
    for di, d in enumerate(distances):
        m = g[:, :, di, :]; tot = m.sum(axis=(0, 1))
        if (tot == 0).any():
            for p in PROPS + ("entropy",): out[f"{prefix}_{p}_d{d}"] = np.nan
            continue
        m = m / tot
        for p in PROPS: out[f"{prefix}_{p}_d{d}"] = float(np.nanmean(graycoprops(m[:, :, None, :], p)))
        ent = [float(-(pa * np.log2(pa)).sum()) for pa in (m[:, :, a][m[:, :, a] > 0] for a in range(len(ANGLES)))]
        out[f"{prefix}_entropy_d{d}"] = float(np.mean(ent))
    return out


def measure(im, st, offs, mc, bg, px):
    """im: top view crop, st: planes crop (high to low offset), mc: outline mask crop."""
    out = {}
    grow = int(round(GROW_UM / px)); big = ndi.binary_dilation(mc, iterations=grow)
    lo, hi = np.percentile(im[big], [1, 99.5])
    if hi <= lo or mc.sum() < 20: return None
    sc = np.clip((im - lo) / (hi - lo), 0, 1); q = 1 + np.minimum((sc * LEVELS).astype(int), LEVELS - 1)
    out.update(glcm(q, mc))
    smooth = (ndi.gaussian_filter(mc.astype(float), 2.0) >= 0.5) & mc                 # outline smoothed before the skeleton
    sk = skeletonize(smooth if smooth.sum() >= 10 else mc); strip = ndi.binary_dilation(sk, iterations=STRIP_PX) & mc
    out.update(glcm(q, strip, distances=(2,), prefix="strip_glcm"))
    lbp = local_binary_pattern((sc * 255).astype(np.uint8), 8, 1, method="uniform")[mc]
    h = np.bincount(lbp.astype(int), minlength=10) / max(len(lbp), 1)
    out["lbp_uniform_share"] = float(1 - h[9]); out["lbp_entropy"] = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    Arr, Arc, Acc = structure_tensor(sc, sigma=1, order="rc")
    w = lambda a: ndi.uniform_filter(a, 3)
    Arr, Arc, Acc = w(Arr), w(Arc), w(Acc)
    tr = Arr + Acc; disc = np.sqrt((Arr - Acc) ** 2 + 4 * Arc ** 2)
    coh = np.where(tr > 1e-9, disc / np.maximum(tr, 1e-9), 0)
    out["coherence"] = float(np.average(coh[mc], weights=tr[mc] + 1e-12))
    v = im[mc] - bg
    out["intensity_cv"] = float(v.std() / v.mean()) if v.mean() > 0 else np.nan
    out["intensity_skew"] = float(stats.skew(v)); out["intensity_kurtosis"] = float(stats.kurtosis(v))
    hq = np.bincount(q[mc], minlength=LEVELS + 1)[1:] / mc.sum(); out["histogram_entropy"] = float(-(hq[hq > 0] * np.log2(hq[hq > 0])).sum())
    ring = ndi.binary_dilation(mc, iterations=int(round(1.5 / px))) & ~ndi.binary_dilation(mc, iterations=grow)
    sv = np.median(im[ring] - bg) if ring.any() else np.nan
    out["bundle_to_surface"] = float(np.median(v) / sv) if sv and sv > 0 else np.nan
    rp = regionprops(label(mc))
    rp = max(rp, key=lambda r: r.area)
    # outline running onto the rim below the bundle: area enclosed once small gaps are closed (not a phenotype; used by the
    # reliability ranking in phenotype_statistics.py)
    out["outline_ring_um2"] = float((ndi.binary_fill_holes(ndi.binary_closing(mc, iterations=3)).sum() - mc.sum()) * px * px)
    out.update(area_um2=float(mc.sum() * px * px), perimeter_um=float(rp.perimeter * px), solidity=float(rp.solidity),
               eccentricity=float(rp.eccentricity), major_um=float(rp.axis_major_length * px), minor_um=float(rp.axis_minor_length * px))
    out["skeleton_um"] = float(sk.sum() * px)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    out["branch_points"] = int(label((nb >= 3) & sk, connectivity=2).max())        # clusters of junction pixels
    dt = ndi.distance_transform_edt(mc)[sk & mc] * 2 * px
    out["thickness_um"] = float(np.median(dt)) if len(dt) else np.nan
    out["thickness_cv"] = float(dt.std() / dt.mean()) if len(dt) and dt.mean() > 0 else np.nan
    pieces = label((sc >= 0.5) & mc); areas = np.bincount(pieces.ravel())[1:] * px * px; areas = areas[areas >= MIN_PIECE_UM2]
    out["pieces"] = int(len(areas)); out["largest_piece_share"] = float(areas.max() / areas.sum()) if len(areas) else np.nan
    offs = np.asarray(offs, float) * getattr(S, "Z_SCALE", 1.0)   # nominal plane offsets to true distance along the optical axis
    pk = st.max(axis=0); hit = st >= 0.5 * pk[None]; th = offs[np.argmax(hit, axis=0)]
    out["tip_height_um"] = float(np.median(th[mc])); out["tip_height_sd_um"] = float(np.std(th[mc]))
    out["top_p90_um"] = float(np.percentile(th[mc], 90))
    dz = abs(offs[0] - offs[1]) if len(offs) > 1 else 0.3 * getattr(S, "Z_SCALE", 1.0)
    out["volume_um3"] = float((hit & mc[None]).sum() * px * px * dz)
    return out


def field_cells(fld, conditions=CONDITIONS, seed=0):
    z = np.load(os.path.join(PAINT, fld + ".npz")); phal = z["phal"].astype(np.float32); off = z["offsets_um"]
    lab = np.load(os.path.join(BUND, fld + "_bundles.npy")); tab = pd.read_csv(os.path.join(BUND, fld + "_bundles.csv"))
    px = S.PX_UM; rng = np.random.default_rng(seed)
    views = {}
    for name, (a, b) in {"base": (TOP_LOW_UM, TOP_HIGH_UM), "planes": (0.6, 5.4)}.items():
        sel = (off >= a) & (off <= b); o = np.argsort(-off[sel]); st = phal[sel][o]; views[name] = (st, off[sel][o], st.max(axis=0))
    prob = None
    if os.path.exists(os.path.join(PROB, fld + ".npz")):
        prob = np.load(os.path.join(PROB, fld + ".npz"))["prob"]; keep_planes = off >= TOP_LOW_UM; grow_c = int(round(0.5 / px))
    smaps = FQ.surface_maps(fld)
    st0, of0, top0 = views["base"]
    hp = top0 - ndi.gaussian_filter(top0, 1.0)
    bgm = (ndi.binary_dilation(lab > 0, iterations=10) == 0) & (top0 < np.percentile(top0, 50))
    noise_sd = 1.4826 * np.median(np.abs(hp[bgm] - np.median(hp[bgm])))
    rows = []
    for _, r in tab.iterrows():
        if "found" in r and not bool(r["found"]): continue
        m = lab == int(r.cell)
        if m.sum() < 20: continue
        ys, xs = np.nonzero(m); pad = int(round(2.0 / px))
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0]); x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
        mc0 = m[y0:y1, x0:x1]
        conf = np.nan; ldep, lslp = FQ.local_surface(fld, [xs.mean()], [ys.mean()], smaps)
        if prob is not None:   # mean detector probability over the largest connected piece of voxels >= 0.5 (outline grown 0.5 um)
            p = prob[:, y0:y1, x0:x1].astype(np.float32) / 255.0
            v = (p >= 0.5) & ndi.binary_dilation(mc0, iterations=grow_c)[None] & keep_planes[:, None, None]
            cc, n = ndi.label(v, structure=np.ones((3, 3, 3)))
            if n: conf = float(p[cc == 1 + int(np.argmax(ndi.sum(v, cc, range(1, n + 1))))].mean())
        if SMOOTH_OUTLINE:
            mc0 = smooth_outline(mc0)
            if mc0.sum() < 20: continue
        for cond in conditions:
            st, offs, top = views["planes" if cond == "planes" else "base"]
            im = top[y0:y1, x0:x1].copy(); sc_ = st[:, y0:y1, x0:x1]
            mc = mc0
            if cond == "eroded": mc = ndi.binary_erosion(mc0)
            if cond == "grown": mc = ndi.binary_dilation(mc0)
            if cond == "noise":
                sc_ = sc_ + rng.normal(0, noise_sd, sc_.shape).astype(np.float32); im = sc_.max(axis=0)
            bg = np.percentile(top, 1)
            o = measure(im, sc_, offs, mc, bg, px)
            if o is None: continue
            o.update(field=fld, cell=int(r.cell), row=r.row, shape_check=r.get("shape_check", ""), condition=cond,
                     x=float(xs.mean()), y=float(ys.mean()), noise_sd=float(noise_sd), confidence=conf,
                     local_depth_um=float(ldep[0]), local_slope_deg=float(lslp[0]))
            rows.append(o)
    return rows


def main(fields=None, conditions=CONDITIONS):
    if fields is None:
        fields = [os.path.basename(f)[:-12] for f in sorted(glob.glob(os.path.join(BUND, "*_bundles.npy")))]
    allrows = []
    for f in fields:
        if not os.path.exists(os.path.join(PAINT, f + ".npz")): print(f"{f}: skipped, no paint stack"); continue
        rr = field_cells(f, conditions); allrows += rr; print(f"{f}: {len(rr)} rows", flush=True)
    out = pd.DataFrame(allrows); out.to_csv(os.path.join(S.RESULTS, "bundle_phenotypes_cells.csv"), index=False)
    print("wrote", len(out)); return out


if __name__ == "__main__":
    main()
