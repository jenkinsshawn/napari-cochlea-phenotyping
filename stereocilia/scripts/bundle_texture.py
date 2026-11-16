"""Texture and uniformity of each stereocilia bundle (inner hair cells and outer rows 1-3).

Input (official run folders from pipeline_settings):
  WORK\\paint_stacks\\<field>.npz       phalloidin flattened to the cell surface (planes -1.8 to +6 um)
  WORK\\bundles_detector\\<field>_bundles.npy / .csv   bundle mask per cell (label = cell) and its row

Each bundle is read from the top-down maximum of the planes 0.3 to 6 um above the surface. Values are rescaled
per bundle to its own 1st and 99.5th percentiles inside the bundle mask grown by 0.5 um, so gain and offset drop out.

Per bundle:
  glcm_<prop>_d<k>    grey-level co-occurrence (16 levels, inside the mask only, 4 directions averaged) at k pixels
                      (k = 1, 2, 3; 0.11, 0.22, 0.33 um): contrast, homogeneity, energy, correlation, entropy
  intensity_cv        SD / mean of background-subtracted phalloidin inside the mask
  tip_height_sd_um    SD over the mask of each pixel's top height (highest plane at half its own peak)
  pieces              separate pieces of the bundle above half its rescaled range (>= 0.05 um2 each)
  largest_piece_share area share of the largest of those pieces
  solidity            mask area / convex hull area
  area_um2            mask area

Output: RESULTS\\bundle_texture_cells.csv
Run (imaging env, scripts folder):  python bundle_texture.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label

PAINT = os.path.join(S.WORK, "paint_stacks")
BUND = os.path.join(S.WORK, "bundles_detector")
LEVELS = 16
DISTANCES = (1, 2, 3)
ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
TOP_LOW_UM, TOP_HIGH_UM = 0.3, 6.0
GROW_UM = 0.5
MIN_PIECE_UM2 = 0.05
PROPS = ("contrast", "homogeneity", "energy", "correlation")


def glcm_props(q, inside):
    """q: integer image 1..LEVELS; pixels outside the mask are set to level 0 and their pairs dropped."""
    g = graycomatrix(np.where(inside, q, 0).astype(np.uint8), DISTANCES, ANGLES, levels=LEVELS + 1, symmetric=True, normed=False)
    g = g[1:, 1:, :, :].astype(float)
    out = {}
    for di, d in enumerate(DISTANCES):
        m = g[:, :, di, :]; tot = m.sum(axis=(0, 1))
        if (tot == 0).any():
            for p in PROPS + ("entropy",): out[f"glcm_{p}_d{d}"] = np.nan
            continue
        m = m / tot
        for p in PROPS:
            out[f"glcm_{p}_d{d}"] = float(np.nanmean(graycoprops(m[:, :, None, :], p)))
        ent = []
        for a in range(len(ANGLES)):
            pa = m[:, :, a][m[:, :, a] > 0]; ent.append(float(-(pa * np.log2(pa)).sum()))
        out[f"glcm_entropy_d{d}"] = float(np.mean(ent))
    return out


def field_cells(fld):
    z = np.load(os.path.join(PAINT, fld + ".npz")); phal = z["phal"].astype(np.float32); off = z["offsets_um"]
    lab = np.load(os.path.join(BUND, fld + "_bundles.npy")); tab = pd.read_csv(os.path.join(BUND, fld + "_bundles.csv"))
    px = S.PX_UM
    sel = (off >= TOP_LOW_UM) & (off <= TOP_HIGH_UM)
    order = np.argsort(-off[sel]); stack = phal[sel][order]; offs = off[sel][order]      # high to low offset
    top = stack.max(axis=0)
    bg = np.percentile(top, 1)
    grow = int(round(GROW_UM / px))
    rows = []
    for _, r in tab.iterrows():
        if "found" in r and not bool(r["found"]): continue
        m = lab == int(r.cell)
        if m.sum() < 20: continue
        ys, xs = np.nonzero(m)
        y0, y1 = max(ys.min() - grow - 2, 0), min(ys.max() + grow + 3, m.shape[0]); x0, x1 = max(xs.min() - grow - 2, 0), min(xs.max() + grow + 3, m.shape[1])
        mc = m[y0:y1, x0:x1]; im = top[y0:y1, x0:x1]; st = stack[:, y0:y1, x0:x1]
        big = ndi.binary_dilation(mc, iterations=grow)
        lo, hi = np.percentile(im[big], [1, 99.5])
        if hi <= lo: continue
        sc = np.clip((im - lo) / (hi - lo), 0, 1)
        q = 1 + np.minimum((sc * LEVELS).astype(int), LEVELS - 1)
        out = dict(field=fld, cell=int(r.cell), row=r.row, shape_check=r.get("shape_check", ""), x=float(xs.mean()), y=float(ys.mean()),
                   area_um2=float(m.sum() * px * px))
        out.update(glcm_props(q, mc))
        v = im[mc] - bg
        out["intensity_cv"] = float(v.std() / v.mean()) if v.mean() > 0 else np.nan
        pk = st.max(axis=0); hit = st >= 0.5 * pk[None]
        idx = np.argmax(hit, axis=0)                                           # first plane from the top at half peak
        th = offs[idx]
        out["tip_height_sd_um"] = float(np.std(th[mc]))
        pieces = label((sc >= 0.5) & mc)
        areas = np.bincount(pieces.ravel())[1:] * px * px; areas = areas[areas >= MIN_PIECE_UM2]
        out["pieces"] = int(len(areas)); out["largest_piece_share"] = float(areas.max() / areas.sum()) if len(areas) else np.nan
        rp = regionprops(mc.astype(np.uint8)); out["solidity"] = float(rp[0].solidity) if rp else np.nan
        rows.append(out)
    return rows


def main(fields=None):
    if fields is None:
        fields = [os.path.basename(f)[:-12] for f in sorted(glob.glob(os.path.join(BUND, "*_bundles.npy")))]
    allrows = []
    for f in fields:
        if not os.path.exists(os.path.join(PAINT, f + ".npz")):
            print(f"{f}: skipped, no paint stack"); continue
        rr = field_cells(f); allrows += rr; print(f"{f}: {len(rr)} bundles")
    out = pd.DataFrame(allrows)
    out.to_csv(os.path.join(S.RESULTS, "bundle_texture_cells.csv"), index=False)
    print("wrote", os.path.join(S.RESULTS, "bundle_texture_cells.csv"), len(out))
    return out


if __name__ == "__main__":
    main()
