"""Outline and measure the stereocilia bundle of every inner and outer hair cell.

For each field:
  1. Which way is up: the bundles stand on the side of lower image numbers in every field
     checked by eye (BUNDLE_SIDE). Two automatic rules were tried and each was wrong in some
     fields: how peaked phalloidin is on each side (cell borders are as peaked as bundles in
     AR1064 base and AR454 apex (3)), and which side holds the Calb2-filled cell bodies.
  2. Bundle layer: each hair cell's bundle depth is found as the image where phalloidin in and
     around the cell is most uneven (see bundle_layer); for every pixel near a hair cell, the
     brightest phalloidin within 0.9 um of that depth, and the height at which it occurs. A white top-hat (radius 2 um) removes the
     broad glow of the cuticular plates.
  3. Bundle outlines: the surface and body outlines of one inner hair cell are joined; pixels
     above the Otsu threshold of the layer, within 2.5 um of a hair cell, are given to the nearest hair cell (watershed from the hair cell outlines); each
     cell keeps its largest piece of at least 1 um2.
  4. Measures per bundle (one row per hair cell):
       area_um2, span_um (longest distance across), width_um (short axis)
       vertex_x/y and two arm ends from the bundle's center line: the ends are its two most
       distant points, the vertex is the point farthest from the line joining them
       arm1_um, arm2_um, v_angle_deg (180 = straight line), v_depth_um (vertex to end line)
       orientation_deg: direction from the cell center to the vertex in the row frame,
       0 = straight across the row toward the outer side, positive = toward increasing
       distance along the row; offset_um: distance from cell center to bundle center
       height_median_um, height_p90_um: height of the bundle pixels' brightest image above the
       cell's own plate (approximate: the microscope blurs about 0.7 um in depth)
       thickness_um = area / span; shape_check: "doubtful" for outer hair cell bundles thicker
       than 2.2 um or longer than 7.5 um (usually a cuticular plate picked up instead)
       phal_mean, phal_sum (bundle layer), pmca2_mean (bundle layer, PMCA2 fields only)
Output: E:\\Webber_confocal\\work\\bundles\\<field>_bundles.csv, figures\\bundles\\<field>.png,
results\\bundle_fields.csv (one row per field: which way up, threshold, bundles found per row).

With --detector, bundle pixels come from the trained detector (train_bundle_detector.py
--predict) instead of the brightness rule, counted only 0.6 um or more above each cell's own
plate; outputs go to ..._detector folders and results\bundle_fields_detector.csv.

Run from the scripts folder:
    python measure_bundles.py --detector                (every field with a detector result)
    python measure_bundles.py --only "AR524 Ocm, Calb2 Phall P1 60X MIDDLE"
    python measure_bundles.py                   (all fields, about 15 minutes)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, glob, os, sys, warnings
import numpy as np
import pandas as pd
import nd2
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import white_tophat, disk, skeletonize, binary_dilation
from skimage.measure import regionprops, label
from skimage.segmentation import watershed, find_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RAW = S.RAW
FIELDS = os.path.join(S.WORK, 'fields')
CELLS = os.path.join(S.WORK, 'cells')
ANN = os.path.join(S.ANNOTATIONS, 'cells')
TYPES = os.path.join(CELLS, "types")
OUT = os.path.join(S.WORK, 'bundles')
LOW_UM, HIGH_UM = 0.6, 3.0        # bundle layer, um above the cell surface
REACH_UM = 2.5                    # how far outside its cell a bundle may reach
BUNDLE_SIDE = -1                  # bundles stand on the side of lower image numbers (checked by eye)
USE_DETECTOR = "--detector" in sys.argv
PROB = os.path.join(S.WORK, 'bundle_prob')
STACKS = os.path.join(S.WORK, 'paint_stacks')
HAIR_ROWS = ("IHC", "OHC1", "OHC2", "OHC3")
COLORS = {"IHC": [1, 0.2, 1], "OHC1": [0.1, 1, 0.1], "OHC2": [0.1, 0.7, 1], "OHC3": [1, 1, 0.1]}


def stretch(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(float) - lo) / max(hi - lo, 1), 0, 1)


def side_stack(stack, surf, k0, k1, sign):
    """Brightest value and its index offset, from k0 to k1 images away from the surface on one side."""
    nz = stack.shape[0]
    best = np.zeros(surf.shape, np.float32); where = np.zeros(surf.shape, np.float32)
    s = np.round(surf).astype(int)
    for k in range(k0, k1 + 1):
        idx = np.clip(s + sign * k, 0, nz - 1)
        v = np.take_along_axis(stack, idx[None], 0)[0]
        better = v > best
        best[better] = v[better]; where[better] = k
    return best, where


def body_side(calb2, surf, dz):
    """Side of the surface where the cell bodies are: Calb2 (inner hair cell bodies) 1-5 um on
    each side of the surface is compared; the bundles are on the other side. Returns the sign
    of the bundle side (-1 = lower image numbers) and the two Calb2 means."""
    k0, k1 = int(round(1.0 / dz)), int(round(5.0 / dz))
    m = {}
    for sign in (-1, 1):
        b, _ = side_stack(calb2, surf, k0, k1, sign)
        m[sign] = float(np.percentile(b, 95))
    return (-1 if m[1] >= m[-1] else 1), m


def up_sign(stack, surf, dz, ohc_mask):
    k0, k1 = int(round(1.0 / dz)), int(round(3.0 / dz))
    peak = {}
    for sign in (-1, 1):
        b, _ = side_stack(stack, surf, k0, k1, sign)
        v = b[ohc_mask]
        peak[sign] = np.percentile(v, 99) / max(np.median(v), 1)
    return (-1 if peak[-1] >= peak[1] else 1), peak


def local_frame(types, cell_xy, row):
    """Unit vectors along and across the row (across points to the outer side) near a cell."""
    same = types[types.row.isin(HAIR_ROWS)]
    d = np.hypot(same.x - cell_xy[0], same.y - cell_xy[1])
    near = same[d < 25 / S.PX_UM]
    if len(near) < 3:
        near = same
    P = near[["x", "y"]].values - near[["x", "y"]].values.mean(0)
    along = np.linalg.svd(P)[2][0]
    # orient along with increasing along_um, and across toward increasing across_um
    if np.corrcoef(near[["x", "y"]].values @ along, near.along_um)[0, 1] < 0:
        along = -along
    across = np.array([-along[1], along[0]])
    if np.corrcoef(near[["x", "y"]].values @ across, near.across_um)[0, 1] < 0:
        across = -across
    return along, across


def shape(mask, px):
    sk = skeletonize(mask)
    pts = np.argwhere(sk if sk.sum() >= 3 else mask)[:, ::-1].astype(float)    # x, y
    if len(pts) < 3:
        return None
    d = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)) if len(pts) < 1500 else None
    if d is None:
        sub = pts[np.linspace(0, len(pts) - 1, 1500).astype(int)]
        d = np.hypot(*(sub[:, None, :] - sub[None, :, :]).transpose(2, 0, 1)); pts = sub
    i, j = np.unravel_index(d.argmax(), d.shape)
    e1, e2 = pts[i], pts[j]
    chord = e2 - e1; L = np.linalg.norm(chord)
    if L == 0:
        return None
    dist = np.abs(chord[0] * (pts[:, 1] - e1[1]) - chord[1] * (pts[:, 0] - e1[0])) / L
    v = pts[dist.argmax()]
    depth = dist.max() * px
    a1, a2 = np.linalg.norm(e1 - v) * px, np.linalg.norm(e2 - v) * px
    if depth < 0.3 or min(a1, a2) < 0.3:
        angle = 180.0; v = (e1 + e2) / 2
    else:
        u1, u2 = (e1 - v) / np.linalg.norm(e1 - v), (e2 - v) / np.linalg.norm(e2 - v)
        angle = float(np.degrees(np.arccos(np.clip(u1 @ u2, -1, 1))))
    return dict(span_um=L * px, vertex=v, end1=e1, end2=e2, arm1_um=a1, arm2_um=a2, v_angle_deg=angle, v_depth_um=depth)


def cell_surface(stack, surf, cells, ids, dz, px):
    """Each hair cell's own plate depth: the offset from the field surface (within +-3 um) where
    the cell's mean phalloidin peaks. Pixels within reach of a hair cell take its offset."""
    nz = stack.shape[0]; s = np.round(surf).astype(int); K = int(round(3.0 / dz))
    offs = np.arange(-K, K + 1); prof = np.zeros((len(offs), len(ids)))
    for j, k in enumerate(offs):
        v = np.take_along_axis(stack, np.clip(s + k, 0, nz - 1)[None], 0)[0]
        prof[j] = ndi.mean(v, cells, ids)
    best = offs[prof.argmax(0)]
    lut = np.zeros(cells.max() + 1); lut[ids] = best
    hair = np.isin(cells, ids)
    dist, (iy, ix) = ndi.distance_transform_edt(~hair, return_indices=True)
    near_cell = cells[iy, ix]
    shift = np.where(dist * px <= REACH_UM + 1, lut[near_cell], 0)
    return surf + shift


def bundle_layer(stack, surf, cells, ids, dz, px, sign, kfix=None):
    """Each hair cell's bundle depth is the image, from 3 um below to 4 um above the field surface,
    where phalloidin in and within 1 um of the cell is most uneven (standard deviation over mean),
    at least 0.6 um above the cell's plate (the image where the cell's own mean peaks): a bundle
    is a thin bright line, a cuticular plate is evenly bright. The layer is the brightest
    phalloidin within 0.9 um of that depth; pixels within reach of a cell use that cell's depth.
    Height is the distance above the field surface (um) of the brightest image."""
    nz = stack.shape[0]; s = np.round(surf).astype(int)
    hair = np.isin(cells, ids)
    dist, (iy, ix) = ndi.distance_transform_edt(~hair, return_indices=True)
    near_cell = np.where(dist * px <= REACH_UM + 1, cells[iy, ix], 0)
    region = np.where(dist * px <= 1.0, cells[iy, ix], 0)
    offs = np.arange(-int(round(3.0 / dz)), int(round(4.0 / dz)) + 1)           # positive = toward the bundle side
    if kfix is None:
        score = np.zeros((len(offs), len(ids))); mean = np.zeros((len(offs), len(ids)))
        own = np.where(hair, cells, 0)
        for j, k in enumerate(offs):
            v = np.take_along_axis(stack, np.clip(s + sign * k, 0, nz - 1)[None], 0)[0]
            score[j] = ndi.standard_deviation(v, region, ids) / np.maximum(ndi.mean(v, region, ids), 1)
            mean[j] = ndi.mean(v, own, ids)
        # the plate is where the cell's own mean peaks; the bundle must lie at least 0.6 um above it
        plate = offs[mean.argmax(0)]
        above = offs[:, None] >= plate[None, :] + int(round(0.6 / dz))
        score = np.where(above, score, -np.inf)
        k_b = offs[score.argmax(0)]
        k_b = np.where(np.isfinite(score.max(0)), k_b, plate + int(round(1.5 / dz)))
        kfix = dict(zip(ids, k_b)); bundle_layer.plate = dict(zip(ids, plate))
    lut = np.full(cells.max() + 1, int(round(1.5 / dz))); 
    for c, k in kfix.items(): lut[c] = k
    center = lut[near_cell]; center[near_cell == 0] = int(round(1.5 / dz))
    w = int(round(0.9 / dz))
    best = np.zeros(surf.shape, np.float32); where = np.zeros(surf.shape, np.float32)
    for k in range(offs.min() - w, offs.max() + w + 1):
        use = np.abs(k - center) <= w
        v = np.take_along_axis(stack, np.clip(s + sign * k, 0, nz - 1)[None], 0)[0]
        better = use & (v > best)
        best[better] = v[better]; where[better] = k
    return best, where, kfix


def detector_layer(field, cells, ids, px):
    """Bundle probability from train_bundle_detector.py, counted only 0.6 um or more above each
    hair cell's own plate (the image of the flattened stack where the cell's mean phalloidin
    peaks). Returns the highest probability, its height above the plate (um) and phalloidin at
    that height."""
    pz = np.load(os.path.join(PROB, field + ".npz")); prob = pz["prob"].astype(np.float32) / 255
    stk = np.load(os.path.join(STACKS, field + ".npz")); hts = stk["offsets_um"]; phal = stk["phal"].astype(np.float32)
    means = np.stack([ndi.mean(p, cells, ids) for p in phal]); plate = means.argmax(0)
    hair = np.isin(cells, ids)
    d, (iy, ix) = ndi.distance_transform_edt(~hair, return_indices=True)
    lut = np.full(cells.max() + 1, -1); lut[ids] = plate
    pzi = lut[cells[iy, ix]]; pzi[d * px > REACH_UM + 1] = -1
    zi = np.arange(len(hts))[:, None, None]
    ok = (zi <= pzi[None] - 2) & (pzi[None] >= 0)            # image 0 is highest; 2 images = 0.6 um
    p = np.where(ok, prob, 0); k = p.argmax(0); best = p.max(0)
    hz = hts[k] - np.where(pzi >= 0, hts[np.clip(pzi, 0, len(hts) - 1)], 0)
    ph_at = np.take_along_axis(phal, k[None], 0)[0]
    return best, hz.astype(np.float32), ph_at


def measure_field(field):
    z = np.load(os.path.join(FIELDS, field + ".npz")); st = list(z["stains"])
    dz, px = float(z["voxel_um"][0]), float(z["voxel_um"][2])
    surf = z["surface"].astype(float)
    cells = np.load(S.cell_labels_path(field)).astype(np.int32)   # corrected outlines where they exist, otherwise v1
    types = pd.read_csv(os.path.join(TYPES, field + "_types.csv"))
    hair = types[types.row.isin(HAIR_ROWS)]
    arr = S.read_raw(field)
    phal = arr[:, st.index("Phalloidin")].astype(np.float32)
    pm = arr[:, st.index("PMCA2")].astype(np.float32) if "PMCA2" in st else None
    calb2 = arr[:, st.index("Calb2")].astype(np.float32)
    del arr
    # inner hair cells: surface and body outlines of one cell act as one marker
    hair = hair.copy()
    hair["unit"] = np.where(hair.row == "IHC", "I" + hair.inner_cell_id.astype(str), "O" + hair.cell.astype(str))
    first = hair.groupby("unit").cell.transform("min")
    remap = dict(zip(hair.cell, first)); lut = np.zeros(cells.max() + 1, np.int32)
    for k, v in remap.items(): lut[k] = v
    cells = lut[cells]
    hair = hair[hair.cell == first]
    ohc_mask = np.isin(cells, hair.cell[hair.row.str.startswith("OHC")].values)
    sign = BUNDLE_SIDE; peak = {-1: np.nan, 1: np.nan}
    layer, where, kstar = bundle_layer(phal, surf, cells, hair.cell.values, dz, px, sign)
    plate = bundle_layer.plate
    # height above the cell's own plate: the brightest image of each bundle pixel, minus the plate image
    hair_m = np.isin(cells, hair.cell.values)
    _, (iy, ix) = ndi.distance_transform_edt(~hair_m, return_indices=True)
    plut = np.zeros(cells.max() + 1); 
    for c, k in plate.items(): plut[c] = k
    height = (where - plut[cells[iy, ix]]) * dz
    pml = bundle_layer(pm, surf, cells, hair.cell.values, dz, px, sign, kfix=kstar)[0] if pm is not None else None
    th = white_tophat(gaussian(layer, 1.0, preserve_range=True), disk(int(round(2.0 / px))))
    hair_lab = np.where(np.isin(cells, hair.cell.values), cells, 0)
    near = binary_dilation(hair_lab > 0, disk(int(round(REACH_UM / px))))
    thr = threshold_otsu(th[near])
    fg = (th > thr) & near
    if USE_DETECTOR:
        best, hz, ph_at = detector_layer(field, cells, hair.cell.values, px)
        thr = float(os.environ.get("BUNDLE_CUTOFF", 0.85)); th = best      # outer hair cells; 0.5 took in rings of cell top
        thr_in = float(os.environ.get("BUNDLE_CUTOFF_INNER", 0.65))        # inner hair cell bundles score lower
        ihc_ids = hair.cell[hair.row == "IHC"].values
        dd, (iy2, ix2) = ndi.distance_transform_edt(hair_lab == 0, return_indices=True)
        near_inner = np.isin(hair_lab[iy2, ix2], ihc_ids)
        fg = np.where(near_inner, best > thr_in, best > thr) & near
        height, layer = hz, ph_at
    own = watershed(-th, markers=hair_lab, mask=fg | (hair_lab > 0))
    own[~fg] = 0
    rows, out = [], np.zeros_like(cells)
    for _, c in hair.iterrows():
        m = own == c.cell
        if m.sum() * px * px < 1.0:
            rows.append(dict(cell=c.cell, row=c.row, found=False)); continue
        lab = label(m); big = np.bincount(lab.ravel())[1:].argmax() + 1; m = lab == big
        if m.sum() * px * px < 1.0:
            rows.append(dict(cell=c.cell, row=c.row, found=False)); continue
        out[m] = c.cell
        rp = regionprops(m.astype(int))[0]
        sh = shape(m, px)
        cy, cx = c.y, c.x
        along, across = local_frame(types, (cx, cy), c.row)
        r = dict(cell=c.cell, row=c.row, found=True, area_um2=m.sum() * px * px,
                 width_um=rp.minor_axis_length * px, offset_um=np.hypot(rp.centroid[1] - cx, rp.centroid[0] - cy) * px,
                 height_median_um=float(np.median(height[m])), height_p90_um=float(np.percentile(height[m], 90)),
                 bundle_depth_um=float(kstar[c.cell] * dz), plate_depth_um=float(plate[c.cell] * dz),
                 phal_mean=float(layer[m].mean()), phal_sum=float(layer[m].sum()),
                 pmca2_mean=float(pml[m].mean()) if pml is not None else np.nan)
        if sh:
            vec = sh["vertex"] - np.array([cx, cy])
            r.update({k: v for k, v in sh.items() if k not in ("vertex", "end1", "end2")})
            r.update(vertex_x=sh["vertex"][0], vertex_y=sh["vertex"][1],
                     orientation_deg=float(np.degrees(np.arctan2(vec @ along, vec @ across))))
        rows.append(r)
    b = pd.DataFrame(rows); b.insert(0, "field", field)
    # shape check: outer hair cell bundles are thin lines; thick or long pieces are usually plates
    b["thickness_um"] = b.area_um2 / b.span_um
    ohc = b.row.str.startswith("OHC")
    b["shape_check"] = np.where(~b.found.astype(bool), "not found",
                       np.where(ohc & ((b.thickness_um > 2.2) | (b.span_um > 7.5)), "doubtful", "ok"))
    info = dict(field=field, up_side="lower image numbers" if sign < 0 else "higher image numbers",
                calb2_lower=round(peak[-1], 1), calb2_higher=round(peak[1], 1), threshold=round(float(thr), 1))
    return b, out, layer, hair_lab, info


def picture(b, out, layer, hair_lab, field, path):
    rgb = np.dstack([stretch(layer) * 0.85] * 3)
    rgb[find_boundaries(hair_lab, mode="inner")] = [0.35, 0.35, 0.35]
    lut = {c: COLORS.get(r, [1, 1, 1]) for c, r in zip(b.cell, b.row)}
    bd = find_boundaries(out, mode="inner")
    for c, col in lut.items():
        rgb[bd & (out == c)] = col
    fig, ax = plt.subplots(figsize=(11, 11)); ax.imshow(rgb); ax.axis("off")
    if "vertex_x" in b:
        v = b.dropna(subset=["vertex_x"])
        ax.plot(v.vertex_x, v.vertex_y, "r.", ms=3)
    n = b[b.found].row.value_counts().to_dict(); tot = b.row.value_counts().to_dict()
    ax.set_title(f"{field}\nbundles found: inner {n.get('IHC', 0)}/{tot.get('IHC', 0)}, rows 1/2/3 "
                 f"{n.get('OHC1', 0)}/{tot.get('OHC1', 0)}, {n.get('OHC2', 0)}/{tot.get('OHC2', 0)}, {n.get('OHC3', 0)}/{tot.get('OHC3', 0)} "
                 "(outlines colored by row, red dot = vertex)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--detector", action="store_true")
    a = ap.parse_args()
    global OUT
    if a.detector:
        OUT = OUT + "_detector"
    os.makedirs(OUT, exist_ok=True); fig_dir = os.path.join(S.FIGURES, "bundles_detector" if a.detector else "bundles"); os.makedirs(fig_dir, exist_ok=True)
    fields = [os.path.basename(f)[:-10] for f in sorted(glob.glob(os.path.join(TYPES, "*_types.csv")))]
    fields = [f for f in fields if a.only in f]
    if a.detector:
        fields = [f for f in fields if os.path.exists(os.path.join(PROB, f + ".npz"))]
    log = os.path.join(S.RESULTS, "bundle_fields_detector.csv" if a.detector else "bundle_fields.csv")
    infos = pd.read_csv(log).to_dict("records") if os.path.exists(log) else []
    for f in fields:
        b, out, layer, hair_lab, info = measure_field(f)
        b.round(3).to_csv(os.path.join(OUT, f + "_bundles.csv"), index=False)
        np.save(os.path.join(OUT, f + "_bundles.npy"), out.astype(np.uint16))
        picture(b, out, layer, hair_lab, f, os.path.join(fig_dir, f + ".png"))
        n = b[b.found].row.value_counts().to_dict()
        info.update({f"found_{r}": n.get(r, 0) for r in HAIR_ROWS}, **{f"cells_{r}": int((b.row == r).sum()) for r in HAIR_ROWS})
        infos = [i for i in infos if i["field"] != f] + [info]
        pd.DataFrame(infos).to_csv(log, index=False)
        print(f"  {f}: {info}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
