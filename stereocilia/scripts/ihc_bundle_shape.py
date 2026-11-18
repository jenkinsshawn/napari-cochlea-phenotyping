"""3D shape of each inner hair cell bundle from the front-view outline (ihc_stereocilia.py), on volumes resampled to
0.2166 um in all directions (nominal planes; distances along the optical axis are corrected by Z_SCALE).

Input: WORK\ihc_masks_v2\<field>.npz (outline: cells x Z x Y x X, one per cell in the order of
WORK\ihc_stacks_v2\<field>_cells.csv), RESULTS\ihc_stereocilia_cells_v2.csv.
Measures per bundle: surface_um2 (marching cubes), compactness (surface of a sphere of the same volume / surface;
1 = sphere), solidity_3d (volume / convex hull volume), thickness_um (median local thickness: twice the distance
from the outline's medial voxels to its edge), mask_volume_um3 (check against volume_um3 of ihc_stereocilia); local_depth_um and local_slope_deg (surface depth and slope at
the cell, field_quality.local_surface) for noise adjustment.
Output: RESULTS\ihc_bundle_shape_cells.csv.   Run (imaging env, scripts folder):  python ihc_bundle_shape.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import field_quality as FQ
import glob
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import skeletonize

ISO = 0.2166
ZS = float(getattr(S, "Z_SCALE", 1.0))   # outlines are stored with nominal plane spacing; true distance along the optical axis = x Z_SCALE
SP = (ISO * ZS, ISO, ISO)
MASKS = os.path.join(S.WORK, "ihc_masks_v2"); STACKS = os.path.join(S.WORK, "ihc_stacks_v2")


def shape(m):
    o = {}; n = int(m.sum())
    if n < 20: return o
    v = n * SP[0] * SP[1] * SP[2]; o["mask_volume_um3"] = v
    verts, faces, _, _ = marching_cubes(np.pad(m, 1).astype(np.float32), 0.5, spacing=SP); sa = mesh_surface_area(verts, faces)
    o["surface_um2"] = float(sa); o["compactness"] = float(np.pi ** (1 / 3) * (6 * v) ** (2 / 3) / sa) if sa > 0 else np.nan
    pts = np.argwhere(m) * np.array(SP)
    try: o["solidity_3d"] = float(v / ConvexHull(pts + np.random.default_rng(0).normal(0, 1e-6, pts.shape)).volume)
    except Exception: pass
    dt = ndi.distance_transform_edt(m, sampling=SP); sk = skeletonize(m)
    if sk.any(): o["thickness_um"] = float(np.median(dt[sk]) * 2)
    return o


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(MASKS, "*.npz"))):
        fld = os.path.basename(f)[:-4]; z = np.load(f); out = z["outline"]
        ct = pd.read_csv(os.path.join(STACKS, fld + "_cells.csv")); cells = ct.cell.values
        dep, slp = FQ.local_surface(fld, ct.x.values, ct.y.values)
        for i, c in enumerate(cells[:len(out)]):
            rows.append(dict(field=fld, cell=int(c), local_depth_um=float(dep[i]), local_slope_deg=float(slp[i]), **shape(out[i])))
        print(fld, len(cells), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(S.RESULTS, "ihc_bundle_shape_cells.csv"), index=False)


if __name__ == "__main__":
    main()
