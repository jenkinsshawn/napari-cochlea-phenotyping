"""Size, shape, DAPI content and chromatin texture of every nucleus (outlines from nuclei_cellpose.py).

For each field the DAPI planes are sampled again exactly as for the outlines (nuclei_cellpose.flattened_planes: 10 um
above to 26 um below the surface every 0.6 um, pixels binned 2 x 2 to 0.217 um, scaled to the field's 1st and 99.8th
percentiles). Distances along the optical axis are multiplied by Z_SCALE. Per nucleus:
  volume_um3, equivalent_diameter_um, elongation (longest / shortest principal axis), sphericity (surface of a sphere of
  the same volume / surface of the outline)
  dapi_total_rel, dapi_mean_rel   summed and mean DAPI, each divided by the median over the field's nuclei
  dapi_cv                         coefficient of variation of DAPI within the nucleus
  chromocenter_count              bright DAPI foci: connected voxels above the nucleus mean + 1.5 SD (after smoothing over
                                  1 voxel), at least 0.1 um3
  chromocenter_share              share of the nucleus volume in those foci
  chromatin_glcm_contrast, chromatin_glcm_homogeneity   GLCM on the nucleus's widest plane, 16 levels from its own 1st to
                                  99.5th percentile, 0.22 um apart, 4 directions averaged
Output: RESULTS\nuclei_chromatin_all.csv (x, y in full-resolution field pixels; depth_um below the surface, x Z_SCALE;
elongation left empty for outlines on fewer than 3 planes).
Run (imaging env, scripts folder):  python nucleus_chromatin.py      (reads the raw stacks again, about 40 s per field)
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.feature import graycomatrix, graycoprops
import nuclei_cellpose as NC

ZS = float(getattr(S, "Z_SCALE", 1.0))
LAB = os.path.join(S.WORK, "nuclei_v3_gpu")
MIN_FOCUS_UM3, FOCUS_SD = 0.1, 1.5


def nucleus(P, m, sp):
    o = {}; n = int(m.sum()); vv = sp[0] * sp[1] * sp[2]; v = P[m]
    o["volume_um3"] = n * vv; o["equivalent_diameter_um"] = (6 * o["volume_um3"] / np.pi) ** (1 / 3)
    pts = np.argwhere(m) * np.array(sp); ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    o["elongation"] = float(np.sqrt(ev[0] / ev[2])) if m.any(axis=(1, 2)).sum() >= 3 and ev[2] > 1e-6 else np.nan
    try:
        ve, fa, _, _ = marching_cubes(np.pad(m, 1).astype(np.float32), 0.5, spacing=sp); sa = mesh_surface_area(ve, fa)
        o["sphericity"] = float(np.pi ** (1 / 3) * (6 * o["volume_um3"]) ** (2 / 3) / sa)
    except Exception: pass
    o["dapi_sum"] = float(v.sum() * vv); o["dapi_mean"] = float(v.mean()); o["dapi_cv"] = float(v.std() / v.mean()) if v.mean() > 0 else np.nan
    sm = ndi.gaussian_filter(P, 1.0); thr = sm[m].mean() + FOCUS_SD * sm[m].std(); fo = (sm > thr) & m
    lab, k = ndi.label(fo); sizes = ndi.sum(fo, lab, range(1, k + 1)) * vv if k else np.array([])
    o["chromocenter_count"] = int((sizes >= MIN_FOCUS_UM3).sum()); o["chromocenter_share"] = float(sizes[sizes >= MIN_FOCUS_UM3].sum() / o["volume_um3"])
    zw = int(np.argmax(m.sum(axis=(1, 2)))); pl = P[zw]; mm = m[zw]
    if mm.sum() >= 30:
        lo, hi = np.percentile(pl[mm], [1, 99.5]); q = np.clip((pl - lo) / max(hi - lo, 1e-6) * 15, 0, 15).astype(np.uint8) + 1; q[~mm] = 0
        g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=17, symmetric=True)[1:, 1:]
        g = g / max(g.sum(axis=(0, 1), keepdims=True).max(), 1)
        o["chromatin_glcm_contrast"] = float(graycoprops(g, "contrast").mean()); o["chromatin_glcm_homogeneity"] = float(graycoprops(g, "homogeneity").mean())
    return o


def field_nuclei(fld):
    lf = os.path.join(LAB, fld + "_nuclei.npz")
    if not os.path.exists(lf): return []
    z = np.load(lf); L = z["labels"]; depth = z["depth_um"]; px = float(z["px_um"])
    P = NC.flattened_planes(fld)
    if P is None: return []
    nz = min(len(P), len(L)); P, L = P[:nz], L[:nz]
    sp = (float(np.median(np.diff(depth))) * ZS, px, px); rows = []
    for i, sl in enumerate(ndi.find_objects(L), 1):
        if sl is None: continue
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl); m = L[sl] == i
        if m.sum() < 30: continue
        o = nucleus(P[sl].astype(np.float32), m, sp); zz, yy, xx = np.nonzero(m)
        o.update(field=fld, label=i, x=float((xx.mean() + sl[2].start) * 2 + 0.5), y=float((yy.mean() + sl[1].start) * 2 + 0.5),
                 depth_um=float(depth[int(round(zz.mean())) + sl[0].start] * ZS))
        rows.append(o)
    d = pd.DataFrame(rows)
    if len(d):
        for c in ["dapi_sum", "dapi_mean"]: d[c.replace("_sum", "_total") + "_rel"] = d[c] / d[c].median()
    return [d]


def main(fields=None):
    """Each field is saved to WORK\\nuclei_chromatin\\<field>.csv as it finishes and skipped on a restart; images marked
    'not measured' in the settings are skipped."""
    import gc
    skip = {f for f, why in getattr(S, "DAMAGED", {}).items() if str(why).startswith("not measured")}
    fields = fields or sorted(os.path.basename(f)[:-11] for f in glob.glob(os.path.join(LAB, "*_nuclei.npz")))
    done_dir = os.path.join(S.WORK, "nuclei_chromatin"); os.makedirs(done_dir, exist_ok=True); out = []
    for f in fields:
        if f in skip: print(f, ": skipped (not measured)", flush=True); continue
        fp = os.path.join(done_dir, f + ".csv")
        if not os.path.exists(fp):
            try:
                d = field_nuclei(f)
                if d: d[0].to_csv(fp, index=False)
            except Exception as ex:
                print(f, ": failed:", repr(ex)[:200], flush=True); continue
            gc.collect()
        if os.path.exists(fp): out.append(pd.read_csv(fp)); print(f, flush=True)
    D = pd.concat(out, ignore_index=True).drop(columns=["dapi_sum", "dapi_mean"], errors="ignore")
    D.to_csv(os.path.join(S.RESULTS, "nuclei_chromatin_all.csv"), index=False); print(f"nuclei: {len(D)} in {D.field.nunique()} fields")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
