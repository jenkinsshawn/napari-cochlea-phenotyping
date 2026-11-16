"""Hair cell nuclei: each hair cell paired with one Cellpose nucleus, then size measured.

Inputs: work\\nuclei\\<field>_nuclei.npz (nuclei_cellpose.py), hair cells from results\\hair_cell_rows_cells.csv (outer
rows) and work\\ihc_stacks_v2\\<field>_cells.csv (inner hair cells).
Pairing: nuclei whose centre lies within 6 um (from above) of the hair cell top and at hair cell depth (inner 3-16 um,
outer 8-22 um below the surface); one nucleus per hair cell, assigned to minimise the total distance.
Measures per nucleus: area of the widest plane (um2), volume (um3), depth of the centre (um).
Checked against Shawn's outlines of 22 nuclei (annotations\\nucleus_outlines): widest-plane area median -6%,
91% within 20%, 77% within 15%, Spearman 0.93.
Output: results\\hair_cell_nuclei_cells.csv

Run (imaging env):  python nuclei_measure.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, re
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__)); RESULTS = S.RESULTS
NUC = os.path.join(S.WORK, 'nuclei'); STACKS2 = os.path.join(S.WORK, 'ihc_stacks_v2')
PXN = 0.2166; DZN = 0.6; RMAX = 6.0; BAND = {"IHC": (3, 16), "OHC1": (8, 22), "OHC2": (8, 22), "OHC3": (8, 22)}


def field_nuclei(field):
    z = np.load(os.path.join(NUC, field + "_nuclei.npz")); L = z["labels"]; dep = z["depth_um"]
    ids = np.unique(L[L > 0]); com = np.array(ndi.center_of_mass(L > 0, L, ids)); vol = ndi.sum(np.ones(L.shape, np.uint8), L, ids) * PXN ** 2 * DZN
    objs = ndi.find_objects(L); area = np.array([(L[objs[i - 1]] == i).sum((1, 2)).max() * PXN ** 2 for i in ids])
    d = np.interp(com[:, 0], np.arange(len(dep)), dep)
    return pd.DataFrame(dict(label=ids, z=com[:, 0], y=com[:, 1] * 2, x=com[:, 2] * 2, depth_um=d, area_um2=area, volume_um3=vol))


def pair(cells, N):
    rows = []
    for rw, c in cells.groupby("row"):
        lo, hi = BAND[rw]; n = N[N.depth_um.between(lo, hi)].reset_index(drop=True)
        if len(n) == 0 or len(c) == 0: continue
        D = np.hypot(c.x.values[:, None] - n.x.values[None], c.y.values[:, None] - n.y.values[None]) * 0.1083
        C = np.where(D <= RMAX, D, 1e3); i, j = linear_sum_assignment(C); ok = C[i, j] < 1e3
        r = c.iloc[i[ok]].reset_index(drop=True); m = n.iloc[j[ok]].reset_index(drop=True)
        rows.append(pd.concat([r[["field", "row", "x", "y"]], m[["label", "depth_um", "area_um2", "volume_um3"]], pd.Series(D[i[ok], j[ok]], name="offset_um")], axis=1))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    R = pd.read_csv(os.path.join(RESULTS, "hair_cell_rows_cells.csv")); out = []
    for f in sorted(x[:-len("_nuclei.npz")] for x in os.listdir(NUC) if x.endswith("_nuclei.npz")):
        p = os.path.join(STACKS2, f + "_cells.csv")
        cells = R[(R.field == f) & R.row.isin(["OHC1", "OHC2", "OHC3"])][["field", "row", "x", "y"]]
        if os.path.exists(p): cells = pd.concat([cells, pd.read_csv(p)[["x", "y"]].assign(field=f, row="IHC")], ignore_index=True)
        if len(cells) == 0: continue
        P = pair(cells, field_nuclei(f)); P["cells_in_field"] = P.row.map(cells.row.value_counts()); out.append(P)
        print(f"{f}: {len(P)} of {len(cells)} hair cells paired with a nucleus", flush=True)
    O = pd.concat(out, ignore_index=True); O.to_csv(os.path.join(RESULTS, "hair_cell_nuclei_cells.csv"), index=False); return O


if __name__ == "__main__":
    main()
