"""Hair cell surface, row packing, bundle polarity and outer hair cell nuclei (outer rows 1-3).

Input (official run folders from pipeline_settings):
  RESULTS\\hair_cell_rows_cells.csv     outer hair cells with row and position along the row (u)
  cell outlines from pipeline_settings.cell_labels_path (hand-corrected where they exist)
  RESULTS\\ohc_bundles_v2_cells.csv     bundle direction relative to the local row axis (direction_deg; 0 = expected)
  RESULTS\\ihc_nuclei_all_nuclei.csv    Cellpose nuclei with depth plane and Calb2 shell

Per cell (hair_cell_surface_polarity_cells.csv): surface area, perimeter, eccentricity, solidity, elongation (long /
short axis), long-axis angle to the row, nearest hair cell distance, bundle direction.
Per field and row (cell_phenotype_fields.csv): medians of those, surface area CV, gaps per 100 um (a gap = spacing
over 1.5 x the row median), spacing CV, polarity (median absolute direction, share over 30 deg off, signed mean,
circular SD, median difference to the next bundle along the row), and nuclei for rows 2 and 3 (nearest nucleus
within 5 um, 8-26 um below the surface, Calb2 shell under 0.3; row 1 not assessed: under 30% of cells pair).

Run (imaging env, scripts folder):  python cell_phenotypes.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

PX = S.PX_UM
ROWS = ["OHC1", "OHC2", "OHC3"]
GAP_FACTOR = 1.5
NUC_RADIUS_UM, NUC_DEPTH_UM, NUC_MAX_CALB2 = 5.0, (8.0, 26.0), 0.3
NUC_ROWS = ["OHC2", "OHC3"]


def angdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180) % 360 - 180


def row_tangent(g):
    xs, ys, u = g.x.values.astype(float), g.y.values.astype(float), g.u.values
    out = np.zeros(len(g))
    for i in range(len(g)):
        j = np.argsort(np.abs(u - u[i]))[:5]
        if len(j) >= 3:
            X = np.c_[xs[j] - xs[j].mean(), ys[j] - ys[j].mean()]; w, v = np.linalg.eigh(X.T @ X); out[i] = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    return out


def cells():
    hc = pd.read_csv(os.path.join(S.RESULTS, "hair_cell_rows_cells.csv")); hc = hc[hc.row.isin(ROWS)]
    rows, gaps = [], []
    for fld, h in hc.groupby("field"):
        p = S.cell_labels_path(fld)
        if not os.path.exists(p): print(f"{fld}: no outlines"); continue
        L = np.load(p); lab = L[h.y.values.astype(int), h.x.values.astype(int)]
        pr = pd.DataFrame(regionprops_table(L, properties=("label", "area", "perimeter", "eccentricity", "solidity",
                                                             "axis_major_length", "axis_minor_length", "orientation"))).set_index("label")
        hh = h.assign(label=lab)
        for rw, g in hh.groupby("row"):
            g = g.sort_values("u"); g = g.assign(row_tangent_deg=row_tangent(g))
            sp = np.diff(g.u.values); med = np.median(sp[sp > 1]) if (sp > 1).any() else np.nan; s15 = sp[sp < 15]
            gaps.append(dict(field=fld, row=rw, gaps_per_100um=100 * np.sum(sp > GAP_FACTOR * med) / max(np.ptp(g.u.values), 1),
                             spacing_cv=s15.std() / s15.mean() if len(s15) > 2 else np.nan))
            for _, c in g.iterrows():
                r = dict(field=fld, row=rw, x=c.x, y=c.y, u=c.u, e=c.e)
                if c.label > 0 and c.label in pr.index:
                    q = pr.loc[c.label]
                    r.update(surface_area_um2=q.area * PX * PX, surface_perimeter_um=q.perimeter * PX, surface_eccentricity=q.eccentricity,
                             surface_solidity=q.solidity, surface_elongation=q.axis_major_length / max(q.axis_minor_length, 1e-6),
                             surface_axis_vs_row_deg=abs(((np.degrees(-q.orientation) + 90) - c.row_tangent_deg + 90) % 180 - 90))
                rows.append(r)
    C = pd.DataFrame(rows); G = pd.DataFrame(gaps)
    nn = pd.Series(np.nan, index=C.index)
    for f, idx in C.groupby("field").groups.items():
        g = C.loc[idx]; d, _ = cKDTree(g[["x", "y"]].values).query(g[["x", "y"]].values, k=2); nn.loc[idx] = d[:, 1] * PX
    C["nearest_cell_um"] = nn
    ob = pd.read_csv(os.path.join(S.RESULTS, "ohc_bundles_v2_cells.csv")); ob = ob[ob.measured.astype(bool)]
    C["direction_deg"] = np.nan
    for f, idx in C.groupby("field").groups.items():
        b = ob[ob.field == f]
        if not len(b): continue
        d, i = cKDTree(b[["x", "y"]].values).query(C.loc[idx, ["x", "y"]].values); ok = d <= 2.0 / PX
        C.loc[np.array(idx)[ok], "direction_deg"] = b.direction_deg.values[i[ok]]
    an = pd.read_csv(os.path.join(S.RESULTS, "ihc_nuclei_all_nuclei.csv")); an["depth_um"] = -10 + 0.6 * an.z
    for col in ("nucleus_area_um2", "nucleus_volume_um3", "nucleus_depth_um"): C[col] = np.nan
    for f, idx in C.groupby("field").groups.items():
        n = an[(an.field == f) & an.depth_um.between(*NUC_DEPTH_UM) & (an.calb2_rel < NUC_MAX_CALB2)]
        if not len(n): continue
        d, i = cKDTree(n[["x", "y"]].values).query(C.loc[idx, ["x", "y"]].values); ok = d <= NUC_RADIUS_UM / PX; ii = np.array(idx)[ok]
        C.loc[ii, "nucleus_area_um2"] = n.area_um2.values[i[ok]]; C.loc[ii, "nucleus_volume_um3"] = n.volume_um3.values[i[ok]]
        C.loc[ii, "nucleus_depth_um"] = n.depth_um.values[i[ok]]
    return C, G


def fields(C, G):
    surf = ["surface_area_um2", "surface_perimeter_um", "surface_eccentricity", "surface_solidity", "surface_elongation", "surface_axis_vs_row_deg", "nearest_cell_um"]
    F = C.groupby(["field", "row"])[surf].median().reset_index()
    F["surface_area_cv"] = C.groupby(["field", "row"]).surface_area_um2.agg(lambda s: s.std() / s.mean()).values
    pol = []
    for (f, rw), g in C.dropna(subset=["direction_deg"]).groupby(["field", "row"]):
        if len(g) < 5: continue
        a = g.sort_values("u").direction_deg.values; z = np.exp(1j * np.deg2rad(a)); R = np.abs(z.mean()); nd = np.abs(angdiff(a[1:], a[:-1]))
        pol.append(dict(field=f, row=rw, polarity_abs_dev_deg=np.median(np.abs(a)), polarity_share_off30=np.mean(np.abs(a) > 30),
                        polarity_signed_mean_deg=np.degrees(np.angle(z.mean())), polarity_circ_sd_deg=np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))),
                        polarity_neighbor_diff_deg=np.median(nd) if len(nd) else np.nan))
    nuc = C[C.row.isin(NUC_ROWS)].groupby(["field", "row"])[["nucleus_area_um2", "nucleus_volume_um3", "nucleus_depth_um"]].median().reset_index()
    nuc["nucleus_paired_share"] = C[C.row.isin(NUC_ROWS)].groupby(["field", "row"]).nucleus_area_um2.apply(lambda s: s.notna().mean()).values
    return F.merge(G, on=["field", "row"], how="left").merge(pd.DataFrame(pol), on=["field", "row"], how="left").merge(nuc, on=["field", "row"], how="left")


def main():
    C, G = cells(); F = fields(C, G)
    C.to_csv(os.path.join(S.RESULTS, "hair_cell_surface_polarity_cells.csv"), index=False)
    F.to_csv(os.path.join(S.RESULTS, "cell_phenotype_fields.csv"), index=False)
    print(f"cells {len(C)}, field rows {len(F)}"); return C, F


if __name__ == "__main__":
    main()
