"""Nucleus of each inner and outer hair cell, with its size, shape, DAPI and chromatin (nucleus_chromatin.py).

Inner hair cells: the Calb2-wrapped nuclei of the inner row (ihc_nuclei_calb2.py, ihc_nuclei_cells.csv).
Outer hair cells (rows 1-3): candidate nuclei lie OHC_DEPTH_UM below the surface, are not Calb2-wrapped (calb2_rel <
CALB2_MAX), are at least MIN_VOLUME_UM3 and do not touch the first or last plane. Outer hair cell nuclei sit offset from the cell top, so in each field
and row the median offset from each cell top to its nearest candidate within SEARCH_UM is estimated first; cells and
nuclei are then matched one to one (Hungarian assignment) on the distance after removing that offset, up to MATCH_UM.
Outputs (results folder):
  hair_cell_nuclei_cells.csv    one row per matched hair cell (cell type, nucleus label, offset, nucleus measures, surface
                                depth and slope above the nucleus)
  hair_cell_nuclei_fields.csv   per field and cell type: matched share and the median of each nucleus measure
Run (imaging env, scripts folder):  python hair_cell_nuclei.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import field_quality as FQ
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

OHC_DEPTH_UM = (6.0, 24.0)
CALB2_MAX = 0.3
SEARCH_UM, MATCH_UM = 8.0, 3.5
MIN_VOLUME_UM3 = 50.0   # smaller outlines are pieces of nuclei (hair cell nuclei are about 150-250 um3)
ROWS = ["OHC1", "OHC2", "OHC3"]
MEASURES = ["volume_um3", "equivalent_diameter_um", "elongation", "sphericity", "dapi_total_rel", "dapi_mean_rel", "dapi_cv",
            "chromocenter_count", "chromocenter_share", "chromatin_glcm_contrast", "chromatin_glcm_homogeneity", "depth_um"]


def match_row(cells, nuc, px):
    """cells, nuc: DataFrames with x, y (pixels). Returns rows of (cell index, nucleus index, dx_um, dy_um)."""
    if len(cells) < 3 or len(nuc) < 3: return []
    A = cells[["x", "y"]].values * px; Bn = nuc[["x", "y"]].values * px
    d, j = cKDTree(Bn).query(A, distance_upper_bound=SEARCH_UM); ok = np.isfinite(d)
    if ok.sum() < 3: return []
    off = np.median(Bn[j[ok]] - A[ok], axis=0)
    D = np.linalg.norm((A + off)[:, None, :] - Bn[None, :, :], axis=2); D[D > MATCH_UM] = 1e6
    ri, ci = linear_sum_assignment(D); keep = D[ri, ci] <= MATCH_UM
    return [(cells.index[a], nuc.index[b], off[0], off[1], D[a, b]) for a, b in zip(ri[keep], ci[keep])]


def main():
    px = S.PX_UM
    N = pd.read_csv(os.path.join(S.RESULTS, "nuclei_chromatin_all.csv"))
    A = pd.read_csv(os.path.join(S.RESULTS, "ihc_nuclei_all_nuclei.csv"))[["field", "label", "calb2_rel", "top_plane", "bottom_plane"]]
    N = N.merge(A, on=["field", "label"], how="left")
    whole = (N.top_plane > 0) & (N.bottom_plane < 60)
    I = pd.read_csv(os.path.join(S.RESULTS, "ihc_nuclei_cells.csv"))[["field", "label"]].drop_duplicates()
    ihc = N.merge(I, on=["field", "label"]).assign(cell_type="IHC", match_um=np.nan, offset_x_um=np.nan, offset_y_um=np.nan)
    hc = pd.read_csv(os.path.join(S.RESULTS, "hair_cell_rows_cells.csv")); hc = hc[hc.row.isin(ROWS)]
    cand = N[whole & (N.calb2_rel < CALB2_MAX) & N.depth_um.between(*OHC_DEPTH_UM) & (N.volume_um3 >= MIN_VOLUME_UM3)]
    out = [ihc]; share = []
    for (f, rw), c in hc.groupby(["field", "row"]):
        nf = cand[cand.field == f]
        m = match_row(c, nf, px)
        rows = []
        for ci, ni, ox, oy, dd in m:
            r = nf.loc[ni].to_dict(); r.update(cell_type=rw, cell_x=c.x[ci], cell_y=c.y[ci], offset_x_um=ox, offset_y_um=oy, match_um=dd); rows.append(r)
        out.append(pd.DataFrame(rows)); share.append(dict(field=f, cell_type=rw, cells=len(c), matched=len(rows)))
    C = pd.concat([o for o in out if len(o)], ignore_index=True)
    # a nucleus matched to cells of two rows keeps the closer match
    C = C.sort_values("match_um", na_position="first").drop_duplicates(["field", "label"]).reset_index(drop=True)
    C["local_depth_um"] = np.nan; C["local_slope_deg"] = np.nan   # surface depth and slope above the nucleus, for noise adjustment
    for f, g in C.groupby("field"):
        try: d, s_ = FQ.local_surface(f, g.x.values, g.y.values)
        except (FileNotFoundError, KeyError): continue
        C.loc[g.index, "local_depth_um"] = d; C.loc[g.index, "local_slope_deg"] = s_
    C.to_csv(os.path.join(S.RESULTS, "hair_cell_nuclei_cells.csv"), index=False)
    Fm = C.groupby(["field", "cell_type"])[MEASURES].median().reset_index()
    sh = pd.DataFrame(share)
    if len(sh):
        mt = C[C.cell_type.isin(ROWS)].groupby(["field", "cell_type"]).size().rename("matched_final").reset_index()
        sh = sh.merge(mt, on=["field", "cell_type"], how="left").fillna({"matched_final": 0}); sh["nucleus_matched_share"] = sh.matched_final / sh.cells
        Fm = Fm.merge(sh[["field", "cell_type", "nucleus_matched_share"]], on=["field", "cell_type"], how="left")
    Fm.to_csv(os.path.join(S.RESULTS, "hair_cell_nuclei_fields.csv"), index=False)
    print("hair cell nuclei:", C.groupby("cell_type").size().to_dict(), "| matched share per row:",
          sh.groupby("cell_type").nucleus_matched_share.median().round(2).to_dict() if len(sh) else {})


if __name__ == "__main__":
    main()
