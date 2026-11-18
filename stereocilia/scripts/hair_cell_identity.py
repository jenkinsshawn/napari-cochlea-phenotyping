"""Marker identity of every hair cell: inner-like cells in the outer rows and outer-like cells in the inner row.

Per hair cell (inner hair cells from ihc_stacks_v2 <field>_cells.csv; outer rows 1-3 from hair_cell_rows_cells.csv):
  calb2_rel     Calb2 in a disk of DISK_UM radius at the cell top (maximum projection, minus the field's 5th
                percentile), divided by the median over the field's inner hair cells
  ocm_rel, pmca2_rel   the same for Ocm or PMCA2 where stained, divided by the median over the field's outer hair cells
  nucleus_calb2_rel    Calb2 ring around the cell's nucleus (hair_cell_nuclei.py, ihc_nuclei_calb2.py), when matched
  identity      'inner-like' for an outer-row cell with calb2_rel >= INNER_LIKE (and ocm_rel < OCM_LOW where Ocm is
                stained); 'outer-like' for an inner-row cell with calb2_rel < OUTER_LIKE; otherwise its row type
Per field: inner-like outer-row cells per 100 um of row, their share, the share of outer-like inner-row cells, and the
inner over outer hair cell Calb2 ratio. Calb2 is compared only within a field, so staining and imaging differences
between fields do not enter.
Outputs: RESULTS\hair_cell_identity_cells.csv, RESULTS\hair_cell_identity_fields.csv
Run (imaging env, scripts folder):  python hair_cell_identity.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import numpy as np, pandas as pd
from scipy import ndimage as ndi

DISK_UM = 2.0
INNER_LIKE, OCM_LOW, OUTER_LIKE = 0.7, 0.5, 0.4
ROWS = ["OHC1", "OHC2", "OHC3"]


def disk_means(img, xy, r_px):
    h, w = img.shape; yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]; dk = yy ** 2 + xx ** 2 <= r_px ** 2; out = []
    for x, y in xy:
        x, y = int(round(x)), int(round(y))
        out.append(float(img[y - r_px:y + r_px + 1, x - r_px:x + r_px + 1][dk].mean()) if r_px <= x < w - r_px and r_px <= y < h - r_px else np.nan)
    return np.array(out)


def main():
    hc = pd.read_csv(os.path.join(S.RESULTS, "hair_cell_rows_cells.csv")); hc = hc[hc.row.isin(ROWS)]
    nuc = os.path.join(S.RESULTS, "hair_cell_nuclei_cells.csv"); NU = pd.read_csv(nuc) if os.path.exists(nuc) else None
    rows, fields = [], []
    r_px = int(round(DISK_UM / S.PX_UM))
    for f, g in hc.groupby("field"):
        fz = os.path.join(S.WORK, "fields", f + ".npz"); ic = os.path.join(S.WORK, "ihc_stacks_v2", f + "_cells.csv")
        if not (os.path.exists(fz) and os.path.exists(ic)): continue
        z = np.load(fz, allow_pickle=True); st = [str(s) for s in z["stains"]]
        if "Calb2" not in st: continue
        cells = pd.concat([pd.read_csv(ic)[["x", "y", "u"]].assign(row="IHC"), g[["x", "y", "u", "row"]]], ignore_index=True)
        for stain, col in [("Calb2", "calb2"), ("Ocm", "ocm"), ("PMCA2", "pmca2")]:
            if stain not in st: cells[col] = np.nan; continue
            im = ndi.gaussian_filter(z["maxproj"][st.index(stain)].astype(np.float32), 1.0); cells[col] = disk_means(im, cells[["x", "y"]].values, r_px) - np.percentile(im, 5)
        ihc_med = cells.loc[cells.row == "IHC", "calb2"].median(); ohc_med = cells[cells.row.isin(ROWS)]
        cells["calb2_rel"] = cells.calb2 / ihc_med
        for col in ["ocm", "pmca2"]: cells[col + "_rel"] = cells[col] / ohc_med[col].median() if ohc_med[col].notna().any() else np.nan
        outer = cells.row.isin(ROWS); ocm_ok = cells.ocm_rel.isna() | (cells.ocm_rel < OCM_LOW)
        cells["identity"] = np.where(outer & (cells.calb2_rel >= INNER_LIKE) & ocm_ok, "inner-like",
                                     np.where(~outer & (cells.calb2_rel < OUTER_LIKE), "outer-like", cells.row))
        cells["field"] = f; rows.append(cells)
        oc = cells[outer]; L = sum(np.ptp(oc[oc.row == rw].u) for rw in ROWS if (oc.row == rw).sum() >= 4) / 3
        ni = int((oc.identity == "inner-like").sum())
        fields.append(dict(field=f, cell_type="all", inner_like_outer_cells_per_100um=100 * ni / L if L > 20 else np.nan,
                           inner_like_outer_share=ni / len(oc) if len(oc) else np.nan,
                           outer_like_inner_share=float((cells[~outer].identity == "outer-like").mean()) if (~outer).any() else np.nan,
                           ihc_over_ohc_calb2=float(ihc_med / oc.calb2.median()) if oc.calb2.median() > 0 else np.nan))
    C = pd.concat(rows, ignore_index=True)
    if NU is not None and "cell_x" in NU:   # Calb2 ring of the matched nucleus (outer rows), for the agreement check
        k = NU.dropna(subset=["cell_x"])[["field", "cell_x", "cell_y", "calb2_rel"]].rename(columns={"cell_x": "x", "cell_y": "y", "calb2_rel": "nucleus_calb2_rel"})
        C = C.merge(k, on=["field", "x", "y"], how="left")
    C.to_csv(os.path.join(S.RESULTS, "hair_cell_identity_cells.csv"), index=False)
    F = pd.DataFrame(fields); F.to_csv(os.path.join(S.RESULTS, "hair_cell_identity_fields.csv"), index=False)
    print("identity:", C.identity.value_counts().to_dict())


if __name__ == "__main__":
    main()
