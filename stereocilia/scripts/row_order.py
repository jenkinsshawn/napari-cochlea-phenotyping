"""Straightness and order of the outer hair cell rows in each field.

For each outer hair cell, the line through the other cells of its row within NEIGHBOR_UM along the row (at least one
on each side, at least 3 in all; principal axis of their centers) is its local row line. Per field and row:
  row_line_deviation_um       median distance of cells from their local row line (um)
  row_line_deviation_p90_um   90th percentile of that distance
  share_off_line              share of cells more than OFF_LINE_UM from their local row line
  share_out_of_order          share of cells closer to the local line of another row than to their own
  row_zigzag_um               median distance of each cell from the midpoint of its two neighbors along the row
Input: RESULTS\hair_cell_rows_cells.csv (x, y in pixels, u along the row in um, row). The same measures are computed
for hand-clicked rows (ANNOTATIONS\reference_tables\ohc_rows_corrected_*.csv) when present, for the agreement check.
Writes RESULTS\row_order_fields.csv and RESULTS\row_order_cells.csv (and row_order_reference_fields.csv).
Run (imaging env, scripts folder):  python row_order.py
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import numpy as np, pandas as pd

NEIGHBOR_UM = 18.0
OFF_LINE_UM = 2.0
ROWS = ["OHC1", "OHC2", "OHC3"]


def line_dist(p, pts):
    """Distance (um) of point p from the principal axis through pts (um)."""
    c = pts.mean(axis=0); _, _, vt = np.linalg.svd(pts - c); d = vt[0]
    q = p - c; return float(abs(q[0] * d[1] - q[1] * d[0]))


def local_line(r, u0, xy, exclude=None):
    k = np.abs(r.u.values - u0) <= NEIGHBOR_UM
    if exclude is not None: k &= r.index.values != exclude
    left = (r.u.values < u0) & k; right = (r.u.values > u0) & k
    if k.sum() < 3 or not left.any() or not right.any(): return None
    return xy[k]


def field_cells(d, px):
    out = []
    rows = {rw: d[d.row == rw].sort_values("u") for rw in ROWS}
    xyr = {rw: r[["x", "y"]].values * px for rw, r in rows.items()}
    for rw, r in rows.items():
        xy = xyr[rw]; uu = r.u.values
        for j, (i, c) in enumerate(r.iterrows()):
            p = xy[j]; pts = local_line(r, c.u, xy, exclude=i)
            if pts is None: continue
            own = line_dist(p, pts); other = []
            for rw2 in ROWS:
                if rw2 == rw or not len(rows[rw2]): continue
                p2 = local_line(rows[rw2], c.u, xyr[rw2])
                if p2 is not None: other.append(line_dist(p, p2))
            zz = np.nan
            if 0 < j < len(r) - 1 and uu[j + 1] - uu[j - 1] <= 2 * NEIGHBOR_UM: zz = float(np.linalg.norm(p - (xy[j - 1] + xy[j + 1]) / 2))
            out.append(dict(row=rw, u=c.u, line_dev_um=own, zigzag_um=zz, out_of_order=bool(other and min(other) < own)))
    return pd.DataFrame(out)


def summarize(cells):
    g = cells.groupby(["field", "row"])
    return pd.DataFrame({"cells": g.size(), "row_line_deviation_um": g.line_dev_um.median(), "row_line_deviation_p90_um": g.line_dev_um.quantile(0.9),
                         "share_off_line": g.line_dev_um.apply(lambda s: (s > OFF_LINE_UM).mean()), "share_out_of_order": g.out_of_order.mean(),
                         "row_zigzag_um": g.zigzag_um.median()}).reset_index()


def run(table):
    cs = []
    for f, d in table.groupby("field"):
        c = field_cells(d, S.PX_UM)
        if len(c): cs.append(c.assign(field=f))
    cells = pd.concat(cs, ignore_index=True); return cells, summarize(cells)


def main():
    hc = pd.read_csv(os.path.join(S.RESULTS, "hair_cell_rows_cells.csv")); hc = hc[hc.row.isin(ROWS)]
    cells, fields = run(hc)
    cells.to_csv(os.path.join(S.RESULTS, "row_order_cells.csv"), index=False); fields.to_csv(os.path.join(S.RESULTS, "row_order_fields.csv"), index=False)
    refs = sorted(glob.glob(os.path.join(S.ANNOTATIONS, "reference_tables", "ohc_rows_corrected_*.csv")))
    if refs:
        _, rf = run(pd.read_csv(refs[-1])); rf.to_csv(os.path.join(S.RESULTS, "row_order_reference_fields.csv"), index=False)
    print(f"row order: {fields.field.nunique()} fields, {len(cells)} cells")


if __name__ == "__main__":
    main()
