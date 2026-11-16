"""Outer hair cell bundle shape (bowl fit) for the hair cells of the second row frame.

Hair cells and rows: hair_cell_rows_classify.classify (row classifiers, second frame). Each outer hair cell (a plate
peak) is linked to the cell outline under it (hand-corrected outlines where they exist, else work\\cells\\v1); outlines holding two or more hair cells, or none,
are left out (shared). The bowl fit is bundle_trace.fit_bowl on that outline, unchanged, so cells in the 96-cell
hand reference get the same fit as before (span 0.26 um and bend depth 0.24 um median error; within target in 4
of 6 reference fields). Row direction at each cell: spatial_stats.smooth_row_frame through the inner hair cells of
the second frame. Measures from bundle_trace.bowl_measures; curve_q20 >= 0.55 marks a readable bundle.
Apex fields are measured but not used for group comparisons (the fit misses the targets there).

Output: results\\ohc_bundles_v2_cells.csv (one row per outer hair cell).
Run from the scripts folder:
    python ohc_bundles_v2.py               (all fields)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bundle_trace as bt
import hair_cell_rows as hr
import hair_cell_rows_classify as hrc
import spatial_stats as ss

RESULTS = S.RESULTS
GATE = 0.55


def field_cells(fl, F):
    z, co = hrc.classify(fl, F)
    if z is None: return None
    oh = z[z.row.isin(["OHC1", "OHC2", "OHC3"])].copy()
    lab = F["lab"]; oh["outline"] = lab[oh.y.values.astype(int), oh.x.values.astype(int)]
    n = oh.groupby("outline").size(); oh["shared"] = oh.outline.map(n) > 1
    ic = hr.inner_cells(fl, co)
    if len(ic):
        e, u, ok = co(ic.x.values, ic.y.values)
        ih = pd.DataFrame(dict(cell=-(np.arange(len(ic)) + 1), row="IHC", x=ic.x.values, y=ic.y.values, along_um=ic.u.values, across_um=e))
    else:
        ih = pd.DataFrame(columns=["cell", "row", "x", "y", "along_um", "across_um"])
    t = pd.concat([ih, pd.DataFrame(dict(cell=oh.outline.values, row=oh.row.values, x=oh.x.values, y=oh.y.values, along_um=oh.u.values, across_um=oh.e.values))], ignore_index=True)
    t = ss.smooth_row_frame(t)
    return oh, t


def main(fields=None):
    fields = fields or [os.path.basename(f)[:-10] for f in sorted(glob.glob(os.path.join(S.WORK, 'ihc_stacks', '*_cells.csv')))]
    rows = []; t0 = time.time()
    for i, fl in enumerate(fields, 1):
        try:
            F = bt.field_maps_bowl(fl); r = field_cells(fl, F)
        except Exception as e:
            print(f"{i}/{len(fields)} {fl}: failed ({e})", flush=True); continue
        if r is None:
            print(f"{i}/{len(fields)} {fl}: no frame", flush=True); continue
        oh, t = r; n_fit = 0
        for _, c in oh.iterrows():
            base = dict(field=fl, x=c.x, y=c.y, row=c.row, outline=int(c.outline), shared=bool(c.shared))
            if c.outline == 0 or c.shared or (F["lab"] == c.outline).sum() < 100 or "along_dx" not in t:
                rows.append(dict(base, status="no single outline")); continue
            f = bt.fit_bowl(F["I_bowl"], F["lab"], int(c.outline))
            if f is None:
                rows.append(dict(base, status="no fit")); continue
            m = bt.bowl_measures(f, ss.row_frame(t, int(c.outline)))
            rows.append(dict(base, status="fitted", readable=bool(m.get("curve_q20", 0) >= GATE), **m)); n_fit += 1
        print(f"{i}/{len(fields)} {fl}: {len(oh)} outer hair cells, {n_fit} fitted ({time.time() - t0:.0f} s)", flush=True)
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RESULTS, "ohc_bundles_v2_cells.csv"), index=False)
    print("done", flush=True); return out


if __name__ == "__main__":
    main()
