"""Compares two pipeline runs: field measures, group results, and cell-level agreement (cells matched by position).

Used after a change to the pipeline to see exactly what moved. Cells are paired by position within 2 um in the same
field, because cell numbers can change between runs.

Outputs in the new results folder:
  compare_fields.csv   per measure: fields in both, correlation, median and largest absolute change
  compare_groups.csv   per measure and group: difference and P in each run
  compare_cells.csv    outer hair cell rows (same row) and inner hair cell bundles (same category, height change)

Run (from the scripts folder):
  python compare_runs.py --old "<old results folder>" --new "<new results folder>"
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S

TOL_UM = 2.0


def pair(old, new, keys=("x", "y")):
    out = []
    for f, a in old.groupby("field"):
        b = new[new.field == f]
        if not len(b): continue
        d, j = cKDTree(b[list(keys)].values).query(a[list(keys)].values, distance_upper_bound=TOL_UM / S.PX_UM); ok = np.isfinite(d)
        out.append(a[ok].reset_index(drop=True).join(b.iloc[j[ok]].reset_index(drop=True), rsuffix="_new"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(old_dir, new_dir):
    rows = []
    fo = pd.read_csv(os.path.join(old_dir, "feature_table_fields.csv")); fn = pd.read_csv(os.path.join(new_dir, "feature_table_fields.csv"))
    m = fo.merge(fn, on="field", suffixes=("_old", "_new"))
    for c in [c for c in fo.columns if c in fn.columns and c != "field" and pd.api.types.is_numeric_dtype(fo[c]) and not pd.api.types.is_bool_dtype(fo[c])]:
        a, b = m[c + "_old"].astype(float), m[c + "_new"].astype(float); ok = a.notna() & b.notna(); d = (b - a)[ok].abs()
        rows.append(dict(measure=c, fields=int(ok.sum()), correlation=float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and a[ok].std() > 0 else np.nan,
                         median_abs_change=float(d.median()) if len(d) else np.nan, largest_abs_change=float(d.max()) if len(d) else np.nan))
    F = pd.DataFrame(rows); F.to_csv(os.path.join(new_dir, "compare_fields.csv"), index=False)
    go = pd.read_csv(os.path.join(old_dir, "group_comparisons.csv")); gn = pd.read_csv(os.path.join(new_dir, "group_comparisons.csv"))
    G = go[["measure", "vs_control", "mouse_level_difference", "p_permutation"]].merge(gn[["measure", "vs_control", "mouse_level_difference", "p_permutation"]],
                                                                                    on=["measure", "vs_control"], how="outer", suffixes=("_old", "_new"))
    G.to_csv(os.path.join(new_dir, "compare_groups.csv"), index=False)
    C = []
    ro = pd.read_csv(os.path.join(old_dir, "hair_cell_rows_cells.csv")); rn = pd.read_csv(os.path.join(new_dir, "hair_cell_rows_cells.csv"))
    p = pair(ro, rn); C.append(dict(item="outer hair cell candidates (rows)", old=len(ro), new=len(rn), paired=len(p), same=float((p.row == p.row_new).mean())))
    io = pd.read_csv(os.path.join(old_dir, "ihc_stereocilia_cells_v2.csv")); inn = pd.read_csv(os.path.join(new_dir, "ihc_stereocilia_cells_v2.csv"))
    p = pair(io, inn); st = p[(p.category == "standing") & (p.category_new == "standing")]; dh = (st.height_um_new - st.height_um).abs()
    C.append(dict(item="inner hair cell bundles", old=len(io), new=len(inn), paired=len(p), same=float((p.category == p.category_new).mean()),
                  standing_both=len(st), height_within_0_25um=float((dh <= 0.25).mean()), height_median_abs_change=float(dh.median())))
    Cc = pd.DataFrame(C); Cc.to_csv(os.path.join(new_dir, "compare_cells.csv"), index=False)
    pd.set_option("display.width", 220)
    print(Cc.round(3).to_string(index=False)); print(F.sort_values("median_abs_change", ascending=False).round(3).to_string(index=False))
    print(G.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--old", default=S.RESULTS); ap.add_argument("--new", required=True); a = ap.parse_args()
    main(a.old, a.new)
