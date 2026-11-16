"""Checks of the pipeline outputs against Shawn's hand references. Run after any change to a measuring script.

Each check recomputes agreement between the current results files and a frozen hand reference, and fails if it
falls more than 3 points below the documented value (so a change can only pass if it does not make a measure worse).

  1. Inner hair cells found       annotations\\cell_points (class 'inner'; clicks under 3 um apart along the row are
                                  one cell) against ihc_stacks_v2\\<field>_cells.csv, matched one to one within 4 um
                                  along the row. Documented: 93.4% found, 3.3% extra (log-scaled Calb2 outlines, 16 Nov 2026; before: 90.5%, 4.8%).
  2. Outer hair cell rows         annotations\\cell_points OHC1-3 against results\\hair_cell_rows_cells.csv, nearest
                                  cell within 2.5 um, same row. The models were trained on these clicks, so this is a
                                  check that the outputs are intact; the held-out figure is 88.9% (leave one mouse out).
  3. Inner hair cell height       annotations\\reference_tables\\ihc_bundle_base_tip_front_clicks.csv (76 cells) against
                                  results\\ihc_stereocilia_cells_v2.csv (base + 0.38 um). Documented: 82.2% within 1 um of 73 matched cells (22 of them fresh held-out), counting bundles whose outline lies on the
                                  bundle side of the cell (16 Nov 2026; 78.8% of 80 without that rule); 92 cells matched in the
                                  run before 16 Nov 2026 (79.3%); in this run 10 clicked cells have no cell center within 2 um.
  4. Row-1 outer hair cell height annotations\\reference_tables\\ohc1_bundle_base_tip_front_clicks.csv (readable test
                                  cells) against results\\ohc1_bundle_height_cells.csv. Documented: 77.1% within 0.75 um.
  5. Imaging settings             results\\imaging_settings_by_field.csv: lists stains whose laser settings differ
                                  between control and mutant fields (brightness of those stains is not comparable).

Run (imaging env, from the scripts folder):  python run_checks.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RESULTS = S.RESULTS
AN = S.ANNOTATIONS
CONTROL = S.CONTROL_GROUP
STACKS2 = os.path.join(S.WORK, 'ihc_stacks_v2')
PX, ZS = S.PX_UM, S.Z_SCALE
IHC_BASE_SHIFT_UM = 0.38
# agreement of the official run (standardized rerun, 13 Nov 2026); a check fails when a change lowers it by more than SLACK
DOCUMENTED = {"ihc_found": 93.4, "ihc_extra_max": 3.3, "ohc_rows": 89.5, "ihc_height": 82.2, "ohc1_height": 77.1}   # run ihc_log_2026-11-14
SLACK = 3.0
CARD = {}   # per clicked cell: field and whether the pipeline agreed, for the validation cards
VALIDATION_REFERENCE = os.path.join(S.MODELS, "validation_reference.csv")   # cards of the validation set, beside its models


class NotAssessed(Exception):
    """The dataset has no hand reference for this check (a new dataset); the check is reported, not failed."""


def _match1d(a, b, tol):
    a = np.sort(a); b = np.sort(b); ma = np.zeros(len(a), bool); mb = np.zeros(len(b), bool)
    for _, i, j in sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if abs(x - y) <= tol):
        if not ma[i] and not mb[j]: ma[i] = mb[j] = True
    return ma, mb, a, b


def _merge(u, gap=3.0):
    u = np.sort(u); out = []; cur = [u[0]] if len(u) else []
    for x in u[1:]:
        if x - cur[-1] <= gap: cur.append(x)
        else: out.append(np.mean(cur)); cur = [x]
    if cur: out.append(np.mean(cur))
    return np.array(out)


def _point_files():
    d = os.path.join(AN, "cell_points")
    if not os.path.isdir(d):
        raise NotAssessed(f"no hand-clicked cells for this dataset ({d})")
    return {f[:-11]: os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith("_points.csv")}


def check_ihc_found():
    import hair_cell_rows_classify as hrc, bundle_trace as bt
    clicked = found = extra = 0; rec = []
    for fld, p in _point_files().items():
        c = pd.read_csv(p); ih = c[c["class"] == "inner"]; q = os.path.join(STACKS2, fld + "_cells.csv")
        if not (len(ih) and os.path.exists(q)): continue
        co = hrc.frame_v2(fld, bt.field_maps_bowl(fld)); e, u, _ = co(ih.x.values.astype(float), ih.y.values.astype(float))
        um = _merge(u[np.isfinite(u)]); ic = pd.read_csv(q); ma, mb, a, b = _match1d(um, ic.u.values, 4.0)
        rec.append(pd.DataFrame(dict(field=fld, ok=ma))); clicked += len(um); found += int(ma.sum()); extra += int(((~mb) & (b >= a.min() - 4) & (b <= a.max() + 4)).sum())
    if not rec: raise NotAssessed("none of this dataset's fields has hand-clicked inner hair cells")
    CARD["inner hair cells found"] = (pd.concat(rec, ignore_index=True), "clicked inner hair cell found within 4 um along the row", DOCUMENTED["ihc_found"])
    f_pct, x_pct = 100 * found / clicked, 100 * extra / (found + extra)
    ok = f_pct >= DOCUMENTED["ihc_found"] - SLACK and x_pct <= DOCUMENTED["ihc_extra_max"] + SLACK
    return ok, f"inner hair cells found {f_pct:.1f}% of {clicked} clicked, {x_pct:.1f}% extra (documented {DOCUMENTED['ihc_found']}% / {DOCUMENTED['ihc_extra_max']}%)"


def check_ohc_rows(tol_um=2.5):
    hc = pd.read_csv(os.path.join(RESULTS, "hair_cell_rows_cells.csv")); n = right = 0; rec = []
    for fld, p in _point_files().items():
        c = pd.read_csv(p); co = c[c["class"].astype(str).str.startswith("OHC")]; h = hc[hc.field == fld]
        if not (len(co) and len(h)): continue
        d, i = cKDTree(h[["x", "y"]].values).query(co[["x", "y"]].values, distance_upper_bound=tol_um / PX); ok = np.isfinite(d)
        hit = np.zeros(len(co), bool); hit[ok] = h.row.values[i[ok]] == co["class"].values[ok]; rec.append(pd.DataFrame(dict(field=fld, ok=hit)))
        n += len(co); right += int((h.row.values[i[ok]] == co["class"].values[ok]).sum())
    if not rec: raise NotAssessed("none of this dataset's fields has hand-clicked outer hair cells")
    CARD["outer hair cell rows"] = (pd.concat(rec, ignore_index=True), "clicked outer hair cell found within 2.5 um and in the clicked row", DOCUMENTED["ohc_rows"])
    pct = 100 * right / n
    return pct >= DOCUMENTED["ohc_rows"] - SLACK, f"outer hair cells found in the clicked row {pct:.1f}% of {n} clicked (documented {DOCUMENTED['ohc_rows']}%; held out, leaving one mouse out: 88.9%)"


def check_ohc_density(max_median_abs=1.9 + 1.0, max_group_bias=1.5):
    """Outer hair cell density per field against the corrected hand reference (annotations\\reference_tables\\
    ohc_rows_corrected_2026-11-16.csv: Shawn's row clicks with his blind re-look of 215 disputed cells applied).
    Reference density per row = 100 x (cells - 1) / first-to-last length after trimming isolated end cells, the same
    rule the pipeline uses. Documented (16 Nov 2026): median absolute field error 1.9 per 100 um; mean error by group
    control -2.5, Smad3 -1.7, beta-catenin -2.9 (the pipeline undercounts; group means within 1.2 of each other)."""
    import hair_cell_rows_classify as hrc
    p = os.path.join(AN, "reference_tables", "ohc_rows_corrected_2026-11-16.csv")
    if not os.path.exists(p): raise NotAssessed("no corrected outer hair cell reference for this dataset")
    ref = pd.read_csv(p); hc = pd.read_csv(os.path.join(RESULTS, "hair_cell_rows_cells.csv"))
    fl = pd.read_csv(os.path.join(RESULTS, "hair_cell_rows_fields.csv")).set_index("field")
    rec = []
    for f, g in ref.groupby("field"):
        if f not in fl.index: continue
        u = g.u.values                                       # position along the row in the official run's row frame
        dens = 0.0
        for rw in ["OHC1", "OHC2", "OHC3"]:
            s = hrc.row_stretch(u[g.row.values == rw])
            if len(s) < 4: dens = np.nan; break
            dens += 100 * (len(s) - 1) / (s[-1] - s[0])
        rec.append(dict(field=f, group=S.group_of(f), reference=dens, pipeline=fl.loc[f, "outer_hair_cells_per_100um"]))
    R = pd.DataFrame(rec).dropna(); R["error"] = R.pipeline - R.reference
    med = R.error.abs().median(); bias = R.groupby("group").error.mean(); spread = bias.max() - bias.min()
    ok = med <= max_median_abs and spread <= max_group_bias
    return ok, (f"outer hair cell density: median field error {med:.2f} per 100 um over {len(R)} fields; mean error by group "
                + ", ".join(f"{k} {v:+.2f}" for k, v in bias.items()) + f" (documented 1.9; group spread {spread:.2f}, limit {max_group_bias})")


def _match_by_position(ref, res, tol_um=2.0):
    """Pairs each reference cell with the nearest result cell of the same field within tol_um (cell numbers can change
    between runs; positions do not)."""
    out = []
    for f, r in ref.groupby("field"):
        q = res[res.field == f]
        if not len(q): continue
        d, j = cKDTree(q[["x", "y"]].values).query(r[["x", "y"]].values, distance_upper_bound=tol_um / PX); ok = np.isfinite(d)
        out.append(r[ok].reset_index(drop=True).join(q.iloc[j[ok]].reset_index(drop=True), rsuffix="_res"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def check_ihc_height():
    p = os.path.join(AN, "reference_tables", "ihc_bundle_base_tip_front_clicks.csv")
    if not os.path.exists(p): raise NotAssessed(f"no inner hair cell height clicks for this dataset ({p})")
    ref = pd.read_csv(p)
    ref = ref[(ref.tip_um - ref.base_um) >= 0.5]
    A = pd.read_csv(os.path.join(RESULTS, "ihc_stereocilia_cells_v2.csv"))
    incomplete = A.reindex(columns=["tip_may_be_cut", "bundle_at_image_edge"]).eq(True).any(axis=1)
    A = A[(A.category == "standing") & ~incomplete]          # complete standing bundles, as in the feature table
    from build_feature_table import outline_in_band, IHC_MIN_IN_BAND   # and outline on the bundle side of the cell, as there
    A = pd.concat([g[g.cell.map(outline_in_band(f)).fillna(0) >= IHC_MIN_IN_BAND] for f, g in A.groupby("field")], ignore_index=True)
    d = _match_by_position(ref, A)
    if not len(d): raise NotAssessed("no clicked inner hair cell matches a complete standing bundle of this dataset")
    e = (d.tip_um_res - d.base_um_res - IHC_BASE_SHIFT_UM) - (d.tip_um - d.base_um); pct = 100 * (e.abs() <= 1).mean()
    CARD["inner hair cell height"] = (pd.DataFrame(dict(field=d.field, ok=(e.abs() <= 1).values)), "complete standing bundle height within 1 um of front-view clicks", DOCUMENTED["ihc_height"])
    fr = d["set"].astype(str) == "test4"; fresh = f"; fresh held-out cells {100 * (e[fr].abs() <= 1).mean():.0f}% of {int(fr.sum())}" if fr.any() else ""
    return pct >= DOCUMENTED["ihc_height"] - SLACK, f"inner hair cell height within 1 um of clicks {pct:.0f}% of {len(d)} complete standing cells matched of {len(ref)} (documented {DOCUMENTED['ihc_height']}%{fresh})"


def check_ohc1_height():
    p = os.path.join(AN, "reference_tables", "ohc1_bundle_base_tip_front_clicks.csv")
    if not os.path.exists(p): raise NotAssessed(f"no row-1 outer hair cell height clicks for this dataset ({p})")
    ref = pd.read_csv(p)
    ref = ref[(ref.split == "test") & (ref.readable == True)]
    A = pd.read_csv(os.path.join(RESULTS, "ohc1_bundle_height_cells.csv"))
    d = _match_by_position(ref, A)
    if not len(d): raise NotAssessed("no clicked row-1 outer hair cell matches a measured bundle of this dataset")
    e = d.height_um - (d.tip_um - d.base_um); pct = 100 * (e.abs() <= 0.75).mean()
    CARD["row-1 outer hair cell height"] = (pd.DataFrame(dict(field=d.field, ok=(e.abs() <= 0.75).values)), "readable test bundle height within 0.75 um of front-view clicks", DOCUMENTED["ohc1_height"])
    return pct >= DOCUMENTED["ohc1_height"] - SLACK, f"row-1 outer hair cell height within 0.75 um of clicks {pct:.0f}% of {len(d)} test cells matched of {len(ref)} (documented {DOCUMENTED['ohc1_height']}%)"


def check_settings():
    p = os.path.join(RESULTS, "imaging_settings_by_field.csv")
    if not os.path.exists(p): return False, "imaging_settings_by_field.csv missing: run imaging_settings.py"
    S = pd.read_csv(p); S["setting"] = S.laser_nm.astype(str) + "@" + S.laser_power_pct.astype(str); bad = []
    for stain, d in S.groupby("stain"):
        c = set(d[d.group == CONTROL].setting)
        if c and any(set(d[d.group == g].setting) != c for g in set(d.group.dropna()) - {CONTROL}): bad.append(stain)
    return True, f"brightness not comparable between {CONTROL or 'control'} and other fields for: " + (", ".join(bad) if bad else "none")


def check_coverage():
    """Every study field (mouse in the mouse table, phalloidin and Calb2 stained) appears in every main output."""
    import glob
    fdir = os.path.join(S.WORK, "fields"); expected = []
    for p in sorted(glob.glob(os.path.join(fdir, "*.npz"))):
        f = os.path.basename(p)[:-4]; st = [str(s) for s in np.load(p)["stains"]]
        if S.group_of(f) is not None and "Phalloidin" in st and "Calb2" in st: expected.append(f)
    outputs = {"rows": os.path.join(RESULTS, "hair_cell_rows_cells.csv"), "inner hair cell stacks": STACKS2,
               "inner hair cell bundles": os.path.join(RESULTS, "ihc_stereocilia_cells_v2.csv"),
               "outer hair cell bundle shape": os.path.join(RESULTS, "ohc_bundles_v2_cells.csv"),
               "row-1 outer hair cell height": os.path.join(RESULTS, "ohc1_bundle_height_cells.csv"),
               "inner hair cell nuclei": os.path.join(RESULTS, "ihc_nuclei_fields.csv"),
               "pillar heads": os.path.join(RESULTS, "pillar_heads_fields.csv"),
               "spatial statistics": os.path.join(RESULTS, "spatial", "fields_spatial.csv"),
               "feature table": os.path.join(RESULTS, "feature_table_fields.csv")}
    gaps = []
    for name, p in outputs.items():
        if os.path.isdir(p): have = {os.path.basename(x)[:-10] for x in glob.glob(os.path.join(p, "*_cells.csv"))}
        elif os.path.exists(p): have = set(pd.read_csv(p, usecols=["field"]).field)
        else: gaps.append(f"{name}: file missing"); continue
        miss = [f for f in expected if f not in have]
        if miss: gaps.append(f"{name}: {len(miss)} missing ({'; '.join(miss[:3])}{'...' if len(miss) > 3 else ''})")
    return not gaps, (f"all {len(expected)} study fields in all {len(outputs)} outputs" if not gaps else "; ".join(gaps))

def check_ranges():
    """Shares and fractions in the feature table lie between 0 and 1, and counts and densities are not negative."""
    F = pd.read_csv(os.path.join(RESULTS, "feature_table_fields.csv")); bad = []
    for c in F.columns:
        if not pd.api.types.is_numeric_dtype(F[c]): continue
        v = F[c].dropna()
        if any(k in c for k in ("share", "fraction")) and ((v < 0) | (v > 1)).any(): bad.append(f"{c}: {int(((v < 0) | (v > 1)).sum())} fields outside 0-1")
        elif any(k in c for k in ("per_100um", "_n_", "count")) and (v < 0).any(): bad.append(f"{c}: negative in {int((v < 0).sum())} fields")
    return not bad, ("all shares within 0-1 and no negative counts" if not bad else "; ".join(bad))


def write_cards():
    """Validation card per measure: agreement with the hand reference, cells and mice, and a 95% interval from
    resampling mice with the small-sample correction (napari_cochlea_phenotyping.cluster_bootstrap)."""
    from napari_cochlea_phenotyping import cluster_bootstrap
    notes = {"inner hair cells found": "the models were trained on these clicks (output check); held-out score in the plan",
             "outer hair cell rows": "the row models were trained on these clicks (output check); held out, leaving one mouse out: 88.9%",
             "inner hair cell height": "66 tuning cells (the +0.38 um base shift and the complete-bundle rule were chosen on them) and the fresh held-out cells (73.1% of 26 in the run before 16 Nov 2026, 75% of 24 now); error grows as bundles get shorter (method slope 0.74 against clicks), so group differences come out smaller than clicked, not larger",
             "row-1 outer hair cell height": "test cells held out from tuning"}
    rows = []
    for name, (d, rule, documented) in CARD.items():
        if not len(d): continue
        mice = d.field.map(S.mouse_of); b = cluster_bootstrap(d.ok.astype(float).values, mice.values, seed=0)
        rows.append(dict(measure=name, agreement_rule=rule, cells=len(d), fields=d.field.nunique(), mice=mice.nunique(),
                         agreement_pct=round(100 * b["estimate"], 1), interval_low_pct=round(100 * max(b["low"], 0), 1),
                         interval_high_pct=round(100 * min(b["high"], 1), 1), documented_pct=documented,
                         interval_method="95%, resampling mice, small-sample corrected", reference=notes.get(name, ""),
                         assessed_on="this dataset"))
    if os.path.exists(VALIDATION_REFERENCE):     # measures without a hand reference here: the validation set's card
        ref = pd.read_csv(VALIDATION_REFERENCE); done = {r["measure"] for r in rows}
        rows += [r for r in ref.to_dict("records") if r["measure"] not in done]
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RESULTS, "validation_cards.csv"), index=False); return out

def main():
    results = []
    for name, fn in [("inner hair cells found", check_ihc_found), ("outer hair cell rows", check_ohc_rows), ("outer hair cell density", check_ohc_density),
                     ("inner hair cell height", check_ihc_height), ("row-1 outer hair cell height", check_ohc1_height),
                     ("imaging settings", check_settings), ("study fields in every output", check_coverage),
                     ("value ranges", check_ranges)]:
        try: ok, msg = fn()
        except NotAssessed as ex: ok, msg = None, f"not assessed: {ex}"
        except Exception as ex: ok, msg = False, f"error: {ex}"
        results.append((name, ok, msg)); print(f"{'PASS' if ok else 'NOT ASSESSED' if ok is None else 'FAIL'}  {name}: {msg}", flush=True)
    try:
        c = write_cards(); print(f"validation cards: {len(c)} measures written to validation_cards.csv", flush=True)
    except Exception as ex:
        print(f"validation cards not written: {ex}", flush=True)
    return results


if __name__ == "__main__":
    r = main(); sys.exit(0 if all(ok is not False for _, ok, _ in r) else 1)
