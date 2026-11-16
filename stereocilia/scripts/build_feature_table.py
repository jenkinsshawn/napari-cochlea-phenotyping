"""Field-level feature table and group comparisons (mouse as the unit), from the saved results.

Inputs (results\\ unless noted):
  ihc_stereocilia_cells_v2.csv            inner hair cell stereocilia (ihc_stereocilia.py --v2)
  work\\ihc_stacks_v2\\<field>_cells.csv    inner hair cell positions along the row
  hair_cell_rows_fields.csv               outer hair cell rows per field (second row frame)
  pillar_heads_fields.csv                 pillar heads per field (reported for middle and base only)
  ohc_bundles_v2_cells.csv                outer hair cell bundle fits (ohc_bundles_v2.py; middle and base only)
  ohc1_bundle_height_fields.csv, ihc_nuclei_fields.csv, ihc_calb2_fields.csv   added 11 Nov 2026 when present
Group comparison for every measure, damaged fields left out: exact permutation test on mouse values (each mouse's
turn-centred field values averaged; every assignment of the mice to the groups), plus a mixed model (group + turn,
random intercept for mouse) where it converges. P values adjusted for all comparisons (Benjamini-Hochberg).
Outputs: results\\feature_table_fields.csv, results\\group_comparisons.csv

Run from the scripts folder:
    python build_feature_table.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S

import itertools, os, re, warnings
import numpy as np
import pandas as pd
import importlib.util
if importlib.util.find_spec("statsmodels") is None:   # needed for the group comparisons; stop before writing anything
    sys.exit("build_feature_table.py needs statsmodels for the group comparisons. In the imaging env: pip install statsmodels. Nothing was written.")

HERE = os.path.dirname(os.path.abspath(__file__)); RESULTS = S.RESULTS
STACKS2 = os.path.join(S.WORK, 'ihc_stacks_v2')
DAMAGED = list(S.DAMAGED)   # fields left out of group comparisons (pipeline_settings.py)
IHC_BASE_SHIFT_UM = 0.38  # median gap between method base and 55 front-view base clicks, 11 Nov 2026
# Standing-bundle measures use only bundles whose automatic outline lies mostly on the bundle side of the cell: from 1 um
# behind to 8 um in front of the cell center across the row (the side-view band of the painting tools, set before this
# check). Against the front-view clicks (16 Nov 2026), 82% of in-band bundles were within 1 um and 43% of the others (median
# error +1.25 um); off-center outlines are about 21% of standing bundles (control 12-13%, Smad3 26-27%).
IHC_BAND_UM = (-1.0, 8.0)
IHC_MIN_IN_BAND = 0.5


def outline_in_band(field):
    """Share of each inner hair cell's automatic bundle outline inside IHC_BAND_UM (index = cell number)."""
    p = os.path.join(S.WORK, "ihc_masks_v2", field + ".npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} is missing: run ihc_stereocilia.py --v2")
    M = np.load(p)["outline"]; iso = 2 * S.PX_UM; c = M.shape[2] // 2
    y0, y1 = c + int(round(IHC_BAND_UM[0] / iso)), c + int(round(IHC_BAND_UM[1] / iso))
    tot = M.sum(axis=(1, 2, 3)); inb = M[:, :, y0:y1, :].sum(axis=(1, 2, 3))
    return pd.Series(np.where(tot > 0, inb / np.maximum(tot, 1), np.nan), index=np.arange(1, len(M) + 1))
STATUS = {
    "ihc_per_100um": "validated: 93.4% of clicked inner hair cells found, 3.3% extra; field values within 0.5 per 100 um of the clicked density in 15 of 26 fields (median error 0.37); differences below about 0.6 per 100 um are not resolved, because the clicked density itself differs by that much between first-to-last and median-spacing definitions",
    "ihc_spacing_um": "validated (as above)",
    "ihc_bundle_length_um": "not at target: standing length within 1 um of hand clicks in 64% of cells; folded 79% within 1 um of paint (beta-catenin +0.6 um)",
    "ihc_share_lying_flat": "descriptive: depends on mounting",
    "ihc_bundles_complete_share": "quality check: share of inner hair cell bundles not cut by the image edge or the cut-out top (only complete bundles enter the bundle measures)",
    "ihc_bundle_length_standing_um": "not at target: within 1 um of hand clicks in 64% of 36 cells",
    "ihc_bundle_height_standing_um": "usable, more variable: base raised 0.38 um; bundles with their outline on the bundle side of the cell; within 1 um of front-view clicks in 82% of 73 matched cells (82% of 22 fresh held-out cells; median absolute error 0.48 um against Shawn's single-click spread of 0.28 um); error grows as bundles get shorter (method slope 0.74 against clicks), no group-specific error once height is accounted for, so group differences come out smaller than clicked",
    "ihc_bundle_lean_standing_deg": "not at target: within 10 deg of hand clicks in 64% of 36 cells; depends on mounting",
    "ihc_bundle_brightness": "quality check",
    "outer_hair_cells_per_100um": "validated: rows 88.9% right leaving one mouse out (26 clicked fields; 19 fields at 85% or more)",
    "OHC1_per_100um": "validated (rows)", "OHC2_per_100um": "validated (rows)", "OHC3_per_100um": "validated (rows)",
    "OHC1_spacing_um": "validated (rows)", "OHC2_spacing_um": "validated (rows)", "OHC3_spacing_um": "validated (rows)",
    "row1_to_2_um": "validated (rows)", "row2_to_3_um": "validated (rows)",
    "calb2_edge_to_row1_um": "validated, more variable: field median within 1 um of Shawn's row-1 clicks in 77% of 26 fields (-0.55 um), r 0.97",
    "pillar_head_width_um": "not validated: middle and base 82% of lines found, 21% extra (left-out mouse); apex 59% found, 45% extra; field values changed between runs (correlation 0.56)",
    "pillar_heads_per_100um": "not validated: as pillar head width; within 5% of clicks in 55% of fields; between-run correlation 0.57",
    "pillar_width_sd_um": "not validated (needs single-head accuracy)",
    "ohc_bundle_span_um": "partly checked: middle and base only, within target in 4 of 6 reference fields",
    "ohc_bundle_bend_depth_um": "partly checked: middle and base only, as bundle span (6 reference fields)",
    "ohc_bundles_readable_share": "quality check (readability gate)",
    "ohc1_bundle_height_um": "validated, more variable: first-row height within 0.75 um of front-view clicks in 77% of 35 readable test cells (+0.21 um); Shawn's repeats 5 of 6; unreadable bundles shift field medians by about 0",
    "ihc_nucleus_area_um2": "validated: Cellpose widest-plane area within 20% of Shawn's outlines in 91% of 22 nuclei (median -6%); inner hair cell nuclei identified by a Calb2 shell, in the inner row, not cut by the stack",
    "ihc_nucleus_volume_um3": "not validated: volume not checked against outlines (nuclei found as for nucleus area)",
    "ihc_nuclei_measured_share": "quality check: inner hair cells with a measured nucleus",
    "ihc_calb2_over_background": "brightness ratio: compare between groups only where Calb2 was imaged with the same settings (see the imaging_settings column)",
}
# status level of each measure (shown in the plugin's Results tab and written as the status column)
STATUS_LEVEL = {**{k: "validated" for k in ["ihc_per_100um", "ihc_spacing_um", "outer_hair_cells_per_100um", "OHC1_per_100um", "OHC2_per_100um",
                                            "OHC3_per_100um", "OHC1_spacing_um", "OHC2_spacing_um", "OHC3_spacing_um", "row1_to_2_um",
                                            "row2_to_3_um", "ihc_nucleus_area_um2"]},
                **{k: "validated, more variable" for k in ["ihc_bundle_height_standing_um", "calb2_edge_to_row1_um", "ohc1_bundle_height_um"]},
                **{k: "partly checked" for k in ["ohc_bundle_span_um", "ohc_bundle_bend_depth_um"]},
                **{k: "quality check" for k in ["ihc_share_lying_flat", "ihc_bundles_complete_share", "ihc_bundle_brightness",
                                                "ohc_bundles_readable_share", "ihc_nuclei_measured_share"]},
                **{k: "not validated" for k in ["ihc_bundle_length_um", "ihc_bundle_length_standing_um", "ihc_bundle_lean_standing_deg",
                                                "pillar_head_width_um", "pillar_heads_per_100um", "pillar_width_sd_um", "ihc_nucleus_volume_um3",
                                                "ihc_calb2_over_background"]}}
assert set(STATUS_LEVEL) == set(STATUS), "every measure needs a status level"


def group_of(f):
    return S.group_of(f)


def mouse_of(f):
    return S.mouse_of(f)


def build():
    D2 = pd.read_csv(os.path.join(RESULTS, "ihc_stereocilia_cells_v2.csv")); D2 = D2[D2.field.map(group_of).notna()]   # only fields of mice in the mouse table (pipeline_settings.MICE)
    RF = pd.read_csv(os.path.join(RESULTS, "hair_cell_rows_fields.csv"))
    pp = os.path.join(RESULTS, "pillar_heads_fields.csv")   # pillar heads are not yet a pipeline step (made in a session, 10 Nov 2026)
    PIL = pd.read_csv(pp).set_index("field") if os.path.exists(pp) else pd.DataFrame()
    if PIL.empty: print("pillar_heads_fields.csv not found: pillar measures left empty", flush=True)
    ob = os.path.join(RESULTS, "ohc_bundles_v2_cells.csv"); OB = pd.read_csv(ob) if os.path.exists(ob) else None
    rows = []
    for f in sorted(D2.field.unique()):
        turn = re.search(r"(apex|middle|base)", f.lower()).group(1)
        c = pd.read_csv(os.path.join(STACKS2, f + "_cells.csv")).sort_values("u"); d = D2[D2.field == f]
        incomplete = d.reindex(columns=["tip_may_be_cut", "bundle_at_image_edge"]).eq(True).any(axis=1)   # missing flag = complete
        cm = d[~incomplete]; st = cm[cm.category == "standing"]   # complete bundles only (12 Nov 2026): a bundle cut by the image edge cannot be measured
        st = st[st.cell.map(outline_in_band(f)).fillna(0) >= IHC_MIN_IN_BAND]   # outline on the bundle side of the cell (16 Nov 2026)
        sp = np.diff(c.u.values); L = np.ptp(c.u.values)
        r = dict(field=f, mouse=mouse_of(f), group=group_of(f), turn=turn, damaged=f in DAMAGED,
                 ihc_cells=len(c), ihc_per_100um=100 * (len(c) - 1) / L if L > 0 else np.nan, ihc_spacing_um=np.median(sp[sp < 15]) if len(sp) else np.nan,
                 ihc_bundle_length_um=cm.length_um.median(), ihc_share_lying_flat=(d.category == "lying flat").mean(),
                 ihc_bundles_complete_share=len(cm) / len(d) if len(d) else np.nan,
                 ihc_bundle_length_standing_um=st.length_um.median(), ihc_bundle_height_standing_um=st.height_um.median() - IHC_BASE_SHIFT_UM,
                 ihc_bundle_lean_standing_deg=st.lean_deg.median(), ihc_bundle_brightness=d.contrast.median())
        q = RF[RF.field == f]
        if len(q):
            q = q.iloc[0]
            for k in ["outer_hair_cells_per_100um", "OHC1_per_100um", "OHC2_per_100um", "OHC3_per_100um", "OHC1_spacing_um", "OHC2_spacing_um",
                      "OHC3_spacing_um", "row1_to_2_um", "row2_to_3_um"]: r[k] = q[k]
            r["calb2_edge_to_row1_um"] = q["OHC1_e_um"]
        if turn != "apex" and f in PIL.index:
            for k in ["pillar_head_width_um", "pillar_heads_per_100um", "pillar_width_sd_um"]: r[k] = PIL.loc[f, k]
        if turn != "apex" and OB is not None:
            b = OB[(OB.field == f) & (OB.status == "fitted")]
            if len(b):
                rd = b[b.readable == True]
                r.update(ohc_bundle_span_um=rd.span_um.median(), ohc_bundle_bend_depth_um=rd.bend_depth_um.median(), ohc_bundles_readable_share=b.readable.mean())
        rows.append(r)
    FT = pd.DataFrame(rows)
    for fn, cols in [("ohc1_bundle_height_fields.csv", ["ohc1_bundle_height_um"]),
                     ("ihc_nuclei_fields.csv", ["ihc_nucleus_area_um2", "ihc_nucleus_volume_um3", "ihc_nuclei_measured_share"]),
                     ("ihc_calb2_fields.csv", ["ihc_calb2_over_background"])]:
        p = os.path.join(RESULTS, fn)
        if os.path.exists(p): FT = FT.merge(pd.read_csv(p)[["field"] + cols], on="field", how="left")
    FT.to_csv(os.path.join(RESULTS, "feature_table_fields.csv"), index=False); return FT


BRIGHTNESS = {"ihc_calb2_over_background": "Calb2"}   # measures that compare brightness, and the stain they use
MIN_MICE = 6                                            # mice with a value needed before a measure is compared


def imaging_note(stain, control, g):
    """Whether a stain was imaged with the same laser settings in both groups (imaging_settings_by_group.csv)."""
    p = os.path.join(RESULTS, "imaging_settings_by_group.csv")
    if not os.path.exists(p): return "imaging settings not read"
    t = pd.read_csv(p); t = t[t.stain == stain]
    a = set(t[t.group == control].laser_setting); b = set(t[t.group == g].laser_setting)
    if not a or not b: return "imaging settings not read"
    return "same" if a == b else f"differ ({control}: {'; '.join(sorted(a))}; {g}: {'; '.join(sorted(b))})"


def fit_mixed(formula, d):
    """Mixed model (mouse as random intercept, REML). Each optimizer is tried on its own and the converged fit with the
    highest likelihood is kept: these fits lie near the boundary (little spread between mice), and a single optimizer
    chain succeeds or stops on a singular matrix depending on the linear-algebra library. None when none converges."""
    import statsmodels.formula.api as smf
    best = None
    for m in ("lbfgs", "powell", "bfgs", "nm", "cg"):
        try:
            f = smf.mixedlm(formula, d, groups="mouse").fit(reml=True, method=m)
        except Exception:
            continue
        if f.converged and np.isfinite(f.llf) and (best is None or f.llf > best.llf + 1e-9):
            best = f
    return best


def model_note(fit):
    """Warn when the model puts the spread between mice at zero: it then treats fields as independent, and its
    interval and P value are too narrow (the mouse-level permutation P is the one to use)."""
    if fit is None: return "model did not converge"
    v = float(fit.cov_re.iloc[0, 0]); r = float(fit.scale)
    return "spread between mice estimated at zero: model interval and P too narrow; use the mouse-level P" if v <= 1e-6 * r else ""


def perm_test(mv, g, control="control"):
    mice = list(mv.mouse); val = dict(zip(mv.mouse, mv.v)); grp = dict(zip(mv.mouse, mv.group))
    nc = sum(grp[m] == control for m in mice); ng = sum(grp[m] == g for m in mice)
    obs = np.mean([val[m] for m in mice if grp[m] == g]) - np.mean([val[m] for m in mice if grp[m] == control]); st = []
    for ctrl in itertools.combinations(mice, nc):
        rest = [m for m in mice if m not in ctrl]
        for gg in itertools.combinations(rest, ng):
            st.append(np.mean([val[m] for m in gg]) - np.mean([val[m] for m in ctrl]))
    st = np.array(st); return obs, float((np.abs(st) >= abs(obs) - 1e-12).mean())


def compare(FT):
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    warnings.filterwarnings("ignore")
    F = FT[~FT.damaged & FT.group.notna()]; out = []
    control = S.CONTROL_GROUP; present = sorted(set(F.group))
    if not control or control not in present:
        G = pd.DataFrame(columns=["measure", "vs_control", "p_permutation", "p_permutation_adjusted"])
        G.to_csv(os.path.join(RESULTS, "group_comparisons.csv"), index=False)
        print(f"no group comparisons: {'no control group chosen' if not control else f'no fields of the control group {control!r}'}", flush=True)
        return G
    others = [g for g in present if g != control]; trt = f"C(group, Treatment({control!r}))"
    for y in STATUS:
        if y not in F: continue
        d = F.dropna(subset=[y])
        if d.mouse.nunique() < MIN_MICE: continue
        fit = fit_mixed(f"{y} ~ {trt} + C(turn)", d)
        d = d.assign(v=d[y] - d.groupby("turn")[y].transform("mean")); mv = d.groupby(["mouse", "group"]).v.mean().reset_index()
        for g in others:
            k = f"{trt}[T.{g}]"
            o, p = perm_test(mv, g, control) if (mv.group == g).sum() >= 2 and (mv.group == control).sum() >= 2 else (np.nan, np.nan)
            ok = fit is not None and k in fit.params and np.isfinite(fit.params[k])
            ci = fit.conf_int().loc[k] if ok else [np.nan, np.nan]
            out.append(dict(measure=y, vs_control=g, control_median=d[d.group == control][y].median(), group_median=d[d.group == g][y].median(),
                            mouse_level_difference=o, p_permutation=p, model_difference=fit.params[k] if ok else np.nan, model_ci_low=ci[0], model_ci_high=ci[1],
                            p_model=fit.pvalues[k] if ok else np.nan, fields=len(d), mice=d.mouse.nunique(), status=STATUS_LEVEL[y], validation=STATUS[y],
                            imaging_settings=imaging_note(BRIGHTNESS[y], control, g) if y in BRIGHTNESS else "",
                            model_note=model_note(fit)))
    G = pd.DataFrame(out); G["p_permutation_adjusted"] = multipletests(G.p_permutation.fillna(1), method="fdr_bh")[1]
    G = G.sort_values("p_permutation"); G.to_csv(os.path.join(RESULTS, "group_comparisons.csv"), index=False); return G


if __name__ == "__main__":
    FT = build(); G = compare(FT)
    print(f"{len(FT)} fields, {len(G)} comparisons" + (f"; smallest adjusted P {G.p_permutation_adjusted.min():.3f}" if len(G) else ""))
