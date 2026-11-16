"""All phenotypes of the confocal set, one statistics pass: field-level table, group tests, whole-panel tests,
bundle-level models.

Reads (RESULTS unless noted): bundle_phenotypes_cells.csv, hair_cell_rows_cells.csv, cell_phenotype_fields.csv,
ihc_stereocilia_cells_v2.csv, feature_table_fields.csv; ANNOTATIONS\\bundle_scores (inner hair cell organization
scores, both sets, repeats and 'cannot tell' left out); ANNOTATIONS\\reference_tables\\ohc_rows_corrected_2026-11-16.csv.

Families: bundle panel (field median, 10th and 90th percentile, IQR, share outside the control 5-95% range with
each control mouse judged against the other, gradient along the field per 100 um), hair cell surface, row packing,
bundle polarity, outer hair cell nuclei (rows 2-3), inner hair cell front view, inner hair cell organization (hand
scores), density (pipeline), density (corrected hand reference).

Robustness (bundle panel): each measure under outline eroded/grown 1 px, planes 0.6-5.4 um, added noise. Status
robust = Spearman with base >= 0.8 under every condition and field medians moving <= 0.5 between-mouse SD;
moderate = Spearman >= 0.6; else unstable.

Tests per phenotype, all turns (field values minus the turn mean, then mouse means) and each turn: three-group
F with P exact over all relabelings of mice keeping group sizes; each mutant against control, difference of mouse
means, Welch 95% interval and exact P with labels reassigned across all mice (560 arrangements for 2+3+3 mice;
the P from relabeling only the two groups compared is kept as *_P_two_groups_only); genotype x turn (field-level F for the interaction, mice relabeled).
Benjamini-Hochberg within each family and cell type (all-turn P). Constant phenotypes are not tested.
Whole panel, per cell type: robust field medians, turn-centred and standardized, mouse profiles; pseudo-F on
Euclidean distances with exact relabeling P; principal components; groups of measures with |Spearman| >= 0.8.
Bundle-level models (robust measures): standardized value ~ genotype + turn, random mouse intercept and field
variance component (REML); variance shares mouse / field / bundle.

Writes: phenotype_field_values_long.csv, bundle_phenotype_robustness.csv, phenotype_statistics_all.csv,
phenotype_panel_multivariate.csv, phenotype_panel_principal_components.csv, phenotype_bundle_level_models.csv
Run (imaging env, scripts folder):  python phenotype_statistics.py
"""
import os, sys, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import hair_cell_rows_classify as hrc
from itertools import combinations
import numpy as np, pandas as pd
from scipy import stats as st
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

R = S.RESULTS; GR = ["control", "Smad3", "beta-catenin"]; TYPES = ["IHC", "OHC1", "OHC2", "OHC3"]
META = ["field", "cell", "row", "shape_check", "condition", "x", "y", "noise_sd"]


def field_info():
    ft = pd.read_csv(os.path.join(R, "feature_table_fields.csv")).set_index("field")
    return ft


def robustness(PH, FT, MEAS):
    base = PH[PH.condition == "base"].set_index(["field", "cell"]); out = []
    for rw in TYPES:
        b = base[base.row == rw]; dam = b.index.get_level_values(0).map(FT.damaged).astype(bool)
        fm = b[~dam].groupby(level=0)[MEAS].median(); turn = fm.index.map(FT.turn); mouse = fm.index.map(FT.mouse)
        for m in MEAS:
            res = fm[m] - fm.groupby(turn)[m].transform("mean"); mm = res.groupby(mouse).mean(); sd = mm.std()
            g = res.groupby(mouse); within = np.nanmean(g.var()); between = mm.var()
            r = dict(cell_type=rw, measure=m, between_mouse_sd=sd, field_icc=between / (between + within) if np.isfinite(within) and between + within > 0 else np.nan)
            sps, shifts = [], []
            for cond in ["eroded", "grown", "planes", "noise"]:
                c = PH[(PH.condition == cond) & (PH.row == rw)].set_index(["field", "cell"])[m]
                j = pd.concat([b[m], c], axis=1, keys=["b", "c"]).dropna()
                rho = st.spearmanr(j.b, j.c)[0] if len(j) > 10 else np.nan
                sh = (j.c.groupby(level=0).median() - j.b.groupby(level=0).median()).median()
                r[f"spearman_{cond}"] = rho; r[f"shift_{cond}"] = sh / sd if sd > 0 else np.nan; sps.append(rho); shifts.append(abs(sh / sd) if sd > 0 else np.nan)
            r["min_spearman"] = np.nanmin(sps); r["max_shift_in_mouse_sd"] = np.nanmax(shifts)
            r["status"] = "robust" if (r["min_spearman"] >= 0.8 and r["max_shift_in_mouse_sd"] <= 0.5) else ("moderate" if r["min_spearman"] >= 0.6 else "unstable")
            out.append(r)
    return pd.DataFrame(out)


def long_table(FT):
    PH = pd.read_csv(os.path.join(R, "bundle_phenotypes_cells.csv")); MEAS = [c for c in PH.columns if c not in META]
    hc = pd.read_csv(os.path.join(R, "hair_cell_rows_cells.csv"))
    B = PH[PH.condition == "base"].copy(); B["u"] = np.nan
    for f, idx in B.groupby("field").groups.items():
        h = hc[hc.field == f]; d, i = cKDTree(h[["x", "y"]].values).query(B.loc[idx, ["x", "y"]].values); B.loc[idx, "u"] = h.u.values[i]
    for c in ["group", "turn", "mouse"]: B[c] = B.field.map(FT[c])
    B = B[~B.field.map(FT.damaged).astype(bool)]
    L = []
    def add(df, ctcol, fam, ms, prefix=""):
        for m in ms:
            d = df[["field", ctcol, m]].rename(columns={ctcol: "cell_type", m: "value"}).dropna(); d["measure"] = prefix + m; d["family"] = fam; L.append(d)
    g = B.groupby(["field", "row"])
    add(g[MEAS].median().reset_index(), "row", "bundle panel: median", MEAS)
    add(g[MEAS].quantile(0.1).reset_index(), "row", "bundle panel: 10th percentile", MEAS, "p10 ")
    add(g[MEAS].quantile(0.9).reset_index(), "row", "bundle panel: 90th percentile", MEAS, "p90 ")
    ctl = B[B.group == "control"]; cm = sorted(ctl.mouse.unique())
    qs = {None: ctl.groupby(["row", "turn"])[MEAS].quantile([0.05, 0.95]).unstack()}
    for m_ in cm: qs[m_] = ctl[ctl.mouse != m_].groupby(["row", "turn"])[MEAS].quantile([0.05, 0.95]).unstack()
    ext = []
    for (f, rw), gg in g:
        t = FT.turn[f]; mo = FT.mouse[f]; qq = qs[mo] if mo in cm else qs[None]; r = dict(field=f, row=rw)
        for m in MEAS:
            v = gg[m].dropna()
            if len(v) < 5 or (rw, t) not in qq.index: continue
            lo, hi = qq.loc[(rw, t), (m, 0.05)], qq.loc[(rw, t), (m, 0.95)]
            r[m + "__iqr"] = np.subtract(*np.percentile(v, [75, 25])); r[m + "__out"] = float(((v < lo) | (v > hi)).mean())
            k = gg.dropna(subset=[m, "u"]); r[m + "__grad"] = 100 * np.polyfit(k.u, k[m], 1)[0] if len(k) >= 6 and np.ptp(k.u) > 20 else np.nan
        ext.append(r)
    E = pd.DataFrame(ext)
    for suf, fam in [("__iqr", "bundle panel: within-field spread (IQR)"), ("__out", "bundle panel: share outside control 5-95% range"), ("__grad", "bundle panel: gradient along field per 100 um")]:
        cols = [c for c in E.columns if c.endswith(suf)]; add(E[["field", "row"] + cols].rename(columns={c: c[:-len(suf)] for c in cols}), "row", fam, [c[:-len(suf)] for c in cols])
    CF = pd.read_csv(os.path.join(R, "cell_phenotype_fields.csv"))
    add(CF, "row", "hair cell surface", ["surface_area_um2", "surface_perimeter_um", "surface_eccentricity", "surface_solidity", "surface_elongation", "surface_axis_vs_row_deg", "surface_area_cv", "nearest_cell_um"])
    add(CF, "row", "row packing", ["gaps_per_100um", "spacing_cv"])
    add(CF, "row", "bundle polarity", ["polarity_abs_dev_deg", "polarity_share_off30", "polarity_signed_mean_deg", "polarity_circ_sd_deg", "polarity_neighbor_diff_deg"])
    add(CF[CF.row.isin(["OHC2", "OHC3"])], "row", "outer hair cell nuclei (not validated)", ["nucleus_area_um2", "nucleus_volume_um3", "nucleus_depth_um"])
    ist = pd.read_csv(os.path.join(R, "ihc_stereocilia_cells_v2.csv")); ist["cell_type"] = "IHC"
    fv = ist[ist.category != "no bundle"].groupby(["field", "cell_type"])[["width_along_um", "depth_across_um", "volume_um3", "length_um"]].median().reset_index()
    fv["share_lying_flat"] = ist.groupby("field").category.apply(lambda s: (s == "lying flat").mean()).reindex(fv.field).values
    add(fv, "cell_type", "inner hair cell front view", ["width_along_um", "depth_across_um", "volume_um3", "length_um", "share_lying_flat"], "front view ")
    ft = FT.reset_index().assign(cell_type="IHC"); add(ft, "cell_type", "inner hair cell front view", ["ihc_bundle_height_standing_um", "ihc_bundle_lean_standing_deg"])
    sc = []
    for name, key in [("organization", "key_organization.csv"), ("organization_ihc", "key_organization_ihc.csv")]:
        a = os.path.join(S.ANNOTATIONS, "bundle_scores", f"answers_{name}.csv"); k = os.path.join(S.ANNOTATIONS, "bundle_scores", key)
        if not (os.path.exists(a) and os.path.exists(k)): continue
        y = pd.read_csv(k).merge(pd.read_csv(a)[["item", "answer"]], on="item")
        rep = [c for c in y.columns if c.startswith("repeat_of")][0]
        y = y[(y.row == "IHC") & y[rep].isna()]; y["score"] = y.answer.map({"orderly": 1, "mildly disorganized": 2, "disorganized": 3}); sc.append(y.dropna(subset=["score"]))
    if sc:
        y = pd.concat(sc); og = y.groupby("field").agg(org_score=("score", "mean"), disorganized_share=("score", lambda s: (s == 3).mean())).reset_index(); og["cell_type"] = "IHC"
        add(og, "cell_type", "inner hair cell organization (hand scores)", ["org_score", "disorganized_share"])
    for rw in ["OHC1", "OHC2", "OHC3"]:
        add(ft.rename(columns={f"{rw}_per_100um": "per_100um"}).assign(cell_type=rw), "cell_type", "density (pipeline)", ["per_100um"])
    add(ft.assign(cell_type="all"), "cell_type", "density (pipeline)", ["outer_hair_cells_per_100um", "ihc_per_100um"])
    rp = os.path.join(S.ANNOTATIONS, "reference_tables", "ohc_rows_corrected_2026-11-16.csv")
    if os.path.exists(rp):
        ref = pd.read_csv(rp); rows = []
        for f, gg in ref.groupby("field"):
            r = dict(field=f); tot = 0.0
            for rw in ["OHC1", "OHC2", "OHC3"]:
                s = hrc.row_stretch(gg.u[gg.row == rw].values); r[rw] = 100 * (len(s) - 1) / (s[-1] - s[0]) if len(s) >= 4 else np.nan
            r["outer_hair_cells_per_100um"] = r["OHC1"] + r["OHC2"] + r["OHC3"]; rows.append(r)
        rf = pd.DataFrame(rows)
        for rw in ["OHC1", "OHC2", "OHC3"]:
            add(rf.rename(columns={rw: "per_100um"}).assign(cell_type=rw), "cell_type", "density (corrected hand reference)", ["per_100um"])
        add(rf.assign(cell_type="all"), "cell_type", "density (corrected hand reference)", ["outer_hair_cells_per_100um"])
    LONG = pd.concat(L, ignore_index=True)
    for c in ["group", "turn", "mouse", "damaged"]: LONG[c] = LONG.field.map(FT[c])
    LONG = LONG[~LONG.damaged.astype(bool) & LONG.group.isin(GR)]
    return LONG, PH, B, MEAS


def assignments(mg):
    codes = mg.map({g: i for i, g in enumerate(GR)}).values; sizes = [(codes == k).sum() for k in range(3)]; n = len(codes); idx = list(range(n)); out = []
    for a in combinations(idx, sizes[0]):
        rest = [i for i in idx if i not in a]
        for b in combinations(rest, sizes[1]):
            lab = np.full(n, 2); lab[list(a)] = 0; lab[list(b)] = 1; out.append(lab)
    return np.array(out), codes


def fstat(x, lab):
    ks = [k for k in range(3) if (lab == k).any()]; gm = x.mean()
    ssb = sum((lab == k).sum() * (x[lab == k].mean() - gm) ** 2 for k in ks); ssw = sum(((x[lab == k] - x[lab == k].mean()) ** 2).sum() for k in ks)
    return (ssb / (len(ks) - 1)) / (ssw / (len(x) - len(ks))) if ssw > 0 and len(x) > len(ks) else np.nan


def welch(a, b):
    d = b.mean() - a.mean()
    if len(a) < 2 or len(b) < 2: return d, np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = np.sqrt(va + vb)
    if va + vb == 0: return d, d, d
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)); t = st.t.ppf(0.975, df); return d, d - t * se, d + t * se


def analyze(d):
    out = []
    for turn in [None, "apex", "middle", "base"]:
        dd = d if turn is None else d[d.turn == turn]
        if not len(dd): continue
        dd = dd.assign(res=dd.value - dd.groupby("turn").value.transform("mean")) if turn is None else dd.assign(res=dd.value)
        mm = dd.groupby(["mouse", "group"]).res.mean().reset_index(); r = dict(turn=turn or "all turns", mice=len(mm))
        for g in GR: r[f"{g}_mean"] = dd[dd.group == g].groupby("mouse").value.mean().mean()
        A = None
        if mm.group.nunique() == 3 and len(mm) > 3:
            A, codes = assignments(mm.set_index("mouse").group); x = mm.res.values; f0 = fstat(x, codes)
            if np.isfinite(f0): r["F"] = f0; r["P_three_group"] = float(np.mean(np.array([fstat(x, a) for a in A]) >= f0 - 1e-12))
        for gi, (g, tag) in enumerate([("Smad3", "Smad3"), ("beta-catenin", "beta_catenin")], start=1):
            sub = mm[mm.group.isin(["control", g])]
            if sub.group.nunique() < 2: continue
            x = sub.res.values; k = (sub.group == g).values; obs = x[k].mean() - x[~k].mean()
            null2 = np.array([x[list(c)].mean() - np.delete(x, list(c)).mean() for c in combinations(range(len(x)), k.sum())])
            _, lo, hi = welch(x[~k], x[k])
            r.update({f"{tag}_difference": obs, f"{tag}_ci_low": lo, f"{tag}_ci_high": hi, f"{tag}_P_two_groups_only": float(np.mean(np.abs(null2) >= abs(obs) - 1e-12))})
            if A is not None:                 # P with group labels reassigned across all mice (as in build_feature_table)
                xa = mm.res.values; nul = np.array([xa[a == gi].mean() - xa[a == 0].mean() for a in A])
                r[f"{tag}_P"] = float(np.mean(np.abs(nul) >= abs(obs) - 1e-12))
            else:
                r[f"{tag}_P"] = r[f"{tag}_P_two_groups_only"]
        out.append(r)
    mg = d.groupby("mouse").group.first()
    if mg.nunique() == 3 and d.turn.nunique() == 3:
        A, codes = assignments(mg); mi = d.mouse.map({m: i for i, m in enumerate(mg.index)}).values
        T = pd.get_dummies(d.turn).values.astype(float); y = d.value.values
        def intF(lab):
            G = np.eye(3)[lab[mi]]; X0 = np.c_[T, G[:, 1:]]; X1 = np.c_[X0, (T[:, 1:, None] * G[:, None, 1:]).reshape(len(y), -1)]
            r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]; r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
            df2 = len(y) - np.linalg.matrix_rank(X1)
            return ((r0 @ r0 - r1 @ r1) / (X1.shape[1] - X0.shape[1])) / ((r1 @ r1) / df2) if df2 > 0 and r1 @ r1 > 0 else np.nan
        f0 = intF(codes)
        if np.isfinite(f0): out[0]["P_genotype_by_turn"] = float(np.mean(np.array([intF(a) for a in A]) >= f0 - 1e-12))
    return out


def bh(p):
    p = np.asarray(p, float); ok = np.isfinite(p); q = np.full(len(p), np.nan)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); n = len(v); adj = np.minimum.accumulate((v[o] * n / np.arange(1, n + 1))[::-1])[::-1]; tmp = np.empty(n); tmp[o] = np.minimum(adj, 1); q[ok] = tmp
    return q


def multivariate(LONG, ROB, FT):
    mv, pcs, fams = [], [], []
    for ct in TYPES:
        rob = ROB[(ROB.cell_type == ct) & (ROB.status == "robust")].measure
        W = LONG[(LONG.family == "bundle panel: median") & (LONG.cell_type == ct) & LONG.measure.isin(rob)].pivot_table(index="field", columns="measure", values="value")
        W = W.loc[:, W.std() > 0]; W = W - W.groupby(W.index.map(FT.turn)).transform("mean"); Z = (W - W.mean()) / W.std()
        Mm = Z.groupby(Z.index.map(FT.mouse)).mean().dropna(axis=1); grp = pd.Series([FT[FT.mouse == m].group.iloc[0] for m in Mm.index], index=Mm.index)
        D2 = ((Mm.values[:, None, :] - Mm.values[None, :, :]) ** 2).sum(-1); n = len(Mm)
        def pf(lab):
            sst = D2.sum() / (2 * n); ssw = sum(D2[np.ix_(lab == k, lab == k)].sum() / (2 * (lab == k).sum()) for k in range(3)); return ((sst - ssw) / 2) / (ssw / (n - 3))
        A, codes = assignments(grp); f0 = pf(codes); r = dict(cell_type=ct, measures=Mm.shape[1], mice=n, pseudoF=f0, P_three_group=float(np.mean(np.array([pf(a) for a in A]) >= f0 - 1e-12)))
        for g in ["Smad3", "beta-catenin"]:
            sub = Mm[grp.isin(["control", g])]; k = (grp[sub.index] == g).values; obs = np.linalg.norm(sub.values[k].mean(0) - sub.values[~k].mean(0))
            null = [np.linalg.norm(sub.values[list(c)].mean(0) - np.delete(sub.values, list(c), 0).mean(0)) for c in combinations(range(len(sub)), k.sum())]
            r[f"P_{g}_vs_control"] = float(np.mean(np.array(null) >= obs - 1e-12))
        mv.append(r)
        X = Z.fillna(0).values; X = X - X.mean(0); U, s, Vt = np.linalg.svd(X, full_matrices=False); ev = s ** 2 / (s ** 2).sum()
        pc1 = pd.Series(U[:, 0] * s[0], index=Z.index); load = pd.Series(Vt[0], index=Z.columns).sort_values()
        bym = pc1.groupby(pc1.index.map(FT.mouse)).mean(); byg = bym.groupby([FT[FT.mouse == m].group.iloc[0] for m in bym.index]).mean()
        pcs.append(dict(cell_type=ct, PC1_share=ev[0], PC2_share=ev[1], PC1_lowest_loadings=", ".join(load.index[:4]), PC1_highest_loadings=", ".join(load.index[-4:]), **{f"PC1_{g}": byg.get(g) for g in GR}))
        Cm = W.corr(method="spearman").abs().fillna(0); Dc = 1 - Cm.values; np.fill_diagonal(Dc, 0)
        cl = fcluster(linkage(squareform(Dc, checks=False), "average"), t=0.2, criterion="distance")
        fams += [dict(cell_type=ct, measure=m, related_measure_group=int(c)) for m, c in zip(Cm.columns, cl)]
    return pd.DataFrame(mv), pd.DataFrame(pcs), pd.DataFrame(fams)


def bundle_models(B, ROB):
    import statsmodels.formula.api as smf
    warnings.filterwarnings("ignore"); out = []
    fml = 'y ~ C(genotype, Treatment("control")) + C(turn)'; kS = 'C(genotype, Treatment("control"))[T.Smad3]'; kB = 'C(genotype, Treatment("control"))[T.beta-catenin]'
    for ct in TYPES:
        for m in ROB[(ROB.cell_type == ct) & (ROB.status == "robust")].measure:
            d = B[B.row == ct][[m, "group", "turn", "mouse", "field"]].dropna().rename(columns={m: "y", "group": "genotype"})
            if d.y.std() == 0: continue
            d["y"] = (d.y - d.y.mean()) / d.y.std()
            try:
                fit = smf.mixedlm(fml, d, groups="mouse", re_formula="1", vc_formula={"field": "0 + C(field)"}).fit(reml=True)
                vm, vf, ve = float(fit.cov_re.iloc[0, 0]), float(fit.vcomp[0]), float(fit.scale); tot = vm + vf + ve
                out.append(dict(cell_type=ct, measure=m, bundles=len(d), Smad3_sd_units=fit.params[kS], Smad3_P_model=fit.pvalues[kS], beta_catenin_sd_units=fit.params[kB],
                                beta_catenin_P_model=fit.pvalues[kB], share_mouse=vm / tot, share_field=vf / tot, share_bundle=ve / tot, converged=fit.converged, singular=vm < 1e-4))
            except Exception as ex:
                out.append(dict(cell_type=ct, measure=m, error=str(ex)[:120]))
    return pd.DataFrame(out)


def main():
    FT = field_info()
    LONG, PH, B, MEAS = long_table(FT); LONG.to_csv(os.path.join(R, "phenotype_field_values_long.csv"), index=False)
    ROB = robustness(PH, FT, MEAS); ROB.to_csv(os.path.join(R, "bundle_phenotype_robustness.csv"), index=False)
    res = []
    for (fam, ct, m), d in LONG.groupby(["family", "cell_type", "measure"]):
        tested = d.value.nunique() > 2 and d.value.std() > 1e-9
        if d.mouse.nunique() < 6: continue
        for r in (analyze(d) if tested else [dict(turn="all turns")]): r.update(family=fam, cell_type=ct, measure=m, tested=tested); res.append(r)
    T = pd.DataFrame(res); a = T.turn == "all turns"
    for col in ["P_three_group", "Smad3_P", "beta_catenin_P", "P_genotype_by_turn"]:
        T[col + "_adj"] = np.nan
        for _, idx in T[a].groupby(["family", "cell_type"]).groups.items(): T.loc[idx, col + "_adj"] = bh(T.loc[idx, col].values)
    MV, PCS, FAM = multivariate(LONG, ROB, FT)
    T = T.merge(FAM, on=["cell_type", "measure"], how="left").merge(ROB[["cell_type", "measure", "status", "min_spearman", "field_icc"]].rename(columns={"status": "robustness"}), on=["cell_type", "measure"], how="left")
    T.to_csv(os.path.join(R, "phenotype_statistics_all.csv"), index=False)
    MV.to_csv(os.path.join(R, "phenotype_panel_multivariate.csv"), index=False); PCS.to_csv(os.path.join(R, "phenotype_panel_principal_components.csv"), index=False)
    MX = bundle_models(B, ROB); MX.to_csv(os.path.join(R, "phenotype_bundle_level_models.csv"), index=False)
    n = T[a & T.tested].P_three_group.notna().sum()
    print(f"phenotypes tested {n}; three-group P < 0.05: {(T[a].P_three_group < 0.05).sum()}; smallest adjusted {T[a].P_three_group_adj.min():.3f}")
    print(MV.round(4).to_string(index=False))
    return T, MV, PCS, MX


if __name__ == "__main__":
    main()
