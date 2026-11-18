r"""Results report for any run: a short Word document and four CSV tables, written from the statistics tables.

Reads (RESULTS): phenotype_statistics_all.csv, phenotype_statistics_all_noise_adjusted.csv (when present),
phenotype_panel_multivariate.csv, validation_cards.csv, fields_index.csv.
Writes RESULTS\report\: Cochlea_results.docx (Cochlea_results.md when python-docx is not installed) and
  Table_1_fields.csv            fields, mice, groups, cochlear turn and fields left out
  Table_2_agreement.csv         agreement of each automatic measure with hand measurements
  Table_3_main_results.csv      the measures named in the report, per family
  Table_4_all_tests.csv         every measure and test, with the noise-adjusted results beside them
Run (imaging env, scripts folder):  python report_run.py
"""
import os, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import numpy as np, pandas as pd
from measure_names import readable

R = S.RESULTS; OUT = os.path.join(R, "report"); MAIN = "flagged fields left out"
FAMILY_ORDER = [("Stereocilia bundles, top view", lambda f: f.startswith("bundle panel")),
                ("Inner hair cell bundles in 3D, front view", lambda f: f in ("inner hair cell front view", "inner hair cell height (hand clicks)")),
                ("Inner hair cell bundles scored by hand", lambda f: f.startswith("inner hair cell organization")),
                ("Outer hair cell rows: straightness and order", lambda f: f == "outer hair cell row order"),
                ("Hair cell spacing, packing and numbers", lambda f: f.startswith("spatial: spacing") or f.startswith("density") or f == "row packing"),
                ("Hair cell direction and bundle polarity", lambda f: f.startswith("spatial: orientation") or f == "bundle polarity"),
                ("Hair cell surface", lambda f: f == "hair cell surface"),
                ("Nuclei and chromatin", lambda f: "nuclei" in f),
                ("Hair cell identity", lambda f: f.startswith("hair cell identity"))]
TYPE = {"IHC": "inner hair cells", "OHC1": "outer row 1", "OHC2": "outer row 2", "OHC3": "outer row 3", "all": "all hair cells"}


def f_(v, nd=2): return "" if v is None or v != v else f"{v:.{nd}f}".replace("-", "\u2212")
def p_(v): return "" if v is None or v != v else (f"{v:.3f}" if v >= 0.001 else f"{v:.1e}")
def nd_(x):
    v = np.nanmax(np.abs([x[c] for c in x.index if c.endswith("_mean")])) if any(c.endswith("_mean") for c in x.index) else 1
    return 3 if v < 1 else (2 if v < 10 else 1)


def groups(T):
    g = [c[:-5] for c in T.columns if c.endswith("_mean")]; return g[0], g[1:]


def sentence(x, ctrl, others, adj=None):
    nd = nd_(x); means = ", ".join(f"{f_(x[g + '_mean'], nd)} in {g}" for g in [ctrl] + others)
    same = adj is None or all(abs((adj.get(c, np.nan) if adj.get(c, np.nan) == adj.get(c, np.nan) else 0) - (x[c] if x[c] == x[c] else 0)) < 1e-9
                              for c in ["P_three_group"] + [g.replace("-", "_") + s for g in others for s in ("_difference", "_P")] if c in x)
    parts = []
    for g in others:
        t = g.replace("-", "_")
        if f"{t}_difference" not in x: continue
        ci = f"95% interval {f_(x[t + '_ci_low'], nd)} to {f_(x[t + '_ci_high'], nd)}; " if x.get(t + "_ci_low") == x.get(t + "_ci_low") else ""
        s = f"{g} minus {ctrl} {f_(x[t + '_difference'], nd)} ({ci}P {p_(x[t + '_P'])})"
        if not same: s += f", after noise adjustment {f_(adj[t + '_difference'], nd)} (P {p_(adj[t + '_P'])})"
        parts.append(s)
    three = f"three-group P {p_(x['P_three_group'])}" + (f", {p_(adj['P_three_group'])} after noise adjustment" if not same else "")
    tail = "; not changed by the noise adjustment" if adj is not None and same else ""
    return f"{readable(x.measure)} in {TYPE.get(x.cell_type, x.cell_type)}: {means} ({three}); " + "; ".join(parts) + tail + "."


def main(top=3):
    T_all = pd.read_csv(os.path.join(R, "phenotype_statistics_all.csv")); T = T_all[(T_all.fields == MAIN) & (T_all.turn == "all turns") & T_all.tested.astype(bool)]
    ap = os.path.join(R, "phenotype_statistics_all_noise_adjusted.csv"); A = pd.read_csv(ap) if os.path.exists(ap) else None
    key = ["fields", "family", "cell_type", "measure", "turn"]
    Ai = A.set_index(key) if A is not None else None
    ctrl, others = groups(T)
    FI = pd.read_csv(os.path.join(R, "fields_index.csv")); VC = pd.read_csv(os.path.join(R, "validation_cards.csv")) if os.path.exists(os.path.join(R, "validation_cards.csv")) else None
    os.makedirs(OUT, exist_ok=True)
    used = FI[FI.measured.astype(bool)]; n_mice = used.groupby("group").mouse.nunique()
    text = []; H = lambda s: text.append(("h", s)); P = lambda s: text.append(("p", s)); main_rows = []
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-6))).strftime("%d %B %Y")
    text.append(("t", "Automatic measurement of hair cells and stereocilia: results")); P(f"Results folder: {R}. Written {now}.")
    H("Data")
    P(f"We measured {len(used)} fields from {used.mouse.nunique()} mice (" + ", ".join(f"{n_mice.get(g, 0)} {g}" for g in [ctrl] + others) + f"). "
      f"{int(used.flagged.astype(bool).sum())} fields marked as damaged were measured but left out of the comparisons (Table 1).")
    H("Methods")
    P("Each field was measured by the Harp pipeline: cell surface, hair cell outlines and rows, stereocilia bundle outlines in the planes 0.3 to 6 \u00b5m above "
      "the surface, inner hair cell bundles in 3D from the full stack, nuclei from DAPI, and Calb2, Ocm or PMCA2 per hair cell. Each measure was summarized per field as the "
      "median over cells or bundles. Field values were centered on the mean of their cochlear turn and averaged within each mouse. Each group was compared with "
      f"{ctrl} as the difference of mouse means with a 95% Welch interval and an exact permutation P value (genotype labels reassigned across mice), and all groups together "
      "with a one-way F statistic and the same permutations. P values were adjusted for the false discovery rate within each family of measures and cell type.")
    if A is not None:
        P("Noise adjustment: each bundle, cell and nucleus lies at its own depth below the first plane of the stack and on its own slope of the tissue surface. For each "
          "measure and cell type, the effects of the local surface depth and slope were estimated within fields, where genotype is constant, and each value was shifted to "
          "the typical depth and slope when these effects were clear (P below 0.01, Wald test with a leave-one-field-out covariance). The tests were then repeated on the "
          "adjusted values; both results are given.")
    if VC is not None and len(VC):
        H("Agreement with hand measurements")
        def card(r):
            if getattr(r, "correlation", np.nan) == getattr(r, "correlation", np.nan) and not (r.agreement_pct == r.agreement_pct):
                return f"{r.measure.capitalize()}: Spearman correlation {f_(r.correlation, 2)} over {int(r.cells)} ({r.agreement_rule})."
            if not (r.agreement_pct == r.agreement_pct):
                return f"{r.measure.capitalize()}: {r.agreement_rule}."
            return (f"{r.measure.capitalize()}: {f_(r.agreement_pct, 1)}% of {int(r.cells)} cells ({r.agreement_rule}; 95% interval "
                    f"{f_(r.interval_low_pct, 1)} to {f_(r.interval_high_pct, 1)}%).")
        P(" ".join(card(r) for r in VC.itertuples()) + " (Table 2)")
    H("Results")
    P(f"Of {len(T):,} measures tested, {int((T.P_three_group < 0.05).sum())} had a three-group P value below 0.05, against about {round(0.05 * len(T))} expected by chance"
      + (f"; after noise adjustment, {int((A[(A.fields == MAIN) & (A.turn == 'all turns') & A.tested.astype(bool)].P_three_group < 0.05).sum())}." if A is not None else "."))
    for title, sel in FAMILY_ORDER:
        d = T[T.family.map(sel)]
        if not len(d): continue
        H(title)
        n05 = int((d.P_three_group < 0.05).sum()); lead = f"Of {len(d):,} measures, {n05} had a three-group P value below 0.05"
        if A is not None:
            da = A[(A.fields == MAIN) & (A.turn == "all turns") & A.family.map(sel) & A.tested.astype(bool)]; lead += f" ({int((da.P_three_group < 0.05).sum())} after noise adjustment)"
        best = d.sort_values("P_three_group").head(top); out = [lead + ". The smallest P values were:"]
        for _, x in best.iterrows():
            ad = Ai.loc[tuple(x[key])] if Ai is not None and tuple(x[key]) in Ai.index else None
            out.append(sentence(x, ctrl, others, ad)); main_rows.append(x)
        P(" ".join(out))
    H("Tables (CSV files in the report folder)")
    for n, dsc in [("Table 1", "fields, mice, groups, cochlear turn and fields left out"), ("Table 2", "agreement of each automatic measure with hand measurements"),
                   ("Table 3", "the measures named above, all turns and each turn"), ("Table 4", "every measure and test, with the noise-adjusted results beside them")]:
        P(f"{n}: {dsc}.")
    # tables
    FI.to_csv(os.path.join(OUT, "Table_1_fields.csv"), index=False)
    if VC is not None: VC.to_csv(os.path.join(OUT, "Table_2_agreement.csv"), index=False)
    def named(df): df = df.copy(); df.insert(0, "measure name", [readable(m) for m in df.measure]); df.insert(1, "cells", df.pop("cell_type").map(lambda c: TYPE.get(c, c))); return df
    if main_rows:
        mk = pd.DataFrame(main_rows)[["family", "cell_type", "measure"]]
        named(T_all[T_all.fields == MAIN].merge(mk, on=["family", "cell_type", "measure"])).to_csv(os.path.join(OUT, "Table_3_main_results.csv"), index=False)
    T4 = T_all if A is None else T_all.merge(A[key + ["P_three_group"] + [c for c in A.columns if c.endswith("_difference") or (c.endswith("_P") and "two_groups" not in c)]],
                                            on=key, how="left", suffixes=("", " (noise adjusted)"))
    named(T4).to_csv(os.path.join(OUT, "Table_4_all_tests.csv"), index=False)
    try:
        from docx import Document
        from docx.shared import Pt
        doc = Document(); st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
        for kind, s in text:
            if kind == "p": doc.add_paragraph(s)
            else: doc.add_paragraph().add_run(s).bold = True
        path = os.path.join(OUT, "Cochlea_results.docx"); doc.save(path)
    except ImportError:
        path = os.path.join(OUT, "Cochlea_results.md")
        open(path, "w", encoding="utf-8").write("\n\n".join(("**" + s + "**") if k != "p" else s for k, s in text))
        print("python-docx is not installed (pip install python-docx); wrote the report as Markdown")
    print("report:", path)


if __name__ == "__main__":
    main()
