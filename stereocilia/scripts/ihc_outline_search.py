"""Tuning search for the inner hair cell bundle outline and measures (standing and lying-flat bundles).

References
  standing: base/tip clicks on 42 standing cells (ihc_base_tip: 24 + 2 repeats; ihc_base_tip_test: 18 held out),
            cells clicked in both views. Base height from the front view (where separate stereocilia begin), tip
            height = mean of the two views, across-row positions from the side view, with the side-view base moved
            along the side-view base-to-tip line to the front-view base height (keeps base height and position
            on the same point of the bundle).
  folded:   painted extent of 19 lying-flat bundles (ihc_paint_folded, ihc_paint_folded2), original cut-outs.
  synthetic: the 200-cell synthetic set (synthetic_ihc_bundles.grid).

Targets (set before scoring): standing height and length within 1 um of the reference in >= 85% of cells, lean
within 10 degrees in >= 85%, folded length within 1 um of the paint in >= 85%, synthetic errors no worse than the
current rule. Settings are chosen leaving one mouse out, so each mouse's agreement comes from settings chosen
without it.

Run:  python ihc_outline_search.py [n_settings]     (default 3000; writes results\\tuning\\ihc_outline_search\\)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, sys, time, json
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ihc_stereocilia as isc
import synthetic_ihc_bundles as syn

ANN = S.ANNOTATIONS; WORK = S.WORK
STACKS_OLD, STACKS_V2 = os.path.join(WORK, "ihc_stacks"), os.path.join(WORK, "ihc_stacks_v2")
OUT = os.path.join(os.path.dirname(HERE), "results", "tuning", "ihc_outline_search"); CACHE = os.path.join(WORK, "ihc_search_cache")
ISO, ZS = isc.ISO, isc.Z_SCALE
NTHREADS = 24
TOL = dict(height=1.0, length=1.0, lean=10.0, folded=1.0)

CURRENT = dict(core_p=0.85, core_tb=98, core_v=90, weak_p=0.5, weak_tb=93, weak_v=75, tube=2.0, down=0.0,
               flat_deg=65, flat_tip=0.5, tip_pct=98, base_pct=2, base_from="core", length_rule="endpoints", band=0.3)
SPACE = dict(core_p=[0.8, 0.85, 0.9], core_tb=[97, 98, 99], core_v=[85, 90, 95], weak_p=[0.3, 0.4, 0.5, 0.6, 0.7],
             weak_tb=[85, 90, 93, 96], weak_v=[60, 70, 75, 80, 85], tube=[1.5, 2.0, 2.5, 3.0], down=[0.0, 0.5, 1.0, 2.0],
             flat_deg=[55, 60, 65, 70], flat_tip=[0.0, 0.5, 1.0], tip_pct=[95, 97, 98, 99, 99.5], base_pct=[1, 2, 5],
             base_from=["core", "mask"], length_rule=["endpoints", "axis_extent"], band=[0.3, 0.5])
TB_P = sorted(set(SPACE["core_tb"] + SPACE["weak_tb"])); V_P = sorted(set(SPACE["core_v"] + SPACE["weak_v"]))


def now():
    return pd.Timestamp.now(tz="America/Chicago").strftime("%H:%M CST")


# ---------------------------------------------------------------- references
def standing_reference():
    rows = []
    for folder in ["ihc_base_tip", "ihc_base_tip_test"]:
        d = os.path.join(ANN, folder); key = pd.read_csv(os.path.join(d, "sample_key.csv"))
        if "repeat" not in key: key["repeat"] = False
        f = pd.read_csv(os.path.join(d, "clicks_face.csv")); s = pd.read_csv(os.path.join(d, "clicks.csv"))
        j = key.merge(f[["item", "status", "base_height_um", "tip_height_um"]], on="item").merge(
            s[["item", "status", "base_height_um", "tip_height_um", "base_across_um", "tip_across_um"]], on="item", suffixes=("_front", "_side"))
        j = j[(~j.repeat.astype(bool)) & (j.status_front == "clicked") & (j.status_side == "clicked")].copy()
        j["set"] = "tuning" if folder == "ihc_base_tip" else "held-out"; rows.append(j)
    j = pd.concat(rows, ignore_index=True)
    bf, bs = j.base_height_um_front * ZS, j.base_height_um_side * ZS; tf, ts = j.tip_height_um_front * ZS, j.tip_height_um_side * ZS
    by = j.base_across_um + (j.tip_across_um - j.base_across_um) * (bf - bs) / (ts - bs)
    j["ref_base"] = bf; j["ref_tip"] = (tf + ts) / 2; j["ref_height"] = j.ref_tip - j.ref_base
    j["ref_length"] = np.hypot(j.tip_across_um - by, j.ref_height); j["ref_lean"] = np.degrees(np.arctan2(j.tip_across_um - by, j.ref_height))
    j["mouse"] = j.field.str.split(" ").str[0]; j["kind"] = "standing"; j["stacks"] = STACKS_V2
    return j[["set", "item", "field", "cell", "group", "turn", "mouse", "kind", "stacks", "ref_base", "ref_tip", "ref_height", "ref_length", "ref_lean"]]


def folded_reference():
    p = os.path.join(os.path.dirname(HERE), "results", "tuning", "ihc_folded_paint_both_sets_vs_auto_2026-11-10.csv")
    f = pd.read_csv(p); f["mouse"] = f.field.str.split(" ").str[0]; f["kind"] = "folded"; f["stacks"] = STACKS_OLD
    f["turn"] = f.field.str.extract(r"(APEX|MIDDLE|BASE)", expand=False).str.lower()
    return f[["set", "item", "field", "cell", "group", "turn", "mouse", "kind", "stacks", "paint_ext"]]


# ---------------------------------------------------------------- cached inputs per cell
def prepare(vol, half):
    v = isc.iso(vol).astype(np.float32); F = isc.features(v)
    P = isc.model().predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(v.shape).astype(np.float32)
    Tb = isc.sato(v, sigmas=[1.0, 1.5, 2.0], black_ridges=False).astype(np.float32)
    return dict(v=v, P=P, Tb=Tb, half=np.array(half, float), tbp=np.percentile(Tb, TB_P), vp=np.percentile(v, V_P))


def real_inputs(ref):
    os.makedirs(CACHE, exist_ok=True); out = {}
    def one(r):
        fn = os.path.join(CACHE, f"{r.kind}_{r.set}_{int(r.item)}.npz")
        if os.path.exists(fn): z = np.load(fn); return dict(z)
        z = np.load(os.path.join(r.stacks, r.field + ".npz")); cl = pd.read_csv(os.path.join(r.stacks, r.field + "_cells.csv"))
        hl, hr = isc.halves(cl); j = int(np.where(cl.cell.values == r.cell)[0][0]); d = prepare(z["phal"][j], (hl[j], hr[j]))
        np.savez(fn, **d); return d
    with ThreadPoolExecutor(NTHREADS) as ex: res = list(ex.map(one, ref.itertuples()))
    for k, d in zip(ref.index, res): out[k] = d
    return out


def synthetic_inputs():
    os.makedirs(CACHE, exist_ok=True); g = syn.grid()
    def one(a):
        H, ln, b, h0, sz, s = a; fn = os.path.join(CACHE, f"synthetic_{s}.npz")
        vol, tr = syn.make(H, ln, b, h0, sz, s)
        if os.path.exists(fn): d = dict(np.load(fn))
        else: d = prepare(vol, (4.25, 4.25)); np.savez(fn, **d)
        return d, dict(H=H, lean_set=ln, bright=b, sig_z=sz, seed=s, **tr)
    with ThreadPoolExecutor(NTHREADS) as ex: res = list(ex.map(one, g))
    return [r[0] for r in res], pd.DataFrame([r[1] for r in res])


# ---------------------------------------------------------------- outline and measures with settings
def pct_tb(d, q): return d["tbp"][TB_P.index(q)]
def pct_v(d, q): return d["vp"][V_P.index(q)]


def outline(d, s):
    v, P, Tb, half = d["v"], d["P"], d["Tb"], tuple(d["half"])
    core = isc.in_window((P >= s["core_p"]) & (Tb >= pct_tb(d, s["core_tb"])) & (v >= pct_v(d, s["core_v"])), half); core = isc.main_piece(core)
    if core.sum() < 50: return core, core, False
    mu, dd = isc.lean_axis(v, core); flat = abs(np.degrees(np.arctan2(dd[0], dd[1]))) > s["flat_deg"]
    zz, yy, _ = np.indices(v.shape); Q = np.stack([yy * ISO, isc.heights(zz)], -1) - mu; t = Q @ dd; n = np.abs(Q @ np.array([-dd[1], dd[0]]))
    lo, hi = np.percentile(t[core], [2, 98])
    if flat:
        ohc_hi = dd[0] > 0; ft = s["flat_tip"]
        allowed = (n <= s["tube"]) & (t >= lo - (0.0 if not ohc_hi else ft)) & (t <= hi + (0.0 if ohc_hi else ft))
    else:
        allowed = (t >= lo - s["down"]) & (n <= s["tube"])
    weak = (P >= s["weak_p"]) & (Tb >= pct_tb(d, s["weak_tb"])) & (v >= pct_v(d, s["weak_v"]))
    L = isc.cclab((weak & allowed & isc.territory(v.shape, half)) | core); keep = np.isin(L, np.unique(L[core])); keep[L == 0] = False
    return keep, core, flat


def measure(d, s):
    v = d["v"]; m, core, flat = outline(d, s)
    if core.sum() < 50: return dict(category="no bundle")
    zz, yy, _ = np.where(m); h = isc.heights(zz); y = yy * ISO; mu, dd = isc.lean_axis(v, m); t = (np.c_[y, h] - mu) @ dd
    if s["base_from"] == "core":
        zc, yc, _ = np.where(core); hb, yb = isc.heights(zc), yc * ISO; tb = (np.c_[yb, hb] - mu) @ dd
    else: hb, yb, tb = h, y, t
    e = s["band"]; bsel = tb <= np.percentile(tb, s["base_pct"]) + e; tsel = t >= np.percentile(t, s["tip_pct"]) - e
    bh, by, th, ty = hb[bsel].mean(), yb[bsel].mean(), h[tsel].mean(), y[tsel].mean()
    ext = float(np.percentile(t, s["tip_pct"]) - np.percentile(t, s["base_pct"]))
    if flat: return dict(category="lying flat", length_um=ext, base_um=bh)
    length = float(np.hypot(ty - by, th - bh)) if s["length_rule"] == "endpoints" else ext
    return dict(category="standing", base_um=bh, tip_um=th, height_um=th - bh, length_um=length, lean_deg=float(np.degrees(np.arctan2(ty - by, th - bh))))


# ---------------------------------------------------------------- scoring
def score_real(ref, M):
    D = pd.concat([ref.reset_index(drop=True), pd.DataFrame(M)], axis=1); st = D[D.kind == "standing"]; fo = D[D.kind == "folded"]
    e = dict(height=st.height_um - st.ref_height, length=st.length_um - st.ref_length, lean=st.lean_deg - st.ref_lean, folded=fo.length_um - fo.paint_ext)
    return D, e


def summarize(D, cells=None):
    """Share within tolerance (missing measure counts as a miss), median error and median absolute error."""
    D = D if cells is None else D.loc[cells]; out = {}
    st = D[D.kind == "standing"]; fo = D[D.kind == "folded"]
    for k, a, b, sub in [("height", "height_um", "ref_height", st), ("length", "length_um", "ref_length", st), ("lean", "lean_deg", "ref_lean", st), ("folded", "length_um", "paint_ext", fo)]:
        if len(sub) == 0: continue
        er = sub[a] - sub[b]; out[f"{k}_within"] = float((er.abs() <= TOL[k]).mean()); out[f"{k}_bias"] = float(er.median()); out[f"{k}_abs"] = float(er.abs().median())
    out["objective"] = float(np.mean([out[f"{k}_within"] for k in ["height", "length", "lean", "folded"] if f"{k}_within" in out]))
    return out


def score_synthetic(T, M):
    S = pd.concat([T.reset_index(drop=True), pd.DataFrame(M)], axis=1); up = S[S.lean_set.isin([0, -20, -40, -60])]; fl = S[S.lean_set.isin([-75, -90])]
    r = {}
    for k in ["height", "length", "base", "tip"]:
        if k + "_um" in up: er = up[k + "_um"] - up["true_" + k]; r[f"syn_{k}_bias"] = float(er.median()); r[f"syn_{k}_abs"] = float(er.abs().median())
    if "lean_deg" in up: er = up.lean_deg - up.true_lean; r["syn_lean_bias"] = float(er.median()); r["syn_lean_abs"] = float(er.abs().median())
    if "length_um" in fl: er = fl.length_um - fl.true_length; r["syn_folded_length_bias"] = float(er.median()); r["syn_folded_length_abs"] = float(er.abs().median())
    r["syn_standing_read_standing"] = float((up.category == "standing").mean()); r["syn_flat_read_flat"] = float((fl.category == "lying flat").mean())
    return r


def run_setting(s, inputs):
    with ThreadPoolExecutor(NTHREADS) as ex: return list(ex.map(lambda d: measure(d, s), inputs))


def sample_settings(n, seed=1):
    rng = np.random.default_rng(seed); out = [dict(CURRENT)]; seen = {json.dumps(CURRENT, sort_keys=True)}
    while len(out) < n:
        s = {k: (lambda v: v.item() if hasattr(v, "item") else v)(rng.choice(np.array(v, dtype=object))) for k, v in SPACE.items()}
        if s["core_p"] < s["weak_p"] or s["core_tb"] < s["weak_tb"] or s["core_v"] < s["weak_v"]: continue
        k = json.dumps(s, sort_keys=True)
        if k not in seen: seen.add(k); out.append(s)
    return out


def main(n=3000):
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); log = open(os.path.join(OUT, "progress.txt"), "a", encoding="utf-8")
    def say(msg): print(f"{now()}  {msg}", flush=True); log.write(f"{now()}  {msg}\n"); log.flush()
    ref = pd.concat([standing_reference(), folded_reference()], ignore_index=True); ref.to_csv(os.path.join(OUT, "reference_cells.csv"), index=False)
    say(f"reference: {int((ref.kind == 'standing').sum())} standing cells, {int((ref.kind == 'folded').sum())} folded cells, {ref.mouse.nunique()} mice")
    R = real_inputs(ref); inputs = [R[k] for k in ref.index]; say(f"real cells prepared ({time.time() - t0:.0f} s)")
    SY, T = synthetic_inputs(); say(f"synthetic cells prepared: {len(SY)} ({time.time() - t0:.0f} s)")
    settings = sample_settings(n); rows = []; percell = []
    for i, s in enumerate(settings):
        M = run_setting(s, inputs); D, _ = score_real(ref, M); r = dict(setting=i, **s, **summarize(D))
        for mouse in ref.mouse.unique(): r.update({f"{mouse}|{k}": v for k, v in summarize(D, D.index[D.mouse == mouse]).items()})
        rows.append(r); percell.append(D[["height_um", "length_um", "lean_deg", "category"]].assign(setting=i, cell_row=D.index))
        if i % 50 == 0 or i == len(settings) - 1:
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "settings_scores.csv"), index=False)
            say(f"{i + 1}/{len(settings)} settings; best objective so far {max(x['objective'] for x in rows):.3f} (current rule {rows[0]['objective']:.3f})")
    S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "settings_scores.csv"), index=False)
    PC = pd.concat(percell, ignore_index=True); PC.to_csv(os.path.join(OUT, "per_cell_all_settings.csv.gz"), index=False)
    # leave one mouse out: choose on the other mice (pooled cells), score the left-out mouse's cells
    lomo = []
    for mouse in ref.mouse.unique():
        others = ref.index[ref.mouse != mouse]; best, bi = -1, 0
        for i, s in enumerate(settings):
            Di = PC[PC.setting == i].set_index("cell_row"); D = pd.concat([ref, Di[["height_um", "length_um", "lean_deg", "category"]]], axis=1)
            o = summarize(D, others)["objective"]
            if o > best: best, bi = o, i
        Di = PC[PC.setting == bi].set_index("cell_row"); D = pd.concat([ref, Di[["height_um", "length_um", "lean_deg", "category"]]], axis=1)
        lomo.append(D[D.mouse == mouse].assign(chosen_setting=bi, chosen_on_others=best))
    L = pd.concat(lomo); L.to_csv(os.path.join(OUT, "leave_one_mouse_out_cells.csv"), index=False)
    say("leave one mouse out, pooled: " + json.dumps({k: round(v, 3) for k, v in summarize(L).items()}))
    # synthetic check for the current rule and the 25 best settings on all cells
    top = [0] + [int(i) for i in S.sort_values("objective", ascending=False).setting.head(25) if i != 0]; syn_rows = []
    for i in top:
        syn_rows.append(dict(setting=i, objective=S.objective[i], **score_synthetic(T, run_setting(settings[i], SY))))
        pd.DataFrame(syn_rows).to_csv(os.path.join(OUT, "synthetic_top_settings.csv"), index=False)
    say(f"synthetic check done for {len(top)} settings; total {(time.time() - t0) / 3600:.1f} h")
    json.dump(settings, open(os.path.join(OUT, "settings.json"), "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
