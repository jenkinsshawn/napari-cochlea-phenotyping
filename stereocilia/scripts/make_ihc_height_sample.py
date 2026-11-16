r"""Fresh held-out sample of inner hair cells for the bundle height clicks (click_ihc_base_tip.py --face --test4).

Draws standing inner hair cell bundles of the official run (S.RESULTS\ihc_stereocilia_cells_v2.csv) that no earlier
sample named (no earlier sampled cell within 3 um), from fields not marked damaged. In each group and turn it takes 3
complete bundles and 1 incomplete bundle (tip may be cut or bundle at the image edge) where available, taking mice in
turn, in a fixed random order. The complete bundles give an independent test of the height agreement; the incomplete
ones test whether the complete-bundle rule picks out the cells that agree.

Each row stores the cell's position (x, y in field pixels), so the clicking tool finds the cut-out by position in
whichever run the settings point at.

Output: S.ANNOTATIONS\ihc_base_tip_test4\sample.csv (item, field, cell, x, y) and sample_key.csv (adds group, turn,
mouse, complete). An existing sample is never overwritten.

Run (imaging env, from the scripts folder):  python make_ihc_height_sample.py
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import pipeline_settings as S

OUT = os.path.join(S.ANNOTATIONS, "ihc_base_tip_test4")
N_COMPLETE, N_INCOMPLETE, EXCLUDE_UM, SEED = 3, 1, 3.0, 20261113


def earlier_cells():
    """Positions of every cell named by an earlier inner hair cell click sample."""
    rows = []
    for d in glob.glob(os.path.join(S.ANNOTATIONS, "ihc_base_tip*")) + [os.path.join(S.ANNOTATIONS, "ihc_clicks")]:
        if os.path.abspath(d) == os.path.abspath(OUT) or not os.path.exists(os.path.join(d, "sample.csv")): continue
        s = pd.read_csv(os.path.join(d, "sample.csv"))
        if {"x", "y"} <= set(s.columns):
            rows += list(zip(s.field, s.x, s.y)); continue
        kind = "ihc_stacks" if os.path.basename(d) == "ihc_clicks" else "ihc_stacks_v2"
        for r in s.itertuples():
            c = pd.read_csv(os.path.join(S.ANNOTATIONS, "click_reference_cutouts", kind, r.field + "_cells.csv")).iloc[int(r.cell) - 1]
            rows.append((r.field, c.x, c.y))
    return pd.DataFrame(rows, columns=["field", "x", "y"])


def candidates():
    B = pd.read_csv(os.path.join(S.RESULTS, "ihc_stereocilia_cells_v2.csv")); B = B[B.category == "standing"].copy()
    B["complete"] = ~B.reindex(columns=["tip_may_be_cut", "bundle_at_image_edge"]).eq(True).any(axis=1)
    B["group"] = B.field.map(S.group_of); B["mouse"] = B.field.map(S.mouse_of)
    B["turn"] = B.field.str.extract(r"(apex|middle|base)", flags=2)[0].str.lower()
    P = earlier_cells()
    def clicked(r):
        q = P[P.field == r.field]
        return len(q) > 0 and (np.hypot(q.x - r.x, q.y - r.y) * S.PX_UM).min() <= EXCLUDE_UM
    B = B[B.group.notna() & ~B.field.isin(dict(S.DAMAGED)) & B.turn.notna()]
    return B[[not clicked(r) for r in B.itertuples()]]


def draw(B):
    rng = np.random.default_rng(SEED); out = []
    for (g, t), grp in B.groupby(["group", "turn"], sort=True):
        for comp, n in ((True, N_COMPLETE), (False, N_INCOMPLETE)):
            pool = grp[grp.complete == comp].sample(frac=1, random_state=int(rng.integers(1e9)))
            mice = list(pool.mouse.drop_duplicates()); picked = []
            while len(picked) < n and len(pool):            # one cell per mouse in turn
                for m in list(mice):
                    q = pool[pool.mouse == m]
                    if not len(q): mice.remove(m); continue
                    picked.append(q.iloc[0]); pool = pool.drop(q.index[0])
                    if len(picked) == n: break
                if not mice: break
            out += picked
    d = pd.DataFrame(out).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shown in random order
    d.insert(0, "item", np.arange(1, len(d) + 1))
    return d


def main():
    if os.path.exists(os.path.join(OUT, "sample.csv")):
        print(f"{OUT} already has a sample; it is never overwritten."); return
    d = draw(candidates()); os.makedirs(OUT, exist_ok=True)
    d[["item", "field", "cell", "x", "y"]].to_csv(os.path.join(OUT, "sample.csv"), index=False)
    d[["item", "field", "cell", "x", "y", "group", "turn", "mouse", "complete"]].to_csv(os.path.join(OUT, "sample_key.csv"), index=False)
    print(f"{len(d)} cells ({int(d.complete.sum())} complete, {int((~d.complete).sum())} incomplete) from {d.mouse.nunique()} mice "
          f"and {d.field.nunique()} fields written to {OUT}")
    print(d.groupby(["group", "turn"]).complete.agg(["sum", "size"]).rename(columns={"sum": "complete", "size": "cells"}).to_string())


if __name__ == "__main__":
    main()
