"""Napari tool for checking the direction of outer hair cells by hand.

Shows one outer hair cell at a time, as two views of the same 13 um square side by side:
    left   the flat top of the cell (mean of the images within 0.3 um of the cell's plate),
           where the notch shows: a small dark spot in the bright plate, usually near one edge,
           where the kinocilium sits (the fonticulus)
    right  the bundle, the brightest phalloidin 1.8 to 3.6 um above the plate, where the V shows
The cell's outline is drawn faintly in both views. Click once, in whichever view is clearer:
the center of the dark notch, or the point where the two arms of the V meet (the bottom of the
curve if the bundle is U-shaped; not the open inside of the V). Both give the same direction
from the cell center. A second click replaces the first. The automatic answer is not shown.

Keys:
    Enter or Space   save and go to the next cell (needs a click, or press n)
    n                cannot tell; save and go to the next cell
    Left             previous cell
    b                show or hide the outline
Cells: 40 outer hair cells per field in three fields (one control, one beta-catenin mutant, one
Smad3 mutant): up to 20 that spatial_stats.py flagged as more than 45 degrees off their row and
the rest from the other outer hair cells, in random order (fixed seed, so the same cells every time).
Clicks save on every step to E:\\Webber_confocal\\annotations\\directions\\direction_clicks.csv;
the tool reopens at the first unanswered cell.

Compare the clicks with the notch measure (writes results\\spatial\\direction_check.csv and prints
a summary):
    python click_directions.py --compare

Run from the scripts folder:
    python click_directions.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, os, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
PAINT = os.path.join(S.WORK, 'paint_stacks')
CELLS = S.CLICK_OUTLINES   # the outlines the saved clicks name cells in (first-run cells\v1, frozen copy)
import click_outlines as CO   # run tables keyed by click-outline cell numbers
TYPES = os.path.join(S.WORK, 'cells', 'types')
BUNDLES = os.path.join(S.WORK, 'bundles_detector')
OUT = os.path.join(S.ANNOTATIONS, 'directions')
CLICKS = os.path.join(OUT, "direction_clicks.csv")
SPATIAL = os.path.join(S.RESULTS, "spatial", "cells_spatial.csv")
FIELDS = ["AR524 Ocm, Calb2 Phall P1 60X APEX",              # control
          "AR1383-2 Calb2 PMCA2 Phall P1 60X APEX (4)",      # beta-catenin mutant
          "JW5436 Calb2 PMCA2 Phall P1 60X middle"]          # Smad3 mutant
PER_FIELD, FLAGGED = 40, 20
HALF = 60                                                    # crop half-width in pixels (6.5 um)
GAP = 6
PX = 0.1083


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def sample_cells():
    c = CO.spatial(SPATIAL)
    rng = np.random.default_rng(20261109); out = []
    for f in FIELDS:
        s = c[(c.field == f) & c.row.str.startswith("OHC") & c.notch_deg.notna()]
        fl = s[s.dev_deg.abs() > 45]; ok = s[~(s.dev_deg.abs() > 45)]
        a = fl.sample(min(FLAGGED, len(fl)), random_state=int(rng.integers(1e9)))
        b = ok.sample(min(PER_FIELD - len(a), len(ok)), random_state=int(rng.integers(1e9)))
        d = pd.concat([a, b]).sample(frac=1, random_state=int(rng.integers(1e9)))
        out.append(d[["field", "cell", "row"]])
    return pd.concat(out).reset_index(drop=True)


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


class Field:
    """Image stack, outlines, cell centers and plate depths of one field, loaded once."""
    def __init__(self, f):
        z = np.load(os.path.join(PAINT, f + ".npz")); self.ph = z["phal"].astype(np.float32); self.off = z["offsets_um"]
        self.lab = np.load(os.path.join(CELLS, f + ".npy"))
        self.types = CO.types(f); self.plate = CO.plate_depths(f)

    def views(self, cell):
        from skimage.segmentation import find_boundaries
        m = self.lab == cell; ys, xs = np.nonzero(m); cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        pdp = self.plate.get(cell, np.nan); pdp = 0.6 if not np.isfinite(pdp) else pdp
        flat = self.ph[np.abs(self.off - pdp) <= 0.31].mean(0)
        up = (self.off >= pdp + 1.8) & (self.off <= pdp + 3.6)
        bun = self.ph[up].max(0) if up.any() else self.ph[: max(1, int(np.argmax(self.off <= pdp)))].max(0)
        H, W = self.lab.shape; y0, x0 = cy - HALF, cx - HALF

        def crop(a):
            o = np.zeros((2 * HALF, 2 * HALF), np.float32)
            ya, yb, xa, xb = max(y0, 0), min(y0 + 2 * HALF, H), max(x0, 0), min(x0 + 2 * HALF, W)
            o[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]; return o
        edge = crop(find_boundaries(m, mode="inner").astype(np.float32))
        img = np.zeros((2 * HALF, 4 * HALF + GAP), np.float32); ol = np.zeros_like(img)
        img[:, :2 * HALF] = scaled(crop(flat)); img[:, 2 * HALF + GAP:] = scaled(crop(bun))
        ol[:, :2 * HALF] = edge; ol[:, 2 * HALF + GAP:] = edge
        return img, ol, (y0, x0)


class Tool:
    def __init__(self, cells):
        import napari
        self.napari = napari
        self.cells, self.fields = cells, {}
        os.makedirs(OUT, exist_ok=True)
        self.ans = pd.read_csv(CLICKS) if os.path.exists(CLICKS) else pd.DataFrame(columns=["field", "cell", "row", "answer", "panel", "x", "y", "saved"])
        done = set(zip(self.ans.field, self.ans.cell))
        self.k = next((i for i, r in cells.iterrows() if (r.field, r.cell) not in done), len(cells) - 1)
        self.click = None
        self.v = napari.Viewer(title="Outer hair cell direction check")
        self.v.mouse_drag_callbacks.append(self.on_mouse)
        self.keys = {"Enter": self.next, "Space": self.next, "n": self.cant_tell, "Left": self.prev, "b": self.toggle}
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 14
        self.load()

    def field(self, f):
        if f not in self.fields:
            self.fields[f] = Field(f)
        return self.fields[f]

    def load(self):
        r = self.cells.iloc[self.k]
        img, ol, self.origin = self.field(r.field).views(int(r.cell))
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.v.add_image(img, name="cell", colormap="gray")
        self.v.add_image(ol, name="outline", colormap="yellow", blending="additive", opacity=0.35)
        self.pts = self.v.add_points(np.zeros((0, 2)), name="click", size=5, face_color="red", border_color="white")
        for lyr in self.v.layers:
            for k, fn in self.keys.items():
                lyr.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
            try:
                lyr.mouse_pan = True
            except Exception:
                pass
        prev = self.ans[(self.ans.field == r.field) & (self.ans.cell == r.cell)]
        self.click = None
        if len(prev) and prev.answer.iloc[0] == "clicked":
            y, x = prev.y.iloc[0] - self.origin[0], prev.x.iloc[0] - self.origin[1]
            if prev.panel.iloc[0] == "bundle": x += 2 * HALF + GAP
            self.click = (y, x); self.pts.data = np.array([[y, x]])
        self.v.reset_view(); self.status()

    def status(self, msg=""):
        r = self.cells.iloc[self.k]; n = len(self.ans)
        self.v.text_overlay.text = (f"Cell {self.k + 1} of {len(self.cells)}  ({n} answered)   field {r.field}, {r.row}\n"
                                    "Left: flat top of the cell; click the small dark spot (notch).\n"
                                    "Right: bundle 1.8-3.6 um higher; or click the point where the two arms of the V meet (not inside the V).\n"
                                    "One click in either view  |  Enter/Space next  n cannot tell  Left previous  b outline\n" + msg)

    def on_mouse(self, viewer, event):
        start = np.array(event.position[-2:]); yield
        moved = False
        while event.type == "mouse_move":
            if np.linalg.norm(np.array(event.position[-2:]) - start) > 3:
                moved = True
            yield
        if moved or event.button != 1:
            return
        y, x = start
        if not (0 <= y < 2 * HALF) or not (0 <= x < 4 * HALF + GAP) or (2 * HALF <= x < 2 * HALF + GAP):
            return self.status("Click inside one of the two views")
        self.click = (float(y), float(x)); self.pts.data = np.array([[y, x]]); self.status("Clicked; Enter for next")

    def record(self, answer):
        r = self.cells.iloc[self.k]
        rec = dict(field=r.field, cell=int(r.cell), row=r.row, answer=answer, panel="", x=np.nan, y=np.nan, saved=central_time())
        if answer == "clicked":
            y, x = self.click; panel = "flat top" if x < 2 * HALF else "bundle"
            if panel == "bundle": x -= 2 * HALF + GAP
            rec.update(panel=panel, x=round(x + self.origin[1], 1), y=round(y + self.origin[0], 1))
        self.ans = pd.concat([self.ans[~((self.ans.field == r.field) & (self.ans.cell == r.cell))], pd.DataFrame([rec])], ignore_index=True)
        self.ans.to_csv(CLICKS, index=False)

    def advance(self):
        if self.k < len(self.cells) - 1:
            self.k += 1; self.load()
        else:
            self.status("Last cell done; you can close the window.")

    def next(self, viewer):
        if self.click is None:
            return self.status("No click yet: click the notch or V tip, or press n")
        self.record("clicked"); self.advance()

    def cant_tell(self, viewer):
        self.record("cannot tell"); self.advance()

    def prev(self, viewer):
        if self.k > 0:
            self.k -= 1; self.load()

    def toggle(self, viewer):
        self.v.layers["outline"].visible = not self.v.layers["outline"].visible


def compare(match_um=2.5):
    """Hand-clicked bundle directions against the notch directions of the current run (S.RESULTS\\spatial).
    The clicks name cells in the frozen click outlines (CELLS); each clicked cell is matched to the current run's cell
    whose center lies nearest its center, within match_um. Matching by position keeps the check valid when the
    run's outlines are renumbered (hand-corrected outlines, a rebuilt run)."""
    from measure_bundles import local_frame
    from spatial_stats import smooth_row_frame, ROWS
    a = pd.read_csv(CLICKS); c = pd.read_csv(SPATIAL); rows = []
    for f, g in a.groupby("field"):
        lab = np.load(os.path.join(CELLS, f + ".npy")); types = pd.read_csv(os.path.join(TYPES, f + "_types.csv"))
        types = smooth_row_frame(types[types.row.isin(ROWS)].copy())       # same row frame as spatial_stats.py
        cf = c[c.field == f].set_index("cell"); tx, ty = types.x.values.astype(float), types.y.values.astype(float)
        for _, r in g.iterrows():
            rec = dict(field=f, cell=r.cell, row=r.row, answer=r.answer, panel=r.panel)
            ys, xs = np.nonzero(lab == r.cell)
            if not len(xs):
                rec.update(matched=False); rows.append(rec); continue
            cx, cy = xs.mean(), ys.mean(); dist = np.hypot(tx - cx, ty - cy) * PX; k = int(np.argmin(dist))
            if dist[k] > match_um or types.cell.iloc[k] not in cf.index:
                rec.update(matched=False, match_um=float(dist[k])); rows.append(rec); continue
            t = types.iloc[k]; s = cf.loc[t.cell]
            rec.update(matched=True, run_cell=int(t.cell), match_um=float(dist[k]), notch_deg=s.notch_deg,
                       notch_dev_deg=s.dev_deg, flagged=abs(s.dev_deg) > 45)
            if r.answer == "clicked":
                along, across = local_frame(types, (t.x, t.y), t.row); vec = np.array([r.x - cx, r.y - cy])
                hand = float(np.degrees(np.arctan2(vec @ along, vec @ across)))
                rowmean = s.notch_deg - s.dev_deg
                rec.update(hand_deg=hand, difference_deg=abs((s.notch_deg - hand + 180) % 360 - 180),
                           hand_dev_deg=(hand - rowmean + 180) % 360 - 180)
            rows.append(rec)
    d = pd.DataFrame(rows); d["hand_over_45"] = d.hand_dev_deg.abs() > 45
    os.makedirs(os.path.join(S.RESULTS, "spatial"), exist_ok=True)
    d.to_csv(os.path.join(S.RESULTS, "spatial", "direction_check.csv"), index=False)
    print(f"clicked cells matched to the run's cells within {match_um} um: {int(d.matched.sum())} of {len(d)}")
    k = d[(d.answer == "clicked") & d.matched]
    print(f"answered {len(d)}, clicked {len(k)}, cannot tell {(d.answer == 'cannot tell').sum()}")
    for f, g in [("all fields", k)] + list(k.groupby("field")):
        print(f"{f}: notch within 45 degrees of the click {np.mean(g.difference_deg < 45):.2f}, within 90 {np.mean(g.difference_deg < 90):.2f}, "
              f"median difference {g.difference_deg.median():.0f} degrees (n {len(g)})")
    for fl, g in k.groupby("flagged"):
        print(f"{'flagged' if fl else 'not flagged'} by the notch measure: hand direction also more than 45 degrees off the row in "
              f"{g.hand_over_45.mean():.2f} of {len(g)}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        return compare()
    t = Tool(sample_cells())
    t.napari.run()


if __name__ == "__main__":
    sys.exit(main())
