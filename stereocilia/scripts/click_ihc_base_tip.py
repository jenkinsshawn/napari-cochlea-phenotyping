"""Two clicks per standing inner hair cell bundle on a side view: its base and the tip of its longest stereocilia.

Each cell opens from the side, across the row (height up; outer hair cells to the right; brightest phalloidin
within 4 um of the cell along the row). Click:
  1. where the stereocilia leave the cell (base)      2. the far end of the longest stereocilia (its tip)
After the second click the cell is saved and the next one opens. The last two cells repeat the first two.

Keys:  u  cannot tell (this cell is skipped)    Backspace  undo the last click (or go back a cell)
Drag to pan, scroll to zoom; a click that moves more than a few pixels is a pan, not a mark.

Input:  E:\\Webber_confocal\\annotations\\ihc_base_tip\\sample.csv (item, field, cell; cut-outs in work\\ihc_stacks_v2)
Output: E:\\Webber_confocal\\annotations\\ihc_base_tip\\clicks.csv (one row per cell; heights in um, nominal)
Saved cells are skipped when the tool is restarted.

Run from the scripts folder:
    python click_ihc_base_tip.py          (side view across the row)
    python click_ihc_base_tip.py --face   (face-on view along the row, cell between the yellow lines; saved to clicks_face.csv)
    add --test to either line for the held-out set of 18 new cells (annotations\\ihc_base_tip_test)
    add --test3 for the third held-out set of 27 new cells (annotations\\ihc_base_tip_test3), face-on view
    add --test4 for the fourth held-out set of 35 new cells (annotations\\ihc_base_tip_test4, from make_ihc_height_sample.py)
    add --repeat for 12 already-clicked cells shown again in a new order (annotations\\ihc_base_tip_repeat)
    add --ohc1 for first-row outer hair cells (annotations\\ohc1_base_tip; cut-outs from ohc1_stacks.py)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import sys
import napari
FACE = "--face" in sys.argv     # face-on view: along the row (height up, row left to right), saved to clicks_face.csv
ACROSS_UM = (-1.0, 8.0)          # band across the row shown in the face-on view (the bundle side of the cell)

import click_outlines as CO   # sample cut-out numbers to the current run's
STACKS = os.path.join(S.WORK, 'ihc_stacks_v2')
DIR = os.path.join(S.ANNOTATIONS, 'ihc_base_tip') + ("_test4" if "--test4" in sys.argv else "_test3" if "--test3" in sys.argv else "_test2" if "--test2" in sys.argv else "_test" if "--test" in sys.argv else "_repeat" if "--repeat" in sys.argv else "")   # --test: held-out set of 18 new cells; --repeat: 12 cells clicked again
PX, DZ = 0.1083, 0.3
SIDE_HALF_UM = 4.0
CELL_HALF_UM = 4.3
if "--ohc1" in sys.argv:   # first-row outer hair cells (ohc1_stacks.py cut-outs, 12 um square)
    STACKS = os.path.join(S.WORK, 'ohc1_stacks'); DIR = os.path.join(S.ANNOTATIONS, 'ohc1_base_tip')
    CELL_HALF_UM = 3.5; ACROSS_UM = (-3.0, 3.0); SIDE_HALF_UM = 3.0
    if "--relook" in sys.argv:
        DIR = DIR + "_relook"   # second look at the first-row bundles marked cannot tell


PROMPTS = ["Click the LEFT end of the bundle", "Click the RIGHT end of the bundle",
           "Click where the stereocilia LEAVE THE CELL (base)", "Click the FAR END of the longest stereocilia (its tip, even if folded over)"]
COLORS = ["cyan", "cyan", "lime", "magenta"]


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


class Tool:
    def __init__(self):
        self.items = pd.read_csv(os.path.join(DIR, "sample.csv"))
        self.out = os.path.join(DIR, "clicks_face.csv" if FACE else "clicks.csv")
        self.res = pd.read_csv(self.out) if os.path.exists(self.out) else pd.DataFrame()
        done = set(self.res["item"]) if len(self.res) else set()
        todo = [k for k, it in enumerate(self.items["item"]) if it not in done]
        self.k = todo[0] if todo else len(self.items) - 1
        self.cache = {}
        self.v = napari.Viewer(title="Inner hair cell bundles: base and tip")
        self.v.mouse_drag_callbacks.append(self.on_mouse)
        self.keys = {"u": self.cannot_tell, "Backspace": self.undo}
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 16
        self.open_cell()

    def volume(self, field, cell, xy=None):
        if field not in self.cache:
            z = np.load(os.path.join(STACKS, field + ".npz")); self.cache = {field: (z["phal"], z["offsets_um"])}
        k = CO.cutout_number(os.path.basename(STACKS), field, cell, xy=xy)   # by position: stored, or first-run cut-out
        if k is None:
            raise ValueError(f"{field} cell {cell}: no cut-out at this position in {STACKS}")
        ph, offs = self.cache[field]; return ph[k - 1].astype(np.float32), offs

    def open_cell(self, msg=""):
        it = self.items.iloc[self.k]
        try:
            self.vol, self.offs = self.volume(it["field"], it["cell"], (it["x"], it["y"]) if "x" in it.index else None)
        except ValueError as ex:   # the current run has no cut-out of this cell: say so; u skips it
            self.vol, self.offs = np.zeros((2, 40, 40), np.float32), np.zeros(2); msg = f"{ex}. Press u to skip."
        self.clicks = [(np.nan, np.nan), (np.nan, np.nan)]
        self.show(msg)

    def show(self, msg=""):
        step = len(self.clicks); nz, ny, nx = self.vol.shape; c = nx // 2
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        if step < 2:
            self.v.add_image(scaled(self.vol.max(0)), name="from above", colormap="gray")
            g = [np.array([[0, c + s * CELL_HALF_UM / PX], [ny, c + s * CELL_HALF_UM / PX]]) for s in (-1, 1)]
            self.v.add_shapes(g, shape_type="line", edge_color="yellow", edge_width=1, name="cell edges", opacity=0.8)
            pts = [p for p in self.clicks]
        else:
            if FACE:
                y0, y1 = ny // 2 + int(ACROSS_UM[0] / PX), ny // 2 + int(ACROSS_UM[1] / PX)
                side = self.vol[:, y0:y1, :].max(1)[::-1]                                     # height up, row left to right
                self.v.add_image(scaled(side), name="face on, along the row", colormap="gray", scale=(DZ / PX, 1))
                g = [np.array([[0, c + s_ * CELL_HALF_UM / PX], [nz * DZ / PX, c + s_ * CELL_HALF_UM / PX]]) for s_ in (-1, 1)]
                self.v.add_shapes(g, shape_type="line", edge_color="yellow", edge_width=1, name="cell edges", opacity=0.8)
            else:
                a = int(SIDE_HALF_UM / PX); side = self.vol[:, :, c - a:c + a].max(2)[::-1]       # height up
                self.v.add_image(scaled(side), name="side, across the row", colormap="gray", scale=(DZ / PX, 1))
            pts = [p for p in self.clicks[2:]]
        lyr = self.v.add_points(np.array([q[:2] for q in pts]) if pts else np.zeros((0, 2)), name="clicks", size=7,
                                face_color=[COLORS[i + (0 if step < 2 else 2)] for i in range(len(pts))] if pts else "white")
        for l in self.v.layers:
            for k, fn in self.keys.items():
                l.bind_key(k, (lambda g_: (lambda layer: g_(self.v)))(fn), overwrite=True)
        self.v.reset_view()
        n_done = len(self.res) if len(self.res) else 0
        self.v.text_overlay.text = (f"Cell {self.k + 1} of {len(self.items)}   (saved {n_done})\n{msg}\n"
                                    f"STEP {step - 1} of 2:  {PROMPTS[step]}\n"
                                    "u cannot tell this cell    Backspace undo last click")

    def on_mouse(self, viewer, event):
        start = np.array(event.position[-2:]); yield
        moved = False
        while event.type == "mouse_move":
            if np.linalg.norm(np.array(event.position[-2:]) - start) > 3:
                moved = True
            yield
        if moved or event.button != 1:
            return
        self.clicks.append((float(start[0]), float(start[1])))
        if len(self.clicks) == 4:
            self.save(); return self.advance("Saved")
        self.show()

    def save(self, status="clicked"):
        it = self.items.iloc[self.k]; nz = self.vol.shape[0]; row = dict(item=int(it["item"]), status=status, saved=central_time())
        if status == "clicked":
            (y1, x1), (y2, x2), (rb, yb), (rt, yt) = self.clicks
            f = DZ / PX
            def height(r_world):                     # side view: world row -> plane (flipped, height up) -> height above the field surface
                return float(self.offs[0] + (nz - 1 - r_world / f) * DZ)
            row.update(left_along_um=x1 * PX, left_across_um=y1 * PX, right_along_um=x2 * PX, right_across_um=y2 * PX,
                       base_height_um=height(rb), tip_height_um=height(rt), view="face" if FACE else "side",
                       **({"base_along_um": yb * PX, "tip_along_um": yt * PX} if FACE else {"base_across_um": yb * PX, "tip_across_um": yt * PX}))
        self.res = pd.concat([self.res[self.res["item"] != row["item"]] if len(self.res) else self.res, pd.DataFrame([row])], ignore_index=True)
        self.res.to_csv(self.out, index=False)

    def advance(self, msg):
        if self.k < len(self.items) - 1:
            self.k += 1; self.open_cell(msg)
        else:
            self.clicks = [(np.nan, np.nan), (np.nan, np.nan)]; self.v.text_overlay.text = msg + ".  All cells done; you can close the window."

    def cannot_tell(self, viewer):
        self.save("cannot tell"); self.advance("Marked: cannot tell")

    def undo(self, viewer):
        if len(self.clicks) > 2:
            self.clicks.pop(); self.show("Undid the last click")
        elif self.k > 0:
            self.k -= 1; self.open_cell("Back one cell")


if __name__ == "__main__":
    Tool(); napari.run()
