"""Four clicks per inner hair cell bundle: its two ends from above, then its base and tip from the side.

Each cell opens from above (brightest phalloidin at any height; the row runs left to right; the cell
is between the yellow lines). The prompt at the top says what to click:
  1. the left end of the bundle      2. the right end of the bundle
Then the view changes to the side, across the row (height up; outer hair cells to the right):
  3. where the stereocilia leave the cell      4. the far end of the longest stereocilia (its tip),
     wherever it points: up, leaning, or folded over by the mount
After the fourth click the cell is saved and the next one opens.

Keys:  u  cannot tell (this cell is skipped)    Backspace  undo the last click (or go back a cell)
Drag to pan, scroll to zoom; a click that moves more than a few pixels is a pan, not a mark.

Input:  E:\\Webber_confocal\\annotations\\ihc_clicks\\sample.csv (item, field, cell)
Output: E:\\Webber_confocal\\annotations\\ihc_clicks\\clicks.csv (one row per cell; positions in um)
Saved cells are skipped when the tool is restarted.

Run from the scripts folder:
    python click_ihc_bundles.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import napari

import click_outlines as CO   # sample cut-out numbers to the current run's
STACKS = os.path.join(S.WORK, 'ihc_stacks')
DIR = os.path.join(S.ANNOTATIONS, 'ihc_clicks')
PX, DZ = 0.1083, 0.3
SIDE_HALF_UM = 4.0
CELL_HALF_UM = 4.3
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
        self.out = os.path.join(DIR, "clicks.csv")
        self.res = pd.read_csv(self.out) if os.path.exists(self.out) else pd.DataFrame()
        done = set(self.res["item"]) if len(self.res) else set()
        todo = [k for k, it in enumerate(self.items["item"]) if it not in done]
        self.k = todo[0] if todo else len(self.items) - 1
        self.cache = {}
        self.v = napari.Viewer(title="Inner hair cell bundles: four clicks")
        self.v.mouse_drag_callbacks.append(self.on_mouse)
        self.keys = {"u": self.cannot_tell, "Backspace": self.undo}
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 16
        self.open_cell()

    def volume(self, field, cell):
        if field not in self.cache:
            z = np.load(os.path.join(STACKS, field + ".npz")); self.cache = {field: (z["phal"], z["offsets_um"])}
        k = CO.cutout_number(os.path.basename(STACKS), field, cell)   # saved samples name first-run cut-outs
        if k is None:
            raise ValueError(f"{field} cell {cell}: no cut-out at this position in {STACKS}")
        ph, offs = self.cache[field]; return ph[k - 1].astype(np.float32), offs

    def open_cell(self, msg=""):
        it = self.items.iloc[self.k]
        try:
            self.vol, self.offs = self.volume(it["field"], it["cell"])
        except ValueError as ex:   # the current run has no cut-out of this cell: say so; u skips it
            self.vol, self.offs = np.zeros((2, 40, 40), np.float32), np.zeros(2); msg = f"{ex}. Press u to skip."
        self.clicks = []
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
                                    f"STEP {step + 1} of 4:  {PROMPTS[step]}\n"
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
                       base_height_um=height(rb), base_across_um=yb * PX, tip_height_um=height(rt), tip_across_um=yt * PX)
        self.res = pd.concat([self.res[self.res["item"] != row["item"]] if len(self.res) else self.res, pd.DataFrame([row])], ignore_index=True)
        self.res.to_csv(self.out, index=False)

    def advance(self, msg):
        if self.k < len(self.items) - 1:
            self.k += 1; self.open_cell(msg)
        else:
            self.clicks = []; self.v.text_overlay.text = msg + ".  All cells done; you can close the window."

    def cannot_tell(self, viewer):
        self.save("cannot tell"); self.advance("Marked: cannot tell")

    def undo(self, viewer):
        if self.clicks:
            self.clicks.pop(); self.show("Undid the last click")
        elif self.k > 0:
            self.k -= 1; self.open_cell("Back one cell")


if __name__ == "__main__":
    Tool(); napari.run()
