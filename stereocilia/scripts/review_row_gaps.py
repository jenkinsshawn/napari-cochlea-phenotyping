"""Review of gaps in the hair cell rows: is a hair cell missing between the two ringed cells?

Shows each gap from above: phalloidin at the surface (gray) with DAPI (magenta); Calb2 can be shown instead.
The two ringed cells are the neighbors on either side of the gap, joined by a red line.
Keys:  1 = missing cell (no hair cell between the rings)    2 = a hair cell is there (gap is a cell nobody marked)
       3 = not a gap (the rings are ordinary neighbors)     r = the two rings are in different rows
       u = cannot tell
       d = DAPI on/off    c = Calb2 on/off    Backspace = back one gap
Answers save after every key to E:\\Webber_confocal\\annotations\\row_gaps\\answers.csv; reopening continues where you stopped.

Run (imaging env):  python review_row_gaps.py   (add --relook for the second look at possible row jumps)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd, napari

import sys
DIR = os.path.join(S.ANNOTATIONS, 'row_gaps') + ("_relook" if "--relook" in sys.argv else ""); FIELDS = os.path.join(S.WORK, 'fields')
HALF = 230          # pixels on each side of the gap centre (about 25 um)
ROWNAME = {"IHC": "inner hair cell row", "OHC1": "outer row 1", "OHC2": "outer row 2", "OHC3": "outer row 3"}


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.7]); return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


class Review:
    def __init__(self):
        self.items = pd.read_csv(os.path.join(DIR, "items.csv")); self.out = os.path.join(DIR, "answers.csv")
        self.res = pd.read_csv(self.out) if os.path.exists(self.out) else pd.DataFrame(columns=["item", "answer", "saved"])
        done = set(self.res["item"]); todo = [k for k, it in enumerate(self.items["item"]) if it not in done]
        self.k = todo[0] if todo else len(self.items) - 1; self.cache = {}
        self.v = napari.Viewer(title="Hair cell row gaps")
        keys = {"1": lambda v: self.answer("missing cell"), "2": lambda v: self.answer("cell present"),
                "3": lambda v: self.answer("not a gap"), "r": lambda v: self.answer("different rows"), "u": lambda v: self.answer("cannot tell"),
                "d": lambda v: self.toggle("DAPI"), "c": lambda v: self.toggle("Calb2"), "Backspace": lambda v: self.back()}
        self.keys = keys
        for k, fn in keys.items(): self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 14; self.show()

    def field(self, f):
        if f not in self.cache:
            z = np.load(os.path.join(FIELDS, f + ".npz"), allow_pickle=True); st = list(z["stains"])
            self.cache = {f: dict(phal=z["surface_layers"][st.index("Phalloidin")], dapi=z["maxproj"][st.index("DAPI")],
                                  calb2=z["maxproj"][st.index("Calb2")] if "Calb2" in st else None)}
        return self.cache[f]

    def show(self, msg=""):
        it = self.items.iloc[self.k]; F = self.field(it["field"])
        cx, cy = (it.xa + it.xb) / 2, (it.ya + it.yb) / 2; ny, nx = F["phal"].shape
        x0, y0 = int(max(cx - HALF, 0)), int(max(cy - HALF, 0)); x1, y1 = int(min(cx + HALF, nx)), int(min(cy + HALF, ny))
        for lyr in list(self.v.layers): self.v.layers.remove(lyr)
        self.v.add_image(scaled(F["phal"][y0:y1, x0:x1]), name="phalloidin", colormap="gray")
        self.v.add_image(scaled(F["dapi"][y0:y1, x0:x1]), name="DAPI", colormap="magenta", blending="additive", opacity=0.6)
        if F["calb2"] is not None:
            self.v.add_image(scaled(F["calb2"][y0:y1, x0:x1]), name="Calb2", colormap="green", blending="additive", visible=False)
        a = np.array([[it.ya - y0, it.xa - x0], [it.yb - y0, it.xb - x0]])
        self.v.add_shapes([a], shape_type="line", edge_color="red", edge_width=1.5, name="gap", opacity=0.8)
        self.v.add_points(a, name="neighbors", size=22, face_color="transparent", border_color="yellow", border_width=0.12)
        for l in self.v.layers:
            for k, fn in self.keys.items(): l.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
        self.v.reset_view()
        self.v.text_overlay.text = (f"Gap {self.k + 1} of {len(self.items)}  ({ROWNAME.get(it.row, it.row)})   saved {len(self.res)}   {msg}\n"
                                    "1 missing cell   2 a cell is there   3 not a gap   r rings are in different rows   u cannot tell\n"
                                    "d DAPI on/off   c Calb2 on/off   Backspace back")

    def toggle(self, name):
        if name in self.v.layers: self.v.layers[name].visible = not self.v.layers[name].visible

    def answer(self, a):
        it = int(self.items.iloc[self.k]["item"])
        self.res = pd.concat([self.res[self.res["item"] != it], pd.DataFrame([dict(item=it, answer=a, saved=central_time())])], ignore_index=True)
        self.res.to_csv(self.out, index=False)
        if self.k < len(self.items) - 1: self.k += 1; self.show(f"saved: {a}")
        else: self.v.text_overlay.text = "All gaps done; you can close the window."

    def back(self):
        if self.k > 0: self.k -= 1; self.show("back one gap")


if __name__ == "__main__":
    Review(); napari.run()
