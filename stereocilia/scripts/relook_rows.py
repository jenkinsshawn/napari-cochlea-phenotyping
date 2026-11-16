"""Blind second look at single cells: is the ringed cell a hair cell, and in which row?

Each item is one cell, marked by a white ring, in the same view as click_cells.py (phalloidin
surface layer in gray, Calb2 in magenta). The tool does not show earlier clicks or any automatic
answer. One key per item; the answer is saved at once and the next item opens.

Keys:
  i   inner hair cell
  1   first outer row      2   second outer row      3   third outer row
  4   outer hair cell beyond the third row
  n   not a hair cell (or no cell there)
  u   cannot tell
  Backspace   back one item (to change an answer)
  c   Calb2 on/off

Input:  E:\\Webber_confocal\\annotations\\row_relook\\items_<set>.csv (item, field, x, y)
Output: E:\\Webber_confocal\\annotations\\row_relook\\answers_<set>.csv (item, answer, saved)
Answered items are skipped when the tool is restarted.

Run from the scripts folder:
    python relook_rows.py              (the tuning set, 105 cells)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import napari

FIELDS = os.path.join(S.WORK, 'fields')
DIR = os.path.join(S.ANNOTATIONS, 'row_relook')
ANSWERS = {"i": "IHC", "1": "OHC1", "2": "OHC2", "3": "OHC3", "4": "OHC beyond row 3", "n": "not a hair cell", "u": "cannot tell"}
RING_UM = 5.0
VIEW_UM = 90.0


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


class Tool:
    def __init__(self, name):
        self.items = pd.read_csv(os.path.join(DIR, f"items_{name}.csv"))
        self.out = os.path.join(DIR, f"answers_{name}.csv")
        self.ans = pd.read_csv(self.out) if os.path.exists(self.out) else pd.DataFrame(columns=["item", "answer", "saved"])
        done = set(self.ans.item)
        todo = [k for k, it in enumerate(self.items.item) if it not in done]
        self.k = todo[0] if todo else len(self.items) - 1
        self.field = None
        self.v = napari.Viewer(title="Second look at single cells")
        self.keys = {k: self.answer(a) for k, a in ANSWERS.items()}
        self.keys.update({"Backspace": self.back, "c": self.toggle_calb2})
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 16
        self.show()

    def load_field(self, f):
        z = np.load(os.path.join(FIELDS, f + ".npz")); st = [str(s) for s in z["stains"]]
        self.px = float(z["voxel_um"][2])
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.v.add_image(scaled(z["surface_layers"][st.index("Phalloidin")]), name="phalloidin", colormap="gray", blending="additive")
        self.v.add_image(scaled(z["maxproj"][st.index("Calb2")]), name="Calb2", colormap="magenta", blending="additive", opacity=0.45)
        self.ring = self.v.add_shapes(name="ring", edge_color="white", face_color="transparent", edge_width=2)
        for lyr in self.v.layers:                                             # a selected layer's keys win over the viewer's
            for k, fn in self.keys.items():
                lyr.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
        self.v.reset_view(); self.zoom0 = self.v.camera.zoom; self.field = f

    def show(self, msg=""):
        it = self.items.iloc[self.k]
        if it.field != self.field:
            self.load_field(it.field)
        r = RING_UM / self.px
        self.ring.data = []; self.ring.add_ellipses(np.array([[it.y, it.x], [r, r]]), edge_color="white", face_color="transparent", edge_width=2)
        self.v.camera.zoom = self.zoom0 * (1024 * self.px) / VIEW_UM
        self.v.camera.center = (float(it.y), float(it.x))
        prev = self.ans[self.ans.item == it["item"]]
        given = f"   (answered: {prev.answer.iloc[0]})" if len(prev) else ""
        n_done = self.ans.item.nunique()
        self.v.text_overlay.text = (f"Cell {self.k + 1} of {len(self.items)}   answered {n_done}{given}\n{msg}\n"
                                    "i inner  1/2/3 outer rows  4 beyond row 3  n not a hair cell  u cannot tell  |  Backspace back  c Calb2")

    def answer(self, a):
        def f(viewer):
            it = self.items.iloc[self.k]
            self.ans = pd.concat([self.ans[self.ans.item != it["item"]],
                                  pd.DataFrame([dict(item=int(it["item"]), answer=a, saved=central_time())])], ignore_index=True)
            self.ans.to_csv(self.out, index=False)
            if self.k < len(self.items) - 1:
                self.k += 1; self.show(f"Saved: {a}")
            else:
                self.show(f"Saved: {a}.  All cells answered; you can close the window.")
        return f

    def back(self, viewer):
        if self.k > 0:
            self.k -= 1; self.show("Back one cell")

    def toggle_calb2(self, viewer):
        self.v.layers["Calb2"].visible = not self.v.layers["Calb2"].visible


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="tune")
    Tool(ap.parse_args().set); napari.run()
