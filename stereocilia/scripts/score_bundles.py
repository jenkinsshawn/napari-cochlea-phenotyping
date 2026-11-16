"""Blind score of stereocilia bundle organization, one bundle at a time (genotype and field are not shown).

Each item is one bundle, ringed in white, shown from above as the brightest phalloidin over the planes 0.3 to 6 um
above the cell surface. Press p to step through the single planes instead (use the slider), p again to go back.

Keys:
  1   orderly bundle (stereocilia in a regular, continuous row or V)
  2   mildly disorganized (gaps, uneven stereocilia, a kink, part of the bundle out of line)
  3   disorganized (fragmented, split, scattered or shapeless)
  u   cannot tell (out of focus, cut off, covered)
  Backspace   back one item (to change an answer)
  c   Calb2 on/off

Input:  E:\\Webber_confocal\\annotations\\bundle_scores\\items_<set>.csv (item, field, x, y)
Output: E:\\Webber_confocal\\annotations\\bundle_scores\\answers_<set>.csv (item, answer, saved)
Answered items are skipped when the tool is restarted.

Run from the scripts folder:
    python score_bundles.py --set organization
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
DIR = os.path.join(S.ANNOTATIONS, 'bundle_scores')
PAINT = os.path.join(S.WORK, 'paint_stacks')
ANSWERS = {"1": "orderly", "2": "mildly disorganized", "3": "disorganized", "u": "cannot tell"}
RING_UM = 3.5
VIEW_UM = 22.0


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
        self.v = napari.Viewer(title="Bundle organization")
        self.keys = {k: self.answer(a) for k, a in ANSWERS.items()}
        self.keys.update({"Backspace": self.back, "c": self.toggle_calb2, "p": self.toggle_planes})
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 16
        self.show()

    def load_field(self, f):
        z = np.load(os.path.join(FIELDS, f + ".npz")); st = [str(s) for s in z["stains"]]
        self.px = float(z["voxel_um"][2])
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        pz = np.load(os.path.join(PAINT, f + ".npz")); off = pz["offsets_um"]; sel = (off >= 0.3) & (off <= 6.0)
        planes = pz["phal"][sel].astype(np.float32)
        self.v.add_image(scaled(planes.max(axis=0)), name="phalloidin", colormap="gray", blending="additive")
        self.v.add_image(scaled(planes), name="planes", colormap="gray", blending="additive", visible=False)
        self.v.add_image(scaled(z["maxproj"][st.index("Calb2")]), name="Calb2", colormap="magenta", blending="additive", opacity=0.45, visible=False)
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
        self.v.text_overlay.text = (f"Bundle {self.k + 1} of {len(self.items)}   answered {n_done}{given}\n{msg}\n"
                                    "1 orderly  2 mildly disorganized  3 disorganized  u cannot tell  |  Backspace back  p planes  c Calb2")

    def answer(self, a):
        def f(viewer):
            it = self.items.iloc[self.k]
            self.ans = pd.concat([self.ans[self.ans.item != it["item"]],
                                  pd.DataFrame([dict(item=int(it["item"]), answer=a, saved=central_time())])], ignore_index=True)
            self.ans.to_csv(self.out, index=False)
            if self.k < len(self.items) - 1:
                self.k += 1; self.show(f"Saved: {a}")
            else:
                self.show(f"Saved: {a}.  All bundles answered; you can close the window.")
        return f

    def back(self, viewer):
        if self.k > 0:
            self.k -= 1; self.show("Back one bundle")

    def toggle_planes(self, viewer):
        on = not self.v.layers["planes"].visible
        self.v.layers["planes"].visible = on; self.v.layers["phalloidin"].visible = not on

    def toggle_calb2(self, viewer):
        self.v.layers["Calb2"].visible = not self.v.layers["Calb2"].visible


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="organization")
    Tool(ap.parse_args().set); napari.run()
