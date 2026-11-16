"""Nucleus outlines: draw the outline of the centred nucleus on the plane where it looks widest.

Each item is a small DAPI stack (14 x 14 um, planes 0.6 um apart, flattened along the tissue surface).
Use the slider under the image to move between planes; the window opens on the centre plane.
Draw: hold the left mouse button and trace around the nucleus (the line closes when you release).
Keys:  Enter = save this outline and go to the next nucleus      u = cannot tell
       Backspace = back one nucleus        Delete = remove the outline you drew (to redraw)
Outlines save to E:\\Webber_confocal\\annotations\\nucleus_outlines\\answers.csv; reopening continues where you stopped.

Run (imaging env):  python outline_nuclei.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd, napari

DIR = os.path.join(S.ANNOTATIONS, 'nucleus_outlines'); PX = 0.2166; DZ = 0.6


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def area_um2(v):
    y, x = v[:, -2] * PX, v[:, -1] * PX; return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


class Outline:
    def __init__(self):
        self.items = pd.read_csv(os.path.join(DIR, "items.csv")); self.st = np.load(os.path.join(DIR, "stacks.npz"))
        self.out = os.path.join(DIR, "answers.csv")
        self.res = pd.read_csv(self.out) if os.path.exists(self.out) else pd.DataFrame(columns=["item", "status", "plane", "area_um2", "vertices", "saved"])
        done = set(self.res["item"]); todo = [k for k, it in enumerate(self.items["item"]) if it not in done]
        self.k = todo[0] if todo else len(self.items) - 1
        self.v = napari.Viewer(title="Nucleus outlines")
        self.keys = {"Enter": lambda v: self.save(), "u": lambda v: self.save("cannot tell"), "Backspace": lambda v: self.back(), "Delete": lambda v: self.clear()}
        for k, fn in self.keys.items(): self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 14; self.show()

    def show(self, msg=""):
        it = self.items.iloc[self.k]; S = self.st[f"item{int(it['item'])}"]
        for lyr in list(self.v.layers): self.v.layers.remove(lyr)
        lo, hi = np.percentile(S, [1, 99.8]); self.v.add_image(np.clip((S - lo) / max(hi - lo, 1e-6), 0, 1), name="DAPI", colormap="gray")
        self.v.add_points(np.array([[int(it.plane_centre), int(it.cy_c), int(it.cx_c)]]), name="centre", size=2, face_color="red", symbol="cross")
        self.sh = self.v.add_shapes(ndim=3, name="outline", edge_color="yellow", face_color="transparent", edge_width=0.6)
        for l in self.v.layers:
            for k, fn in self.keys.items(): l.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
        self.v.dims.set_point(0, int(it.plane_centre)); self.v.layers.selection.active = self.sh
        try: self.sh.mode = "add_polygon_lasso"
        except Exception: self.sh.mode = "add_polygon"
        self.v.reset_view()
        self.v.text_overlay.text = (f"Nucleus {self.k + 1} of {len(self.items)}   saved {len(self.res)}   {msg}\n"
                                    "Outline the nucleus under the red cross, on the plane where it looks widest (slider below).\n"
                                    "Enter save and next   u cannot tell   Delete redraw   Backspace back")

    def clear(self):
        self.sh.data = []
        try: self.sh.mode = "add_polygon_lasso"
        except Exception: self.sh.mode = "add_polygon"

    def save(self, status="outlined"):
        it = int(self.items.iloc[self.k]["item"]); row = dict(item=it, status=status, plane=np.nan, area_um2=np.nan, vertices=0, saved=central_time())
        if status == "outlined":
            if len(self.sh.data) == 0: self.v.text_overlay.text = "No outline drawn yet: trace the nucleus, or press u if you cannot tell."; return
            v = np.asarray(self.sh.data[-1]); row.update(plane=int(round(v[:, 0].mean())), area_um2=area_um2(v), vertices=len(v))
            np.save(os.path.join(DIR, f"outline_{it}.npy"), v)
        self.res = pd.concat([self.res[self.res["item"] != it], pd.DataFrame([row])], ignore_index=True); self.res.to_csv(self.out, index=False)
        if self.k < len(self.items) - 1: self.k += 1; self.show(f"saved: {status}")
        else: self.v.text_overlay.text = "All nuclei done; you can close the window."

    def back(self):
        if self.k > 0: self.k -= 1; self.show("back one nucleus")


if __name__ == "__main__":
    Outline(); napari.run()
