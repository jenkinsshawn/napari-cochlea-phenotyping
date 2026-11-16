"""Napari tool for putting a point on every inner and outer hair cell, by row.

Each field opens at the cell surface: phalloidin in gray, Calb2 in magenta (inner hair cell
row), outlines faint. Fields start with no points, and the marking key starts at 1 (first
outer row) in every field; the current key is shown at the top left.

Keys choose what the next clicks mark:
    i   inner hair cell (magenta)
    1   outer hair cell, first row (next to the band) (green)
    2   outer hair cell, second row (blue)
    3   outer hair cell, third row (yellow)
    e   extra hair cell outside the usual rows (red)
Mouse: left-click puts a point; right-click removes the nearest point; drag pans; wheel zooms.
Other keys:
    c   show or hide Calb2        b   show or hide outlines
    Enter   field finished: save, mark done, open the next field
    .  ,    next / previous field (saves first)
Points save on every click to E:\\Webber_confocal\\annotations\\cell_points\\<field>_points.csv,
and finished fields are listed in fields_done.csv there.

Run from the scripts folder:
    python click_cells.py                 (the six corrected fields first, then the rest)
    python click_cells.py --start AR47    (open at the first field whose name contains AR47)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, glob, json, os, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import napari

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = os.path.join(S.WORK, 'fields')
CELLS = os.path.join(S.WORK, 'cells')
CORR = os.path.join(S.ANNOTATIONS, 'cells')
OUT = os.path.join(S.ANNOTATIONS, 'cell_points')
CLASSES = {"i": ("inner", "magenta"), "1": ("OHC1", "lime"), "2": ("OHC2", "deepskyblue"),
           "3": ("OHC3", "yellow"), "e": ("extra", "red")}
COLOR = {v[0]: v[1] for v in CLASSES.values()}
REMOVE_UM = 3.0


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def field_order():
    sets = json.load(open(os.path.join(HERE, "cell_sets.json"), encoding="utf-8"))
    first = sets["first"]["fields"]
    allf = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(CELLS, "v1", "*.npy")))]
    return first + [f for f in allf if f not in first]


class Tool:
    def __init__(self, fields):
        self.fields, self.k, self.cls = fields, 0, "inner"
        os.makedirs(OUT, exist_ok=True)
        self.v = napari.Viewer(title="Hair cell points")
        self.v.mouse_drag_callbacks.append(self.on_mouse)
        keys = {k: self.choose(c) for k, (c, _) in CLASSES.items()}
        keys.update({"c": self.toggle("Calb2"), "b": self.toggle("outlines"), "Enter": self.finish,
                     ".": self.next, ",": self.prev})
        self.keys = keys
        for k, fn in keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 16
        self.load()

    def path(self):
        return os.path.join(OUT, self.fields[self.k] + "_points.csv")

    def load(self):
        f = self.fields[self.k]
        self.cls = "OHC1"                                                    # each field starts on the first outer row
        z = np.load(os.path.join(FIELDS, f + ".npz")); st = list(z["stains"])
        self.px = float(z["voxel_um"][2])
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.v.add_image(scaled(z["surface_layers"][st.index("Phalloidin")]), name="phalloidin", colormap="gray", blending="additive")
        self.v.add_image(scaled(z["maxproj"][st.index("Calb2")]), name="Calb2", colormap="magenta", blending="additive", opacity=0.45)
        lab = os.path.join(CORR, f + "_cells.npy")
        lab = np.load(lab) if os.path.exists(lab) else np.load(os.path.join(CELLS, "v1", f + ".npy"))
        o = self.v.add_labels(lab.astype(np.int32), name="outlines", opacity=0.25); o.contour = 1
        self.pts = pd.read_csv(self.path()) if os.path.exists(self.path()) else pd.DataFrame({"y": pd.Series(dtype=float), "x": pd.Series(dtype=float), "class": pd.Series(dtype=str)})
        self.layer = self.v.add_points(np.zeros((0, 2)), name="points", size=9, border_color="black")
        for lyr in self.v.layers:                                             # a selected layer's keys win over the viewer's
            for k, fn in self.keys.items():
                lyr.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
        for lyr in self.v.layers:
            try:
                lyr.mouse_pan = True
            except Exception:
                pass
        self.draw(); self.v.reset_view(); self.status()

    def draw(self):
        d = self.pts
        self.layer.data = d[["y", "x"]].values.astype(float) if len(d) else np.zeros((0, 2))
        if len(d):
            self.layer.face_color = [COLOR[c] for c in d["class"]]
        self.layer.refresh()

    def save(self, finished=False):
        self.pts.to_csv(self.path(), index=False)
        if finished:
            log = os.path.join(OUT, "fields_done.csv")
            old = pd.read_csv(log) if os.path.exists(log) else pd.DataFrame()
            n = self.pts["class"].value_counts().to_dict()
            row = pd.DataFrame([dict(field=self.fields[self.k], saved=central_time(), **{c: n.get(c, 0) for c in COLOR})])
            pd.concat([old[old.field != self.fields[self.k]] if len(old) else old, row]).to_csv(log, index=False)

    def status(self, msg=""):
        n = self.pts["class"].value_counts().to_dict() if len(self.pts) else {}
        counts = "  ".join(f"{c} {n.get(c, 0)}" for c in COLOR)
        self.v.text_overlay.text = (f"Field {self.k + 1} of {len(self.fields)}: {self.fields[self.k]}\n"
                                    f"Now marking: {self.cls.upper()}    ({counts})\n{msg}\n"
                                    "i inner  1/2/3 outer rows  e extra  | left-click add, right-click remove | "
                                    "c Calb2  b outlines  Enter finished  . , next/previous")

    def on_mouse(self, viewer, event):
        start = np.array(event.position[-2:]); yield
        moved = False
        while event.type == "mouse_move":
            if np.linalg.norm(np.array(event.position[-2:]) - start) > 3:
                moved = True
            yield
        if moved:
            return
        y, x = start
        if event.button == 1:
            self.pts = pd.concat([self.pts, pd.DataFrame([dict(y=round(y, 1), x=round(x, 1), **{"class": self.cls})])], ignore_index=True)
            msg = f"Added {self.cls}"
        elif event.button == 2 and len(self.pts):
            d = np.hypot(self.pts.y.astype(float) - y, self.pts.x.astype(float) - x)
            if d.min() * self.px > REMOVE_UM:
                return self.status("No point within 3 um to remove")
            i = int(d.idxmin()); msg = f"Removed {self.pts.loc[i, 'class']}"
            self.pts = self.pts.drop(index=i).reset_index(drop=True)
        else:
            return
        self.save(); self.draw(); self.status(msg)

    def choose(self, cls):
        def f(viewer):
            self.cls = cls; self.status()
        return f

    def toggle(self, name):
        def f(viewer):
            self.v.layers[name].visible = not self.v.layers[name].visible
        return f

    def finish(self, viewer):
        self.save(finished=True)
        if self.k < len(self.fields) - 1:
            self.k += 1; self.load(); self.status("Previous field marked finished")
        else:
            self.status("Last field finished; you can close the window.")

    def next(self, viewer):
        self.save()
        if self.k < len(self.fields) - 1: self.k += 1; self.load()

    def prev(self, viewer):
        self.save()
        if self.k > 0: self.k -= 1; self.load()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="")
    a = ap.parse_args()
    fields = field_order()
    t = Tool(fields)
    if a.start:
        t.k = next((i for i, f in enumerate(fields) if a.start in f), 0); t.load()
    napari.run()
    t.save()


if __name__ == "__main__":
    sys.exit(main())
