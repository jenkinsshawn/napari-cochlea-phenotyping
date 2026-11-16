"""Napari tool for clicking the dividing lines between neighboring pillar heads, as a reference for the
automatic pillar measures.

Each field opens at the cell surface: phalloidin in gray, Calb2 in magenta (inner hair cell row, off at the
start; press c). The pillar heads are the band of tall plates between the inner hair cell row and the first outer
row; neighboring heads are separated by thin bright lines running across the band.

Click once on each dividing line, near the middle of the band, along the whole field. Where you cannot see the
lines, skip that stretch. If you see two tiers of heads (one nearer the inner hair cells, one nearer the outer
rows), use key 1 for lines in the tier nearer the inner hair cells and key 2 for lines in the other tier;
otherwise use 1 only.

Keys:
    1   dividing line, tier nearer the inner hair cells, or the only tier (yellow)
    2   dividing line, tier nearer the outer hair cells (cyan)
    c   show or hide Calb2
    Enter   field finished: save, mark done, open the next field
    .  ,    next / previous field (saves first)
Mouse: left-click puts a point; right-click removes the nearest point; drag pans; wheel zooms.
Points save on every click to E:\\Webber_confocal\\annotations\\pillar_points\\<field>_points.csv, and finished
fields are listed in fields_done.csv there. The first four fields cover controls, Smad3 and beta-catenin (an
early stop after three still covers every group); the rest follow if you want to go on.

Run from the scripts folder:
    python click_pillars.py
    python click_pillars.py --start AR47    (open at the first field whose name contains AR47)
    python click_pillars.py --repeat        (AR1383-2 MIDDLE and AR1064 MIDDLE again, for click repeatability)
    python click_pillars.py --more          (AR524 and JW5436 fields, alternating; saved in pillar_points_more)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, glob, os, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import napari

FIELDS = os.path.join(S.WORK, 'fields')
STACKS = os.path.join(S.WORK, 'ihc_stacks')
OUT = os.path.join(S.ANNOTATIONS, 'pillar_points')
CLASSES = {"1": ("line_1", "yellow"), "2": ("line_2", "cyan")}
COLOR = {v[0]: v[1] for v in CLASSES.values()}
REMOVE_UM = 2.0
FIRST = ["AR524 Ocm, Calb2 Phall P1 60X MIDDLE",          # control
         "JW5436 Calb2 PMCA2 Phall P1 60X middle",        # Smad3
         "AR1383-2 Calb2 PMCA2 Phall P1 60X BASE",        # beta-catenin
         "AR1064 Ocm Calb2 Phall P1 60X BASE",            # control
         "AR454 Calb2 PMCA2 Phall P1 60X BASE",           # Smad3
         "AR47 Calb2 PMCA2 Phall P1 60X MIDDLE"]          # beta-catenin


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def field_order():
    allf = [os.path.basename(f)[:-10] for f in sorted(glob.glob(os.path.join(STACKS, "*_cells.csv")))]
    return [f for f in FIRST if f in allf] + [f for f in allf if f not in FIRST]


class Tool:
    def __init__(self, fields):
        self.fields, self.k, self.cls = fields, 0, "line_1"
        os.makedirs(OUT, exist_ok=True)
        self.v = napari.Viewer(title="Pillar head dividing lines")
        self.v.mouse_drag_callbacks.append(self.on_mouse)
        keys = {k: self.choose(c) for k, (c, _) in CLASSES.items()}
        keys.update({"c": self.toggle("Calb2"), "Enter": self.finish, ".": self.next, ",": self.prev})
        self.keys = keys
        for k, fn in keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 16
        self.load()

    def path(self):
        return os.path.join(OUT, self.fields[self.k] + "_points.csv")

    def load(self):
        f = self.fields[self.k]; self.cls = "line_1"
        z = np.load(os.path.join(FIELDS, f + ".npz")); st = list(z["stains"])
        self.px = float(z["voxel_um"][2])
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.v.add_image(scaled(z["surface_layers"][st.index("Phalloidin")]), name="phalloidin", colormap="gray", blending="additive")
        cal = self.v.add_image(scaled(z["maxproj"][st.index("Calb2")]), name="Calb2", colormap="magenta", blending="additive", opacity=0.45)
        cal.visible = False
        self.pts = pd.read_csv(self.path()) if os.path.exists(self.path()) else pd.DataFrame({"y": pd.Series(dtype=float), "x": pd.Series(dtype=float), "class": pd.Series(dtype=str)})
        self.layer = self.v.add_points(np.zeros((0, 2)), name="points", size=7, border_color="black")
        for lyr in self.v.layers:                                             # a selected layer's keys win over the viewer's
            for k, fn in self.keys.items():
                lyr.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
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
            n = self.pts["class"].value_counts().to_dict() if len(self.pts) else {}
            row = pd.DataFrame([dict(field=self.fields[self.k], saved=central_time(), **{c: n.get(c, 0) for c in COLOR})])
            pd.concat([old[old.field != self.fields[self.k]] if len(old) else old, row]).to_csv(log, index=False)

    def status(self, msg=""):
        n = self.pts["class"].value_counts().to_dict() if len(self.pts) else {}
        self.v.text_overlay.text = (f"Field {self.k + 1} of {len(self.fields)}: {self.fields[self.k]}\n"
                                    f"Now marking: {'tier nearer the inner hair cells (or only tier)' if self.cls == 'line_1' else 'tier nearer the outer hair cells'}"
                                    f"    (lines: tier 1 {n.get('line_1', 0)}, tier 2 {n.get('line_2', 0)})\n{msg}\n"
                                    "click each dividing line between pillar heads | 1/2 tier | left-click add, right-click remove | "
                                    "c Calb2  Enter finished  . , next/previous")

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
            msg = "Added"
        elif event.button == 2 and len(self.pts):
            d = np.hypot(self.pts.y.astype(float) - y, self.pts.x.astype(float) - x)
            if d.min() * self.px > REMOVE_UM:
                return self.status("No point within 2 um to remove")
            i = int(d.idxmin()); msg = "Removed"
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


REPEAT = ["AR1383-2 Calb2 PMCA2 Phall P1 60X MIDDLE", "AR1064 Ocm Calb2 Phall P1 60X MIDDLE"]
MORE = ["AR524 Ocm, Calb2 Phall P1 60X MIDDLE", "JW5436 Calb2 PMCA2 Phall P1 60X middle", "AR524 Ocm, Calb2 Phall P1 60X BASE",
        "JW5436 Calb2 PMCA2 Phall P1 60X base", "AR524 Ocm, Calb2 Phall P1 60X APEX", "JW5436 Calb2 PMCA2 Phall P1 60X Apex (3)"]


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="")
    ap.add_argument("--repeat", action="store_true", help="click two fields again (saved in pillar_points_repeat)")
    ap.add_argument("--more", action="store_true", help="AR524 and JW5436 fields (saved in pillar_points_more)")
    a = ap.parse_args()
    fields = field_order()
    if a.repeat:
        OUT = OUT + "_repeat"; fields = REPEAT
    if a.more:
        OUT = OUT + "_more"; fields = MORE
    t = Tool(fields)
    if a.start:
        t.k = next((i for i, f in enumerate(fields) if a.start in f), 0); t.load()
    napari.run()
    t.save()


if __name__ == "__main__":
    sys.exit(main())
