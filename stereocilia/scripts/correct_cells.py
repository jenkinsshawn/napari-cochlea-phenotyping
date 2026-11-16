"""Napari tool for correcting the cell outlines and calling each cell's type.

Opens each field of a set (cell_sets.json) at the cell surface: phalloidin in gray, Calb2 in
magenta, the EGFP stain (Ocm or PMCA2) in green (hidden; press v to show). The outlines start
from the computer's outlines (--tag) or from your saved corrections, if any.

Keys (hold the mouse over a cell):
    x       delete the cell
    a       add a cell here: the computer fills out to the phalloidin border
    g       merge: g on the first cell, then g on the second
    i o s   call the cell inner hair cell / outer hair cell / supporting cell
    u       unsure
    b c v   show or hide phalloidin / Calb2 / EGFP stain
    w       save now
    Enter   field finished: save, mark it done, go to the next field
    .  ,    next / previous field (saves first)
Painting also works with napari's own Labels tools (brush, fill, eraser). Press m in the
Labels layer to get a new unused cell number before painting a new cell.

Saved on every change to E:\\Webber_confocal\\annotations\\cells\\:
    <field>_cells.npy, <field>_cell_types.csv, and fields_corrected.csv (one row per finished field).

Run from the scripts folder:
    python correct_cells.py --set first
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, json, os, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import sobel, gaussian
from skimage.segmentation import watershed
import napari

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = os.path.join(S.WORK, 'fields')
CELLS = os.path.join(S.WORK, 'cells')
ANN = os.path.join(S.ANNOTATIONS, 'cells')
TYPE_KEYS = {"i": "inner", "o": "outer", "s": "supporting", "u": "unsure"}
TYPE_COLORS = {"inner": "magenta", "outer": "lime", "supporting": "orange", "unsure": "red", "": "white"}
ADD_WINDOW_UM = 12.0


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


class Tool:
    def __init__(self, fields, tag, set_name):
        self.fields, self.tag, self.set_name, self.k = fields, tag, set_name, 0
        os.makedirs(ANN, exist_ok=True)
        self.v = napari.Viewer(title="Cell outlines")
        self.merge_first = None
        self.load()
        for key, fn in [("x", self.delete), ("a", self.add), ("g", self.merge), ("w", self.save_now),
                        ("Enter", self.finish), (".", self.next), (",", self.prev),
                        ("b", self.toggle(0)), ("c", self.toggle(1)), ("v", self.toggle(2))] + \
                       [(k, self.call_type(t)) for k, t in TYPE_KEYS.items()]:
            self.v.bind_key(key, fn, overwrite=True)
            self.bind_on_layers(key, fn)
        self.v.text_overlay.visible = True
        self.v.text_overlay.font_size = 12

    # ---------- loading and saving ----------
    def paths(self, field):
        return (os.path.join(ANN, field + "_cells.npy"), os.path.join(ANN, field + "_cell_types.csv"))

    def load(self):
        field = self.fields[self.k]
        z = np.load(os.path.join(FIELDS, field + ".npz"))
        stains = list(z["stains"]); L = z["surface_layers"]
        get = lambda names: L[[i for i, s in enumerate(stains) if s in names][0]] if any(s in names for s in stains) else np.zeros(L.shape[1:])
        self.phal = scaled(get(("Phalloidin",)))
        lab_path, typ_path = self.paths(field)
        if os.path.exists(lab_path):
            labels = np.load(lab_path).astype(np.int32); self.source = "your corrections"
            self.types = pd.read_csv(typ_path).fillna("").set_index("cell")["type"].to_dict() if os.path.exists(typ_path) else {}
        else:
            labels = np.load(os.path.join(CELLS, self.tag, field + ".npy")).astype(np.int32); self.source = f"computer outlines {self.tag}"
            guess = pd.read_csv(os.path.join(CELLS, self.tag, field + "_cells.csv"))
            # pre-filled types: outer from the Ocm or PMCA2 guess; inner only for cells outlined from Calb2 (v1 on)
            src = guess["source"] if "source" in guess else pd.Series(["phalloidin"] * len(guess))
            self.types = {int(c): (t if t == "outer" or (t == "inner" and s == "calb2") else "")
                          for c, t, s in zip(guess.cell, guess.type_guess, src)}
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.imgs = [self.v.add_image(self.phal, name="phalloidin", colormap="gray", blending="additive"),
                     self.v.add_image(scaled(get(("Calb2",))), name="Calb2", colormap="magenta", blending="additive", opacity=0.6),
                     self.v.add_image(scaled(get(("Ocm", "PMCA2"))), name="EGFP stain", colormap="green", blending="additive", visible=False)]
        self.lab = self.v.add_labels(labels, name="cells", opacity=0.35)
        self.lab.contour = 2
        self.pts = None
        self.refresh_types()
        self.v.layers.selection.active = self.lab
        self.v.reset_view()
        self.status()

    def save(self, finished=False):
        field = self.fields[self.k]; lab_path, typ_path = self.paths(field)
        labels = np.asarray(self.lab.data).astype(np.uint16)
        np.save(lab_path, labels)
        present = set(np.unique(labels)) - {0}
        pd.DataFrame({"cell": sorted(present), "type": [self.types.get(int(c), "") for c in sorted(present)]}).to_csv(typ_path, index=False)
        if finished:
            log = os.path.join(ANN, "fields_corrected.csv")
            old = pd.read_csv(log) if os.path.exists(log) else pd.DataFrame()
            row = pd.DataFrame([dict(field=field, set=self.set_name, started_from=self.tag, cells=len(present),
                                     untyped=sum(self.types.get(int(c), "") == "" for c in present), saved=central_time())])
            pd.concat([old[old.field != field] if len(old) else old, row]).to_csv(log, index=False)

    # ---------- display ----------
    def refresh_types(self):
        labels = np.asarray(self.lab.data); ids = np.unique(labels); ids = ids[ids > 0]
        if len(ids):
            cy, cx = np.array(ndi.center_of_mass(np.ones_like(labels), labels, ids)).T
        else:
            cy = cx = np.zeros(0)
        typ = [self.types.get(int(c), "") for c in ids]
        short = {"inner": "I", "outer": "O", "supporting": "S", "unsure": "?", "": ""}
        feats = pd.DataFrame({"t": [short[t] for t in typ]})
        if self.pts is not None and self.pts in self.v.layers:
            self.v.layers.remove(self.pts)
        self.pts = self.v.add_points(np.c_[cy, cx] if len(ids) else np.zeros((0, 2)), features=feats, name="types", size=3,
                                     face_color=[TYPE_COLORS[t] for t in typ] if len(ids) else "white",
                                     text={"string": "{t}", "size": 10, "color": "white", "translation": [-6, 0]})
        self.bind_all_layers()
        self.v.layers.selection.active = self.lab

    def status(self, msg=""):
        field = self.fields[self.k]
        labels = np.asarray(self.lab.data); ids = np.unique(labels); ids = ids[ids > 0]
        counts = pd.Series([self.types.get(int(c), "") or "not called" for c in ids]).value_counts().to_dict()
        self.v.text_overlay.text = (f"Field {self.k + 1} of {len(self.fields)}: {field}   (from {self.source})\n"
                                    f"{len(ids)} cells: {counts}\n{msg}\n"
                                    "x delete  a add  g merge  i/o/s/u type  b/c/v stains  w save  Enter finished  . , next/previous")

    # ---------- helpers ----------
    def cell_under_cursor(self):
        y, x = [int(round(c)) for c in self.v.cursor.position[-2:]]
        labels = np.asarray(self.lab.data)
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            return y, x, int(labels[y, x])
        return y, x, None

    def changed(self, msg):
        self.lab.refresh(); self.save(); self.refresh_types(); self.status(msg)

    def bind_on_layers(self, key, fn):
        self._keys = getattr(self, "_keys", {}); self._keys[key] = fn

    def bind_all_layers(self):
        # a selected layer's own key bindings take precedence over the viewer's, so bind on every layer too
        for lyr in self.v.layers:
            for key, fn in getattr(self, "_keys", {}).items():
                lyr.bind_key(key, (lambda f: (lambda layer: f(self.v)))(fn), overwrite=True)

    # ---------- actions ----------
    def delete(self, viewer):
        y, x, c = self.cell_under_cursor()
        if not c: return self.status("No cell under the cursor")
        data = np.asarray(self.lab.data); data[data == c] = 0; self.lab.data = data
        self.types.pop(c, None); self.changed(f"Deleted cell {c}")

    def add(self, viewer):
        y, x, c = self.cell_under_cursor()
        if c is None: return
        if c: return self.status(f"Cell {c} is already here; delete it first or paint")
        data = np.asarray(self.lab.data)
        px = float(np.load(os.path.join(FIELDS, self.fields[self.k] + ".npz"))["voxel_um"][2])
        r = int(ADD_WINDOW_UM / px / 2)
        y0, y1, x0, x1 = max(0, y - r), min(data.shape[0], y + r), max(0, x - r), min(data.shape[1], x + r)
        win = gaussian(self.phal[y0:y1, x0:x1], 1.5)
        markers = np.zeros(win.shape, np.int32)
        markers[[0, -1], :] = 2; markers[:, [0, -1]] = 2
        markers[data[y0:y1, x0:x1] > 0] = 2                                   # existing cells stay as they are
        markers[max(0, y - y0 - 2):y - y0 + 3, max(0, x - x0 - 2):x - x0 + 3] = 1
        region = watershed(sobel(win), markers) == 1
        region &= data[y0:y1, x0:x1] == 0
        new = int(data.max()) + 1
        data[y0:y1, x0:x1][region] = new; self.lab.data = data
        calb = np.asarray(self.imgs[1].data)[y0:y1, x0:x1][region].mean() if region.any() else 0
        self.types[new] = "inner" if calb > 0.5 else ""
        self.changed(f"Added cell {new} ({int(region.sum() * px * px)} um2)")

    def merge(self, viewer):
        y, x, c = self.cell_under_cursor()
        if not c: return self.status("No cell under the cursor")
        if self.merge_first is None or self.merge_first == c:
            self.merge_first = c; return self.status(f"Merge: cell {c} chosen; press g on the second cell")
        data = np.asarray(self.lab.data); data[data == c] = self.merge_first; self.lab.data = data
        self.types.pop(c, None); first, self.merge_first = self.merge_first, None
        self.changed(f"Merged cell {c} into {first}")

    def call_type(self, typ):
        def f(viewer):
            y, x, c = self.cell_under_cursor()
            if not c: return self.status("No cell under the cursor")
            self.types[c] = typ; self.changed(f"Cell {c}: {typ}")
        return f

    def toggle(self, i):
        def f(viewer):
            self.imgs[i].visible = not self.imgs[i].visible
        return f

    def save_now(self, viewer):
        self.save(); self.status(f"Saved {central_time()}")

    def finish(self, viewer):
        self.save(finished=True)
        if self.k < len(self.fields) - 1:
            self.k += 1; self.load(); self.status("Previous field marked finished")
        else:
            self.status("Last field marked finished. All fields in this set are done; you can close the window.")

    def next(self, viewer):
        self.save()
        if self.k < len(self.fields) - 1: self.k += 1; self.load()

    def prev(self, viewer):
        self.save()
        if self.k > 0: self.k -= 1; self.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="first", help="set name in cell_sets.json")
    ap.add_argument("--tag", default="v1", help="computer outlines to start from")
    ap.add_argument("--start", default="", help="open at the first field whose name contains this text")
    a = ap.parse_args()
    sets = json.load(open(os.path.join(HERE, "cell_sets.json"), encoding="utf-8"))
    fields = sets[a.set]["fields"]
    t = Tool(fields, a.tag, a.set)
    if a.start:
        t.k = next((i for i, f in enumerate(fields) if a.start in f), 0); t.load()
    napari.run()
    t.save()


if __name__ == "__main__":
    sys.exit(main())
