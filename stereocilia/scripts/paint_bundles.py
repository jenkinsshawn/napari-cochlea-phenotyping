"""Napari tool for painting bundle and not-bundle strokes, to train the bundle detector.

Each field opens as a stack of phalloidin images taken at fixed heights above and below the
cell surface (0.3 um apart, from 6.0 um above to 1.8 um below), so every image shows the
whole field at one height. The slider at the bottom moves through the heights; the bundles
are usually clearest about 1 to 2 um above the surface.

Keys:
    b       paint BUNDLE (red)
    x       paint NOT BUNDLE (blue, wide brush): paint over cuticular plates, bright cell tops,
            cell borders and the band at the same height as the bundles
    e       eraser
    [  ]    smaller / larger brush
    v       show or hide PMCA2 (PMCA2 fields only; a hint, not used by the detector)
    Enter   field finished: save, mark done, open the next field
    .  ,    next / previous field (saves first)
Painting works on the image you are looking at only. Strokes save automatically every few
seconds and on every field change, to E:\\Webber_confocal\\annotations\\bundle_strokes\\.

Run from the scripts folder:
    python paint_bundles.py
    python paint_bundles.py --start AR47      (open at the first field whose name contains AR47)
    python paint_bundles.py --fields "<field>,<field>" --height 3.5 --ihc   (only those fields, opened 3.5 um up,
                                               inner hair cells marked with yellow points)
    python paint_bundles.py --fields "<field>,<field>" --rings   (outer hair cell bundles whose outline runs onto the
                                               rim below them marked with orange points; paint the rim NOT BUNDLE)
Each field opens about 1.5 um above the surface; inner hair cell bundles are often 3 to 5 um up.
The stacks are made once beforehand with:  python paint_bundles.py --prepare  (needs nd2);
python paint_bundles.py --prepare --all makes them for every field, for the bundle detector.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, os, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = S.RAW
FIELDS = os.path.join(S.WORK, 'fields')
STACKS = os.path.join(S.WORK, 'paint_stacks')
OUT = os.path.join(S.ANNOTATIONS, 'bundle_strokes')
ABOVE, BELOW = 20, 6          # images above and below the surface (0.3 um each): 6.0 um above to 1.8 um below
                              # (was 13 above, 3.9 um, until 6 Nov 2026; inner hair cell bundles reached the top)
REDO = "--redo" in sys.argv
PAINT_FIELDS = [
    "AR454 Calb2 PMCA2 Phall P1 60X APEX (3)",       # Smad3 apex, where the rule failed
    "AR524 Ocm, Calb2 Phall P1 60X MIDDLE",          # control middle
    "JW5436 Calb2 PMCA2 Phall P1 60X middle",        # Smad3 middle
    "AR1383-2 Calb2 PMCA2 Phall P1 60X APEX (4)",    # beta-catenin apex
    "AR1064 Ocm Calb2 Phall P1 60X BASE",            # control base
    "AR47 Calb2 PMCA2 Phall P1 60X BASE",            # beta-catenin base
    "AR524 Ocm, Calb2 Phall P1 60X APEX",            # control apex
]


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def all_fields():
    import glob
    return [os.path.basename(f)[:-10] for f in sorted(glob.glob(os.path.join(S.WORK, 'cells', 'types', '*_types.csv')))]


def prepare(fields=None):
    """Images at fixed heights from the cell surface, first image highest above it. The bundle
    side is the side of lower image numbers in every field (checked by eye on the painting
    fields; two automatic rules, peaked phalloidin and Calb2 bodies, each failed in some fields)."""
    import nd2
    os.makedirs(STACKS, exist_ok=True)
    for f in (fields or PAINT_FIELDS):
        dst = os.path.join(STACKS, f + ".npz")
        if os.path.exists(dst) and not REDO:
            print("  already made:", f); continue
        z = np.load(os.path.join(FIELDS, f + ".npz")); st = list(z["stains"])
        arr = S.read_raw(f)
        s = np.round(z["surface"]).astype(int); nz = arr.shape[0]
        sys.path.insert(0, HERE)
        from measure_bundles import body_side
        sign, peak = -1, "fixed"            # bundles on the side of lower image numbers in every field checked by eye
        offs = np.array([sign * k for k in range(ABOVE, -BELOW - 1, -1)])
        print(f"  {f}: bundle side = {'lower' if sign < 0 else 'higher'} image numbers {peak}")
        def flat(ch):
            return np.stack([np.take_along_axis(arr[:, ch], np.clip(s + k, 0, nz - 1)[None], 0)[0] for k in offs])
        out = dict(phal=flat(st.index("Phalloidin")), offsets_um=np.arange(ABOVE, -BELOW - 1, -1) * float(z["voxel_um"][0]))
        if "PMCA2" in st:
            out["pmca2"] = flat(st.index("PMCA2"))
        np.savez_compressed(dst, **out); print("  made:", f, out["phal"].shape)


def strokes_to_stack(s, heights):
    """Stroke indices moved to the images of the current stack with the same height above the
    surface (strokes are saved with the heights of the stack they were painted on)."""
    idx = np.array(s["idx"]).astype(np.int64).copy()
    old = np.asarray(s["heights_um"]) if "heights_um" in s else heights
    lut = np.array([int(np.argmin(np.abs(heights - h))) if np.min(np.abs(heights - h)) < 0.05 else -1 for h in old])
    idx[0] = lut[idx[0]]
    return idx


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


class Tool:
    def __init__(self, fields, start_um=1.5, mark_ihc=False, mark_rings=False):
        import napari
        self.fields, self.k, self.start_um, self.mark_ihc, self.mark_rings = fields, 0, start_um, mark_ihc, mark_rings
        os.makedirs(OUT, exist_ok=True)
        self.v = napari.Viewer(title="Bundle strokes")
        self.keys = {"b": self.choose(1), "x": self.choose(2), "e": self.eraser, "v": self.toggle_pm,
                     "Enter": self.finish, ".": self.next, ",": self.prev}
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 14
        self.dirty = False
        try:
            from qtpy.QtCore import QTimer
            self.timer = QTimer(); self.timer.timeout.connect(self.autosave); self.timer.start(5000)
        except Exception:
            self.timer = None
        self.load()

    def add_ihc_marks(self, f):
        """Yellow points at the inner hair cells (from the cell types), so it is clear which bundles to paint."""
        try:
            from scipy import ndimage as ndi
            ty = pd.read_csv(os.path.join(S.WORK, "cells", "types", f + "_types.csv")); ty = ty[ty.row == "IHC"]
            if not len(ty): return
            if "inner_cell_id" in ty and (ty.inner_cell_id > 0).any():          # surface and body pieces of one cell count once
                ty = ty.assign(k=np.where(ty.inner_cell_id > 0, ty.inner_cell_id, -ty.cell)).groupby("k")[["y", "x"]].mean()
            pts = ty[["y", "x"]].values
            self.v.add_points(pts, name="inner hair cells", size=9, face_color="yellow", border_color="black", opacity=0.8)
            self.v.layers.selection.active = self.lab
        except Exception as e:
            print("inner hair cell marks not shown:", e)

    def add_ring_marks(self, f):
        """Orange points at outer hair cell bundles whose outline runs onto the rim below the bundle (ring_bundles.csv):
        paint NOT BUNDLE (x) over the rim next to each point."""
        try:
            r = pd.read_csv(os.path.join(OUT, "ring_bundles.csv")); r = r[r.field == f]
            if not len(r): return
            self.v.add_points(r[["y", "x"]].values, name="outlines running onto the rim", size=11, face_color="orange", border_color="black", opacity=0.8)
            self.v.layers.selection.active = self.lab
        except Exception as e:
            print("ring marks not shown:", e)

    def path(self):
        return os.path.join(OUT, self.fields[self.k] + "_strokes.npz")

    def load(self):
        f = self.fields[self.k]
        z = np.load(os.path.join(STACKS, f + ".npz"))
        self.heights = z["offsets_um"]
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.v.add_image(scaled(z["phal"]), name="phalloidin", colormap="gray")
        self.pm = self.v.add_image(scaled(z["pmca2"]), name="PMCA2", colormap="green", blending="additive",
                                   visible=False, opacity=0.6) if "pmca2" in z else None
        lab = np.zeros(z["phal"].shape, np.uint8)
        if os.path.exists(self.path()):
            s = np.load(self.path()); idx = strokes_to_stack(s, self.heights)
            lab[tuple(idx[:, idx[0] >= 0])] = s["val"][idx[0] >= 0]
        self.lab = self.v.add_labels(lab, name="strokes", opacity=0.7)
        try:
            from napari.utils.colormaps import DirectLabelColormap
            self.lab.colormap = DirectLabelColormap(color_dict={0: "transparent", 1: "red", 2: "dodgerblue", None: "transparent"})
        except Exception:
            try:
                self.lab.color = {1: "red", 2: "dodgerblue"}
            except Exception:
                pass
        self.lab.brush_size = 5; self.lab.selected_label = 1; self.lab.mode = "paint"
        try:
            self.lab.events.paint.connect(self.mark)
        except Exception:
            self.lab.events.data.connect(self.mark)
        for lyr in self.v.layers:                                      # a selected layer's keys win over the viewer's
            for k, fn in self.keys.items():
                lyr.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
        self.v.layers.selection.active = self.lab
        self.v.dims.current_step = (int(np.argmin(np.abs(self.heights - self.start_um))), 0, 0)   # open at the chosen height
        if self.mark_ihc: self.add_ihc_marks(f)
        if self.mark_rings: self.add_ring_marks(f)
        self.v.dims.events.current_step.connect(self.status)
        self.v.reset_view(); self.status()

    def mark(self, event=None):
        self.dirty = True

    def autosave(self):
        if self.dirty:
            self.save(); self.status()

    def save(self, finished=False):
        lab = np.asarray(self.lab.data); idx = np.nonzero(lab)
        np.savez_compressed(self.path(), idx=np.array(idx, dtype=np.int16), val=lab[idx], heights_um=self.heights)
        self.dirty = False
        if finished:
            log = os.path.join(OUT, "fields_done.csv")
            old = pd.read_csv(log) if os.path.exists(log) else pd.DataFrame()
            planes = sorted({float(self.heights[i]) for i in set(idx[0].tolist())})
            row = pd.DataFrame([dict(field=self.fields[self.k], saved=central_time(), bundle_pixels=int((lab == 1).sum()),
                                     not_bundle_pixels=int((lab == 2).sum()), heights_painted=";".join(f"{h:.1f}" for h in planes))])
            pd.concat([old[old.field != self.fields[self.k]] if len(old) else old, row]).to_csv(log, index=False)

    def status(self, event=None):
        lab = np.asarray(self.lab.data); i = self.v.dims.current_step[0]
        h = self.heights[i]
        where = f"{abs(h):.1f} um {'above' if h > 0 else 'below' if h < 0 else 'at'} the cell surface"
        now = {1: "BUNDLE (red)", 2: "NOT BUNDLE (blue)"}.get(self.lab.selected_label, "")
        if self.lab.mode == "erase": now = "ERASER"
        self.v.text_overlay.text = (f"Field {self.k + 1} of {len(self.fields)}: {self.fields[self.k]}\n"
                                    f"Image {i + 1} of {len(self.heights)}: {where}    Painting: {now}\n"
                                    f"Strokes in this field: bundle {int((lab == 1).sum())} px, not bundle {int((lab == 2).sum())} px\n"
                                    "b bundle  x not bundle  e eraser  [ ] brush  v PMCA2  Enter finished  . , next/previous")

    def choose(self, val):
        def f(viewer):
            self.lab.selected_label = val; self.lab.mode = "paint"; self.v.layers.selection.active = self.lab
            self.lab.brush_size = 4 if val == 1 else 14                      # thin for bundles, wide for not-bundle fill
            self.status()
        return f

    def eraser(self, viewer):
        self.lab.mode = "erase"; self.status()

    def toggle_pm(self, viewer):
        if self.pm is not None:
            self.pm.visible = not self.pm.visible

    def finish(self, viewer):
        self.save(finished=True)
        if self.k < len(self.fields) - 1:
            self.k += 1; self.load()
        else:
            self.v.text_overlay.text += "\nLast field finished; you can close the window."

    def next(self, viewer):
        self.save()
        if self.k < len(self.fields) - 1: self.k += 1; self.load()

    def prev(self, viewer):
        self.save()
        if self.k > 0: self.k -= 1; self.load()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--redo", action="store_true"); ap.add_argument("--start", default="")
    ap.add_argument("--all", action="store_true", help="with --prepare: every field with cell types, not only the painting fields")
    ap.add_argument("--fields", default="", help="paint only these fields (full names separated by commas)")
    ap.add_argument("--height", type=float, default=1.5, help="height above the surface each field opens at (um)")
    ap.add_argument("--ihc", action="store_true", help="mark the inner hair cells with yellow points")
    ap.add_argument("--rings", action="store_true", help="mark outer hair cell bundles whose outline runs onto the rim (orange points)")
    a = ap.parse_args()
    if a.prepare:
        return prepare(all_fields() if a.all else None)
    import napari
    fields = [f.strip() for f in a.fields.split(",") if f.strip()] or PAINT_FIELDS
    missing = [f for f in fields if not os.path.exists(os.path.join(STACKS, f + ".npz"))]
    if missing: print("no stack for:", missing); fields = [f for f in fields if f not in missing]
    t = Tool(fields, a.height, a.ihc, a.rings)
    if a.start:
        t.k = next((i for i, f in enumerate(fields) if a.start in f), 0); t.load()
    napari.run()
    t.save()


if __name__ == "__main__":
    sys.exit(main())
