"""Strokes on inner hair cell stereocilia, to train a classifier that finds them in every cell.

Each cell opens as a stack of side slices from the full-height cut-out (ihc_stacks.py): each slice is
one position along the row, showing height (up) against distance across the row (outer hair cells to
the right). Long stereocilia appear as streaks. The slider (or the left and right arrow keys) moves
along the row; the cell's own bundle is in the middle slices.

Keys:
  b   brush for stereocilia: paint along long, clear stereocilia
  x   brush for not stereocilia: the cell top and plate, cell borders, folds, neighboring cells
  e   eraser          [ and ]  brush size
  Enter      save and next cell          Backspace  back one cell
A few strokes in 3-4 slices per cell is enough; strokes do not need to cover everything.

Input:  E:\\Webber_confocal\\annotations\\ihc_strokes\\sample.csv (item, field, cell)
Output: E:\\Webber_confocal\\annotations\\ihc_strokes\\<item>.npz (strokes in cut-out voxel coordinates) and log.csv
Saved cells reopen with their strokes, so a cell can be added to later.

Run from the scripts folder:
    python paint_ihc_strokes.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import napari

STACKS = os.path.join(S.WORK, 'ihc_stacks')
DIR = os.path.join(S.ANNOTATIONS, 'ihc_strokes')
PX, DZ = 0.1083, 0.3


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.9])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def to_view(vol):
    """cut-out (planes low->high, across, along) -> view (along, planes high->low, across)."""
    return np.ascontiguousarray(vol.transpose(2, 0, 1)[:, ::-1, :])


def from_view(lab):
    """view labels (along, planes high->low, across) -> cut-out order (planes low->high, across, along)."""
    return np.ascontiguousarray(lab[:, ::-1, :].transpose(1, 2, 0))


class Tool:
    def __init__(self):
        self.items = pd.read_csv(os.path.join(DIR, "sample.csv"))
        self.log = os.path.join(DIR, "log.csv")
        self.done = pd.read_csv(self.log) if os.path.exists(self.log) else pd.DataFrame(columns=["item", "stereocilia_voxels", "not_voxels", "saved"])
        todo = [k for k, it in enumerate(self.items["item"]) if it not in set(self.done["item"])]
        self.k = todo[0] if todo else 0
        self.cache = {}
        self.v = napari.Viewer(title="Inner hair cell stereocilia strokes")
        self.keys = {"b": self.brush(1), "x": self.brush(2), "e": self.eraser, "Enter": self.save_next, "Backspace": self.back}
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 15
        self.show()

    def volume(self, field, cell):
        if field not in self.cache:
            self.cache = {field: np.load(os.path.join(STACKS, field + ".npz"))["phal"]}
        return self.cache[field][int(cell) - 1].astype(np.float32)

    def show(self, msg=""):
        it = self.items.iloc[self.k]
        vol = self.volume(it["field"], it["cell"]); self.shape = vol.shape
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        view = to_view(vol); sc = (1, DZ / PX, 1)
        self.v.add_image(scaled(view), name="phalloidin", colormap="gray", scale=sc)
        f = os.path.join(DIR, f"{int(it['item'])}.npz"); lab = np.zeros(view.shape, np.uint8)
        if os.path.exists(f):
            z = np.load(f)
            full = np.zeros(vol.shape, np.uint8); full[tuple(z["stereocilia"].T)] = 1; full[tuple(z["not_stereocilia"].T)] = 2
            lab = to_view(full)
        self.lab = self.v.add_labels(lab, name="strokes", scale=sc, opacity=0.6)
        self.lab.brush_size = 3; self.lab.selected_label = 1; self.lab.mode = "paint"
        try:
            self.lab.n_edit_dimensions = 2
        except Exception:
            pass
        for k, fn in self.keys.items():
            self.lab.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
        self.v.layers.selection.active = self.lab
        self.v.dims.set_point(0, view.shape[0] // 2)
        self.v.reset_view(); self.status(msg)

    def status(self, msg=""):
        brush = {1: "STEREOCILIA (b)", 2: "NOT stereocilia (x)"}.get(self.lab.selected_label, "")
        mode = "eraser" if self.lab.mode == "erase" else f"brush: {brush}"
        self.v.text_overlay.text = (f"Cell {self.k + 1} of {len(self.items)}   saved {len(self.done)}   {mode}\n{msg}\n"
                                    "slider or arrow keys: move along the row | b stereocilia  x not stereocilia  e eraser  [ ] size | "
                                    "Enter save, next  Backspace back")

    def brush(self, label):
        def f(viewer):
            self.lab.selected_label = label; self.lab.mode = "paint"; self.status()
        return f

    def eraser(self, viewer):
        self.lab.mode = "erase"; self.status()

    def save_next(self, viewer):
        it = self.items.iloc[self.k]; full = from_view(np.asarray(self.lab.data))
        s = np.argwhere(full == 1).astype(np.int16); n = np.argwhere(full == 2).astype(np.int16)
        np.savez_compressed(os.path.join(DIR, f"{int(it['item'])}.npz"), stereocilia=s, not_stereocilia=n, shape=np.array(self.shape))
        row = pd.DataFrame([dict(item=int(it["item"]), stereocilia_voxels=len(s), not_voxels=len(n), saved=central_time())])
        self.done = pd.concat([self.done[self.done["item"] != it["item"]], row], ignore_index=True); self.done.to_csv(self.log, index=False)
        if self.k < len(self.items) - 1:
            self.k += 1; self.show(f"Saved ({len(s)} stereocilia and {len(n)} other voxels)")
        else:
            self.status("Saved. All cells done; you can close the window.")

    def back(self, viewer):
        if self.k > 0:
            self.k -= 1; self.show("Back one cell")


if __name__ == "__main__":
    Tool(); napari.run()
