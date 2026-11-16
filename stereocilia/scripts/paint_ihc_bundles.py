"""Paint inner hair cell stereocilia in three views, as a reference for the automatic measures.

Each cell opens as three panels of phalloidin from the full stack (ihc_stacks.py), at true scale:
  left    from above: brightest phalloidin at any height (the row runs left to right,
          the outer hair cells are toward the bottom)
  middle  from the side, across the row (height up, outer hair cells to the right)
  right   from the side, along the bundle (height up, the row left to right)
Paint the stereocilia of the cell between the two yellow guide lines (left and right panels; the
middle panel shows only that cell): every stereocilium you can see, from where it leaves the cell
to its tip. Leave the cell body and neighboring cells unpainted. If a view is unclear, leave that
panel empty.

Keys:
  Enter       save and next cell
  u           no readable bundle in this cell; next cell
  Backspace   back one cell
  Drag with the left button to paint; hold the right button (or press e) to erase; [ and ] brush size.

Input:  E:\\Webber_confocal\\annotations\\ihc_paint\\sample.csv (item, field, cell)
Output: E:\\Webber_confocal\\annotations\\ihc_paint\\<item>.npz (mask of each panel) and answers.csv
Saved cells are skipped when the tool is restarted.

A set folder may hold set.json naming its cut-out folder ("stacks"), the automatic outlines of those cut-outs
("masks", from ihc_stereocilia.py) and "prefill": true. With prefill, each cell opens with the automatic outline
already painted, projected the same way as the image; correct it with the brush and the eraser. The prefill is saved
beside the painting, and a cell saved without changes is recorded as 'accepted unchanged'.

Run from the scripts folder:
    python paint_ihc_bundles.py           (sample in ihc_paint)
    python paint_ihc_bundles.py folded    (bundles folded flat by the mount; sample in ihc_paint_folded)
    python paint_ihc_bundles.py train     (training set for a learned bundle model, 100 cells, prefilled; ihc_paint_train)
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
import sys
SET = sys.argv[1] if len(sys.argv) > 1 else ""   # e.g. "folded": sample and paint in ihc_paint_folded
DIR = os.path.join(S.ANNOTATIONS, 'ihc_paint') + (f"_{SET}" if SET else "")
SETCFG = {}
if os.path.exists(os.path.join(DIR, "set.json")):
    import json
    with open(os.path.join(DIR, "set.json"), encoding="utf-8") as _fh:
        SETCFG = json.load(_fh)
STACKS = SETCFG.get("stacks", STACKS)          # cut-outs of this set
MASKS = SETCFG.get("masks", "")                # automatic outlines of the same cut-outs (cells, z, y, x on the 0.2166 um grid)
PREFILL = bool(SETCFG.get("prefill", False)) and bool(MASKS)
PX, DZ = 0.1083, 0.3
GAP = 20
ALONG_HALF_UM = 4.0          # middle panel: brightest along +-4 um of the row (about half the inner hair cell spacing)
CELL_HALF_UM = 4.3           # guide lines at the cell's edges along the row (half the usual spacing, 8.6 um)
ACROSS_UM = (-1.0, 8.0)      # right panel: brightest across this band (the bundle side of the cell)


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def stretch_z(img):
    """(planes, n) at 0.3 um -> (rows, n) at the pixel size, height increasing upward (row 0 = top)."""
    f = DZ / PX; nz = img.shape[0]; rows = int(round(nz * f))
    src = np.clip((np.arange(rows) + 0.5) / f - 0.5, 0, nz - 1)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, nz - 1); w = (src - lo)[:, None]
    out = img[lo] * (1 - w) + img[hi] * w
    return out[::-1]


BAND_HALF_UM = 4.5           # with an outline: right panel shows +-4.5 um across the row around the bundle outline


def across_band(ny, outline=None):
    """Rows (across the row) shown in the right panel: around the bundle's own outline when there is one (its center
    can sit 2-5 um off the cell center when a cell's top and body outlines are averaged), else ACROSS_UM from the center."""
    c = ny // 2
    if outline is not None and outline.any():
        yc = int(round(np.average(np.arange(ny), weights=outline.sum(axis=(0, 2)))))
        h = int(BAND_HALF_UM / PX); return max(yc - h, 0), min(yc + h, ny)
    return c + int(ACROSS_UM[0] / PX), c + int(ACROSS_UM[1] / PX)


def projections(vol, band=None):
    """vol (planes, across, along) -> the three views: from above, from the side across the row, along the bundle."""
    nz, ny, nx = vol.shape; c = ny // 2
    top = vol.max(0)
    a = int(ALONG_HALF_UM / PX); side_across = stretch_z(vol[:, :, c - a:c + a].max(2))
    y0, y1 = band if band is not None else across_band(ny); side_along = stretch_z(vol[:, y0:y1, :].max(1))
    return [top, side_across, side_along]


def outline_on_cutout(m, shape):
    """Automatic outline (z, y, x on the resampled grid of ihc_stereocilia.py) -> the cut-out grid (planes, 222, 222),
    nearest voxel, using the same coordinate mapping as scipy.ndimage.zoom."""
    idx = [np.clip(np.round(np.arange(n) * (mi - 1) / max(n - 1, 1)).astype(int), 0, mi - 1) for n, mi in zip(shape, m.shape)]
    return m[np.ix_(*idx)]


def prefill_mosaic(m, shape, layout, band=None):
    """The automatic outline drawn in the three views, in the same layout as the image."""
    vol = outline_on_cutout(m, shape).astype(np.float32)
    out = np.zeros((max(h for _, _, h, _ in layout), layout[-1][1] + layout[-1][3]), np.uint8)
    for p, (y, x0, h, w) in zip(projections(vol, band), layout):
        out[y:y + h, x0:x0 + w] = (p >= 0.5).astype(np.uint8)
    return out


def views(vol, band=None):
    """vol (planes, across, along) -> three panels and their layout."""
    panels = [scaled(p) for p in projections(vol, band)]
    H = max(p.shape[0] for p in panels); x = 0; layout = []
    for p in panels:
        layout.append((0, x, p.shape[0], p.shape[1])); x += p.shape[1] + GAP
    mosaic = np.zeros((H, x - GAP), np.float32)
    for p, (y, x0, h, w) in zip(panels, layout):
        mosaic[y:y + h, x0:x0 + w] = p
    return mosaic, layout


class Tool:
    def __init__(self):
        self.items = pd.read_csv(os.path.join(DIR, "sample.csv"))
        self.log = os.path.join(DIR, "answers.csv")
        self.ans = pd.read_csv(self.log) if os.path.exists(self.log) else pd.DataFrame(columns=["item", "answer", "saved"])
        done = set(self.ans["item"])
        todo = [k for k, it in enumerate(self.items["item"]) if it not in done]
        self.k = todo[0] if todo else len(self.items) - 1
        self.cache = {}; self.mcache = {}; self.prefill = None
        self.v = napari.Viewer(title="Inner hair cell stereocilia")
        self.keys = {"Enter": self.save_next, "u": self.unreadable, "Backspace": self.back}
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 14
        self.show()

    def volume(self, field, cell):
        if field not in self.cache:
            self.cache = {field: np.load(os.path.join(STACKS, field + ".npz"))["phal"]}
        return self.cache[field][int(cell) - 1].astype(np.float32)

    def outline(self, field, cell):
        if field not in self.mcache:
            self.mcache = {field: np.load(os.path.join(MASKS, field + ".npz"))["outline"]}
        return self.mcache[field][int(cell) - 1]

    def show(self, msg=""):
        it = self.items.iloc[self.k]
        vol = self.volume(it["field"], it["cell"])
        mo = outline_on_cutout(self.outline(it["field"], it["cell"]), vol.shape) if MASKS else None
        self.band = across_band(vol.shape[1], mo); mosaic, self.layout = views(vol, self.band)
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.v.add_image(mosaic, name="phalloidin", colormap="gray")
        old = os.path.join(DIR, f"{int(it['item'])}.npz")
        self.prefill = prefill_mosaic(self.outline(it["field"], it["cell"]), vol.shape, self.layout, self.band) if PREFILL else None
        if os.path.exists(old):
            m = np.load(old)["mosaic"]
        else:
            m = self.prefill.copy() if self.prefill is not None else np.zeros(mosaic.shape, np.uint8)
        g = []
        for (y0, x0, h, w), on in zip(self.layout, (True, False, True)):
            if on:
                for dx in (-CELL_HALF_UM / PX, CELL_HALF_UM / PX):
                    xg = x0 + w / 2 + dx; g.append(np.array([[y0, xg], [y0 + h, xg]]))
        self.v.add_shapes(g, shape_type="line", edge_color="yellow", edge_width=1, name="cell edges", opacity=0.8)
        self.lab = self.v.add_labels(m.astype(np.uint8), name="paint", opacity=0.45)
        self.lab.selected_label = 1; self.lab.brush_size = 4; self.lab.mode = "paint"
        for k, fn in self.keys.items():
            self.lab.bind_key(k, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
        self.v.layers.selection.active = self.lab
        self.v.reset_view()
        prev = self.ans[self.ans["item"] == it["item"]]
        given = f"   (saved: {prev.answer.iloc[0]})" if len(prev) else ""
        self.v.text_overlay.text = (f"Cell {self.k + 1} of {len(self.items)}{given}\n{msg}\n"
                                    "left: from above   middle: side, across the row   right: side, along the bundle\n"
                                    + ("the automatic outline is painted in: correct it  |  " if PREFILL else "")
                                    + "paint the stereocilia of the cell between the yellow lines  |  Enter save, next   u no readable bundle   Backspace back")

    def record(self, answer):
        it = self.items.iloc[self.k]
        self.ans = pd.concat([self.ans[self.ans["item"] != it["item"]],
                              pd.DataFrame([dict(item=int(it["item"]), answer=answer, saved=central_time())])], ignore_index=True)
        self.ans.to_csv(self.log, index=False)

    def save_next(self, viewer):
        it = self.items.iloc[self.k]; m = np.asarray(self.lab.data).astype(np.uint8)
        extra = dict(across_band=np.array(self.band)) if self.prefill is None else dict(prefill=self.prefill, across_band=np.array(self.band))
        np.savez_compressed(os.path.join(DIR, f"{int(it['item'])}.npz"), mosaic=m, layout=np.array(self.layout), **extra)
        unchanged = self.prefill is not None and np.array_equal(m > 0, self.prefill > 0)
        self.record("accepted unchanged" if unchanged and m.any() else "painted" if m.any() else "nothing painted")
        self.advance("Saved")

    def unreadable(self, viewer):
        self.record("no readable bundle"); self.advance("Marked: no readable bundle")

    def advance(self, msg):
        if self.k < len(self.items) - 1:
            self.k += 1; self.show(msg)
        else:
            self.show(msg + ".  All cells done; you can close the window.")

    def back(self, viewer):
        if self.k > 0:
            self.k -= 1; self.show("Back one cell")


if __name__ == "__main__":
    Tool(); napari.run()
