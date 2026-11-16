"""Hand reference for outer hair cell bundles: three clicks per bundle on randomly chosen cells.

Purpose: a reference on typical (randomly chosen, not hand-picked) cells against which every
automatic bundle measure is tuned and scored: direction, axis, span and bend depth. It also
measures how many bundles can be read at all, and, from cells shown twice, how repeatable a
hand reading is (the best agreement any automatic method can be expected to reach).

Sample (made once, before any clicking, and saved; it never changes after that):
    6 fields never used to train the bundle detector or for strokes: 2 control, 2 Smad3 mutant,
    2 beta-catenin mutant (FIELDS below). In each, 20 outer hair cells drawn at random with a
    fixed seed, as evenly as possible across rows 1-3. Cells with no image data at bundle height
    (more than 5% empty pixels in the cell) are left out before drawing and counted in the
    sample file. Then 15 of the 120 are drawn again and shown a second time at the end, in random
    order, without marking (repeatability).
    E:\\Webber_confocal\\annotations\\bundle_points\\sample.csv

What is shown (one cell at a time, the cell's outline faint yellow, native orientation, no
automatic result):
    left    phalloidin
    middle  PMCA2 (blank in fields without PMCA2)
    right   flat top of the cell (mean of the images within 0.3 um of its estimated plate depth)
    The left and middle views can be stepped through heights with the slider or the Left/Right
    arrow keys:
        view 0  brightest phalloidin 1.2 to 4.2 um above the cell's estimated flat top (the
                default; a wider range lets the bright flat top swamp the bundle)
        view 1  brightest phalloidin 1.2 to 4.2 um above the fitted surface (use when view 0
                looks wrong: the flat-top estimate varies a lot between cells and may be off)
        view 2+ single images, from highest to lowest (height above the fitted surface shown)

What to click (in any view; clicks are kept as positions in the field):
    1  one end of the bundle
    2  the bend point: the point of the V, or the middle of the curve's bow; for a straight
       bundle, its middle
    3  the other end
Keys:
    Enter       save as clear, next cell (needs 3 clicks)
    u           save as unsure, next cell (needs 3 clicks)
    n           cannot tell where the bundle is or how it lies; next cell
    Backspace   remove the last click
    p           previous cell
    b           show or hide the outline
    0           back to view 0
Answers save on every step to E:\\Webber_confocal\\annotations\\bundle_points\\bundle_points.csv,
with the view and height on screen at each click; the tool reopens at the first unanswered cell.

Reference measures from the clicks (writes results\\tuning\\bundle_points_reference.csv):
    python click_bundle_points.py --reference
    span_um (end to end), bend_depth_um (bend point to the end-to-end line), arm lengths,
    bend_angle_deg (angle at the bend point), axis_deg (perpendicular to the end-to-end line,
    as an axis, -90..90, 0 = straight across the row), direction_deg (from the middle of the
    ends to the bend point, -180..180, 0 = toward the outer side; left blank when the bend
    depth is under 0.3 um), and for repeated cells the differences between the two readings.

Axis mode (python click_bundle_points.py --axis): the same cells and views, in their own fixed
random order (sample_axis_order.csv), with the 15 repeats again at the end. Two clicks along the
line the bundle faces: 1 at the back of the bundle (its open side), 2 toward the point it faces
(its focal point, where the kinocilium and the notch sit), at least 1.5 um apart. The three shape
clicks give the direction of the bundle's shape; these two give the direction it faces, which can
differ in turned or lopsided bundles. Answers: E:\\Webber_confocal\\annotations\\bundle_points\\bundle_axis.csv.
    python click_bundle_points.py --axis-reference   (results\\tuning\\bundle_axis_reference.csv: facing
    direction, its repeatability, and its difference from the shape direction)

Run from the scripts folder:
    python click_bundle_points.py            (three shape clicks; done 10 November 2026)
    python click_bundle_points.py --axis     (two axis clicks; done 10 November 2026)
    python click_bundle_points.py --axis-survey   (two axis clicks on 20 random outer hair cells in each of the
                                                   other fields, with 20 hidden repeats; sample_axis_survey.csv,
                                                   answers bundle_axis_survey.csv)
    python click_bundle_points.py --axis-survey-reference   (results\\tuning\\bundle_axis_survey.csv)
    (the axis survey was set up but shelved on 10 November 2026: orientation spread is left for a second round)
    python click_bundle_points.py --apex    (three shape clicks on 14 random outer hair cells in each of three apex
                                             fields, AR524 apex, AR454 apex (2) and AR47 apex, plus 8 hidden repeats;
                                             sample_apex.csv, answers bundle_points_apex.csv)
    python click_bundle_points.py --apex-reference   (results\\tuning\\bundle_points_apex_reference.csv)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, os, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
PAINT = os.path.join(S.WORK, 'paint_stacks')
CELLS = S.CLICK_OUTLINES   # the outlines the saved clicks name cells in (first-run cells\v1, frozen copy)
import click_outlines as CO   # run tables keyed by click-outline cell numbers
TYPES = os.path.join(S.WORK, 'cells', 'types')
BUNDLES = os.path.join(S.WORK, 'bundles_detector')
OUT = os.path.join(S.ANNOTATIONS, 'bundle_points')
SAMPLE = os.path.join(OUT, "sample.csv")
ANSWERS = os.path.join(OUT, "bundle_points.csv")
AXIS_SAMPLE = os.path.join(OUT, "sample_axis_order.csv")       # same items, own fixed random order
AXIS_ANSWERS = os.path.join(OUT, "bundle_axis.csv")
AXIS_MIN_UM = 1.5
SURVEY_SAMPLE = os.path.join(OUT, "sample_axis_survey.csv")    # 20 random outer hair cells in each of the other fields
SURVEY_ANSWERS = os.path.join(OUT, "bundle_axis_survey.csv")
SURVEY_PER_FIELD, SURVEY_REPEATS = 20, 20
APEX_FIELDS = ["AR524 Ocm, Calb2 Phall P1 60X APEX", "AR454 Calb2 PMCA2 Phall P1 60X APEX (2)", "AR47 Calb2 PMCA2 Phall P1 60X APEX"]
APEX_SAMPLE = os.path.join(OUT, "sample_apex.csv")
APEX_ANSWERS = os.path.join(OUT, "bundle_points_apex.csv")
APEX_PER_FIELD, APEX_REPEATS = 14, 8
GROUP_OF_MOUSE = {"AR1064": "control", "AR524": "control", "JW5436": "Smad3", "JW5436 (2)": "Smad3", "AR454": "Smad3",
                  "AR1383-2": "beta-catenin", "AR170": "beta-catenin", "AR47": "beta-catenin"}


def mouse_of(field):
    import re
    return re.match(r"^(JW5436 \(2\)|JW5436|AR\d+-?\d*)", field).group(1).strip()
FIELDS = {"AR1064 Ocm Calb2 Phall P1 60X MIDDLE": "control",
          "AR524 Ocm, Calb2 Phall P1 60X BASE": "control",
          "JW5436 Calb2 PMCA2 Phall P1 60X Apex (3)": "Smad3",
          "AR454 Calb2 PMCA2 Phall P1 60X MIDDLE": "Smad3",
          "AR1383-2 Calb2 PMCA2 Phall P1 60X MIDDLE": "beta-catenin",
          "AR47 Calb2 PMCA2 Phall P1 60X MIDDLE": "beta-catenin"}
PER_FIELD, REPEATS, SEED = 20, 15, 20261109
HALF, GAP, PX = 60, 6, 0.1083                      # crop 13 um square
LO_UM, HI_UM = 1.2, 4.2                            # view 0: this range above the estimated flat top


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def plate_depths(field):
    return CO.plate_depths(field)


def window_max(stack, off, depth, lo, hi):
    k = (off >= depth + lo) & (off <= depth + hi)
    return stack[k].max(0) if k.any() else np.zeros(stack.shape[1:], stack.dtype)


def make_sample():
    """Draw the sample once; refuse to redraw if it exists, so the reference never changes."""
    if os.path.exists(SAMPLE):
        return pd.read_csv(SAMPLE)
    rng = np.random.default_rng(SEED); rows = []
    for f, grp in FIELDS.items():
        t = CO.types(f); t = t[t.row.isin(["OHC1", "OHC2", "OHC3"])]
        z = np.load(os.path.join(PAINT, f + ".npz")); ph = z["phal"]; off = z["offsets_um"]
        lab = np.load(os.path.join(CELLS, f + ".npy")); plate = plate_depths(f)
        usable, empty = [], 0
        for _, r in t.iterrows():
            m = lab == r.cell
            d = plate.get(r.cell, np.nan); d = 0.6 if not np.isfinite(d) else d
            w = window_max(ph, off, d, LO_UM, HI_UM)
            if m.sum() < 50 or (w[m] <= 0).mean() > 0.05:
                empty += 1; continue
            usable.append(r)
        u = pd.DataFrame(usable)
        per_row = {r: PER_FIELD // 3 for r in ("OHC1", "OHC2", "OHC3")}
        for r in list(per_row)[: PER_FIELD - 3 * (PER_FIELD // 3)]:
            per_row[r] += 1
        take = []
        for r, n in per_row.items():
            pool = u[u.row == r]
            take.append(pool.sample(min(n, len(pool)), random_state=int(rng.integers(1e9))))
        short = PER_FIELD - sum(len(x) for x in take)
        if short > 0:                                    # a row with too few cells: fill from the others
            rest = u[~u.cell.isin(pd.concat(take).cell)]
            take.append(rest.sample(min(short, len(rest)), random_state=int(rng.integers(1e9))))
        d = pd.concat(take)
        for _, r in d.iterrows():
            rows.append(dict(field=f, group=grp, cell=int(r.cell), row=r.row, left_out_no_data_in_field=empty))
    s = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    s["repeat"] = False
    rep = s.sample(REPEATS, random_state=int(rng.integers(1e9))).copy(); rep["repeat"] = True
    s = pd.concat([s, rep.sample(frac=1, random_state=int(rng.integers(1e9)))]).reset_index(drop=True)
    s.insert(0, "item", np.arange(1, len(s) + 1))
    os.makedirs(OUT, exist_ok=True); s.to_csv(SAMPLE, index=False)
    return s


def scaled(a):
    lo, hi = np.percentile(a, [1, 99.8])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


class Field:
    def __init__(self, f):
        z = np.load(os.path.join(PAINT, f + ".npz"))
        self.ph = z["phal"]; self.off = z["offsets_um"]; self.pm = z["pmca2"] if "pmca2" in z.files else None
        self.lab = np.load(os.path.join(CELLS, f + ".npy")); self.plate = plate_depths(f)

    def views(self, cell):
        """Stack of composite images (views x 2*HALF x 6*HALF+2*GAP) and the outline image."""
        from skimage.segmentation import find_boundaries
        m = self.lab == cell; ys, xs = np.nonzero(m); cy, cx = int(round(ys.mean())), int(round(xs.mean()))
        d = self.plate.get(cell, np.nan); d = 0.6 if not np.isfinite(d) else d
        H, W = self.lab.shape; y0, x0 = cy - HALF, cx - HALF

        def crop(a):
            o = np.zeros((2 * HALF, 2 * HALF), np.float32)
            ya, yb, xa, xb = max(y0, 0), min(y0 + 2 * HALF, H), max(x0, 0), min(x0 + 2 * HALF, W)
            o[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]; return o

        def series(stack):
            v = [window_max(stack, self.off, d, LO_UM, HI_UM), window_max(stack, self.off, 0.0, LO_UM, HI_UM)]
            v += [stack[k] for k in range(len(self.off))]          # stack order: highest first
            return [scaled(crop(a)) for a in v]
        phs = series(self.ph)
        pms = series(self.pm) if self.pm is not None else [np.zeros((2 * HALF, 2 * HALF), np.float32)] * len(phs)
        flat = scaled(crop(self.ph[np.abs(self.off - d) <= 0.31].mean(0)))
        n = len(phs); comp = np.zeros((n, 2 * HALF, 6 * HALF + 2 * GAP), np.float32)
        for i in range(n):
            comp[i, :, :2 * HALF] = phs[i]; comp[i, :, 2 * HALF + GAP:4 * HALF + GAP] = pms[i]; comp[i, :, 4 * HALF + 2 * GAP:] = flat
        edge = crop(find_boundaries(m, mode="inner").astype(np.float32)); ol = np.zeros(comp.shape[1:], np.float32)
        for k in range(3):
            ol[:, k * (2 * HALF + GAP):k * (2 * HALF + GAP) + 2 * HALF] = edge
        labels = ["view 0: brightest 1.2-4.2 um above the cell's estimated flat top", "view 1: brightest 1.2-4.2 um above the fitted surface"] + \
                 [f"single image {h:+.1f} um from the fitted surface" for h in self.off]
        return comp, ol, (y0, x0), labels, self.pm is not None


def to_field(y, x, origin):
    """Composite coordinates to field pixels and the view clicked."""
    k = int(x // (2 * HALF + GAP)); xl = x - k * (2 * HALF + GAP)
    if k > 2 or xl >= 2 * HALF or not (0 <= y < 2 * HALF):
        return None
    return dict(panel=["phalloidin", "PMCA2", "flat top"][k], x=round(xl + origin[1], 1), y=round(y + origin[0], 1))


class Tool:
    N = 3
    TITLE = "Bundle reference: three clicks per bundle"
    PROMPT = "Click 1 end, 2 bend point (point of the V or middle of the bow), 3 other end"
    NEED = "Needs 3 clicks (end, bend point, end), or press n"

    def path(self):
        return ANSWERS

    def __init__(self, sample):
        import napari
        self.napari = napari; self.s = sample; self.fields = {}
        self.clicks, self.labels, self.has_pm, self.origin = [], [], False, (0, 0)
        os.makedirs(OUT, exist_ok=True)
        cols = ["item", "field", "cell", "row", "repeat", "answer"] + [f"{a}{i}" for i in range(1, self.N + 1) for a in ("x", "y", "panel", "view")] + ["saved"]
        self.ans = pd.read_csv(self.path()) if os.path.exists(self.path()) else pd.DataFrame(columns=cols)
        done = set(self.ans["item"])
        self.k = next((i for i in range(len(sample)) if sample["item"].iloc[i] not in done), len(sample) - 1)
        self.v = napari.Viewer(title=self.TITLE)
        self.v.mouse_drag_callbacks.append(self.on_mouse)
        self.keys = {"Enter": lambda v: self.save("clear"), "u": lambda v: self.save("unsure"), "n": lambda v: self.save("cannot tell"),
                     "Backspace": self.undo, "p": self.prev, "b": self.toggle, "0": self.home}
        for k, fn in self.keys.items():
            self.v.bind_key(k, fn, overwrite=True)
        self.v.text_overlay.visible = True; self.v.text_overlay.font_size = 13
        self.v.dims.events.current_step.connect(lambda e: self.status())
        self.load()

    def field(self, f):
        if f not in self.fields:
            self.fields[f] = Field(f)
        return self.fields[f]

    def load(self):
        r = self.s.iloc[self.k]
        self.clicks = []
        comp, ol, self.origin, self.labels, self.has_pm = self.field(r.field).views(int(r.cell))
        for lyr in list(self.v.layers):
            self.v.layers.remove(lyr)
        self.v.add_image(comp, name="cell", colormap="gray")
        self.v.add_image(ol, name="outline", colormap="yellow", blending="additive", opacity=0.3)
        self.line = self.v.add_shapes(name="line", edge_color="cyan", edge_width=1.0, face_color="transparent")
        self.pts = self.v.add_points(np.zeros((0, 2)), ndim=2, name="clicks", size=4, face_color="red", border_color="white",
                                     text={"string": "{n}", "size": 12, "color": "cyan", "anchor": "upper_left"},
                                     features={"n": np.zeros(0, int)})
        for lyr in self.v.layers:
            for key, fn in self.keys.items():
                lyr.bind_key(key, (lambda g: (lambda layer: g(self.v)))(fn), overwrite=True)
            try:
                lyr.mouse_pan = True
            except Exception:
                pass
        self.clicks = []
        prev = self.ans[self.ans["item"] == r["item"]]
        if len(prev) and prev.answer.iloc[0] in ("clear", "unsure"):
            for i in range(1, self.N + 1):
                p = prev.iloc[0]
                k = ["phalloidin", "PMCA2", "flat top"].index(p[f"panel{i}"])
                self.clicks.append(dict(y=p[f"y{i}"] - self.origin[0], x=p[f"x{i}"] - self.origin[1] + k * (2 * HALF + GAP),
                                        view=int(p[f"view{i}"])))
        self.v.dims.set_current_step(0, 0); self.draw(); self.v.reset_view(); self.status()

    def draw(self):
        self.pts.data = np.array([[c["y"], c["x"]] for c in self.clicks]) if self.clicks else np.zeros((0, 2))
        self.pts.features = pd.DataFrame({"n": np.arange(1, len(self.clicks) + 1)})
        self.pts.refresh()
        self.line.data = []
        if len(self.clicks) >= 2:
            self.line.add_paths([np.array([[c["y"], c["x"]] for c in self.clicks])])

    def status(self, msg=""):
        r = self.s.iloc[self.k]; n = self.ans.answer.notna().sum()
        view = int(self.v.dims.current_step[0]) if self.v.layers else 0
        view = view if 0 <= view < len(self.labels) else 0
        self.v.text_overlay.text = (
            f"Cell {self.k + 1} of {len(self.s)}  ({n} answered)   {self.labels[view] if self.labels else ''}\n"
            f"Left: phalloidin   Middle: {'PMCA2' if self.has_pm else 'no PMCA2 in this field'}   Right: flat top   (Left/Right arrows step views)\n"
            f"{self.PROMPT}   [{len(self.clicks)} of {self.N}]\n"
            "Enter clear   u unsure   n cannot tell   Backspace undo click   p previous   b outline   0 view 0\n" + msg)

    def on_mouse(self, viewer, event):
        start = np.array(event.position[-2:]); yield
        moved = False
        while event.type == "mouse_move":
            if np.linalg.norm(np.array(event.position[-2:]) - start) > 3:
                moved = True
            yield
        if moved or event.button != 1:
            return
        if len(self.clicks) >= self.N:
            return self.status(f"Already {self.N} clicks: Backspace to redo one, or Enter / u to save")
        y, x = start
        if to_field(y, x, self.origin) is None:
            return self.status("Click inside one of the three views")
        self.clicks.append(dict(y=float(y), x=float(x), view=int(self.v.dims.current_step[0])))
        self.draw(); self.status()

    def save(self, answer):
        if answer in ("clear", "unsure") and len(self.clicks) != self.N:
            return self.status(self.NEED)
        if answer in ("clear", "unsure") and not self.valid():
            return self.status(self.INVALID)
        r = self.s.iloc[self.k]
        rec = dict(item=int(r["item"]), field=r.field, cell=int(r.cell), row=r.row, repeat=bool(r["repeat"]), answer=answer, saved=central_time())
        if answer != "cannot tell":
            for i, c in enumerate(self.clicks, 1):
                f = to_field(c["y"], c["x"], self.origin)
                rec.update({f"x{i}": f["x"], f"y{i}": f["y"], f"panel{i}": f["panel"], f"view{i}": c["view"]})
        self.ans = pd.concat([self.ans[self.ans["item"] != r["item"]], pd.DataFrame([rec])], ignore_index=True)
        self.ans.to_csv(self.path(), index=False)
        if self.k < len(self.s) - 1:
            self.k += 1; self.load()
        else:
            self.status("Last cell done; you can close the window.")

    INVALID = ""

    def valid(self):
        return True

    def undo(self, viewer):
        if self.clicks:
            self.clicks.pop(); self.draw(); self.status()

    def prev(self, viewer):
        if self.k > 0:
            self.k -= 1; self.load()

    def toggle(self, viewer):
        self.v.layers["outline"].visible = not self.v.layers["outline"].visible

    def home(self, viewer):
        self.v.dims.set_current_step(0, 0)


def make_random_sample(fields, per_field, repeats, seed, path):
    """Outer hair cells drawn at random per field (fixed seed, spread evenly across rows 1-3), cells with no image
    data at bundle height left out and counted, then `repeats` of them again at the end without marking. Made once."""
    if os.path.exists(path):
        return pd.read_csv(path)
    rng = np.random.default_rng(seed); rows = []
    for f in fields:
        t = CO.types(f); t = t[t.row.isin(["OHC1", "OHC2", "OHC3"])]
        z = np.load(os.path.join(PAINT, f + ".npz")); ph = z["phal"]; off = z["offsets_um"]
        lab = np.load(os.path.join(CELLS, f + ".npy")); plate = plate_depths(f)
        usable, empty = [], 0
        for _, r in t.iterrows():
            m = lab == r.cell
            d = plate.get(r.cell, np.nan); d = 0.6 if not np.isfinite(d) else d
            w = window_max(ph, off, d, LO_UM, HI_UM)
            if m.sum() < 50 or (w[m] <= 0).mean() > 0.05:
                empty += 1; continue
            usable.append(r)
        u = pd.DataFrame(usable)
        if u.empty:
            continue
        per_row = {r: per_field // 3 for r in ("OHC1", "OHC2", "OHC3")}
        for r in list(per_row)[: per_field - 3 * (per_field // 3)]:
            per_row[r] += 1
        take = [u[u.row == r].sample(min(n, int((u.row == r).sum())), random_state=int(rng.integers(1e9))) for r, n in per_row.items()]
        short = per_field - sum(len(x) for x in take)
        if short > 0:
            rest = u[~u.cell.isin(pd.concat(take).cell)]
            take.append(rest.sample(min(short, len(rest)), random_state=int(rng.integers(1e9))))
        for _, r in pd.concat(take).iterrows():
            rows.append(dict(field=f, group=GROUP_OF_MOUSE.get(mouse_of(f), ""), cell=int(r.cell), row=r.row, left_out_no_data_in_field=empty))
    s = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True); s["repeat"] = False
    rep = s.sample(repeats, random_state=int(rng.integers(1e9))).copy(); rep["repeat"] = True
    s = pd.concat([s, rep.sample(frac=1, random_state=int(rng.integers(1e9)))]).reset_index(drop=True)
    s.insert(0, "item", np.arange(1, len(s) + 1))
    s.to_csv(path, index=False)
    return s


class ApexTool(Tool):
    """Three shape clicks on random outer hair cells of three apex fields (sample_apex.csv)."""
    TITLE = "Bundle reference, apex fields: three clicks per bundle"

    def path(self):
        return APEX_ANSWERS


def make_survey_sample():
    """20 outer hair cells drawn at random in every field outside the reference sample (made once; never redrawn),
    by the same rules as make_sample, then 20 of them again at the end without marking."""
    if os.path.exists(SURVEY_SAMPLE):
        return pd.read_csv(SURVEY_SAMPLE)
    import glob
    rng = np.random.default_rng(SEED + 2); rows = []
    fields = sorted(os.path.basename(f)[:-10] for f in glob.glob(os.path.join(TYPES, "*_types.csv")))
    fields = [f for f in fields if f not in FIELDS and os.path.exists(os.path.join(PAINT, f + ".npz")) and os.path.exists(os.path.join(BUNDLES, f + "_bundles.csv"))]
    for f in fields:
        t = CO.types(f); t = t[t.row.isin(["OHC1", "OHC2", "OHC3"])]
        z = np.load(os.path.join(PAINT, f + ".npz")); ph = z["phal"]; off = z["offsets_um"]
        lab = np.load(os.path.join(CELLS, f + ".npy")); plate = plate_depths(f)
        usable, empty = [], 0
        for _, r in t.iterrows():
            m = lab == r.cell
            d = plate.get(r.cell, np.nan); d = 0.6 if not np.isfinite(d) else d
            w = window_max(ph, off, d, LO_UM, HI_UM)
            if m.sum() < 50 or (w[m] <= 0).mean() > 0.05:
                empty += 1; continue
            usable.append(r)
        u = pd.DataFrame(usable)
        if u.empty:
            continue
        per_row = {r: SURVEY_PER_FIELD // 3 for r in ("OHC1", "OHC2", "OHC3")}
        for r in list(per_row)[: SURVEY_PER_FIELD - 3 * (SURVEY_PER_FIELD // 3)]:
            per_row[r] += 1
        take = [u[u.row == r].sample(min(n, int((u.row == r).sum())), random_state=int(rng.integers(1e9))) for r, n in per_row.items()]
        short = SURVEY_PER_FIELD - sum(len(x) for x in take)
        if short > 0:
            rest = u[~u.cell.isin(pd.concat(take).cell)]
            take.append(rest.sample(min(short, len(rest)), random_state=int(rng.integers(1e9))))
        for _, r in pd.concat(take).iterrows():
            rows.append(dict(field=f, group=GROUP_OF_MOUSE.get(mouse_of(f), ""), cell=int(r.cell), row=r.row, left_out_no_data_in_field=empty))
    s = pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True); s["repeat"] = False
    rep = s.sample(SURVEY_REPEATS, random_state=int(rng.integers(1e9))).copy(); rep["repeat"] = True
    s = pd.concat([s, rep.sample(frac=1, random_state=int(rng.integers(1e9)))]).reset_index(drop=True)
    s.insert(0, "item", np.arange(1, len(s) + 1))
    s.to_csv(SURVEY_SAMPLE, index=False)
    return s


class AxisTool(Tool):
    """Two clicks along the line the bundle faces: first at the back of the bundle (its open side),
    second toward its focal point (the side it faces, where the kinocilium and the notch sit)."""
    N = 2
    TITLE = "Bundle reference: the axis the bundle faces"
    PROMPT = "Click 1 at the back of the bundle (open side), 2 toward the point it faces (focal point)"
    NEED = "Needs 2 clicks (back of the bundle, then toward the point it faces), or press n"
    INVALID = f"The two clicks must be at least {AXIS_MIN_UM} um apart; Backspace and click again"

    def path(self):
        return AXIS_ANSWERS

    def valid(self):
        a, b = self.clicks
        xa = a["x"] % (2 * HALF + GAP); xb = b["x"] % (2 * HALF + GAP)
        return np.hypot(a["y"] - b["y"], xa - xb) * PX >= AXIS_MIN_UM


class SurveyTool(AxisTool):
    """Axis clicks on the random sample of every other field (sample_axis_survey.csv)."""
    TITLE = "Bundle axis survey: the axis the bundle faces"

    def path(self):
        return SURVEY_ANSWERS


def axis_order(sample):
    """The same items as the shape sample, in their own fixed random order (made once)."""
    if os.path.exists(AXIS_SAMPLE):
        return pd.read_csv(AXIS_SAMPLE)
    rng = np.random.default_rng(SEED + 1)
    first = sample[~sample["repeat"]].sample(frac=1, random_state=int(rng.integers(1e9)))
    rep = sample[sample["repeat"]].sample(frac=1, random_state=int(rng.integers(1e9)))
    s = pd.concat([first, rep]).reset_index(drop=True); s.to_csv(AXIS_SAMPLE, index=False)
    return s


def frame_of(field, cell):
    """Row direction at the cell from the curve fitted through the inner hair cells (spatial_stats.row_frame)."""
    from spatial_stats import smooth_row_frame, row_frame, ROWS
    t = CO.types(field); t = smooth_row_frame(t[t.row.isin(ROWS)].copy())
    return row_frame(t, cell)


def measures(p1, pb, p2, frame):
    along, across = frame; ch = p2 - p1; L = np.linalg.norm(ch)
    if L == 0:
        return {}
    n = np.array([-ch[1], ch[0]]) / L; mid = (p1 + p2) / 2; bis = pb - mid; depth = abs(n @ (pb - p1))
    a1, a2 = np.linalg.norm(p1 - pb), np.linalg.norm(p2 - pb)
    ang = np.degrees(np.arccos(np.clip(((p1 - pb) @ (p2 - pb)) / max(a1 * a2, 1e-9), -1, 1)))
    axis = (np.degrees(np.arctan2(n @ along, n @ across)) + 90) % 180 - 90
    direction = np.degrees(np.arctan2(bis @ along, bis @ across)) if depth * PX >= 0.3 else np.nan
    return dict(span_um=L * PX, bend_depth_um=depth * PX, arm_short_um=min(a1, a2) * PX, arm_long_um=max(a1, a2) * PX,
                bend_angle_deg=ang, axis_deg=axis, direction_deg=direction)


def reference(answers=None, out_name="bundle_points_reference.csv"):
    a = pd.read_csv(answers or ANSWERS); rows = []
    for _, r in a.iterrows():
        rec = dict(item=r["item"], field=r.field, group=FIELDS.get(r.field, GROUP_OF_MOUSE.get(mouse_of(r.field), "")), cell=r.cell, row=r.row,
                   repeat=r["repeat"], answer=r.answer)
        if r.answer in ("clear", "unsure"):
            fr = frame_of(r.field, int(r.cell))
            p = [np.array([r[f"x{i}"], r[f"y{i}"]], float) for i in (1, 2, 3)]
            rec.update(measures(p[0], p[1], p[2], fr))
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["repeat"] = d["repeat"].astype(str).str.strip().str.lower().isin(["true", "1"])     # robust to how the csv stored it
    out = os.path.join(PROJ, "results", "tuning"); os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, out_name), index=False)
    first = d[~d["repeat"]]
    print(f"answered {len(d)} ({len(first)} first readings): " + ", ".join(f"{k} {v}" for k, v in first.answer.value_counts().items()))
    print(first.groupby("group").answer.value_counts(normalize=True).unstack().round(2).to_string())
    rep = d[d["repeat"]].merge(first, on=["field", "cell"], suffixes=("_2", "_1"))
    rep = rep[rep.answer_1.isin(["clear", "unsure"]) & rep.answer_2.isin(["clear", "unsure"])]
    if len(rep):
        dd = lambda a, b, p: np.abs((a - b + p / 2) % p - p / 2)
        print(f"repeat readings ({len(rep)}): span difference median {np.nanmedian(np.abs(rep.span_um_2 - rep.span_um_1)):.2f} um, "
              f"bend depth {np.nanmedian(np.abs(rep.bend_depth_um_2 - rep.bend_depth_um_1)):.2f} um, "
              f"axis {np.nanmedian(dd(rep.axis_deg_2, rep.axis_deg_1, 180)):.0f} deg, direction {np.nanmedian(dd(rep.direction_deg_2, rep.direction_deg_1, 360)):.0f} deg")


def axis_reference(answers=None, out_name="bundle_axis_reference.csv"):
    """Direction each bundle faces (click 1 to click 2) in the row frame, its repeatability, and its
    difference from the direction of the bundle's shape (middle of the ends to the bend point)."""
    a = pd.read_csv(answers or AXIS_ANSWERS); rows = []
    for _, r in a.iterrows():
        rec = dict(item=r["item"], field=r.field, group=FIELDS.get(r.field, GROUP_OF_MOUSE.get(mouse_of(r.field), "")), cell=r.cell, row=r.row,
                   repeat=r["repeat"], answer=r.answer)
        if r.answer in ("clear", "unsure"):
            along, across = frame_of(r.field, int(r.cell))
            v = np.array([r["x2"] - r["x1"], r["y2"] - r["y1"]], float)
            rec["facing_deg"] = float(np.degrees(np.arctan2(v @ along, v @ across)))
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["repeat"] = d["repeat"].astype(str).str.strip().str.lower().isin(["true", "1"])
    shape = os.path.join(PROJ, "results", "tuning", "bundle_points_reference.csv")
    first = d[~d["repeat"]]
    if os.path.exists(shape):
        sh = pd.read_csv(shape); sh = sh[~sh["repeat"].astype(str).str.lower().eq("true")][["field", "cell", "answer", "direction_deg"]]
        first = first.merge(sh.rename(columns={"answer": "shape_answer", "direction_deg": "shape_direction_deg"}), on=["field", "cell"], how="left")
        first["facing_minus_shape_deg"] = (first.facing_deg - first.shape_direction_deg + 180) % 360 - 180
    out = os.path.join(PROJ, "results", "tuning"); os.makedirs(out, exist_ok=True)
    first["facing_toward_bend_deg"] = (first["facing_deg"] + 360) % 360 - 180      # same convention as the shape direction
    first.to_csv(os.path.join(out, out_name), index=False)
    print(f"answered {len(d)} ({len(first)} first readings): " + ", ".join(f"{k} {v}" for k, v in first.answer.value_counts().items()))
    if "facing_minus_shape_deg" in first:
        k = first.facing_minus_shape_deg.dropna()
        print(f"facing direction minus shape direction: median abs {k.abs().median():.1f} deg, more than 30 deg in {np.mean(k.abs() > 30):.2f} (n {len(k)})")
    rep = d[d["repeat"]].merge(d[~d["repeat"]], on=["field", "cell"], suffixes=("_2", "_1")).dropna(subset=["facing_deg_1", "facing_deg_2"])
    if len(rep):
        dd = np.abs((rep.facing_deg_2 - rep.facing_deg_1 + 180) % 360 - 180)
        print(f"repeat readings ({len(rep)}): facing direction difference median {np.median(dd):.1f} deg")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reference", action="store_true"); ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--axis", action="store_true"); ap.add_argument("--axis-reference", action="store_true")
    ap.add_argument("--axis-survey", action="store_true"); ap.add_argument("--axis-survey-reference", action="store_true")
    ap.add_argument("--apex", action="store_true"); ap.add_argument("--apex-reference", action="store_true")
    a = ap.parse_args()
    if a.reference:
        return reference()
    if a.axis_reference:
        return axis_reference()
    if a.apex_reference:
        return reference(APEX_ANSWERS, "bundle_points_apex_reference.csv")
    if a.apex:
        t = ApexTool(make_random_sample(APEX_FIELDS, APEX_PER_FIELD, APEX_REPEATS, SEED + 3, APEX_SAMPLE)); t.napari.run(); return
    if a.axis_survey_reference:
        return axis_reference(SURVEY_ANSWERS, "bundle_axis_survey.csv")
    if a.axis_survey:
        t = SurveyTool(make_survey_sample()); t.napari.run(); return
    s = make_sample()
    if a.axis:
        t = AxisTool(axis_order(s)); t.napari.run(); return
    if a.sample_only:
        print(s.groupby(["group", "field"]).size().to_string()); return
    t = Tool(s)
    t.napari.run()


if __name__ == "__main__":
    sys.exit(main())
