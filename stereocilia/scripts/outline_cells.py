"""Step 3 of the stereocilia plan: outline every cell at the cell surface with Cellpose-SAM.

Input: E:\\Webber_confocal\\work\\fields\\<field>.npz from make_field_pictures.py (each stain
within 2 um of the cell surface). By default Cellpose-SAM sees phalloidin only (cell borders,
cuticular plates, bundles). With --stains three it also sees Calb2 and the EGFP stain, but
Calb2 then pulls the outlines toward inner hair cell bodies, which lie beside their surfaces
in these fields. Calb2 and the EGFP stain are always measured in each outline for the type guess.

Output, per outline version (--tag, for example v0 for the untrained model, r1 after the
first training round):
    E:\\Webber_confocal\\work\\cells\\<tag>\\<field>.npy        cell outlines (0 = no cell)
    E:\\Webber_confocal\\work\\cells\\<tag>\\<field>_cells.csv  one row per cell: area,
        center, mean of each stain in the cell, and a first guess at its type
    figures\\cells_<tag>\\<field>.png                          outlines drawn on phalloidin

Run from the scripts folder:
    python outline_cells.py --tag v0                         (all fields, about 15 minutes)
    python outline_cells.py --tag r1 --model <trained model file>
    python outline_cells.py --tag v0 --only "AR524 Ocm, Calb2 Phall P1 60X BASE"
    python outline_cells.py --tag v1 --base v0      (adds inner hair cells from Calb2 to the v0 outlines)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch and numpy can bring two copies of one math library on Windows
import torch                                             # noqa: F401  (load before numpy's libraries)
import argparse, glob, sys, time
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
FIELDS = os.path.join(S.WORK, 'fields')
CELLS = os.path.join(S.WORK, 'cells')


def three_stain_image(z):
    """Y x X x 3 image: phalloidin, Calb2, EGFP stain, each scaled 0-1 (1st to 99.8th percentile)."""
    stains = list(z["stains"]); L = z["surface_layers"]
    out = []
    for s in ["Phalloidin", "Calb2", ("Ocm", "PMCA2")]:
        names = s if isinstance(s, tuple) else (s,)
        idx = [i for i, n in enumerate(stains) if n in names]
        if not idx:
            out.append(np.zeros(L.shape[1:], np.float32)); continue
        im = L[idx[0]].astype(np.float32)
        lo, hi = np.percentile(im, [1, 99.8])
        out.append(np.clip((im - lo) / max(hi - lo, 1), 0, 1))
    return np.stack(out, -1), stains


def cell_table(masks, img, px):
    lab = np.arange(1, masks.max() + 1)
    if len(lab) == 0:
        return pd.DataFrame()
    area = ndi.sum(np.ones_like(masks), masks, lab) * px * px
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(masks), masks, lab)).T
    t = pd.DataFrame(dict(cell=lab, area_um2=area.round(2), y_px=cy.round(1), x_px=cx.round(1)))
    for k, name in enumerate(["phalloidin", "calb2", "egfp_stain"]):
        t[f"{name}_mean"] = ndi.mean(img[..., k], masks, lab).round(4)
    t = t[t.area_um2 > 0].reset_index(drop=True)
    t["type_guess"] = "other"
    for col, typ in [("calb2_mean", "inner"), ("egfp_stain_mean", "outer")]:
        v = t[col].values
        if v.max() > 0 and len(v) > 5:
            thr = threshold_otsu(v)
            t.loc[(v > thr) & (t.type_guess == "other"), "type_guess"] = typ
    return t


def add_calb2_cells(m, masks, z, stains):
    """Inner hair cells from Calb2 over all depths: Cellpose outlines every Calb2 cell, cells
    brighter than the field's Otsu threshold of cell means are kept (dim ones are outer hair
    cells with weak Calb2), phalloidin outlines more than half covered by them are dropped,
    and the Calb2 outlines are written in on top."""
    im = z["maxproj"][stains.index("Calb2")].astype(np.float32)
    lo, hi = np.percentile(im, [1, 99.8]); im = np.clip((im - lo) / max(hi - lo, 1), 0, 1)
    cm, _, _ = m.eval(im, normalize=False)
    ids = np.arange(1, cm.max() + 1)
    if len(ids) < 3:
        return masks, []
    mean = ndi.mean(im, cm, ids)
    keep = ids[mean > threshold_otsu(mean)]
    inner = np.isin(cm, keep)
    old = np.unique(masks); old = old[old > 0]
    if len(old):
        covered = ndi.mean(inner.astype(float), masks, old)
        masks = np.where(np.isin(masks, old[covered > 0.5]), 0, masks)
    start = int(masks.max()) + 1
    new_ids = []
    for k, c in enumerate(keep):
        masks[cm == c] = start + k; new_ids.append(start + k)
    return masks, new_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--model", default="cpsam", help="cpsam (untrained) or the path of a trained model")
    ap.add_argument("--only", default="")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="use the graphics card if torch can see one")
    ap.add_argument("--base", default="", help="reuse the phalloidin outlines of this version (for example v0) instead of running them again")
    ap.add_argument("--no-calb2", action="store_true", help="skip the inner hair cell outlines from Calb2")
    ap.add_argument("--stains", default="phalloidin", choices=["phalloidin", "three"],
                    help="what Cellpose sees: phalloidin only (default) or phalloidin, Calb2 and the EGFP stain")
    a = ap.parse_args()
    from cellpose import models
    out = os.path.join(CELLS, a.tag); fig_dir = os.path.join(S.FIGURES, f"cells_{a.tag}")
    os.makedirs(out, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    gpu = a.gpu and torch.cuda.is_available()
    m = models.CellposeModel(gpu=gpu, pretrained_model=a.model)
    files = sorted(glob.glob(os.path.join(FIELDS, "*.npz")))
    files = [f for f in files if a.only in os.path.basename(f)]
    print(f"{len(files)} fields, model {a.model}, graphics card {'on' if gpu else 'off'}", flush=True)
    for i, f in enumerate(files, 1):
        field = os.path.splitext(os.path.basename(f))[0]
        dst = os.path.join(out, field + ".npy")
        z = np.load(f)
        if z["surface_layers"].size == 0 or float(z["voxel_um"][2]) > 0.2:
            continue                                                          # no phalloidin, or the 10x overview
        if os.path.exists(dst) and not a.redo:
            print(f"  [{i}/{len(files)}] {field}: already done", flush=True); continue
        t0 = time.time()
        img, stains = three_stain_image(z)
        base = os.path.join(CELLS, a.base, field + ".npy") if a.base else ""
        if base and os.path.exists(base):
            masks = np.load(base).astype(np.int32)
        elif a.stains == "three":
            masks, _, _ = m.eval(img, channel_axis=-1, normalize=False)
        else:   # phalloidin only: Calb2 fills inner hair cell bodies, which sit beside their surfaces
            masks, _, _ = m.eval(img[..., 0], normalize=False)
        masks = masks.astype(np.int32)
        inner_ids = []
        if not a.no_calb2 and "Calb2" in stains:
            masks, inner_ids = add_calb2_cells(m, masks, z, stains)
        masks = masks.astype(np.uint16)
        np.save(dst, masks)
        t = cell_table(masks, img, float(z["voxel_um"][2]))
        t["source"] = np.where(t.cell.isin(inner_ids), "calb2", "phalloidin")
        t.loc[t.source == "calb2", "type_guess"] = "inner"
        t.loc[(t.source == "phalloidin") & (t.type_guess == "inner"), "type_guess"] = "other"
        t.insert(0, "field", field); t.to_csv(os.path.join(out, field + "_cells.csv"), index=False)
        rgb = np.dstack([img[..., 0]] * 3); b = find_boundaries(masks, mode="inner")
        colors = {"inner": [1, 0.3, 1], "outer": [0.2, 1, 0.2], "other": [1, 0.8, 0.1]}
        for typ, c in colors.items():
            sel = np.isin(masks, t.cell[t.type_guess == typ].values) & b
            rgb[sel] = c
        fig, ax = plt.subplots(figsize=(11, 11)); ax.imshow(rgb); ax.axis("off")
        ax.set_title(f"{field}: {len(t)} cells (magenta inner, green outer, yellow other)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, field + ".png"), dpi=70); plt.close(fig)
        n = t.type_guess.value_counts().to_dict()
        print(f"  [{i}/{len(files)}] {field}: {len(t)} cells {n}, {round(time.time() - t0)} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
