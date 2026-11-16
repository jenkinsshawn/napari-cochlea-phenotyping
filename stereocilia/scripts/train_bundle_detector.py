"""Train and run the bundle detector: a pixel classifier for "stereocilia bundle" in phalloidin.

Training examples
    Bundle: Shawn's red strokes from paint_bundles.py (E:\\Webber_confocal\\annotations\\bundle_strokes).
    Not bundle: his blue strokes (second round, painted over the band, borders and gaps at
    bundle heights), except within 0.5 um of a red stroke in the same image; the rest of each
    outer hair cell he painted, more than 1.2 um from its red stroke, at the stroke's heights;
    and examples taken where no bundle can be:
      - every image 0.6 um or more below the cell surface,
      - each hair cell's own cuticular plate: pixels inside the cell outline at the image where
        the cell's mean phalloidin peaks,
      - band cells (pillar cells) at all heights,
      - background more than 3 um from any hair cell,
    always more than 0.6 um from any red stroke.
Pixel features (per image of the flattened stack, 0.3 um apart): smoothed brightness at three
widths, the difference between narrow and wide smoothing, edge strength, two line and blob
measures (Hessian eigenvalues) at two widths, the same brightness and line measures one and two
images above and below, and the change in brightness from the image below (height above the
surface was used in rounds 1 and 2 and is dropped from round 3, because the public stacks have no
surface). Classifier: random forest, 200 trees. With --public, the resampled VASCilia bundle
outlines (prepare_vascilia.py) are added to training; the check still scores Dominika's fields only.

Check: each painted field is left out in turn, the forest is trained on the others, and on the
left-out field it reports, for each kind of example, the share of pixels called bundle
(the share for blue strokes and the rest of painted cells is the fair test of false bundles). Results go to results\\bundle_detector_check.csv.

Run from the scripts folder:
    python train_bundle_detector.py --check        (leave-one-field-out check, about 5 minutes)
    python train_bundle_detector.py --train        (final detector on all painted fields; add --public to include VASCilia)
    python train_bundle_detector.py --predict --only "AR524 Ocm, Calb2 Phall P1 60X MIDDLE"
Prediction writes E:\\Webber_confocal\\work\\bundle_prob\\<field>.npz: probability per image,
its maximum over heights and the height of that maximum.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, glob, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.ensemble import RandomForestClassifier
import joblib
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
STACKS = os.path.join(S.WORK, 'paint_stacks')
STROKES = os.path.join(S.ANNOTATIONS, 'bundle_strokes')
CELLS = os.path.join(S.WORK, 'cells')
ANN = os.path.join(S.ANNOTATIONS, 'cells')
TYPES = os.path.join(CELLS, "types")
PROB = os.path.join(S.WORK, 'bundle_prob')
MODELS = {False: os.path.join(S.MODELS, 'bundle_detector_strokes.joblib'),          # Shawn's strokes only, with height feature
          True: os.path.join(S.MODELS, 'bundle_detector_strokes_public.joblib')}    # plus VASCilia, no height feature
USE_HEIGHT = True
PUBLIC = os.path.join(S.PUBLIC, 'vascilia', 'resampled')
N_PUBLIC = 3000                # training pixels per kind per public stack
PX, DZ = S.PX_UM, S.DZ_UM
N_PER_CLASS = 40000            # training pixels per class per field
RNG = np.random.default_rng(20261106)


def plane_features(im):
    g1, g2, g4 = (ndi.gaussian_filter(im, s) for s in (1, 2, 4))
    grad = ndi.gaussian_gradient_magnitude(im, 1)
    out = [g1, g2, g4, g1 - g4, grad]
    for s in (1.5, 3):
        e = hessian_matrix_eigvals(hessian_matrix(im, sigma=s, order="rc", use_gaussian_derivatives=False))
        out += [e[0], e[1]]
    return np.stack(out).astype(np.float32)          # 9 x Y x X


def stack_features(phal):
    """phal: images x Y x X scaled 0-1, first image highest above the surface."""
    base = np.stack([plane_features(p) for p in phal])  # Z x 9 x Y x X
    return base


def features_at(base, heights, z):
    """Feature image for image z: its own 9 features, brightness (g1) and smaller line measure of
    images z-2..z+2 (above and below), change from the image below, height."""
    nz = base.shape[0]
    f = [base[z]]
    for dzi in (-2, -1, 1, 2):
        k = int(np.clip(z + dzi, 0, nz - 1))
        f.append(base[k][[0, 6]])
    below = int(np.clip(z + 1, 0, nz - 1))
    f.append((base[z][0] - base[below][0])[None])
    if USE_HEIGHT and heights is not None:
        f.append(np.full((1,) + base.shape[2:], heights[z], np.float32))
    return np.concatenate(f)                           # 9 + 8 + 1 (+ 1 height) features


def scaled(im):
    lo, hi = np.percentile(im, [1, 99.8])
    return np.clip((im.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def load_cells(field):
    cells = np.load(S.cell_labels_path(field))   # corrected outlines where they exist, otherwise v1
    types = pd.read_csv(os.path.join(TYPES, field + "_types.csv"))
    return cells.astype(np.int32), types


def examples(field):
    z = np.load(os.path.join(STACKS, field + ".npz")); heights = z["offsets_um"]
    phal = np.stack([scaled(p) for p in z["phal"]])
    s = np.load(os.path.join(STROKES, field + "_strokes.npz")); val = s["val"]
    sys.path.insert(0, HERE); from paint_bundles import strokes_to_stack
    idx = strokes_to_stack(s, heights); keep = idx[0] >= 0; idx, val = idx[:, keep], val[keep]
    red = np.zeros(phal.shape, bool); red[tuple(idx[:, val == 1])] = True
    cells, types = load_cells(field)
    hair_ids = types.cell[types.row.isin(["IHC", "OHC1", "OHC2", "OHC3"])].values
    band_ids = types.cell[types.row == "band"].values
    hair = np.isin(cells, hair_ids); band = np.isin(cells, band_ids)
    near_red = ndi.binary_dilation(red.any(0), iterations=int(round(0.6 / PX)))
    neg = np.zeros(phal.shape, bool)
    below = heights <= -0.6
    neg[below] = True
    neg[:, band] = True
    far = ndi.distance_transform_edt(~hair) * PX > 3
    neg[:, far] = True
    # each hair cell's plate: image where the cell's own mean peaks
    means = np.stack([ndi.mean(p, cells, hair_ids) for p in phal])
    plate_z = dict(zip(hair_ids, means.argmax(0)))
    for c, k in plate_z.items():
        neg[k][cells == c] = True
    neg[:, near_red] = False
    neg &= ~red
    # Shawn's blue strokes, except within 0.5 um of a red stroke in the same image
    blue = np.zeros(phal.shape, bool); blue[tuple(idx[:, val == 2])] = True
    r5 = int(round(0.5 / PX))
    for k in range(phal.shape[0]):
        if red[k].any() and blue[k].any():
            blue[k] &= ~ndi.binary_dilation(red[k], iterations=r5)
    # rest of each painted outer hair cell at the heights of its red stroke: more than 1.2 um from the stroke
    rest = np.zeros(phal.shape, bool); r12 = int(round(1.2 / PX))
    ohc_ids = types.cell[types.row.isin(["OHC1", "OHC2", "OHC3"])].values
    for k in np.unique(np.nonzero(red)[0]):
        cr = np.unique(cells[red[k]]); cr = cr[np.isin(cr, ohc_ids)]
        if len(cr):
            rest[k] |= np.isin(cells, cr) & ~ndi.binary_dilation(red[k], iterations=r12)
    base = stack_features(phal)
    X, y, src, info = [], [], [], {}
    for name, cls, mask, n in (("bundle", 1, red, N_PER_CLASS), ("automatic", 0, neg, N_PER_CLASS),
                               ("blue", 0, blue, N_PER_CLASS), ("cell rest", 0, rest, N_PER_CLASS // 2)):
        zz, yy, xx = np.nonzero(mask)
        if len(zz) == 0:
            info[name] = 0; continue
        pick = RNG.choice(len(zz), min(n, len(zz)), replace=False)
        zz, yy, xx = zz[pick], yy[pick], xx[pick]
        for k in np.unique(zz):
            sel = zz == k
            F = features_at(base, heights, k)
            X.append(F[:, yy[sel], xx[sel]].T); y.append(np.full(sel.sum(), cls)); src += [name] * int(sel.sum())
        info[name] = len(pick)
    return np.concatenate(X), np.concatenate(y), info, np.array(src)


def public_examples():
    """VASCilia P5 stacks resampled by prepare_vascilia.py. Bundle: labeled pixels. Not bundle:
    unlabeled pixels more than 0.3 um from a bundle, half of them among the brightest quarter
    (plates and cell borders), half anywhere. Each stack is turned so that its plate side (more
    bright unlabeled pixels) is the last image, as in the flattened stacks of Dominika's fields."""
    Xs, ys, n = [], [], 0
    for fp in sorted(glob.glob(os.path.join(PUBLIC, "*.npz"))):
        d = np.load(fp); phal = d["phal"].astype(np.float32); lab = d["labels"]
        if phal.shape[0] < 2 or (lab > 0).sum() < 100:
            continue
        bright = phal > np.percentile(phal, 75)
        first, last = (bright[0] & (lab[0] == 0)).mean(), (bright[-1] & (lab[-1] == 0)).mean()
        if first > last:
            phal, lab = phal[::-1].copy(), lab[::-1].copy(); bright = bright[::-1]
        bun = lab > 0
        away = ~ndi.binary_dilation(bun, structure=np.ones((1, 3, 3)), iterations=int(round(0.3 / PX)))
        base = stack_features(phal)
        for cls, mask in ((1, bun), (0, away & bright), (0, away)):
            zz, yy, xx = np.nonzero(mask)
            if len(zz) == 0: continue
            pick = RNG.choice(len(zz), min(N_PUBLIC, len(zz)), replace=False); zz, yy, xx = zz[pick], yy[pick], xx[pick]
            for k in np.unique(zz):
                sel = zz == k; F = features_at(base, None, k)
                Xs.append(F[:, yy[sel], xx[sel]].T); ys.append(np.full(sel.sum(), cls))
        n += 1
    print(f"  public examples from {n} stacks: {sum(len(y) for y in ys)} pixels", flush=True)
    return np.concatenate(Xs), np.concatenate(ys)


def forest():
    return RandomForestClassifier(n_estimators=int(os.environ.get("BUNDLE_TREES", 200)), max_depth=18, min_samples_leaf=5,
                                  n_jobs=int(os.environ.get("BUNDLE_JOBS", -1)),
                                  class_weight="balanced", random_state=0)


def painted_fields():
    return [os.path.basename(f)[:-12] for f in sorted(glob.glob(os.path.join(STROKES, "*_strokes.npz")))]


def check(public=False):
    global USE_HEIGHT; USE_HEIGHT = not public
    fields = painted_fields(); data = {}
    pub = public_examples() if public else None
    for f in fields:
        t0 = time.time(); data[f] = examples(f)
        print(f"  examples {f}: {data[f][2]} ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    for f in fields:
        Xtr = np.concatenate([data[g][0] for g in fields if g != f] + ([pub[0]] if pub else []))
        ytr = np.concatenate([data[g][1] for g in fields if g != f] + ([pub[1]] if pub else []))
        m = forest().fit(Xtr, ytr)
        p = m.predict_proba(data[f][0])[:, 1]; s = data[f][3]
        row = dict(left_out=f)
        for name in ("bundle", "automatic", "blue", "cell rest"):
            sel = s == name
            row[f"{name}_pixels"] = int(sel.sum())
            row[f"{name}_called_bundle"] = round(float((p[sel] > 0.5).mean()), 3) if sel.any() else np.nan
        rows.append(row)
        print("  ", rows[-1], flush=True)
    name = "bundle_detector_check_with_public.csv" if public else "bundle_detector_check.csv"
    pd.DataFrame(rows).to_csv(os.path.join(S.RESULTS, name), index=False)


def train(public=False):
    global USE_HEIGHT; USE_HEIGHT = not public
    fields = painted_fields()
    Xs, ys = [], []
    if public:
        a, b = public_examples(); Xs.append(a); ys.append(b)
    for f in fields:
        a, b, _, _ = examples(f); Xs.append(a); ys.append(b)
    m = forest().fit(np.concatenate(Xs), np.concatenate(ys))
    path = MODELS[public]; os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(dict(forest=m, height=USE_HEIGHT, public=public, fields=fields, trees=m.n_estimators), path, compress=3)
    print("saved", path, "from", len(fields), "fields", "with VASCilia" if public else "", flush=True)


def predict(only, public=False):
    global USE_HEIGHT
    saved = joblib.load(MODELS[public])
    m = saved["forest"]; USE_HEIGHT = saved["height"]; m.n_jobs = int(os.environ.get("BUNDLE_JOBS", -1))
    os.makedirs(PROB, exist_ok=True)
    fields = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(STACKS, "*.npz")))]
    for f in [f for f in fields if only in f]:
        t0 = time.time()
        z = np.load(os.path.join(STACKS, f + ".npz")); heights = z["offsets_um"]
        phal = np.stack([scaled(p) for p in z["phal"]])
        cells, types = load_cells(f)
        near = ndi.distance_transform_edt(~np.isin(cells, types.cell[types.row.isin(["IHC", "OHC1", "OHC2", "OHC3"])].values)) * PX <= 3
        base = stack_features(phal)
        prob = np.zeros(phal.shape, np.float32)
        for k in np.where(heights > -0.6)[0]:
            F = features_at(base, heights, k)
            prob[k][near] = m.predict_proba(F[:, near].T)[:, 1]
        best = prob.max(0); hz = heights[prob.argmax(0)]
        np.savez_compressed(os.path.join(PROB, f + ".npz"), prob=(prob * 255).astype(np.uint8), best=best, height_um=hz.astype(np.float32))
        print(f"  {f}: {time.time() - t0:.0f} s, pixels above 0.5: {(best > 0.5).sum()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true"); ap.add_argument("--only", default="")
    ap.add_argument("--public", action="store_true", help="add the resampled VASCilia bundle outlines to training")
    a = ap.parse_args()
    if a.check: check(a.public)
    if a.train: train(a.public)
    if a.predict: predict(a.only, a.public)
