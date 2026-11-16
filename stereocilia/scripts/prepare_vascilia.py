"""Resample the public VASCilia 3D bundle outlines to look like Dominika's images.

Source: VASCilia 3D stereocilia dataset (Kassim et al. 2026, PLoS Biology; Zenodo 17905676,
CC-BY 4.0), E:\\Webber_confocal\\public\\vascilia\\VASCilia_3D_DataSet.zip. Each stack is a set of
8-bit RGB slices (phalloidin in green) from an Airyscan microscope, 43 nm pixels, with one label
image per slice whose ids are the same bundle across slices. The z step is not stored; 0.11 um
is assumed, as in confocal-phenotyping\\pipeline\\vascilia_to_ims.py.

Only the P5 stacks are used (P21 bundles are mature and less like P1). For each stack:
  1. extra blur so the sharper Airyscan images match this spinning-disk confocal (60x, NA 1.4):
     about 0.21 um extra blur side to side and 0.6 um in depth (full width at half maximum),
     applied at the original sampling;
  2. resampled to 0.1083 um pixels and 0.3 um between images (labels by nearest pixel).
Output: E:\\Webber_confocal\\public\\vascilia\\resampled\\<stack>.npz (phal float16 scaled 0-1 per
image, labels uint16) and index.csv.

Run from the scripts folder:  python prepare_vascilia.py      (about 5 minutes)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import io, os, re, sys, zipfile
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi

ZIP = os.path.join(S.PUBLIC, 'vascilia', 'VASCilia_3D_DataSet.zip')
OUT = os.path.join(S.PUBLIC, 'vascilia', 'resampled')
SRC_XY, SRC_Z = 0.043, 0.11
DST_XY, DST_Z = 0.1083, 0.3
EXTRA_FWHM_XY, EXTRA_FWHM_Z = 0.21, 0.60
FWHM_TO_SIGMA = 1 / 2.3548


def stacks(z):
    names = [n for n in z.namelist() if not n.endswith("/")]
    d = {}
    for n in names:
        p = n.split("/")
        if len(p) >= 4 and p[-2].lower() in ("rawimages", "raw_images", "new_assignment_obj"):
            key = "/".join(p[:-2]); kind = "label" if p[-2] == "new_assignment_obj" else "raw"
            d.setdefault(key, {"raw": [], "label": []})[kind].append(n)
    return d


def num(n):
    m = re.findall(r"(\d+)\.(?:tif|tiff|png)$", n, re.I)
    return int(m[-1]) if m else -1


def main():
    os.makedirs(OUT, exist_ok=True)
    z = zipfile.ZipFile(ZIP); rows = []
    for key, f in sorted(stacks(z).items()):
        age = key.split("/")[1]
        if age != "P5":
            continue
        raw = sorted(f["raw"], key=num); lab = sorted(f["label"], key=num)
        if len(raw) != len(lab) or len(raw) < 3:
            rows.append(dict(stack=key.split("/")[-1], used=False, reason="slice counts differ")); continue
        nums = [num(n) for n in raw]
        img = np.stack([tifffile.imread(io.BytesIO(z.read(n)))[..., 1] for n in raw]).astype(np.float32)
        lb = np.stack([np.array(Image.open(io.BytesIO(z.read(n)))) for n in lab])
        if lb.ndim == 4:
            lb = lb[..., 0]
        sig_xy = EXTRA_FWHM_XY * FWHM_TO_SIGMA / SRC_XY
        sig_z = EXTRA_FWHM_Z * FWHM_TO_SIGMA / SRC_Z
        img = ndi.gaussian_filter(img, (sig_z, sig_xy, sig_xy), mode="nearest")
        fz, fxy = SRC_Z / DST_Z, SRC_XY / DST_XY
        img = ndi.zoom(img, (fz, fxy, fxy), order=1, prefilter=False)
        lbr = ndi.zoom(lb, (fz, fxy, fxy), order=0, prefilter=False)
        for k in range(img.shape[0]):
            lo, hi = np.percentile(img[k], [1, 99.8]); img[k] = np.clip((img[k] - lo) / max(hi - lo, 1e-6), 0, 1)
        stem = key.split("/")[-1]
        np.savez_compressed(os.path.join(OUT, stem + ".npz"), phal=img.astype(np.float16), labels=lbr.astype(np.uint16))
        rows.append(dict(stack=stem, used=True, age=age, slices=len(raw), slice_numbers_consecutive=bool(np.all(np.diff(nums) == 1)),
                         images_after=img.shape[0], size_after=f"{img.shape[1]}x{img.shape[2]}",
                         bundles=int(len(np.unique(lbr)) - 1), bundle_pixel_share=round(float((lbr > 0).mean()), 4)))
        print(" ", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "index.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
