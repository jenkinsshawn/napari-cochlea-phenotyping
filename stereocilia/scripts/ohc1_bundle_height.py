"""First-row outer hair cell bundle height from the ohc1_stacks.py cut-outs (front view).

In the central 7 x 6 um of each cut-out (3.5 um either side along the row, 3 um either side across), per plane:
mean and 99th percentile of phalloidin. Base: the height above the cell-top peak where the mean falls to
background + 0.5 x (peak - background). Tip: the highest height where the 99th percentile is at least
background + 0.2 x (its peak - background). Background: median of the top 8 planes. Heights x 0.84 (Z_SCALE).
Fractions chosen on the tuning cells of Shawn's front-view reference (annotations\\ohc1_base_tip and _relook; 81 cells,
split into tuning and test before clicking). On the 35 readable test cells: height +0.21 um, median abs 0.32 um,
77% within 0.75 um; base and tip each 97% within 0.75 um. Shawn's repeats: 5 of 6 heights within 0.75 um.
Readable bundles: 67 of 81. Readability cannot be predicted (AUC at most 0.70), but leaving unreadable bundles in
shifts field medians by about 0 (median 0.00 um), so field medians are usable (more variable).
Sharpening with the measured blur (deconvolve.py) was tested on 11 Nov 2026 and did not improve agreement.
Output: results\\ohc1_bundle_height_cells.csv and results\\ohc1_bundle_height_fields.csv (field medians)

Run (imaging env):  python ohc1_bundle_height.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob, os
import numpy as np, pandas as pd

STACKS = os.path.join(S.WORK, 'ohc1_stacks'); RESULTS = S.RESULTS
ZS, PX = S.Z_SCALE, S.PX_UM; TIP_F, BASE_F = 0.2, 0.5


def profile(v, off, band=(-3.0, 3.0), half=3.5):
    n = v.shape[1]; c = n // 2; sub = v[:, c + int(band[0] / PX):c + int(band[1] / PX), c - int(half / PX):c + int(half / PX)].reshape(len(off), -1)
    return sub.mean(1), np.percentile(sub, 99, axis=1)


def edges(off, mn, p99, ft=TIP_F, fb=BASE_F):
    bg = np.percentile(p99[-8:], 50); pk = p99.max(); above = np.where(p99 >= bg + ft * (pk - bg))[0]; tip = np.nan
    if len(above):
        a = above.max(); tip = off[a]
        if a + 1 < len(p99): t = bg + ft * (pk - bg); tip = off[a] + (off[a + 1] - off[a]) * (p99[a] - t) / max(p99[a] - p99[a + 1], 1e-9)
    i0 = int(np.argmax(mn)); bgm = np.percentile(mn[-8:], 50); thr = bgm + fb * (mn[i0] - bgm); j = i0
    while j + 1 < len(mn) and mn[j + 1] >= thr: j += 1
    base = off[j] if j + 1 >= len(mn) else off[j] + (off[j + 1] - off[j]) * (mn[j] - thr) / max(mn[j] - mn[j + 1], 1e-9)
    return base, tip


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(STACKS, "*.npz"))):
        f = os.path.basename(p)[:-4]; z = np.load(p); off = z["offsets_um"] * ZS; cells = pd.read_csv(os.path.join(STACKS, f + "_cells.csv"))
        for j, c in cells.iterrows():
            mn, p99 = profile(z["phal"][j].astype(np.float32), off); b, t = edges(off, mn, p99)
            rows.append(dict(field=f, cell=int(c.cell), x=c.x, y=c.y, u=c.u, base_um=b, tip_um=t, height_um=t - b))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(RESULTS, "ohc1_bundle_height_cells.csv"), index=False)
    F = out.groupby("field").agg(ohc1_bundle_height_um=("height_um", "median"), ohc1_cells=("height_um", "size")).reset_index()
    F.to_csv(os.path.join(RESULTS, "ohc1_bundle_height_fields.csv"), index=False); return out


if __name__ == "__main__":
    main()
