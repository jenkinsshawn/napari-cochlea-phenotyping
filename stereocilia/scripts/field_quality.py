"""Technical image quality of each field, for noise adjustment in phenotype_statistics.py.

Reads (pipeline_settings): WORK\fields\<field>.npz (surface, maxproj, stains), WORK\paint_stacks\<field>.npz (phalloidin
planes at fixed heights above the surface), WORK\bundles_detector\<field>_bundles.npy, RESULTS\field_table.csv and
RESULTS\imaging_settings_by_field.csv.
Writes RESULTS\field_quality.csv, one row per field:
  phal_background_ratio   phalloidin 5th percentile / median inside bundles, top view 0.3-6 um above the surface
  phal_noise_to_signal    noise SD (1.4826 x MAD of the fine-detail image away from bundles) / (bundle median - 5th percentile)
  phal_saturated_share    share of phalloidin pixels at the detector maximum, planes -1.8 to 6 um
  dapi_noise_to_signal    the same noise measure for DAPI in its maximum projection (nuclei above the 90th percentile)
  surface_tilt_um         5th to 95th percentile range of the surface depth over the field (x Z_SCALE)
  surface_fit_error_um    from field_table.csv
  surface_depth_um        median depth of the surface below the first plane of the stack (x Z_SCALE)
  phal_dose, dapi_dose    laser power (%) x exposure (ms) of the channel, when given in the imaging settings
local_surface() gives the surface depth and slope at given positions, used for each bundle and cell (local_depth_um,
local_slope_deg) by bundle_phenotypes.py and ihc_bundle_shape.py.
Run (imaging env, scripts folder):  python field_quality.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob
import numpy as np, pandas as pd
from scipy import ndimage as ndi

ZS = float(getattr(S, "Z_SCALE", 1.0))


def surface_maps(fld, smooth_um=2.0):
    """Depth of the cell surface below the first plane of the stack (um, x Z_SCALE) and its local slope (degrees), per pixel."""
    z = np.load(os.path.join(S.WORK, "fields", fld + ".npz"), allow_pickle=True)
    dz = float(z["voxel_um"][0]) if "voxel_um" in z.files else 0.3
    s = z["surface"].astype(float); s = np.where(np.isfinite(s), s, np.nanmedian(s))
    depth = ndi.gaussian_filter(s, smooth_um / S.PX_UM) * dz * ZS
    gy, gx = np.gradient(depth, S.PX_UM)
    return depth, np.degrees(np.arctan(np.hypot(gx, gy)))


def local_surface(fld, x, y, maps=None):
    """Surface depth (um) and slope (degrees) at field pixel positions x, y."""
    depth, slope = maps if maps is not None else surface_maps(fld)
    yy = np.clip(np.round(np.asarray(y, float)).astype(int), 0, depth.shape[0] - 1); xx = np.clip(np.round(np.asarray(x, float)).astype(int), 0, depth.shape[1] - 1)
    return depth[yy, xx], slope[yy, xx]


def noise_sd(img, mask_out):
    hp = img - ndi.gaussian_filter(img, 1.0); v = hp[mask_out]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if v.size > 100 else np.nan


def field_row(fld):
    o = dict(field=fld)
    z = np.load(os.path.join(S.WORK, "fields", fld + ".npz"), allow_pickle=True)
    dz = float(z["voxel_um"][0]) if "voxel_um" in z.files else 0.3
    surf = z["surface"].astype(float); s = surf[np.isfinite(surf)]
    if s.size:
        lo, hi = np.percentile(s, [5, 95]); o["surface_tilt_um"] = float((hi - lo) * dz * ZS); o["surface_depth_um"] = float(np.median(s) * dz * ZS)
    stains = [str(x) for x in z["stains"]]
    if "DAPI" in stains:
        d = z["maxproj"][stains.index("DAPI")].astype(np.float32); b = np.percentile(d, 5); nuc = d > np.percentile(d, 90)
        sig = np.median(d[nuc]) - b
        o["dapi_noise_to_signal"] = float(noise_sd(d, ~ndi.binary_dilation(nuc, iterations=5)) / sig) if sig > 0 else np.nan
    ps = os.path.join(S.WORK, "paint_stacks", fld + ".npz")
    if os.path.exists(ps):
        pz = np.load(ps); off = pz["offsets_um"]; ph = pz["phal"]
        sel = (off >= 0.3) & (off <= 6.0); top = ph[sel].astype(np.float32).max(axis=0)
        sat = float(getattr(S, "SATURATION_VALUE", 0) or 0) or (4095.0 if ph.max() <= 4095 else 65535.0)
        o["phal_saturated_share"] = float((ph >= sat).mean())
        bl = os.path.join(S.WORK, "bundles_detector", fld + "_bundles.npy")
        lab = np.load(bl) if os.path.exists(bl) else np.zeros(top.shape, int)
        b = np.percentile(top, 5); inb = lab > 0
        med = np.median(top[inb]) if inb.sum() > 100 else np.percentile(top, 99)
        o["phal_background_ratio"] = float(b / med) if med > 0 else np.nan
        away = ~ndi.binary_dilation(inb, iterations=10) & (top < np.percentile(top, 50))
        o["phal_noise_to_signal"] = float(noise_sd(top, away) / (med - b)) if med > b else np.nan
    return o


def main():
    flds = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(S.WORK, "fields", "*.npz")))
    Q = pd.DataFrame([field_row(f) for f in flds])
    ft = os.path.join(S.RESULTS, "field_table.csv")
    if os.path.exists(ft): Q = Q.merge(pd.read_csv(ft)[["field", "surface_fit_error_um"]], on="field", how="left")
    im = os.path.join(S.RESULTS, "imaging_settings_by_field.csv")
    if os.path.exists(im):
        I = pd.read_csv(im); I["dose"] = pd.to_numeric(I.laser_power_pct, errors="coerce") * pd.to_numeric(I.exposure_ms, errors="coerce")
        for st, col in [("Phalloidin", "phal_dose"), ("DAPI", "dapi_dose")]:
            Q = Q.merge(I[I.stain == st].groupby("field").dose.first().rename(col), left_on="field", right_index=True, how="left")
    Q.to_csv(os.path.join(S.RESULTS, "field_quality.csv"), index=False); print(f"field quality: {len(Q)} fields")


if __name__ == "__main__":
    main()
