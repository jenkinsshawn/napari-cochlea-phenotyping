"""Cell surface: the depth of the brightest phalloidin layer (cuticular plates and bundle bases) across the field.

Same method as the validated script (make_field_pictures.find_surface), with every size in micrometers so it works
at any pixel size and plane spacing. At 0.108 um pixels and 0.3 um planes it uses exactly the script's sizes (64-pixel
squares, 3-plane depth smoothing), and a regression test checks that the two give identical surfaces.

  1. Light smoothing in depth (0.9 um, an odd number of planes so it stays centered), then for each square of about 6.9 um the 99th percentile of each plane.
  2. The plane where that profile peaks is the square's surface depth, if the square has a clear actin layer:
     (peak - background) / (median - background) >= clear_ratio, with background the field's 1st percentile.
     Removing the background makes this test independent of exposure, gain and camera offset.
  3. Squares without a clear layer take the nearest good depth; squares more than 3 um off their neighbors' median
     are replaced by it; the result is smoothed over one square and interpolated to every pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

TILE_UM = 64 * 0.1083          # 6.93 um: the validated 64-pixel square at 0.1083 um pixels
DEPTH_SMOOTH_UM = 0.9          # 3 planes at 0.3 um
OFF_NEIGHBORS_UM = 3.0
CLEAR_RATIO = 2.0              # pipeline_settings.SURFACE_CLEAR_RATIO


@dataclass
class Surface:
    planes: np.ndarray         # (y, x) surface depth in plane numbers (float)
    square_planes: np.ndarray  # (squares y, squares x) peak plane per square, nan where no clear layer
    good_squares: int
    fit_error_um: float
    tile_px: int

    def depth_um(self, dz_um: float) -> np.ndarray:
        return self.planes * dz_um


def find_surface(phal: np.ndarray, voxel_um, clear_ratio: float = CLEAR_RATIO, local: bool = True,
                 tile_um: float = TILE_UM) -> Surface:
    """phal: (planes, y, x) phalloidin stack; voxel_um: (z, y, x)."""
    dz, dy, dx = (float(v) for v in voxel_um)
    tile = max(4, int(round(tile_um / dx)))
    zs = max(1, 2 * int(round((DEPTH_SMOOTH_UM / dz - 1) / 2)) + 1)   # odd number of planes, so the smoothing is centered
    nz, ny, nx = phal.shape
    ty, tx = ny // tile, nx // tile
    sm = ndi.uniform_filter1d(phal.astype(np.float32), size=zs, axis=0)
    tiles = sm[:, :ty * tile, :tx * tile].reshape(nz, ty, tile, tx, tile)
    prof = np.percentile(tiles.transpose(0, 1, 3, 2, 4).reshape(nz, ty, tx, -1), 99, axis=-1)
    zpk = prof.argmax(0).astype(float)
    bg = float(np.percentile(phal, 1))
    contrast = (prof.max(0) - bg) / np.maximum(np.median(prof, 0) - bg, 1e-3)
    zpk[contrast < clear_ratio] = np.nan
    good = ~np.isnan(zpk)
    if good.sum() >= 6 and local:
        idx = ndi.distance_transform_edt(~good, return_distances=False, return_indices=True)
        filled = zpk[tuple(idx)]
        med = ndi.median_filter(filled, size=3, mode="nearest")
        far = np.abs(filled - med) > max(OFF_NEIGHBORS_UM / dz, 3)
        filled[far] = med[far]
        smooth = ndi.gaussian_filter(filled, sigma=1.0, mode="nearest")
        fit_err = float(np.median(np.abs((zpk - smooth)[good])) * dz)
        ys = (np.arange(ny) + 0.5) / tile - 0.5; xs = (np.arange(nx) + 0.5) / tile - 0.5
        surf = ndi.map_coordinates(smooth, np.meshgrid(np.clip(ys, 0, ty - 1), np.clip(xs, 0, tx - 1), indexing="ij"),
                                   order=1, mode="nearest")
    elif good.sum() >= 6:
        yy, xx = np.mgrid[0:ty, 0:tx]
        A = np.c_[np.ones(good.sum()), xx[good], yy[good], xx[good] ** 2, yy[good] ** 2, xx[good] * yy[good]]
        coef, *_ = np.linalg.lstsq(A, zpk[good], rcond=None)
        resid = zpk[good] - A @ coef
        keep = np.abs(resid) < 3 * max(np.median(np.abs(resid)), 1)
        coef, *_ = np.linalg.lstsq(A[keep], zpk[good][keep], rcond=None)
        Y, X = np.mgrid[0:ny, 0:nx] / tile - 0.5
        surf = coef[0] + coef[1] * X + coef[2] * Y + coef[3] * X ** 2 + coef[4] * Y ** 2 + coef[5] * X * Y
        fit_err = float(np.median(np.abs(resid[keep])) * dz)
    else:
        surf = np.full((ny, nx), np.nanmedian(zpk) if good.any() else nz / 2)
        fit_err = float("nan")
    return Surface(np.clip(surf, 0, nz - 1), zpk, int(good.sum()), fit_err, tile)


MIN_CLEAR_SHARE = 0.75   # warn below this share of squares with a clear actin layer (P1 data: torn field 0.56, all others 0.85-1.00)


def clear_share(s: Surface) -> float:
    return float(np.isfinite(s.square_planes).mean()) if s.square_planes.size else 0.0


def layer_projection(stack: np.ndarray, surface_planes: np.ndarray, half_um: float, dz_um: float) -> np.ndarray:
    """Brightest voxel of each pixel within +-half_um of the surface (as make_field_pictures.slab_projection)."""
    half = int(round(half_um / dz_um)); z = np.arange(stack.shape[0])[:, None, None]
    return np.where(np.abs(z - surface_planes[None]) <= half, stack, 0).max(0)
