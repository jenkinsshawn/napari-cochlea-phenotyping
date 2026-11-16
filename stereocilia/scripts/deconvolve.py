"""Sharpening of confocal cut-outs with the measured microscope blur (Richardson-Lucy deconvolution).

The blur is a Gaussian with the size measured from 381 isolated PMCA2 specks: 0.381 um across and
1.073 um along the optical axis (nominal z), full width at half maximum. Specks have some size of their
own, so this is an upper bound; `scale` shrinks it.

    from deconvolve import sharpen
    out = sharpen(vol, px_um=0.1083, dz_um=0.3, scale=0.85, rounds=(10, 20, 40))   # {10: vol10, 20: vol20, 40: vol40}

For a new microscope or objective, measure the blur again and pass fwhm_xy_um / fwhm_z_um.
"""
import numpy as np
from scipy import fft as sfft

FWHM_XY_UM = 0.381
FWHM_Z_UM = 1.073


def gaussian_psf(px_um=0.1083, dz_um=0.3, scale=1.0, fwhm_xy_um=FWHM_XY_UM, fwhm_z_um=FWHM_Z_UM):
    sx = scale * fwhm_xy_um / 2.3548 / px_um
    sz = scale * fwhm_z_um / 2.3548 / dz_um
    r = np.ceil(3 * np.array([sz, sx, sx])).astype(int)
    z, y, x = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    p = np.exp(-z ** 2 / (2 * sz ** 2) - (y ** 2 + x ** 2) / (2 * sx ** 2))
    return (p / p.sum()).astype(np.float32)


def sharpen(vol, px_um=0.1083, dz_um=0.3, scale=1.0, rounds=(20,), fwhm_xy_um=FWHM_XY_UM, fwhm_z_um=FWHM_Z_UM):
    """Returns {rounds: sharpened volume} on the input's grid, scaled 0-1 after removing the 1st percentile."""
    psf = gaussian_psf(px_um, dz_um, scale, fwhm_xy_um, fwhm_z_um)
    pad = [(s // 2, s // 2) for s in psf.shape]
    v = vol.astype(np.float32); v = np.clip(v - np.percentile(v, 1), 0, None); v /= max(v.max(), 1e-6)
    v = np.pad(v, pad, mode="reflect") + 1e-6
    shape = v.shape
    k = np.zeros(shape, np.float32); k[tuple(slice(0, s) for s in psf.shape)] = psf
    k = np.roll(k, [-(s // 2) for s in psf.shape], axis=(0, 1, 2))
    K = sfft.rfftn(k, workers=1); Kc = np.conj(K)
    conv = lambda a, F: sfft.irfftn(sfft.rfftn(a, workers=1) * F, s=shape, workers=1)
    est = v.copy(); out = {}
    for i in range(1, max(rounds) + 1):
        est *= conv(v / np.maximum(conv(est, K), 1e-6), Kc)
        if i in rounds:
            out[i] = est[tuple(slice(p[0], s - p[1]) for p, s in zip(pad, shape))].copy()
    return out
