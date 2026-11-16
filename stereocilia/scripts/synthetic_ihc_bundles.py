"""Synthetic inner hair cell bundles of known base, tip, length and lean, to check ihc_stereocilia.py.

Each synthetic cut-out has the size and spacing of the real cut-outs from ihc_stacks.py (61 planes at 0.3 um from
3 um below to 15 um above the field surface, 222 x 222 pixels at 0.1083 um; x along the row, y toward the outer
hair cells). It holds three cells 8.5 um apart along the row. Each cell has a cuticular plate (disc 3.5 um in
radius, raised to h0 above the surface) and a bundle of three staircase rows of rods (full, 0.7 and 0.45 of the
tallest-row length; rods every 0.4 um along a shallow arc, rows 0.4 um apart, on the outer hair cell side of the
plate), with scatter between rods in angle (8 degrees toward the outer hair cells, 10 degrees along the row) and
brightness (lognormal, sd 0.3). A sloped band of tissue lies toward the outer hair cells. The volume is blurred with
the point blur measured from isolated PMCA2 specks (0.38 um across the image, 1.07 um along the optical axis, at half
maximum), heights are stretched by 1.19 as in the real images (DAPI nuclei), and background (2000 counts),
smooth texture (sd 600) and brightness-dependent noise (variance 127 x mean + 500^2) are added. Brightness 1 puts
the 99.9th percentile near 31,000 counts, as in a typical bright real cut-out.

Run from the scripts folder:
    python synthetic_ihc_bundles.py     (about 15 minutes; writes results\\tuning\\ihc_stereocilia_synthetic_validation.csv)
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
PX, DZ, NZ, NY, NX = 0.1083, 0.3, 61, 222, 222
STRETCH = 1.19
SIG_XY, SIG_Z = 0.16, 0.45            # Gaussian sd (um) for 0.38 and 1.07 um at half maximum
BG, READ, GAIN = 2000., 500., 127.


def splat(vol, x, y, h, w):
    """Add points (true um; h above the field surface) to vol with linear weights."""
    zi = (h * STRETCH + 3.0) / DZ; yi = y / PX; xi = x / PX
    a0 = np.floor(zi).astype(int); b0 = np.floor(yi).astype(int); c0 = np.floor(xi).astype(int)
    ok = (a0 >= 0) & (a0 < NZ - 1) & (b0 >= 0) & (b0 < NY - 1) & (c0 >= 0) & (c0 < NX - 1)
    fa, fb, fc = (zi - np.floor(zi))[ok], (yi - np.floor(yi))[ok], (xi - np.floor(xi))[ok]
    a0, b0, c0, w = a0[ok], b0[ok], c0[ok], w[ok]
    for da in (0, 1):
        for db in (0, 1):
            for dc in (0, 1):
                np.add.at(vol, (a0 + da, b0 + db, c0 + dc),
                          w * (fa if da else 1 - fa) * (fb if db else 1 - fb) * (fc if dc else 1 - fc))


def bundle_volume(H=5.0, lean=-30, bright=1.0, h0=2.5, sig_z=SIG_Z, W=7.0, seed=0):
    rng = np.random.default_rng(seed); vol = np.zeros((NZ, NY, NX), np.float32)
    n = 60000; yy = rng.uniform(18.5, 24, n); xx = rng.uniform(0, 24, n); hh = (yy - 18.5) * 0.45 - 0.5 + rng.uniform(-0.7, 0.7, n)
    splat(vol, xx, yy, hh, np.full(n, 0.12 * bright)); truth = {}
    for cx, main in [(12.0, True), (3.5, False), (20.5, False)]:
        Hc = H if main else H * rng.uniform(.85, 1.15); ln = lean if main else lean + rng.uniform(-8, 8)
        hb = h0 if main else h0 + rng.uniform(-0.5, 0.5)
        n = 3000; r = 3.5 * np.sqrt(rng.uniform(0, 1, n)); a = rng.uniform(0, 2 * np.pi, n)
        splat(vol, cx + r * np.cos(a), 14.5 + r * np.sin(a), hb + rng.uniform(-0.5, 0.5, n), np.full(n, 0.2 * bright))
        for k, (rf, dy) in enumerate([(1.0, 17.4), (0.7, 17.0), (0.45, 16.6)]):
            for x0 in np.arange(-W / 2 + 0.3 * k, W / 2 - 0.3 * k + 1e-6, 0.4):
                th = np.radians(ln + rng.normal(0, 8)); ph = np.radians(rng.normal(0, 10) + 12 * x0 / (W / 2))
                y0 = dy - 0.06 * x0 ** 2; L = Hc * rf * (1 - 0.15 * (abs(x0) / (W / 2)) ** 2) * rng.uniform(.92, 1.05); s = np.arange(0, L, 0.05)
                dx, dyy, dh = np.sin(ph) * np.cos(th), np.sin(th), np.cos(th) * np.cos(ph)
                splat(vol, cx + x0 + s * dx, y0 + s * dyy, hb + 0.5 + s * dh, np.full(len(s), bright * rng.lognormal(0, 0.3)))
        if main:
            th = np.radians(ln)
            truth = dict(true_base=hb + 0.5, true_tip=hb + 0.5 + Hc * np.cos(th), true_height=Hc * np.cos(th), true_length=Hc, true_lean=ln)
    return ndi.gaussian_filter(vol, (sig_z * STRETCH / DZ, SIG_XY / PX, SIG_XY / PX)), truth


_SCALE = None


def scale():
    global _SCALE
    if _SCALE is None:
        v, _ = bundle_volume(); _SCALE = 29000 / np.percentile(v, 99.9)
    return _SCALE


def make(H, lean, bright, h0, sig_z=SIG_Z, seed=0):
    """Synthetic cut-out (uint16, like ihc_stacks phal) and its truth."""
    v, t = bundle_volume(H, lean, bright, h0=h0, sig_z=sig_z, seed=seed); rng = np.random.default_rng(seed + 999)
    tex = ndi.gaussian_filter(rng.normal(0, 1, v.shape).astype(np.float32), (3, 8, 8)); tex *= 600 / tex.std()
    mean = np.clip(BG + scale() * v + tex, 0, None)
    return np.clip(rng.normal(mean, np.sqrt(GAIN * mean + READ ** 2)), 0, 65535).astype(np.uint16), t


def grid():
    """Standard blur: 4 lengths x 4 leans x 3 brightness levels, three replicates (144); folded: 4 x 2 x 3, two
    replicates (48); 1.4 times the blur along the optical axis: 8."""
    out = []; r0 = np.random.default_rng(7); s = 0
    for H in [3, 4.5, 6, 8]:
        for ln in [0, -20, -40, -60]:
            for b in [1.0, 0.5, 0.25]: s += 1; out.append((H, ln, b, r0.uniform(1.5, 3.5), SIG_Z, s))
    for H in [4.5, 6]:
        for ln in [-20, -40]:
            for b in [1.0, 0.5]: s += 1; out.append((H, ln, b, r0.uniform(1.5, 3.5), 0.65, s))
    r1 = np.random.default_rng(11); s = 1000
    for rep in range(2):
        for H in [3, 4.5, 6, 8]:
            for ln in [0, -20, -40, -60]:
                for b in [1.0, 0.5, 0.25]: s += 1; out.append((H, ln, b, r1.uniform(1.5, 3.5), SIG_Z, s))
    r2 = np.random.default_rng(23); s = 2000
    for rep in range(2):
        for H in [3, 4.5, 6, 8]:
            for ln in [-75, -90]:
                for b in [1.0, 0.5, 0.25]: s += 1; out.append((H, ln, b, r2.uniform(1.0, 2.5), SIG_Z, s))
    return out


def validate(path=None):
    import ihc_stereocilia as isc
    rows = []
    for H, ln, b, h0, sz, s in grid():
        vol, tr = make(H, ln, b, h0, sz, s); m, core, r = isc.run_cell(vol, (4.25, 4.25))
        rows.append(dict(H=H, lean_set=ln, bright=b, h0=h0, sig_z=sz, seed=s, **tr, **r))
    out = pd.DataFrame(rows)
    for c in ["tip", "base", "height", "length"]: out["err_" + c] = out[c + "_um"] - out["true_" + c]
    out["err_lean"] = out.lean_deg - out.true_lean
    path = path or os.path.join(os.path.dirname(HERE), "results", "tuning", "ihc_stereocilia_synthetic_validation.csv")
    out.to_csv(path, index=False); return out


if __name__ == "__main__":
    validate()
