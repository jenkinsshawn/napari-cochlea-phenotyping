"""Cut-outs of every inner hair cell from the full .nd2 stack, for inner hair cell bundle work.

The stored paint stacks stop 6 um above the field surface, below the tips of inner hair cell
stereocilia (8-14 um), so these cut-outs are read from the original .nd2 files.

For each field:
  1. Inner hair cells: Cellpose outlines of the Calb2 projection grouped into cells
     (hair_cell_rows.inner_cells), positioned along the Calb2 band.
  2. Row direction at each cell: the line through the inner hair cells within 20 um along the band.
  3. Planes from 3 um below to 15 um above the field surface (fields npz 'surface', a plane number
     per pixel; above = lower plane number), each sampled at the surface + offset.
  4. A 24 x 24 um square around each cell, turned so x runs along the row and y increases toward
     the outer hair cells, at the original pixel size (0.108 um).
  5. Plate depth of each cell: the offset (within -3 to +8 um) where mean phalloidin in the central
     6 x 6 um of the cut-out peaks.
Output: E:\\Webber_confocal\\work\\ihc_stacks\\<field>.npz with
        phal (cells, planes, 222, 222) uint16, pmca2 the same where stained, offsets_um,
        and <field>_cells.csv (cell, x, y, u, angle_deg, plate_um).
Fields already done are skipped.

Run from the scripts folder:
    python ihc_stacks.py           (all fields, about 15-20 minutes)
    python ihc_stacks.py --v2      (inner hair cells from the second row frame; writes work\\ihc_stacks_v2)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import pipeline_settings as PS   # raw images (PS: this file uses S for surfaces)
import glob, os, sys, time
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hair_cell_rows as hr
import bundle_trace as bt

RAW = S.RAW
OUT = os.path.join(S.WORK, 'ihc_stacks')
FRAME = "v1"     # "v2": inner hair cells from the second row frame (hair_cell_rows_classify.frame_v2), written to ihc_stacks_v2
PX, DZ = S.PX_UM, S.DZ_UM
LO_UM, HI_UM = -3.0, 15.0
HALF_UM = 12.0


def raw_path(field):
    try:
        return PS.raw_path(field)
    except FileNotFoundError:
        return None


def flattened(stack, surf, offs):
    """stack (planes, y, x); surf plane number per pixel; offs planes above the surface."""
    S = np.round(surf).astype(int); nz = stack.shape[0]
    return np.stack([np.take_along_axis(stack, np.clip(S - k, 0, nz - 1)[None], 0)[0] for k in offs])


def row_angles(C, half_um=20.0):
    ang = []
    for i in range(len(C)):
        nb = C[np.abs(C.u - C.u.iloc[i]) <= half_um]
        P = nb[["x", "y"]].values.astype(float)
        if len(P) < 3:
            P = C[["x", "y"]].values.astype(float)
        v = np.linalg.svd(P - P.mean(0))[2][0]
        ang.append(np.arctan2(v[1], v[0]))
    return np.array(ang)


def cutout(vol, x, y, ang, sign, half_um=None):
    n = int(round(2 * (HALF_UM if half_um is None else half_um) / PX)); g = (np.arange(n) - n / 2) * PX / PX
    gx, gy = np.meshgrid(g, g)                                   # gx along the row, gy across
    ax, ay = np.cos(ang), np.sin(ang); nx, ny = -ay * sign, ax * sign
    X = x + gx * ax + gy * nx; Y = y + gx * ay + gy * ny
    return np.stack([map_coordinates(p.astype(np.float32), [Y, X], order=1, cval=0) for p in vol]).astype(np.uint16)


def run_field(field):
    dst = os.path.join(OUT, field + ".npz")
    if os.path.exists(dst):
        return "already done"
    rp = raw_path(field)
    if rp is None:
        return "no image file"
    zf, st = hr._field(field)
    if "Calb2" not in st or "Phalloidin" not in st:
        return "no Calb2 and phalloidin"
    F = bt.field_maps_bowl(field)
    if FRAME == "v2":
        import hair_cell_rows_classify as hrc; z, co = hrc.classify(field, F)          # rows from the row classifiers
    else:
        z, stks, co = hr.outer_rows(field, F)
    if co is None:
        return "no Calb2 band"
    C = hr.inner_cells(field, co)
    if len(C) == 0:
        return "no inner hair cells"
    import nd2
    arr = PS.read_raw(field)
    offs = np.arange(int(round(LO_UM / DZ)), int(round(HI_UM / DZ)) + 1)
    surf = zf["surface"]
    ph = flattened(arr[:, st.index("Phalloidin")], surf, offs)
    pm = flattened(arr[:, st.index("PMCA2")], surf, offs) if "PMCA2" in st else None
    del arr
    # across direction points toward the outer hair cells (positive distance from the Calb2 edge)
    ohc = z[z.row.isin(["OHC1", "OHC2", "OHC3"])][["x", "y"]].values.mean(0) if (z.row.isin(["OHC1", "OHC2", "OHC3"])).any() else None
    ang = row_angles(C)
    cuts, cutp, rows = [], [], []
    for i, c in C.reset_index(drop=True).iterrows():
        a = ang[i]; sign = 1.0
        if ohc is not None:
            nvec = np.array([-np.sin(a), np.cos(a)])
            sign = 1.0 if (ohc - np.array([c.x, c.y])) @ nvec > 0 else -1.0
        cp = cutout(ph, c.x, c.y, a, sign); cuts.append(cp)
        if pm is not None:
            cutp.append(cutout(pm, c.x, c.y, a, sign))
        n = cp.shape[1]; m = slice(n // 2 - int(3 / PX), n // 2 + int(3 / PX))
        prof = cp[:, m, m].reshape(len(offs), -1).mean(1); okp = (offs * DZ >= -3) & (offs * DZ <= 8)
        plate = float(offs[okp][np.argmax(prof[okp])] * DZ)
        rows.append(dict(cell=i + 1, x=c.x, y=c.y, u=c.u, angle_deg=float(np.degrees(a)), across_sign=sign, plate_um=plate))
    os.makedirs(OUT, exist_ok=True)
    out = dict(phal=np.stack(cuts), offsets_um=offs * DZ)
    if cutp:
        out["pmca2"] = np.stack(cutp)
    np.savez_compressed(dst, **out)
    pd.DataFrame(rows).assign(field=field).to_csv(os.path.join(OUT, field + "_cells.csv"), index=False)
    return f"{len(rows)} inner hair cells"


WORKERS = 4   # fields processed at once (each full stack takes about 2-3 GB of memory)


def main():
    fields = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(os.path.join(S.WORK, 'fields'), "*.npz")))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def one(f):
        t = time.time()
        try:
            msg = run_field(f)
        except Exception as e:
            msg = f"failed: {e}"
        return f, msg, time.time() - t
    with ThreadPoolExecutor(WORKERS) as ex:        # fields are independent; several are read at once
        for i, fut in enumerate(as_completed([ex.submit(one, f) for f in fields]), 1):
            f, msg, s = fut.result(); print(f"{i}/{len(fields)} {f}: {msg} ({s:.0f} s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    if "--v2" in sys.argv:
        FRAME = "v2"; OUT = OUT + "_v2"; os.makedirs(OUT, exist_ok=True)
    main()
