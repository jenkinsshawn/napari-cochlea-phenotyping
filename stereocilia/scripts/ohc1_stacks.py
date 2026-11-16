"""Cut-outs of every first-row outer hair cell from the full .nd2 stack, for outer hair cell bundle height.

For each field:
  1. First-row outer hair cells from results\\hair_cell_rows_cells.csv (row OHC1, version 2 row models).
  2. Row direction at each cell: the line through the first-row cells within 20 um along the row.
  3. Planes from 2 um below to 8 um above the field surface (fields npz 'surface'), as in ihc_stacks.py.
  4. A 12 x 12 um square around each cell, turned so x runs along the row and y increases away from the inner
     hair cells, at the original pixel size (0.108 um).
Output: E:\\Webber_confocal\\work\\ohc1_stacks\\<field>.npz with phal (cells, planes, 110, 110) uint16, pmca2 where
stained, offsets_um; and <field>_cells.csv (cell, x, y, u, angle_deg, across_sign). Fields already done are skipped.

Run from the scripts folder:  python ohc1_stacks.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ihc_stacks as ist
import hair_cell_rows as hr

OUT = os.path.join(S.WORK, 'ohc1_stacks'); RESULTS = S.RESULTS
LO_UM, HI_UM, HALF_UM = -2.0, 8.0, 6.0


def run_field(field, R):
    dst = os.path.join(OUT, field + ".npz")
    if os.path.exists(dst): return "already done"
    rp = ist.raw_path(field)
    if rp is None: return "no image file"
    zf, st = hr._field(field)
    if "Phalloidin" not in st: return "no phalloidin"
    C = R[(R.field == field) & (R.row == "OHC1")].sort_values("u").reset_index(drop=True)
    if len(C) < 3: return "fewer than 3 first-row cells"
    far = R[(R.field == field) & (R.row == "OHC3")][["x", "y"]].values.mean(0) if ((R.field == field) & (R.row == "OHC3")).any() else None
    import nd2
    arr = S.read_raw(field)
    offs = np.arange(int(round(LO_UM / ist.DZ)), int(round(HI_UM / ist.DZ)) + 1); surf = zf["surface"]
    ph = ist.flattened(arr[:, st.index("Phalloidin")], surf, offs); pm = ist.flattened(arr[:, st.index("PMCA2")], surf, offs) if "PMCA2" in st else None
    del arr
    ang = ist.row_angles(C); cuts, cutp, rows = [], [], []   # cut-out width passed to ist.cutout, so fields can run at once
    try:
        for i, c in C.iterrows():
            a = ang[i]; sign = 1.0
            if far is not None: sign = 1.0 if (far - np.array([c.x, c.y])) @ np.array([-np.sin(a), np.cos(a)]) > 0 else -1.0
            cuts.append(ist.cutout(ph, c.x, c.y, a, sign, HALF_UM))
            if pm is not None: cutp.append(ist.cutout(pm, c.x, c.y, a, sign, HALF_UM))
            rows.append(dict(cell=i + 1, x=c.x, y=c.y, u=c.u, angle_deg=float(np.degrees(a)), across_sign=sign))
    finally:
        pass
    os.makedirs(OUT, exist_ok=True); out = dict(phal=np.stack(cuts), offsets_um=offs * ist.DZ)
    if cutp: out["pmca2"] = np.stack(cutp)
    np.savez_compressed(dst, **out); pd.DataFrame(rows).assign(field=field).to_csv(os.path.join(OUT, field + "_cells.csv"), index=False)
    return f"{len(rows)} first-row outer hair cells"


def main():
    R = pd.read_csv(os.path.join(RESULTS, "hair_cell_rows_cells.csv"))
    from concurrent.futures import ThreadPoolExecutor
    fs = sorted(R.field.unique())
    with ThreadPoolExecutor(4) as ex:              # fields are independent; several are read at once
        for i, (f, msg) in enumerate(zip(fs, ex.map(lambda f: run_field(f, R), fs)), 1): print(f"{i}/{len(fs)} {f}: {msg}", flush=True)


if __name__ == "__main__":
    main()
