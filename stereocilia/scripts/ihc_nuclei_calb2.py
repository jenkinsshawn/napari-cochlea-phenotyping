"""Inner hair cell nuclei and their Calb2, from the Cellpose nucleus outlines (nuclei_cellpose.py --gpu).

Steps for each field:
  1. Calb2 is read from the raw stack and flattened along the fitted phalloidin surface on the same planes as the
     nucleus outlines (nuclei_cellpose.DEPTHS_UM, 2 x 2 binned, 0.2166 um pixels).
  2. For every nucleus: Calb2 in a ring 1-4 binned pixels (0.2-0.9 um) outside the outline in each plane, excluding
     other nuclei, minus the field's 20th-percentile Calb2; widest-plane area; volume; top, bottom and widest planes.
  3. A nucleus is wrapped in Calb2 (an inner hair cell nucleus) when its ring Calb2 is at least half of the field's
     99th percentile of ring Calb2 (calb2_rel >= 0.5; the split is clean, see the plan).
  4. Kept: wrapped nuclei within 8 um across the row of the field's inner hair cells (ihc_stacks_v2 <field>_cells.csv,
     row frame from hair_cell_rows_classify.frame_v2) that do not touch the first or last plane (whole).
Checked against Shawn's 22 hand outlines: widest-plane area median -6%, 91% within 20%.

Calb2 brightness (ring Calb2 over the field background) is written too, but it is NOT comparable between imaging
sessions with different Calb2 settings (see imaging_settings.py). Compare it only within one session.

Outputs (results folder):
  ihc_nuclei_all_nuclei.csv   every nucleus with its ring Calb2
  ihc_nuclei_cells.csv        kept inner hair cell nuclei
  ihc_nuclei_fields.csv       field medians: ihc_nucleus_area_um2, ihc_nucleus_volume_um3, measured share
  ihc_calb2_fields.csv        field median ihc_calb2_over_background

Run (imaging env, from the scripts folder):  python ihc_nuclei_calb2.py   [--fields "<field name>" ...]
About 40 minutes for 28 fields, mostly reading the raw stacks. Reproduces the 11 Nov 2026 session results exactly (294 nuclei).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import pipeline_settings as PS   # raw images (PS: this file uses S for surfaces)
import os, sys
import numpy as np, pandas as pd
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import nuclei_cellpose as nc
import hair_cell_rows_classify as hrc
import bundle_trace as bt

NUCLEI = os.path.join(S.WORK, 'nuclei_v3_gpu')
FIELDS = os.path.join(S.WORK, 'fields')
STACKS2 = os.path.join(S.WORK, 'ihc_stacks_v2')
RESULTS = S.RESULTS
DEPTHS_UM = np.arange(-10.0, 26.01, 0.6)      # planes of the nuclei_v3_gpu run
PXN = 2 * S.PX_UM                                   # binned pixel (um)
RING = (1, 4)                                  # ring inner and outer dilation, binned pixels
BG_PCT = 20                                    # field background percentile of flattened Calb2
WRAPPED = 0.5                                  # calb2_rel threshold for an inner hair cell nucleus
IN_ROW_UM = 8.0                                # across-row distance from the inner hair cell row


def flat_stain(field, stain):
    import nd2
    z = np.load(os.path.join(FIELDS, field + ".npz"), allow_pickle=True); st = [str(s) for s in z["stains"]]
    D = PS.read_raw(field, st.index(stain)).astype(np.float32)
    S = ndi.gaussian_filter(z["surface"].astype(np.float32), 20)
    D = D[:, :1024, :1024].reshape(D.shape[0], 512, 2, 512, 2).mean((2, 4)); S = S[::2, ::2]
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float32)
    return np.stack([ndi.map_coordinates(D, [S + d / 0.3, yy, xx], order=1, mode="nearest") for d in DEPTHS_UM])


def field_nuclei(field):
    L = np.load(os.path.join(NUCLEI, field + "_nuclei.npz"))["labels"]; C2 = flat_stain(field, "Calb2")
    bg = float(np.percentile(C2, BG_PCT)); objs = ndi.find_objects(L); rows = []
    for i in np.unique(L[L > 0]):
        sl = objs[i - 1]; sl = tuple(slice(max(s.start - 6, 0), min(s.stop + 6, L.shape[d])) for d, s in enumerate(sl))
        m = L[sl] == i; sh = np.zeros_like(m)
        for k in range(m.shape[0]):
            if m[k].any(): sh[k] = ndi.binary_dilation(m[k], iterations=RING[1]) & ~ndi.binary_dilation(m[k], iterations=RING[0])
        sh &= (L[sl] == 0); pa = m.sum((1, 2)); zi = np.arange(len(pa)) + sl[0].start; w = np.where(pa > 0)[0]
        rows.append(dict(field=field, label=int(i), calb2_shell=float(C2[sl][sh].mean() - bg) if sh.any() else np.nan,
                         area_um2=float(pa.max() * PXN ** 2), volume_um3=float(m.sum() * PXN ** 2 * 0.6),
                         z=float(np.average(zi, weights=pa)), top_plane=int(zi[w[0]]), bottom_plane=int(zi[w[-1]]),
                         widest_plane=int(zi[np.argmax(pa)]), n_planes=L.shape[0],
                         y=float(np.mean(np.where(m)[1]) + sl[1].start) * 2, x=float(np.mean(np.where(m)[2]) + sl[2].start) * 2))
    N = pd.DataFrame(rows); N["bg"] = bg
    N["calb2_rel"] = N.calb2_shell / np.nanpercentile(N.calb2_shell, 99)
    return N


def select_ihc(N, field):
    d = N[N.calb2_rel >= WRAPPED].copy()
    co = hrc.frame_v2(field, bt.field_maps_bowl(field)); e, u, _ = co(d.x.values.astype(float), d.y.values.astype(float))
    ic = pd.read_csv(os.path.join(STACKS2, field + "_cells.csv")); ei, ui, _ = co(ic.x.values.astype(float), ic.y.values.astype(float))
    d = d.assign(e=e, u=u, e_ihc=np.nanmedian(ei), n_ihc=len(ic))
    d["in_row"] = (d.e - d.e_ihc).abs() <= IN_ROW_UM; d["whole"] = (d.top_plane > 0) & (d.bottom_plane < d.n_planes - 1)
    # detected inner hair cells with a kept nucleus within 4 um along the row (13 Nov 2026: the share was nuclei / cells,
    # which passes 1 when the cell list misses cells, e.g. 6 nuclei for 4 detected cells in the torn JW5436 (2) APEX)
    ku = d.u.values[(d.in_row & d.whole).values]; ui = ui[np.isfinite(ui)]
    if len(ku) and len(ui):                         # one nucleus per cell: optimal pairing within 4 um along the row
        from scipy.optimize import linear_sum_assignment
        D = np.abs(ku[:, None] - ui[None, :]); r, c = linear_sum_assignment(np.where(D <= 4.0, D, 1e6))
        n_pair = int((D[r, c] <= 4.0).sum())
        kept_idx = d.index[(d.in_row & d.whole).values]; d["paired_cell"] = False
        d.loc[kept_idx[r[D[r, c] <= 4.0]], "paired_cell"] = True
    else:
        n_pair = 0; d["paired_cell"] = False
    d["n_ihc_with_nucleus"] = n_pair; d["n_nuclei_without_cell"] = len(ku) - n_pair
    return d


def main(fields=None):
    fields = fields or sorted(f[:-11] for f in os.listdir(NUCLEI) if f.endswith("_nuclei.npz"))
    fields = [f for f in fields if os.path.exists(os.path.join(STACKS2, f + "_cells.csv"))]
    ALL, KEEP = [], []
    from concurrent.futures import ThreadPoolExecutor
    def one(f):
        N = field_nuclei(f); return N, select_ihc(N, f)
    with ThreadPoolExecutor(4) as ex:              # fields are independent; several are read at once
        for f, (N, d) in zip(fields, ex.map(one, fields)):
            ALL.append(N); KEEP.append(d)
            print(f"{f}: {len(N)} nuclei, {len(d)} wrapped in Calb2, {int((d.in_row & d.whole).sum())} kept", flush=True)
    ALL = pd.concat(ALL, ignore_index=True); IH = pd.concat(KEEP, ignore_index=True); U = IH[IH.in_row & IH.whole].copy()
    U["calb2_over_bg"] = (U.calb2_shell + U.bg) / U.bg
    FN = U.groupby("field").agg(ihc_nucleus_area_um2=("area_um2", "median"), ihc_nucleus_volume_um3=("volume_um3", "median"),
                                ihc_nuclei_measured=("label", "size"), ihc_cells=("n_ihc", "first"),
                                ihc_cells_with_nucleus=("n_ihc_with_nucleus", "first"),
                                ihc_nuclei_without_detected_cell=("n_nuclei_without_cell", "first")).reset_index()
    FN["ihc_nuclei_measured_share"] = FN.ihc_cells_with_nucleus / FN.ihc_cells   # share of detected inner hair cells with a measured nucleus
    FC = U.groupby("field").agg(ihc_calb2_over_background=("calb2_over_bg", "median")).reset_index()
    ALL.to_csv(os.path.join(RESULTS, "ihc_nuclei_all_nuclei.csv"), index=False)
    U.drop(columns=["bg", "calb2_over_bg"]).to_csv(os.path.join(RESULTS, "ihc_nuclei_cells.csv"), index=False)
    FN.to_csv(os.path.join(RESULTS, "ihc_nuclei_fields.csv"), index=False); FC.to_csv(os.path.join(RESULTS, "ihc_calb2_fields.csv"), index=False)
    return ALL, U, FN, FC


if __name__ == "__main__":
    main([a for a in sys.argv[sys.argv.index("--fields") + 1:]] if "--fields" in sys.argv else None)
