"""Nuclei from DAPI with Cellpose (Cellpose-SAM): 2D outlines on each plane, joined across planes.

For each field, the DAPI stack is flattened along the tissue surface (the field file's surface map) and sampled
from 10 um above to 26 um below the surface every 0.6 um (61 planes), at half the pixel size (0.217 um). Hair cell nuclei
sit about 10-20 um below the surface (outer) and 5-15 um (inner).

Output, one file per field:  E:\\Webber_confocal\\work\\nuclei_v3_gpu\\<field>_nuclei.npz (with --gpu)
    labels   (planes, 512, 512) uint32, the same number on every plane of one nucleus
    depth_um (planes,) depth below the surface;  px_um  pixel size
Finished fields are skipped, so the run can be stopped and restarted. Progress goes to run_log.csv.

Run (imaging env):  python nuclei_cellpose.py --gpu      (graphics card; writes work\\nuclei_v3_gpu)
                    python nuclei_cellpose.py            (CPU only, about 4 min per field)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import pipeline_settings as PS   # raw images (PS: this file uses S for surfaces)
import glob, os, sys, time
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy import ndimage as ndi

RAW = S.RAW; FIELDS = os.path.join(S.WORK, 'fields'); OUT = os.path.join(S.WORK, 'nuclei_v3')   # v3 (11 Nov 2026): planes from 10 um above the surface; earlier runs in work\nuclei, nuclei_v2
# planes from 10 um above to 26 um below the surface: half of the inner hair cell nuclei reached the -3 um top plane in v2
DEPTHS_UM = np.arange(-10.0, 26.01, 0.6); DZ = S.DZ_UM; PX = S.PX_UM; BIN = 2
STITCH = 0.25
GPU = "--gpu" in sys.argv
if GPU: OUT = OUT + "_gpu"   # --gpu: outlines on the graphics card, written to their own folder so every field uses one device


def central_time():
    now = datetime.now(timezone.utc); dst = 3 <= now.month <= 10
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def flattened_planes(field, crop=None):
    """DAPI planes at fixed depths below the smoothed surface, binned 2 x 2 (planes, y, x), scaled 0-1."""
    import nd2
    z = np.load(os.path.join(FIELDS, field + ".npz"), allow_pickle=True); st = [str(s) for s in z["stains"]]
    if "DAPI" not in st or np.ndim(z["surface"]) != 2: return None   # e.g. AR1065: no phalloidin, so no surface map
    D = PS.read_raw(field, st.index("DAPI")).astype(np.float32)
    S = ndi.gaussian_filter(z["surface"].astype(np.float32), 20)
    if crop is not None: y0, y1, x0, x1 = crop; D = D[:, y0:y1, x0:x1]; S = S[y0:y1, x0:x1]
    ny, nx = D.shape[1] // BIN * BIN, D.shape[2] // BIN * BIN
    D = D[:, :ny, :nx].reshape(D.shape[0], ny // BIN, BIN, nx // BIN, BIN).mean((2, 4)); S = S[:ny:BIN, :nx:BIN]
    yy, xx = np.mgrid[0:D.shape[1], 0:D.shape[2]].astype(np.float32)
    P = np.stack([ndi.map_coordinates(D, [S + d / DZ, yy, xx], order=1, mode="nearest") for d in DEPTHS_UM])
    lo, hi = np.percentile(P, [1, 99.8]); return np.clip((P - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def run_field(field, model, crop=None):
    P = flattened_planes(field, crop)
    if P is None: return None
    masks = model.eval(P, z_axis=0, do_3D=False, stitch_threshold=STITCH, batch_size=8)[0]
    return np.asarray(masks).astype(np.uint32)


def main():
    from cellpose import models
    os.makedirs(OUT, exist_ok=True); log = os.path.join(OUT, "run_log.csv")
    if not os.path.exists(log): open(log, "w").write("field,nuclei,seconds,finished_central\n")
    import torch
    use_gpu = GPU and torch.cuda.is_available()
    print("graphics card:", torch.cuda.get_device_name(0) if use_gpu else "not used (CPU)", flush=True)
    model = models.CellposeModel(gpu=use_gpu)
    fields = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(FIELDS, "*.npz")))
    fields = [f for f in fields if os.path.exists(os.path.join(RAW, f + ".nd2"))]
    for i, f in enumerate(fields, 1):
        dst = os.path.join(OUT, f + "_nuclei.npz")
        if os.path.exists(dst): print(f"{i}/{len(fields)} {f}: already done", flush=True); continue
        t0 = time.time(); L = run_field(f, model)
        if L is None: print(f"{i}/{len(fields)} {f}: skipped, no DAPI or no surface map", flush=True); continue
        np.savez_compressed(dst, labels=L, depth_um=DEPTHS_UM, px_um=PX * BIN); n = len(np.unique(L)) - 1
        open(log, "a").write(f'"{f}",{n},{time.time() - t0:.0f},{central_time()}\n')
        print(f"{i}/{len(fields)} {f}: {n} nuclei, {(time.time() - t0) / 60:.1f} min, finished {central_time()}", flush=True)
    print("all fields done", central_time(), flush=True)


def test():
    """One 28 um square of AR524 base, outer hair cell rows: prints counts and saves nuclei_test.png in OUT."""
    from cellpose import models
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True); f = "AR524 Ocm, Calb2 Phall P1 60X BASE"; crop = (355, 611, 235, 491)
    t0 = time.time(); P = flattened_planes(f, crop); L = np.asarray(models.CellposeModel(gpu=False).eval(P, z_axis=0, do_3D=False, stitch_threshold=STITCH, batch_size=8)[0])
    ids = np.unique(L[L > 0]); areas = [(L == i).sum((1, 2)).max() * (PX * BIN) ** 2 for i in ids]; planes = [(L == i).any((1, 2)).sum() for i in ids]
    print(f"{len(ids)} nuclei in {time.time() - t0:.0f} s; planes per nucleus median {np.median(planes):.0f}; widest-plane area quartiles (um2) {np.percentile(areas, [25, 50, 75]).round(1)}", flush=True)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for a, k in zip(axs.T, [5, 13, 20, 30]):
        a[0].imshow(P[k], cmap="gray"); a[0].set_title(f"{DEPTHS_UM[k]:.1f} um below surface", fontsize=8)
        a[1].imshow(P[k], cmap="gray"); a[1].contour(L[k] > 0, [0.5], colors="yellow", linewidths=0.6)
        for b in a: b.axis("off")
    fig.savefig(os.path.join(OUT, "nuclei_test.png"), dpi=90); np.save(os.path.join(OUT, "nuclei_test_labels.npy"), L.astype(np.uint32))
    print("saved", os.path.join(OUT, "nuclei_test.png"), flush=True)


if __name__ == "__main__":
    test() if "--test" in sys.argv else main()
