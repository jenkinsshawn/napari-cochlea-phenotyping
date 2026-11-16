"""Inner hair cell outlines from the Calb2 maximum projection, one outline per cell.

In the Calb2 projection the top and the body of each inner hair cell show as one cell, so
one outline per inner hair cell needs no linking of top and body outlines. Peak and border
rules on Calb2 intensity merged or split neighboring cells (checked 10 November 2026), so
the cells are outlined with Cellpose.

Two sets of outlines are made from the same projection with different brightness scaling:
  <field>_ihc.npy      scaled so the 99.5th percentile is 1 and clipped at 1.5. The row frame (the Calb2 band
                       in hair_cell_rows_classify) is built from these, and the row classifiers were trained on it.
  <field>_ihc_log.npy  log scaling up to the 99.99th percentile, without the flat clip. The inner hair cells
                       themselves (hair_cell_rows.inner_cells) are found from these. The clip left the brightest
                       cells, often at the ends of the row, as flat plateaus that Cellpose did not outline. Against
                       the clicked inner hair cells of 28 fields (13 Nov 2026), found rose from 90.5% to 93.4% and
                       extra fell from 4.8% to 3.3% (beta-catenin fields 82.3% to 87.9%); 9 fields improved, 16
                       were unchanged and 1 lost one cell. Outer hair cell rows were unchanged (89.5%).

Input:  E:\\Webber_confocal\\work\\fields\\<field>.npz (maxproj, Calb2 channel)
Output: E:\\Webber_confocal\\work\\cells\\ihc_calb2\\<field>_ihc.npy and <field>_ihc_log.npy (outline labels, int32)
        E:\\Webber_confocal\\work\\cells\\ihc_calb2\\run_log.csv (field, outlines, seconds, Central time; log-scaled
        outlines are listed as '<field> (log scaling)')
Fields already done are skipped, so the run can be stopped and restarted.

Run from the scripts folder:
    python ihc_cellpose.py            (all 32 field files, about 15-45 minutes on the CPU)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import glob, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

FIELDS = os.path.join(S.WORK, 'fields')
OUT = os.path.join(S.WORK, 'cells', 'ihc_calb2')
PX_UM = S.PX_UM
IHC_DIAMETER_UM = 10.0          # used only by Cellpose 3 models, which need a cell size


def calb2_projection(path):
    z = np.load(path); st = list(z["stains"])
    c = z["maxproj"][st.index("Calb2")].astype(float)
    c = (c - np.median(c)) / (np.percentile(c, 99.5) - np.median(c) + 1e-9)
    return np.clip(c, 0, 1.5).astype(np.float32)


def calb2_projection_log(path):
    """Calb2 projection on a log scale from the median to the 99.99th percentile (no flat clip of bright cells)."""
    z = np.load(path); st = list(z["stains"])
    c = z["maxproj"][st.index("Calb2")].astype(float); md = np.median(c)
    x = np.clip((c - md) / (np.percentile(c, 99.99) - md + 1e-9), 0, 1)
    return (np.log1p(50 * x) / np.log1p(50)).astype(np.float32)


OUTPUTS = (("_ihc.npy", calb2_projection, ""), ("_ihc_log.npy", calb2_projection_log, " (log scaling)"))


def main():
    from importlib.metadata import version
    from cellpose import models
    v = version("cellpose"); major = int(v.split(".")[0])
    print("cellpose", v, flush=True)
    import torch
    gpu = "--gpu" in sys.argv and torch.cuda.is_available()      # --gpu: graphics card when torch can see one
    print("graphics card:", torch.cuda.get_device_name(0) if gpu else "not used (CPU)", flush=True)
    model = models.CellposeModel(gpu=gpu) if major >= 4 else models.CellposeModel(gpu=gpu, model_type="cyto3")
    diameter = None if major >= 4 else IHC_DIAMETER_UM / PX_UM
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, "run_log.csv")
    if not os.path.exists(log):
        open(log, "w").write("field,outlines,seconds,finished_central\n")
    fields = sorted(glob.glob(os.path.join(FIELDS, "*.npz")))
    for i, f in enumerate(fields, 1):
        name = os.path.basename(f)[:-4]
        st = [str(s) for s in np.load(f)["stains"]]
        if "Calb2" not in st or "Phalloidin" not in st:          # e.g. AR1065 (stains not named, no phalloidin)
            print(f"{i}/{len(fields)} {name}: skipped, no Calb2 and phalloidin", flush=True); continue
        for suffix, projection, label in OUTPUTS:
            dst = os.path.join(OUT, name + suffix)
            if os.path.exists(dst):
                print(f"{i}/{len(fields)} {name}{label}: already done", flush=True); continue
            t = time.time()
            try:
                masks = model.eval(projection(f), diameter=diameter)[0].astype(np.int32)
            except Exception as e:
                print(f"{i}/{len(fields)} {name}{label}: failed, {e}", flush=True); continue
            np.save(dst, masks); s = time.time() - t
            now = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M %Z")
            open(log, "a").write(f"\"{name}{label}\",{int(masks.max())},{s:.0f},{now}\n")
            print(f"{i}/{len(fields)} {name}{label}: {int(masks.max())} outlines, {s:.0f} s, {now}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
