"""Imaging settings of every field, read from the .nd2 metadata, and a check of which groups share them.

For each field and stain: imaging date, laser line (nm) and power (%), camera exposure (ms), objective, pinhole,
pixel size. Brightness can only be compared between fields imaged with the same laser line and power; exposure is
reported so that brightness per ms can be used within one laser setting. Shape and count measures use thresholds
relative to each cell or field and do not need equal settings.

Outputs (results folder):
  imaging_settings_by_field.csv     one row per field and stain
  imaging_settings_by_group.csv     for each stain, the settings used in each group (fields per setting)
The script prints a warning for every stain whose settings differ between the control group and another group.

Run (imaging env, from the scripts folder):  python imaging_settings.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
RAW = S.RAW
CONTROL = S.CONTROL_GROUP   # (S is reused below for the settings table)
FIELDS = os.path.join(S.WORK, 'fields')
RESULTS = S.RESULTS


def field_settings(field):
    import nd2
    z = np.load(os.path.join(FIELDS, field + ".npz"), allow_pickle=True); stains = [str(s) for s in z["stains"]]
    with nd2.ND2File(os.path.join(RAW, field + ".nd2")) as h:
        text = str(h.text_info.get("description", "")); date = str(h.text_info.get("date", "")); chans = h.metadata.channels
        blocks = re.split(r"Plane #\d+:", text)[1:]; rows = []
        for ci, stain in enumerate(stains):
            b = blocks[ci] if ci < len(blocks) else ""; ch = chans[ci]
            ex = re.search(r"Exposure:\s*([\d.]+)\s*ms", b); lz = re.findall(r"ExW:\s*(\d+);\s*Power:\s*([\d.\-]+)", b)
            rows.append(dict(field=field, stain=stain, date=date.split()[0] if date else "", channel=ch.channel.name,
                             laser_nm=";".join(a for a, _ in lz), laser_power_pct=";".join(p for _, p in lz),
                             exposure_ms=float(ex.group(1)) if ex else np.nan, objective=ch.microscope.objectiveName,
                             pinhole_um=ch.microscope.pinholeDiameterUm, pixel_um=ch.volume.axesCalibration[0]))
    return rows


def main():
    import build_feature_table as bft
    fields = sorted(f[:-4] for f in os.listdir(FIELDS) if f.endswith(".npz"))
    S = pd.DataFrame([r for f in fields if os.path.exists(os.path.join(RAW, f + ".nd2")) for r in field_settings(f)])
    S["mouse"] = S.field.map(bft.mouse_of); S["group"] = S.field.map(bft.group_of)
    S["laser_setting"] = S.laser_nm + " nm at " + S.laser_power_pct + "%"
    G = S.groupby(["stain", "group", "laser_setting"]).agg(fields=("field", "size"), exposure_ms=("exposure_ms", lambda x: "/".join(str(int(v)) if v == int(v) else str(v) for v in sorted(set(x)))),
                                                           dates=("date", lambda x: ", ".join(sorted(set(x))))).reset_index()
    S.to_csv(os.path.join(RESULTS, "imaging_settings_by_field.csv"), index=False); G.to_csv(os.path.join(RESULTS, "imaging_settings_by_group.csv"), index=False)
    for stain, d in S.groupby("stain"):
        ctrl = set(d[d.group == CONTROL].laser_setting)
        for g in sorted(set(d.group.dropna()) - {CONTROL}):
            other = set(d[d.group == g].laser_setting)
            if ctrl and other and ctrl != other:
                print(f"WARNING {stain}: {CONTROL} imaged with {sorted(ctrl)}, {g} with {sorted(other)}; do not compare {stain} brightness between these groups")
    return S, G


if __name__ == "__main__":
    main()
