"""Step 1 of the stereocilia plan: one picture per field, the channel table and the cell surface.

For every .nd2 field this script
  1. reads the 3D stack with its pixel sizes,
  2. makes a maximum projection of each color channel,
  3. finds the depth of the cell surface across the field from the phalloidin channel
     (the depth of brightest phalloidin in 64 x 64 pixel squares; squares far from their
     neighbors are replaced by the neighbors' median, then the depths are smoothed),
  4. projects phalloidin only in a thin layer around that surface, which removes the
     cell borders deeper in the tissue and leaves cuticular plates and bundles,
  5. saves the projections and surface to E:\\Webber_confocal\\work\\fields\\<field>.npz,
     one picture per field to figures\\fields\\<field>.png, and one row per field to
     results\\field_table.csv (channel to stain map, surface depth and tilt).

Run from the scripts folder:
    python make_field_pictures.py            (all fields, about 30 minutes)
    python make_field_pictures.py --only AR170 --limit 1   (one test field)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_settings as S
import argparse, glob, os, re, sys, time, types
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import nd2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RAW = S.RAW
WORK = os.path.join(S.WORK, 'fields')
FIG = os.path.join(S.FIGURES, "fields")
TABLE = os.path.join(S.RESULTS, "field_table.csv")
TILE = 64            # pixels per tile for surface finding (about 7 um)
SLAB_UM = 2.0        # half thickness of the layer around the surface, um
LOCAL = True         # surface follows each square's depth (False: one smooth curved surface for the field)
CHECKED = {"AR1064 Ocm Calb2 Phall P1 60X APEX.nd2", "AR454 Calb2 PMCA2 Phall P1 60X APEX (2).nd2"}


def central_time():
    """Local time in Omaha (CST from November to March, CDT otherwise)."""
    now = datetime.now(timezone.utc)
    dst = 3 <= now.month <= 10   # close enough for log stamps
    return (now + timedelta(hours=-5 if dst else -6)).strftime("%Y-%m-%d %H:%M ") + ("CDT" if dst else "CST")


def stains_from_name(name):
    low = name.lower()
    keys = [("Ocm", r"ocm|oncm"), ("Calb2", "calb2"), ("PMCA2", "pmca2"), ("Phalloidin", "phall"),
            ("Prestin", "prestin"), ("CtBP2", "ctbp2")]
    return [s for s, k in keys if re.search(k, low)]


def channel_map(filters, stains):
    """Stain for each color channel, from the filter name and the stains in the file name."""
    out = []
    for f in filters:
        f = f.upper()
        if f == "DAPI":
            out.append("DAPI")
        elif f == "CY5":
            out.append("Phalloidin" if "Phalloidin" in stains else "to check")
        elif f == "TRITC":
            out.append("Calb2" if "Calb2" in stains and "Phalloidin" in stains else "to check")
        elif f in ("EGFP", "FITC"):
            out.append("Ocm" if "Ocm" in stains else "PMCA2" if "PMCA2" in stains else "to check")
        else:
            out.append("to check")
    return out


def find_surface(phal, z_um, voxel_um=None):
    """Depth (in images) of the brightest phalloidin layer per tile, smoothed into a surface. The method lives in the
    plugin package (napari_cochlea_phenotyping._surface.find_surface, sizes in um); with the default voxel size it is
    identical to the validated version that was here until 12 Nov 2026."""
    from napari_cochlea_phenotyping._surface import find_surface as _fs
    vox = voxel_um if voxel_um is not None else (z_um, S.PX_UM, S.PX_UM)
    s = _fs(phal, vox, clear_ratio=S.SURFACE_CLEAR_RATIO, local=LOCAL)
    return s.planes, s.square_planes, s.good_squares, s.fit_error_um

def slab_projection(stack, surf, half):
    """Maximum of each pixel within +-half images of the surface."""
    nz = stack.shape[0]
    z = np.arange(nz)[:, None, None]
    inside = np.abs(z - surf[None]) <= half
    return np.where(inside, stack, 0).max(0)


def stretch(im):
    lo, hi = np.percentile(im, [1, 99.7])
    return np.clip((im.astype(float) - lo) / max(hi - lo, 1), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="process only files whose name contains this text")
    ap.add_argument("--limit", type=int, default=0, help="process at most this many files")
    ap.add_argument("--redo", action="store_true", help="redo fields already done")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    files = [S.raw_path(f) for f in S.raw_fields()]
    files = [f for f in files if a.only in os.path.basename(f)]
    if a.limit:
        files = files[:a.limit]
    old = pd.read_csv(TABLE) if os.path.exists(TABLE) else pd.DataFrame()
    rows = [] if old.empty else old.to_dict("records")
    done = {r["file"] for r in rows}
    print(f"{central_time()}  {len(files)} fields to check", flush=True)
    for i, p in enumerate(files, 1):
        name = os.path.basename(p); field = next(f for f in S.raw_fields() if os.path.basename(S.raw_path(f)) == name)
        if name in done and not a.redo:
            print(f"  [{i}/{len(files)}] {name}: already done", flush=True); continue
        t0 = time.time()
        arr = S.read_raw(field)                                              # Z, C, Y, X
        filters, (vz_, vy_, vx_) = S.raw_channels(field); vx = types.SimpleNamespace(z=vz_, y=vy_, x=vx_)
        stains = stains_from_name(name)
        cmap = S.stains_for(field, filters) or channel_map(filters, stains)   # stains confirmed in the plugin, else guessed
        maxp = arr.max(0)
        zprof = arr.reshape(arr.shape[0], arr.shape[1], -1).mean(-1)
        rec = dict(file=name, field=field, z_images=arr.shape[0], pixel_um=round(vx.x, 4), z_step_um=round(vx.z, 3),
                   channels=";".join(filters), stains=";".join(cmap),
                   channel_check="by eye" if name in CHECKED else "from file name")
        if "Phalloidin" in cmap:
            ci = cmap.index("Phalloidin")
            surf, zpk, ntiles, fit_err = find_surface(arr[:, ci], vx.z, voxel_um=(vx.z, vx.y, vx.x))
            half = int(round(SLAB_UM / vx.z))
            slab = slab_projection(arr[:, ci], surf, half)
            layers = np.stack([slab_projection(arr[:, c], surf, half) for c in range(arr.shape[1])])
            rec.update(surface_tiles=ntiles, surface_fit_error_um=round(fit_err, 2),
                       surface_depth_min_um=round(float(surf.min() * vx.z), 1),
                       surface_depth_max_um=round(float(surf.max() * vx.z), 1),
                       surface_depth_range_um=round(float((surf.max() - surf.min()) * vx.z), 1))
        else:
            surf = zpk = slab = layers = None
        np.savez_compressed(os.path.join(WORK, field + ".npz"), maxproj=maxp, zprofile=zprof,
                            channels=np.array(filters), stains=np.array(cmap),
                            surface=np.zeros(0) if surf is None else surf.astype(np.float32),
                            surface_tiles=np.zeros(0) if zpk is None else zpk,
                            phalloidin_layer=np.zeros(0) if slab is None else slab,
                            surface_layers=np.zeros(0) if slab is None else layers,
                            voxel_um=np.array([vx.z, vx.y, vx.x]))
        # picture: four channels, phalloidin near the surface, surface depth
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        for c in range(min(4, maxp.shape[0])):
            ax = axs.flat[c]; ax.imshow(stretch(maxp[c]), cmap="gray")
            ax.set_title(f"{filters[c]}: {cmap[c]} (all depths)"); ax.axis("off")
        ax = axs.flat[4]
        if slab is not None:
            ax.imshow(stretch(slab), cmap="gray"); ax.set_title(f"Phalloidin within {SLAB_UM:g} um of the cell surface")
        ax.axis("off")
        ax = axs.flat[5]
        if surf is not None:
            im = ax.imshow(surf * vx.z, cmap="viridis"); fig.colorbar(im, ax=ax, fraction=0.046, label="Surface depth (um)")
            ax.set_title("Cell surface depth")
        ax.axis("off")
        fig.suptitle(name, fontsize=14); fig.tight_layout()
        fig.savefig(os.path.join(FIG, field + ".png"), dpi=70); plt.close(fig)
        rec["seconds"] = round(time.time() - t0)
        rows = [r for r in rows if r["file"] != name] + [rec]
        pd.DataFrame(rows).to_csv(TABLE, index=False)                         # saved after every field
        print(f"  [{i}/{len(files)}] {name}: {rec['seconds']} s, stains {rec['stains']}", flush=True)
        del arr
    print(f"{central_time()}  finished; table {TABLE}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
