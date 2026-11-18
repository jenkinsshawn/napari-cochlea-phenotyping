"""Settings for the confocal measurement pipeline: folders, image scale, and which mouse and group each field belongs to.

Every pipeline script reads its folders and scale from here. For a new dataset, copy this file, edit the copy, and
point the scripts at it with the environment variable CONFOCAL_SETTINGS, or with run_pipeline.py --settings <file>:

    set CONFOCAL_SETTINGS=E:\\NewStudy\\pipeline_settings_newstudy.py
    python run_pipeline.py

The copy only needs the values that change; anything it leaves out keeps the value below.

Field names: each .nd2 file name starts with the mouse ID (e.g. "AR524 Ocm, Calb2 Phall P1 60X APEX.nd2"). MICE lists
the mouse IDs as they begin the file names, with the group of each; the longest matching start wins, so
"JW5436 (2)" is matched before "JW5436". Fields of mice not listed are left without a group and are not compared.
"""
import os

# Folders ---------------------------------------------------------------------------------------------------------
DATA_ROOT = r"E:\Webber_confocal"
RAW_FOLDER = "Dominika_Smad3_bcat"    # the .nd2 files, one per field (inside DATA_ROOT)
# WORK, ANNOTATIONS, MODELS, PUBLIC and RESULTS are set from DATA_ROOT at the end of this file unless a settings copy
# gives them directly.

# Image scale (Nikon spinning disk, Plan Apo 60x oil, Kinetix camera) ------------------------------------------------
PX_UM = 0.1083        # pixel size across the image (um); read from the .nd2 metadata for a new microscope
DZ_UM = 0.3           # z step (um, nominal)
BUNDLE_SIDE = -1      # bundles lie toward lower plane numbers (checked by eye in every field); +1 flips the planes on reading
Z_SCALE = 0.84        # nominal z distances x 0.84 = true distances (oil objective into watery tissue; measured on the
                      # stacked cell layers of three fields). Measure again for a different objective or mounting.

# Bundle outlines ---------------------------------------------------------------------------------------------------
SMOOTH_BUNDLE_OUTLINE = True  # bundle measures use the detector outline with holes filled, its largest piece only, and a
                              # smoothed edge (Gaussian, 1.3 px); 143 of 192 bundle measures robust against 107 unsmoothed

# Standardization (tested 12 Nov 2026 against brightness changes; see the plan) ------------------------------------
SURFACE_CLEAR_RATIO = 2.0     # a 7 um square shows clear phalloidin when (its brightest layer - field background) is at
                              # least this many times (its median layer - field background); background = 1st percentile
IHC_BACKGROUND_RATIO = 0.042  # inner hair cell cut-outs are scaled so the background sits at this fraction of the
                              # 99.9th percentile (the median over the 354 cut-outs the classifier was built on)

# Mice and groups --------------------------------------------------------------------------------------------------
MICE = {              # mouse ID as it begins the file name: group
    "AR1064": "control", "AR524": "control",
    "JW5436": "Smad3", "JW5436 (2)": "Smad3", "AR454": "Smad3",
    "AR1383-2": "beta-catenin", "AR170": "beta-catenin", "AR47": "beta-catenin",
}
CONTROL_GROUP = "control"
DAMAGED = {           # fields left out of group comparisons, with the reason
    "AR47 Calb2 PMCA2 Phall P1 60X BASE": "folded",
    "AR1064 Ocm Calb2 Phall P1 60X APEX": "folded",
    "JW5436 (2)Calb2 PMCA2 Phall P1 60X APEX": "torn",
}

# A copy for another dataset replaces any of the values above -------------------------------------------------------
_override = os.environ.get("CONFOCAL_SETTINGS")
_given = set()
if _override:
    _vals = {"os": os, "__file__": _override}
    with open(_override, encoding="utf-8") as _fh:
        exec(compile(_fh.read(), _override, "exec"), _vals)
    _given = {k for k in _vals if k.isupper()}
    globals().update({k: v for k, v in _vals.items() if k.isupper()})

# Settings the user confirmed in the napari plugin (dataset_settings.json): scale, bundle direction, stains per image,
# mouse and group. A settings copy names the file with DATASET_SETTINGS = r"...\dataset_settings.json"; values the copy
# gives directly win.
if "DATASET_SETTINGS" not in _given: DATASET_SETTINGS = ""
_ds = None
if DATASET_SETTINGS:
    import json as _json
    with open(DATASET_SETTINGS, encoding="utf-8") as _fh:
        _ds = _json.load(_fh)
    _vz, _vy, _vx = (float(v) for v in _ds["voxel_um"])
    if "PX_UM" not in _given: PX_UM = _vx
    if "DZ_UM" not in _given: DZ_UM = _vz
    if "BUNDLE_SIDE" not in _given: BUNDLE_SIDE = int(_ds.get("bundle_side", -1))
    # the plugin's mouse table replaces this file's (Dominika's) mice, control group and damaged fields
    if "MICE" not in _given: MICE = {m: g for m, g in _ds.get("group_of_mouse", {}).items() if g}
    if "CONTROL_GROUP" not in _given: CONTROL_GROUP = _ds.get("control_group", "")
    if "DAMAGED" not in _given: DAMAGED = dict(_ds.get("excluded_fields", {}))
if BUNDLE_SIDE not in (-1, 1):
    raise ValueError(f"BUNDLE_SIDE must be -1 or 1, got {BUNDLE_SIDE}")

# Folders that follow DATA_ROOT unless given directly
# The official run is ihc_log_2026-11-14 (16 Nov 2026): the standardized rerun of 12-13 Nov 2026 with inner hair cells
# found from log-scaled Calb2 outlines. Earlier runs are kept unchanged for reference: standardized_2026-11-12
# (work_standardized_2026-11-12, results_standardized_2026-11-12) and the first full run (work\, stereocilia\results\).
OFFICIAL_RUN = "ihc_log_2026-11-14"
if "RAW" not in _given: RAW = os.path.join(DATA_ROOT, RAW_FOLDER)
if "WORK" not in _given:                                                          # stacks, outlines, cut-outs
    WORK = os.path.join(DATA_ROOT, "work_" + OFFICIAL_RUN) if not _override else os.path.join(DATA_ROOT, "work")
if "ANNOTATIONS" not in _given: ANNOTATIONS = os.path.join(DATA_ROOT, "annotations")  # hand references
if "CLICK_OUTLINES" not in _given:   # the outlines the saved clicks name cells in (click_bundle_points, click_directions)
    CLICK_OUTLINES = os.path.join(ANNOTATIONS, "click_reference_outlines_v1")
if "MODELS" not in _given: MODELS = os.path.join(DATA_ROOT, "models")            # trained classifiers
if "PUBLIC" not in _given: PUBLIC = os.path.join(DATA_ROOT, "public")            # public data (VASCilia)
if "RESULTS" not in _given:
    RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_" + OFFICIAL_RUN) \
        if not _override else os.path.join(DATA_ROOT, "results")
if "FIGURES" not in _given:   # figures go inside the results folder
    FIGURES = os.path.join(RESULTS, "figures")


def mouse_of(field):
    """Mouse ID of a field: the longest ID in MICE that the field name starts with and that ends at a word boundary
    (a space, a bracket or the end of the name); otherwise the first word of the name."""
    if _ds and field in _ds.get("mouse_of_field", {}):
        return _ds["mouse_of_field"][field]
    def ends_word(m): return m[-1] in " )" or len(field) == len(m) or field[len(m)] in " (,"
    hits = [m for m in MICE if field.startswith(m) and ends_word(m)]
    return max(hits, key=len) if hits else field.split(" ")[0]


def group_of(field):
    """Group of a field, or None when its mouse is not listed in MICE."""
    return MICE.get(mouse_of(field))


def cell_labels_path(field, tag="v1"):
    """Cell outlines that every step uses for a field: the hand-corrected outlines (ANNOTATIONS\\cells\\<field>_cells.npy)
    when they exist, otherwise the computer outlines of tag (WORK\\cells\\<tag>). One rule for all steps (12 Nov 2026):
    cell types and bundle measures used the corrected outlines while bundle_trace.field_maps used the computer
    outlines, so cell numbers did not match on the corrected fields."""
    corr = os.path.join(ANNOTATIONS, "cells", field + "_cells.npy")
    return corr if os.path.exists(corr) else os.path.join(WORK, "cells", tag, field + ".npy")


# Raw images ---------------------------------------------------------------------------------------------------------
RAW_EXTENSIONS = (".nd2", ".ome.tif", ".ome.tiff", ".tif", ".tiff")


def raw_path(field):
    """The raw image file of a field (RAW\\<field>.nd2, or a TIFF of the same name)."""
    for ext in RAW_EXTENSIONS:
        p = os.path.join(RAW, field + ext)
        if os.path.exists(p):
            return p
    import glob as _glob
    for ext in RAW_EXTENSIONS:                      # files in subfolders of RAW
        hits = [p for p in _glob.glob(os.path.join(RAW, "**", "*" + ext), recursive=True) if os.path.basename(p)[:-len(ext)] == field]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no image file for {field!r} in {RAW} ({', '.join(RAW_EXTENSIONS)})")


def raw_fields():
    """Field names of the raw images in RAW."""
    if not os.path.isdir(RAW):
        return []
    out = set()
    for f in os.listdir(RAW):
        for ext in RAW_EXTENSIONS:
            if f.lower().endswith(ext):
                out.add(f[:-len(ext)]); break
    return sorted(out)


def raw_channels(field):
    """Filter name of each channel and the voxel size (z, y, x) in um, from the file."""
    p = raw_path(field)
    if p.lower().endswith(".nd2"):
        import nd2
        with nd2.ND2File(p) as f:
            v = f.voxel_size(); return [c.channel.name for c in f.metadata.channels], (v.z, v.y, v.x)
    from napari_cochlea_phenotyping._io import read_info
    i = read_info(p); return [c.name for c in i.channels], tuple(i.voxel_um)


def read_raw(field, channel=None):
    """Raw image (z, c, y, x), or one channel (z, y, x) when channel is an index; planes are put in the pipeline's
    order (bundles toward lower plane numbers) using BUNDLE_SIDE."""
    p = raw_path(field)
    if p.lower().endswith(".nd2"):
        import nd2
        with nd2.ND2File(p) as f:
            a = f.asarray() if channel is None else f.to_dask()[:, channel].compute()
    else:
        from napari_cochlea_phenotyping._io import read_stack
        a = read_stack(p); a = a if channel is None else a[:, channel]
    return a[::-1].copy() if BUNDLE_SIDE == 1 else a


def stains_for(field, filters):
    """Stain of each channel as the user confirmed it in the plugin, or None when there are no plugin settings."""
    if not _ds:
        return None
    m = _ds.get("stains_by_panel", {}).get(_ds.get("panel_of_field", {}).get(field, ""), _ds.get("stains_by_channel_name", {}))
    return [m.get(c, "to check") for c in filters]


# The plugin package holds the shared step code (one copy for the scripts and the napari plugin). Use the installed
# package if there is one, otherwise the copy beside the scripts (Projects\confocal-phenotyping\napari-cochlea-phenotyping).
import sys as _sys
try:
    import napari_cochlea_phenotyping as _ncp  # noqa: F401
except ImportError:
    _src = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "napari-cochlea-phenotyping", "src")
    if os.path.isdir(_src) and _src not in _sys.path: _sys.path.insert(0, _src)
