"""Results of a finished run, read from its results folder: group comparisons with their warnings, validation cards
and the tables per cell and per field. No Qt here, so the rules can be tested on their own.

Which P is shown: the mouse-level permutation P (every mouse's turn-adjusted mean, labels shuffled across mice). The
mixed-model P is kept as a second column; where the model estimates no spread between mice its P is too narrow, and
the warning says so.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from math import comb
from typing import Dict, List, Optional

import pandas as pd

# Plain names for the measures (anything not listed is shown with underscores replaced)
LABELS = {
    "ihc_per_100um": "inner hair cells per 100 um", "ihc_spacing_um": "inner hair cell spacing (um)",
    "outer_hair_cells_per_100um": "outer hair cells per 100 um",
    "OHC1_per_100um": "row 1 outer hair cells per 100 um", "OHC2_per_100um": "row 2 outer hair cells per 100 um",
    "OHC3_per_100um": "row 3 outer hair cells per 100 um", "OHC1_spacing_um": "row 1 spacing (um)",
    "OHC2_spacing_um": "row 2 spacing (um)", "OHC3_spacing_um": "row 3 spacing (um)",
    "row1_to_2_um": "row 1 to row 2 distance (um)", "row2_to_3_um": "row 2 to row 3 distance (um)",
    "calb2_edge_to_row1_um": "Calb2 band edge to row 1 (um)",
    "ihc_bundle_height_standing_um": "inner hair cell bundle height, standing bundles (um)",
    "ihc_bundle_length_um": "inner hair cell bundle length (um)", "ihc_bundle_length_standing_um": "inner hair cell bundle length, standing (um)",
    "ihc_bundle_lean_standing_deg": "inner hair cell bundle lean, standing (degrees)",
    "ihc_share_lying_flat": "share of inner hair cell bundles lying flat", "ihc_bundles_complete_share": "share of inner hair cell bundles complete",
    "ihc_bundle_brightness": "inner hair cell bundle brightness", "ohc1_bundle_height_um": "row 1 outer hair cell bundle height (um)",
    "ohc_bundle_span_um": "outer hair cell bundle span (um)", "ohc_bundle_bend_depth_um": "outer hair cell bundle bend depth (um)",
    "ohc_bundles_readable_share": "share of outer hair cell bundles readable",
    "pillar_head_width_um": "pillar head width (um)", "pillar_heads_per_100um": "pillar heads per 100 um", "pillar_width_sd_um": "pillar head width spread (um)",
    "ihc_nucleus_area_um2": "inner hair cell nucleus area (um2)", "ihc_nucleus_volume_um3": "inner hair cell nucleus volume (um3)",
    "ihc_nuclei_measured_share": "share of inner hair cells with a measured nucleus", "ihc_calb2_over_background": "inner hair cell Calb2 over background",
}

# Status levels, in the order they are listed
LEVELS = ["validated", "validated, more variable", "partly checked", "quality check", "not validated"]
LEVEL_COLORS = {"validated": "#2e7d32", "validated, more variable": "#558b2f", "partly checked": "#b26a00",
                "quality check": "#757575", "not validated": "#c62828"}

TABLES = {   # results files shown in the tab: label -> file name in the results folder
    "group comparisons": "group_comparisons.csv", "fields": "feature_table_fields.csv",
    "inner hair cells": "ihc_stereocilia_cells_v2.csv", "outer hair cells and rows": "hair_cell_rows_cells.csv",
    "outer hair cell bundles": "ohc_bundles_v2_cells.csv", "row 1 bundle heights": "ohc1_bundle_height_cells.csv",
    "inner hair cell nuclei": "ihc_nuclei_cells.csv", "validation cards": "validation_cards.csv",
    "imaging settings": "imaging_settings_by_field.csv",
}


def label(measure: str) -> str:
    return LABELS.get(measure, measure.replace("_", " "))


def status_level(status: Optional[str], validation: Optional[str] = None) -> str:
    """The status level of a measure: the 'status' column when the run wrote one, otherwise read from the start of its
    validation note (runs before 14 Nov 2026)."""
    if isinstance(status, str) and status in LEVELS:
        return status
    t = (validation or "").strip().lower() if isinstance(validation, str) else ""
    if "not checked" in t or "not validated" in t: return "not validated"
    if t.startswith(("validated, more variable", "usable, more variable")): return "validated, more variable"
    if t.startswith(("validated", "usable")): return "validated"
    if t.startswith(("not at target", "not validated", "not usable", "brightness ratio")): return "not validated"
    if t.startswith(("quality check", "descriptive")): return "quality check"
    if t.startswith(("middle and base", "partly")): return "partly checked"
    if t.startswith("as "): return "partly checked"
    return "not validated"


def smallest_p(n_mice: int, n_control: int, n_group: int) -> float:
    """Smallest P the mouse-level permutation test can give: one arrangement out of all ways of choosing the control
    and the compared group from the mice in the comparison."""
    if n_control < 1 or n_group < 1 or n_control + n_group > n_mice:
        return float("nan")
    return 1.0 / (comb(n_mice, n_control) * comb(n_mice - n_control, n_group))


@dataclass
class Comparison:
    measure: str
    group: str
    control_median: float
    group_median: float
    difference: float
    p: float
    p_adjusted: float
    p_model: float
    mice_control: int
    mice_group: int
    level: str
    note: str
    warnings: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return label(self.measure)


def _mouse_counts(ft: Optional[pd.DataFrame], measure: str, group: str, control: str):
    if ft is None or measure not in ft:
        return None
    d = ft[ft[measure].notna() & ft.group.notna()]
    if "damaged" in d: d = d[~d.damaged.astype(bool)]
    return d.mouse.nunique(), d[d.group == control].mouse.nunique(), d[d.group == group].mouse.nunique()


def comparisons(G: pd.DataFrame, ft: Optional[pd.DataFrame] = None, control: str = "control") -> List[Comparison]:
    """One entry per row of group_comparisons.csv, with the warnings that apply to it."""
    out = []
    for _, r in G.iterrows():
        lvl = status_level(r.get("status"), r.get("validation"))
        w = []
        if lvl == "not validated": w.append("not validated: " + str(r.get("validation", "")))
        elif lvl == "partly checked": w.append("partly checked: " + str(r.get("validation", "")))
        note = r.get("model_note")
        if isinstance(note, str) and note.strip(): w.append(note.strip())
        im = r.get("imaging_settings")
        if isinstance(im, str) and im.strip(): w.append(im.strip())
        n = _mouse_counts(ft, r["measure"], r["vs_control"], control); nc = ng = 0
        if n is not None:
            nm, nc, ng = n; sp = smallest_p(nm, nc, ng)
            if sp == sp and sp > 0.05:
                w.append(f"with {nc} control and {ng} {r['vs_control']} mice the smallest possible P is {sp:.2f}")
        if not (r.get("p_permutation") == r.get("p_permutation")):
            w.append("no mouse-level P: fewer than 2 mice in a group")
        out.append(Comparison(measure=r["measure"], group=r["vs_control"], control_median=r.get("control_median"),
                              group_median=r.get("group_median"), difference=r.get("mouse_level_difference"),
                              p=r.get("p_permutation"), p_adjusted=r.get("p_permutation_adjusted"), p_model=r.get("p_model"),
                              mice_control=nc, mice_group=ng, level=lvl, note=str(r.get("validation", "")), warnings=w))
    out.sort(key=lambda c: (LEVELS.index(c.level), c.p if c.p == c.p else 2.0))
    return out


@dataclass
class Results:
    folder: str
    tables: Dict[str, str]
    control: str = "control"
    run_info: dict = field(default_factory=dict)

    def read(self, name: str) -> Optional[pd.DataFrame]:
        p = self.tables.get(name)
        return pd.read_csv(p) if p and os.path.exists(p) else None

    def comparisons(self) -> List[Comparison]:
        G = self.read("group comparisons")
        if G is None or G.empty: return []
        return comparisons(G, self.read("fields"), self.control)

    def missing(self) -> List[str]:
        return [k for k, p in self.tables.items() if not os.path.exists(p)]


def open_results(folder: str) -> Results:
    """The results of the run in folder (a run's RESULTS folder)."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no results folder at {folder}")
    tables = {k: os.path.join(folder, f) for k, f in TABLES.items()}
    control, info = "control", {}
    ds = os.path.join(folder, "dataset_settings.json")
    if os.path.exists(ds):
        with open(ds, encoding="utf-8") as fh:
            control = json.load(fh).get("control_group") or control
    runs = sorted(f for f in os.listdir(folder) if f.startswith("run_info_") and f.endswith(".json"))
    if runs:
        with open(os.path.join(folder, runs[-1]), encoding="utf-8") as fh:
            info = json.load(fh)
        control = (info.get("settings", {}) or {}).get("CONTROL_GROUP") or control
    return Results(folder=folder, tables=tables, control=control, run_info=info)
