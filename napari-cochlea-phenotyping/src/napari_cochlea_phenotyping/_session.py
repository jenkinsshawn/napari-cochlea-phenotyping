"""State of the Standardize step, kept apart from the panel so it can be tested without a window.

A session holds the images of one dataset, the voxel size and stain table the user confirms, the bundle direction,
and the brightness and QC results. save() refuses until the stains and the bundle direction are confirmed.
"""
from __future__ import annotations

import glob
import os
from collections import Counter
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ._io import ImageInfo, read_info, read_stack
from ._standardize import DatasetSettings, brightness, guess_stains, qc_flags, stains_in_name

SUPPORTED = (".nd2", ".tif", ".tiff")
KNOWN_STAINS = ["Phalloidin", "Calb2", "PMCA2", "Ocm", "DAPI", "Myo7a", "Prestin", "CtBP2", "other", "to check"]


class StandardizeSession:
    """Stains are kept per stain panel: the images whose file names list the same stains and whose channels have the
    same filter names. One filter can carry different stains in different panels (EGFP is Ocm in some Webber lab
    images and PMCA2 in others), so each panel's channels are confirmed separately."""
    def __init__(self):
        self.infos: Dict[str, ImageInfo] = {}          # field -> metadata
        self.errors: Dict[str, str] = {}               # path -> why it could not be read
        self.voxel_um: Optional[tuple] = None
        self.panel_of: Dict[str, str] = {}             # field -> stain panel label
        self.stains: Dict[Tuple[str, str], str] = {}   # (panel, channel name) -> stain
        self.stain_reasons: Dict[Tuple[str, str], str] = {}
        self.stains_confirmed = False
        self.bundle_side: int = -1
        self.bundle_side_confirmed = False
        self.mouse_of: Dict[str, str] = {}             # field -> mouse ID (guessed from the file name, editable)
        self.group_of_mouse: Dict[str, str] = {}       # mouse ID -> group ("" = not compared)
        self.control_group: str = ""
        self.qc: Dict[str, dict] = {}                  # field -> brightness per stain + flags
        self.excluded: Dict[str, str] = {}             # field -> reason it is left out of group comparisons (folded, torn...)

    # images -------------------------------------------------------------------------------------------------
    def add_paths(self, paths: List[str]) -> List[str]:
        """Add image files or folders (searched one level deep). Returns the fields added."""
        files = []
        for p in paths:
            if os.path.isdir(p):
                files += [f for f in sorted(glob.glob(os.path.join(p, "*"))) if f.lower().endswith(SUPPORTED)]
            elif p.lower().endswith(SUPPORTED):
                files.append(p)
        added = []
        for f in files:
            try:
                info = read_info(f)
            except Exception as e:              # report every unreadable file instead of stopping
                self.errors[f] = f"{type(e).__name__}: {e}"; continue
            if info.field in self.infos and self.infos[info.field].path != f:
                self.errors[f] = f"another file already uses the name {info.field!r}"; continue
            self.infos[info.field] = info; added.append(info.field)
        self._refresh_defaults()
        return added

    def _refresh_defaults(self):
        vox = [i.voxel_um for i in self.infos.values() if None not in i.voxel_um]
        if vox and self.voxel_um is None:
            self.voxel_um = Counter(tuple(round(v, 4) for v in x) for x in vox).most_common(1)[0][0]
        keys = {f: (tuple(sorted(stains_in_name(f))), tuple(c.name for c in i.channels)) for f, i in self.infos.items()}
        named = Counter(k[0] for k in set(keys.values()))
        def label(k):
            base = ", ".join(k[0]) if k[0] else "no stains in the file name"
            return base if named[k[0]] == 1 else f"{base} ({', '.join(k[1])})"
        counts = Counter(label(k) for k in keys.values())
        for f, i in self.infos.items():
            p = label(keys[f]); self.panel_of[f] = p
            for g in guess_stains(i):
                key = (p, g.channel_name)
                if key not in self.stains or not self.stains_confirmed:
                    self.stains[key] = g.stain
                    self.stain_reasons[key] = g.reason + (f" ({counts[p]} images)" if counts[p] > 1 else "")
            if f not in self.mouse_of:
                self.mouse_of[f] = guess_mouse(f)
            self.group_of_mouse.setdefault(self.mouse_of[f], "")

    def panels(self) -> Dict[str, List[str]]:
        """Stain panel -> its fields."""
        out: Dict[str, List[str]] = {}
        for f, p in sorted(self.panel_of.items()):
            out.setdefault(p, []).append(f)
        return out

    def fields_off_scale(self, tol: float = 0.02) -> List[str]:
        if not self.voxel_um:
            return list(self.infos)
        return [f for f, i in self.infos.items() if None in i.voxel_um or
                any(abs(a - b) > tol * b for a, b in zip(i.voxel_um, self.voxel_um))]

    # user choices -------------------------------------------------------------------------------------------
    def set_stain(self, channel_name: str, stain: str, panel: Optional[str] = None):
        """Set the stain of a channel in one panel, or in every panel that has the channel when panel is None."""
        keys = [k for k in self.stains if k[1] == channel_name and (panel is None or k[0] == panel)]
        if not keys:
            raise KeyError(f"no channel {channel_name!r}" + (f" in panel {panel!r}" if panel else ""))
        for k in keys:
            self.stains[k] = stain
        self.stains_confirmed = False

    def panels_without_phalloidin(self) -> List[str]:
        return [p for p in self.panels() if "Phalloidin" not in {s for (q, _), s in self.stains.items() if q == p}]

    def confirm_stains(self):
        if not any(s == "Phalloidin" for s in self.stains.values()):
            raise ValueError("set the phalloidin channel before confirming")
        if any(s == "to check" for s in self.stains.values()):
            raise ValueError("every channel needs a stain (choose 'other' for a channel you will not use)")
        self.stains_confirmed = True

    def set_mouse(self, field: str, mouse: str):
        self.mouse_of[field] = mouse.strip(); self.group_of_mouse.setdefault(mouse.strip(), "")

    def set_group(self, mouse: str, group: str):
        self.group_of_mouse[mouse] = group.strip()

    def groups(self) -> List[str]:
        return sorted({g for g in self.group_of_mouse.values() if g})

    def exclude(self, field: str, reason: str):
        """Measure the image but leave it out of group comparisons (damaged tissue)."""
        if field not in self.infos:
            raise KeyError(field)
        self.excluded[field] = reason.strip() or "left out"

    def include(self, field: str):
        self.excluded.pop(field, None)

    def set_bundle_side(self, side: int):
        if side not in (-1, 1):
            raise ValueError("bundle side must be -1 or 1")
        self.bundle_side = side; self.bundle_side_confirmed = True

    def stains_of(self, field: str) -> List[str]:
        p = self.panel_of.get(field, "")
        return [self.stains.get((p, c.name), "to check") for c in self.infos[field].channels]

    # brightness and QC --------------------------------------------------------------------------------------
    def check_field(self, field: str, reader: Callable = read_stack) -> dict:
        info = self.infos[field]; arr = reader(info.path); st = self.stains_of(field)
        stats = {s: brightness(arr[:, i], info.saturation_value) for i, s in enumerate(st) if s not in ("other",)}
        flags = qc_flags(info, st, stats, expected_voxel_um=self.voxel_um)
        self.qc[field] = dict(stats=stats, flags=flags); return self.qc[field]

    def qc_rows(self) -> List[dict]:
        rows = []
        for f, i in sorted(self.infos.items()):
            q = self.qc.get(f); ph = (q or {}).get("stats", {}).get("Phalloidin", {})
            rows.append(dict(field=f, planes=i.shape_zcyx[0], pixel_um=i.voxel_um[2], plane_um=i.voxel_um[0],
                             stains=";".join(self.stains_of(f)), checked=q is not None,
                             phalloidin_contrast=ph.get("contrast", np.nan),
                             most_saturated_pct=max([100 * s.get("saturated_share", 0) for s in (q or {}).get("stats", {}).values()] or [np.nan]),
                             flags="; ".join((q or {}).get("flags", i.warnings))))
        return rows

    # saving -------------------------------------------------------------------------------------------------
    def settings(self, confirmed_by: str = "", device: Optional[dict] = None) -> DatasetSettings:
        missing = []
        if not self.stains_confirmed: missing.append("confirm the stain table")
        if not self.bundle_side_confirmed: missing.append("confirm the bundle direction on the side view")
        if not self.voxel_um: missing.append("set the voxel size")
        if self.control_group and self.control_group not in self.groups():
            missing.append(f"the control group {self.control_group!r} has no mice")
        if missing:
            raise ValueError("before saving: " + "; ".join(missing))
        by_panel: Dict[str, Dict[str, str]] = {}
        for (p, ch), s in self.stains.items():
            by_panel.setdefault(p, {})[ch] = s
        flat = {}
        for p, m in by_panel.items():                       # one map for all images when the panels agree
            for ch, s in m.items():
                flat[ch] = s if flat.get(ch, s) == s else None
        flat = {ch: s for ch, s in flat.items() if s is not None}
        return DatasetSettings(tuple(self.voxel_um), flat, self.bundle_side, confirmed_by=confirmed_by,
                               device=device or {}, stains_by_panel=by_panel, panel_of_field=dict(self.panel_of),
                               mouse_of_field=dict(self.mouse_of), group_of_mouse=dict(self.group_of_mouse),
                               control_group=self.control_group, excluded_fields=dict(self.excluded))

    def save(self, folder: str, confirmed_by: str = "", device: Optional[dict] = None) -> List[str]:
        import pandas as pd
        os.makedirs(folder, exist_ok=True)
        s = self.settings(confirmed_by, device); p1 = os.path.join(folder, "dataset_settings.json"); s.save(p1)
        p2 = os.path.join(folder, "standardize_qc.csv"); pd.DataFrame(self.qc_rows()).to_csv(p2, index=False)
        return [p1, p2]


def guess_mouse(field: str) -> str:
    """Mouse ID at the start of a file name: letters and digits, an optional -number and an optional (number), e.g.
    'AR1383-2', 'JW5436 (2)'; otherwise the first word."""
    m = re.match(r"^\s*([A-Za-z]+\d+(?:-\d+)?(?:\s*\(\d+\))?)", field)
    return m.group(1).strip() if m else (field.split() or [field])[0]


def side_view(stack_zyx: np.ndarray) -> np.ndarray:
    """Side view for checking the bundle direction: the brightest voxel along y for each plane and x, (z, x)."""
    return np.asarray(stack_zyx).max(axis=1)
