"""Standardize: confirm the scale, the stain in each channel and the bundle direction, measure brightness, and flag
problems before any measurement runs.

The stain guess is only a starting point. It combines the filter name with the stains named in the file name (the
rule the Webber lab files follow) and falls back to the emission wavelength. The user confirms or edits it in the
panel, and the confirmed table is saved in the dataset settings file with the scale and bundle direction, so every
later step reads the same choices.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ._io import ImageInfo

SETTINGS_VERSION = 2   # 2: stains per stain panel, mouse and group of each image
NAME_KEYS = [("Ocm", r"ocm|oncm"), ("Calb2", r"calb2"), ("PMCA2", r"pmca2"), ("Phalloidin", r"phall"),
             ("Prestin", r"prestin"), ("CtBP2", r"ctbp2"), ("Myo7a", r"myo7a|myo7"), ("DAPI", r"dapi|hoechst")]


def stains_in_name(name: str) -> List[str]:
    low = name.lower()
    return [s for s, k in NAME_KEYS if re.search(k, low)]


@dataclass
class StainGuess:
    channel: int
    channel_name: str
    stain: str            # stain name, or "to check"
    reason: str


def guess_stains(info: ImageInfo) -> List[StainGuess]:
    """Webber lab rule (identical to make_field_pictures.channel_map for those files): DAPI filter = DAPI; Cy5 =
    phalloidin; TRITC = Calb2 when the name lists Calb2 and phalloidin; EGFP/FITC = Ocm, else PMCA2, as named.
    Unknown filter names fall back to the emission wavelength, and are always marked to check."""
    named = stains_in_name(info.field); out = []
    for c in info.channels:
        f = c.name.upper(); s, why = "to check", f"filter {c.name}"
        if f == "DAPI":
            s, why = "DAPI", "DAPI filter"
        elif f == "CY5":
            if "Phalloidin" in named: s, why = "Phalloidin", "Cy5 filter, phalloidin in the file name"
        elif f == "TRITC":
            if "Calb2" in named and "Phalloidin" in named: s, why = "Calb2", "TRITC filter, Calb2 in the file name"
        elif f in ("EGFP", "FITC"):
            if "Ocm" in named: s, why = "Ocm", f"{c.name} filter, Ocm in the file name"
            elif "PMCA2" in named: s, why = "PMCA2", f"{c.name} filter, PMCA2 in the file name"
        else:
            em = c.emission_nm
            if em:
                band = "blue" if em < 490 else "green" if em < 560 else "red" if em < 640 else "far red"
                why = f"unknown filter {c.name!r}, emission {em:.0f} nm ({band}); please choose the stain"
            else:
                why = f"unknown filter {c.name!r} and no emission wavelength; please choose the stain"
        if s == "to check" and not why.startswith("unknown"):
            why += f"; the file name lists {', '.join(named) or 'no stains'}"
        out.append(StainGuess(c.index, c.name, s, why))
    return out


def brightness(stack: np.ndarray, saturation_value: Optional[int] = None) -> Dict[str, float]:
    """One channel (z, y, x). Background = 1st percentile of non-zero voxels (zero is fill outside the scanned area),
    high = 99.9th percentile, contrast = (high - background) / (median - background). Saturated share = share of
    voxels at the camera's maximum value."""
    v = np.asarray(stack).ravel(); real = v[v > 0] if (v > 0).any() else v
    p1, med, p999 = (float(x) for x in np.percentile(real, [1, 50, 99.9]))
    out = dict(background=p1, median=med, high=p999, contrast=(p999 - p1) / max(med - p1, 1e-9),
               zero_share=float(np.mean(v == 0)))
    out["saturated_share"] = float(np.mean(v >= saturation_value)) if saturation_value else float("nan")
    return out


def scale_to_background(x, background, high, floor=0.0):
    """Brightness with background removed and the high level set to 1: floor + (1 - floor)(x - bg)/(high - bg).
    Zero fill stays zero. floor = 0.042 reproduces the inner hair cell step's scaling (pipeline_settings
    IHC_BACKGROUND_RATIO)."""
    x = np.asarray(x, float); y = floor + (1 - floor) * (x - background) / max(high - background, 1e-9)
    return np.where(x == 0, 0.0, y)


def scale_real_voxels(v, floor: float = 0.0, low_pct: float = 1.0, high_pct: float = 99.9):
    """Brightness scaled with both levels taken over real voxels (v > 0): the low_pct percentile maps to floor and
    the high_pct percentile to 1; zero fill (outside the image) stays 0. Exposure, gain, background level and the
    share of fill all cancel. Used for inner hair cell cut-outs with floor = 0.042 (pipeline_settings
    IHC_BACKGROUND_RATIO); tested against the hand clicks 12 Nov 2026."""
    v = np.asarray(v); real = v > 0
    if not real.any():
        return np.zeros(v.shape)
    lo = np.percentile(v[real], low_pct); hi = np.percentile(v[real], high_pct)
    return np.where(real, floor + (1 - floor) * (v - lo) / max(hi - lo, 1e-6), 0.0)


QC_LIMITS = dict(saturated_share=0.001, zero_share=0.05, min_contrast=2.0, scale_tolerance=0.02)


def qc_flags(info: ImageInfo, stains: List[str], stats: Dict[str, Dict[str, float]],
             expected_voxel_um: Optional[tuple] = None) -> List[str]:
    """Plain-language warnings for one field. They do not exclude the field; they are shown in the panel and report."""
    flags = list(info.warnings)
    if "Phalloidin" not in stains:
        flags.append("no phalloidin channel: surface, cells and bundles cannot be measured")
    if len(set(s for s in stains if s != "to check")) < len([s for s in stains if s != "to check"]):
        flags.append("two channels have the same stain")
    if any(s == "to check" for s in stains):
        flags.append("a channel's stain is not confirmed")
    if expected_voxel_um and None not in info.voxel_um:
        for name, a, b in zip(("plane spacing", "pixel height", "pixel width"), info.voxel_um, expected_voxel_um):
            if abs(a - b) > QC_LIMITS["scale_tolerance"] * b:
                flags.append(f"{name} {a:.4g} um differs from the dataset's {b:.4g} um")
    for s, st in stats.items():
        if st.get("saturated_share", 0) > QC_LIMITS["saturated_share"]:
            flags.append(f"{s}: {100 * st['saturated_share']:.2f}% of voxels saturated")
        if st.get("zero_share", 0) > QC_LIMITS["zero_share"]:
            flags.append(f"{s}: {100 * st['zero_share']:.1f}% of voxels are empty fill (field edge or stitching)")
        if s == "Phalloidin" and st.get("contrast", 99) < QC_LIMITS["min_contrast"]:
            flags.append(f"phalloidin contrast {st['contrast']:.1f} is low (bright layer barely above the rest)")
    return flags


@dataclass
class DatasetSettings:
    """What the user confirmed for a set of images. Saved as JSON next to the results."""
    voxel_um: tuple                                   # (z, y, x)
    stains_by_channel_name: Dict[str, str]            # e.g. {"Cy5": "Phalloidin", "DAPI": "DAPI"}
    bundle_side: int = -1                             # -1: bundles toward lower plane numbers; +1: higher
    confirmed_by: str = ""
    confirmed_on: str = ""
    plugin_version: str = ""
    device: dict = field(default_factory=dict)
    notes: str = ""
    stains_by_panel: Dict[str, Dict[str, str]] = field(default_factory=dict)   # panel -> channel name -> stain
    panel_of_field: Dict[str, str] = field(default_factory=dict)
    mouse_of_field: Dict[str, str] = field(default_factory=dict)
    group_of_mouse: Dict[str, str] = field(default_factory=dict)
    control_group: str = ""
    excluded_fields: Dict[str, str] = field(default_factory=dict)   # field -> reason (measured, left out of group comparisons)
    settings_version: int = SETTINGS_VERSION

    def validate(self):
        if len(self.voxel_um) != 3 or any((v is None or not (0 < float(v) < 100)) for v in self.voxel_um):
            raise ValueError(f"voxel size must be three positive sizes in um, got {self.voxel_um}")
        if self.bundle_side not in (-1, 1):
            raise ValueError("bundle_side must be -1 or 1")
        all_stains = set(self.stains_by_channel_name.values()) | {s for m in self.stains_by_panel.values() for s in m.values()}
        if "Phalloidin" not in all_stains:
            raise ValueError("no channel is set to Phalloidin")
        if self.control_group and self.control_group not in set(self.group_of_mouse.values()):
            raise ValueError(f"control group {self.control_group!r} has no mice")
        unknown = [f for f in self.excluded_fields if self.mouse_of_field and f not in self.mouse_of_field]
        if unknown:
            raise ValueError(f"left-out images not in the dataset: {', '.join(unknown)}")

    def stains_for(self, field: str, channel_names: List[str]) -> List[str]:
        """Stain of each channel of one image: its panel's map, else the map for all images."""
        m = self.stains_by_panel.get(self.panel_of_field.get(field, ""), self.stains_by_channel_name)
        return [m.get(c, "to check") for c in channel_names]

    def group_of(self, field: str) -> str:
        return self.group_of_mouse.get(self.mouse_of_field.get(field, ""), "")

    def save(self, path: str):
        self.validate()
        if not self.confirmed_on:
            self.confirmed_on = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        if not self.plugin_version:
            from . import __version__
            self.plugin_version = __version__
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "DatasetSettings":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("settings_version", 0) > SETTINGS_VERSION:
            raise ValueError("settings file was written by a newer version of the plugin; please update")
        d["voxel_um"] = tuple(d["voxel_um"]); s = cls(**d); s.validate(); return s
