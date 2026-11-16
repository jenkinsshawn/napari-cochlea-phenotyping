"""Reading confocal images and their metadata: pixel size, plane spacing, channels and imaging settings.

Everything downstream uses the sizes read here (in micrometers), never a fixed pixel size, so images from another
microscope or objective are measured at their own scale. Supported now: Nikon .nd2 (needs the nd2 package) and
.tif/.tiff with ImageJ or OME metadata. Files with several time points or positions are refused with a clear message
rather than read partly.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ChannelInfo:
    index: int
    name: str                                  # filter or channel name in the file (e.g. "Cy5", "DAPI")
    excitation_nm: Optional[float] = None
    emission_nm: Optional[float] = None
    laser_nm: str = ""                         # laser lines used, ";"-separated (nd2 description)
    laser_power_pct: str = ""
    exposure_ms: Optional[float] = None


@dataclass
class ImageInfo:
    path: str
    field: str                                 # file name without extension
    source: str                                # "nd2" or "tif"
    shape_zcyx: tuple
    dtype: str
    bit_depth: Optional[int]                   # significant bits (12 for a 12-bit camera stored as uint16)
    voxel_um: tuple                            # (z, y, x) in micrometers; None entries when the file does not say
    channels: List[ChannelInfo] = field(default_factory=list)
    objective: str = ""
    pinhole_um: Optional[float] = None
    date: str = ""
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self); d["channels"] = [asdict(c) for c in self.channels]; return d

    @property
    def saturation_value(self) -> Optional[int]:
        if self.bit_depth:
            return 2 ** int(self.bit_depth) - 1
        return int(np.iinfo(self.dtype).max) if np.issubdtype(np.dtype(self.dtype), np.integer) else None


def _nd2_settings(text: str, n: int):
    """Laser lines, powers and exposure per channel from the nd2 description text ("Plane #1: ... ExW: 640; Power: 22.0
    ... Exposure: 200 ms"), as in imaging_settings.py."""
    blocks = re.split(r"Plane #\d+:", text)[1:]; out = []
    for ci in range(n):
        b = blocks[ci] if ci < len(blocks) else ""
        ex = re.search(r"Exposure:\s*([\d.]+)\s*ms", b); lz = re.findall(r"ExW:\s*(\d+);\s*Power:\s*([\d.\-]+)", b)
        out.append(dict(laser_nm=";".join(a for a, _ in lz), laser_power_pct=";".join(p for _, p in lz),
                        exposure_ms=float(ex.group(1)) if ex else None))
    return out


def _nd2():
    try:
        import nd2
    except ImportError as e:
        raise ImportError("reading .nd2 files needs the nd2 package: pip install nd2  "
                          "(or pip install napari-cochlea-phenotyping[nd2])") from e
    return nd2


def field_name(path: str) -> str:
    """File name without its image extension (".ome.tif" counts as one extension)."""
    b = os.path.basename(path)
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".nd2"):
        if b.lower().endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def _read_nd2_info(path: str) -> ImageInfo:
    nd2 = _nd2()
    with nd2.ND2File(path) as f:
        sizes = dict(f.sizes); warn = []
        extra = {k: v for k, v in sizes.items() if k not in ("Z", "C", "Y", "X") and v > 1}
        if extra:
            raise ValueError(f"{os.path.basename(path)} has {extra}; split time points or positions into separate files first")
        vx = f.voxel_size()
        chans = f.metadata.channels or []
        text = str((f.text_info or {}).get("description", "")); date = str((f.text_info or {}).get("date", ""))
        sets = _nd2_settings(text, len(chans))
        cinfo = []
        for i, c in enumerate(chans):
            ch = c.channel
            cinfo.append(ChannelInfo(i, str(ch.name), getattr(ch, "excitationLambdaNm", None),
                                     getattr(ch, "emissionLambdaNm", None), **sets[i]))
        m = chans[0].microscope if chans else None
        bits = getattr(f.attributes, "bitsPerComponentSignificant", None)
        shape = (sizes.get("Z", 1), sizes.get("C", 1), sizes["Y"], sizes["X"])
        if sizes.get("Z", 1) == 1:
            warn.append("single plane: bundle height and surface depth cannot be measured")
        return ImageInfo(path, field_name(path), "nd2", shape, str(f.dtype), bits,
                         (float(vx.z), float(vx.y), float(vx.x)), cinfo,
                         str(getattr(m, "objectiveName", "") or ""), getattr(m, "pinholeDiameterUm", None),
                         date.split()[0] if date else "", warn)


def _read_tif_info(path: str) -> ImageInfo:
    import tifffile
    warn = []
    with tifffile.TiffFile(path) as t:
        s = t.series[0]; axes = s.axes.upper(); shp = dict(zip(axes, s.shape)); dtype = str(s.dtype)
        extra = {k: v for k, v in shp.items() if k not in ("Z", "C", "Y", "X", "S", "Q", "I") and v > 1}
        if extra:
            raise ValueError(f"{os.path.basename(path)} has axes {extra}; save one stack per file")
        z = shp.get("Z", shp.get("Q", shp.get("I", 1))); c = shp.get("C", shp.get("S", 1))
        dz = dy = dx = None; names = []
        if t.is_ome and t.ome_metadata:
            om = t.ome_metadata
            g = lambda k: (float(re.search(fr'{k}="([\d.eE+-]+)"', om).group(1)) if re.search(fr'{k}="([\d.eE+-]+)"', om) else None)
            dx, dy, dz = g("PhysicalSizeX"), g("PhysicalSizeY"), g("PhysicalSizeZ")
            names = re.findall(r'<Channel[^>]*?Name="([^"]*)"', om)
        elif t.is_imagej:
            ij = t.imagej_metadata or {}
            p = t.pages[0].tags
            if "XResolution" in p:
                num, den = p["XResolution"].value; dx = den / num if num else None
            if "YResolution" in p:
                num, den = p["YResolution"].value; dy = den / num if num else None
            dz = ij.get("spacing"); unit = str(ij.get("unit", "")).strip().lower().replace("\\u00b5", "\u00b5")
            if unit not in ("micron", "microns", "um", "\u00b5m"):
                warn.append(f"pixel size unit in the file is {ij.get('unit')!r}; please confirm the scale")
        if None in (dx, dy, dz):
            warn.append("the file does not give the full voxel size; enter it in the Standardize tab")
    cinfo = [ChannelInfo(i, names[i] if i < len(names) else f"channel {i + 1}") for i in range(c)]
    return ImageInfo(path, field_name(path), "tif", (z, c, shp["Y"], shp["X"]), dtype, None,
                     (dz, dy, dx), cinfo, warnings=warn)


def read_info(path: str) -> ImageInfo:
    """Metadata only; no pixel data is read."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nd2":
        return _read_nd2_info(path)
    if ext in (".tif", ".tiff"):
        return _read_tif_info(path)
    raise ValueError(f"{ext} files are not supported yet (supported: .nd2, .tif, .tiff)")


def read_stack(path: str) -> np.ndarray:
    """Pixel data as (z, c, y, x)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".nd2":
        nd2 = _nd2()
        with nd2.ND2File(path) as f:
            arr = f.asarray(); axes = [k for k, v in f.sizes.items()]
        order = [a for a in axes]
        for a in ("Z", "C"):
            if a not in order:
                arr = arr[None] if a == "Z" else np.expand_dims(arr, 1 if "Z" in order else 0); order.insert(0 if a == "Z" else 1, a)
        return np.moveaxis(arr, [order.index(a) for a in "ZCYX"], range(4))
    import tifffile
    with tifffile.TiffFile(path) as t:
        s = t.series[0]; arr = s.asarray(); axes = list(s.axes.upper())
    axes = ["Z" if a in "QI" else "C" if a == "S" else a for a in axes]
    for a in ("Z", "C"):
        if a not in axes:
            arr = arr[None]; axes.insert(0, a)
    return np.moveaxis(arr, [axes.index(a) for a in "ZCYX"], range(4))
