"""Cells tab. Now: the cell surface of the selected image, shown as layers, with a warning when the actin layer is
clear in too few places. Next: hair cell outlines and rows (being moved in from the scripts)."""
from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ._surface import MIN_CLEAR_SHARE, clear_share, find_surface, layer_projection

OK, WARN = "#2e7d32", "#b26a00"


class CellsTab(QWidget):
    def __init__(self, viewer=None, session_getter: Callable = lambda: None, selected_getter: Callable = lambda: None,
                 background: bool = True):
        super().__init__()
        self.viewer, self.session_getter, self.selected_getter, self.background = viewer, session_getter, selected_getter, background
        lay = QVBoxLayout(self)
        self.surface_btn = QPushButton("Find surface"); self.surface_btn.clicked.connect(lambda: self.run_surface())
        self.status = QLabel("Select an image in the Standardize tab."); self.status.setWordWrap(True)
        lay.addWidget(self.surface_btn); lay.addWidget(self.status)
        lay.addWidget(QLabel("<span style='color:gray'>Hair cell outlines and rows: next version.</span>")); lay.addStretch(1)
        self.result = None

    def _compute(self, field):
        from ._io import read_stack
        s = self.session_getter(); info = s.infos[field]; st = s.stains_of(field)
        if "Phalloidin" not in st:
            raise ValueError(f"{field} has no phalloidin channel")
        arr = read_stack(info.path); ph = arr[:, st.index("Phalloidin")]; dz = info.voxel_um[0]
        sf = find_surface(ph, info.voxel_um)
        return dict(field=field, info=info, surface=sf, layer=layer_projection(ph, sf.planes, 2.0, dz), share=clear_share(sf))

    def _show(self, r):
        self.result = r; f, info, sf = r["field"], r["info"], r["surface"]; dz, dy, dx = info.voxel_um
        if self.viewer is not None:
            self.viewer.add_image(r["layer"], name=f"{f} actin layer near surface", scale=(dy, dx), colormap="gray")
            self.viewer.add_image(sf.depth_um(dz), name=f"{f} surface depth (um)", scale=(dy, dx), colormap="viridis", visible=False)
        msg = f"{f}: clear actin layer in {100 * r['share']:.0f}% of the field"
        if r["share"] < MIN_CLEAR_SHARE:
            self.status.setText(f"<span style='color:{WARN}'>{msg}; the tissue may be torn or partly outside the stack</span>")
        else:
            self.status.setText(f"<span style='color:{OK}'>{msg}</span>")

    def run_surface(self, field=None):
        field = field or self.selected_getter()
        if not field:
            self.status.setText(f"<span style='color:{WARN}'>Select an image in the Standardize tab first</span>"); return None
        self.status.setText(f"Finding the surface of {field}...")
        if self.background:
            try:
                from napari.qt.threading import create_worker
                w = create_worker(self._compute, field); w.returned.connect(self._show)
                w.errored.connect(lambda e: self.status.setText(f"<span style='color:{WARN}'>{e}</span>")); w.start(); return None
            except ImportError:
                pass
        try:
            r = self._compute(field)
        except Exception as e:
            self.status.setText(f"<span style='color:{WARN}'>{e}</span>"); return None
        self._show(r); return r
