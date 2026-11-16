"""The Cochlea Phenotyping panel: one tab per step.

The device is found automatically and shown at the top (set COCHLEA_PHENOTYPING_DEVICE=cpu to force the processor).
Each step's output is an editable napari layer that the next step reads.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from . import __version__
from ._device import Device, detect_device

STEPS = [
    ("1 Standardize", None),
    ("2 Cells", "Surface, hair cell outlines and rows, as editable layers."),
    ("3 Measure", "Every image measured with the validated pipeline: cells, rows, bundles, nuclei, pillar heads."),
    ("4 Results", "Tables per cell, field and mouse, validation cards, group comparisons and warnings."),
]


def _scroll(w):
    a = QScrollArea(); a.setWidgetResizable(True); a.setWidget(w); return a


class CochleaPhenotyping(QWidget):
    device_changed = Signal(object)

    def __init__(self, napari_viewer=None, detect_in_background: bool = True):
        super().__init__()
        self.viewer = napari_viewer; self.device: Device | None = None
        lay = QVBoxLayout(self); lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(QLabel(f"<b>Cochlea Phenotyping</b>  <span style='color:gray'>v{__version__}</span>"))
        self.device_label = QLabel("Looking for a graphics card..."); self.device_label.setWordWrap(True)
        self.device_label.setTextFormat(Qt.RichText); lay.addWidget(self.device_label)
        self.tabs = QTabWidget()
        from ._standardize_tab import StandardizeTab
        self.standardize = StandardizeTab(napari_viewer, device_getter=lambda: self.device)
        self.tabs.addTab(_scroll(self.standardize), STEPS[0][0])
        from ._cells_tab import CellsTab
        self.cells = CellsTab(napari_viewer, session_getter=lambda: self.standardize.session,
                              selected_getter=self.standardize.selected_field, background=detect_in_background)
        self.tabs.addTab(_scroll(self.cells), STEPS[1][0])
        from ._run_tab import RunTab
        self.run = RunTab(session_getter=lambda: self.standardize.session, results_getter=lambda: self.standardize.results_folder,
                          excluded_getter=lambda: self.standardize.session.excluded, device_getter=lambda: self.device)
        self.tabs.addTab(_scroll(self.run), STEPS[2][0])
        from ._results_tab import ResultsTab
        self.results = ResultsTab(napari_viewer, session_getter=lambda: self.standardize.session,
                                  selected_getter=self.standardize.selected_field, results_getter=lambda: self.standardize.results_folder)
        self.tabs.addTab(_scroll(self.results), STEPS[3][0])
        self.run.finished.connect(self._run_finished)
        lay.addWidget(self.tabs, 1)
        if detect_in_background:
            try:
                from napari.qt.threading import create_worker
                w = create_worker(detect_device); w.returned.connect(self._set_device); w.start()
            except Exception:
                self._set_device(detect_device())
        else:
            self._set_device(detect_device())

    def _run_finished(self, folder: str):
        """A finished run opens its results in the Results tab."""
        if self.results.load(folder):
            self.tabs.setCurrentIndex(3)

    def _set_device(self, d: Device):
        self.device = d; color = "#2e7d32" if d.gpu else "#8a6d00"
        tip = d.note
        self.device_label.setText(f"<span style='color:{color}'>{d.label()}</span>"); self.device_label.setToolTip(tip)
        self.device_changed.emit(d)
