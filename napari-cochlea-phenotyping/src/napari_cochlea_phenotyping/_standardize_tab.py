"""Standardize tab: the few choices a person has to confirm once per dataset.

Shown: the images, the stain in each channel of each stain panel, the group of each mouse, and which way the bundles
point. The scale is read from the files and
only asks for input when a file does not give it or the images disagree. Brightness standardization and quality
checks always run during the measurement steps and are not settings; their warnings appear with the results.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                            QHBoxLayout, QHeaderView, QLabel, QListWidget, QPushButton, QRadioButton, QTableWidget,
                            QTableWidgetItem, QVBoxLayout, QWidget)

from ._session import KNOWN_STAINS, StandardizeSession, side_view

OK, WARN = "#2e7d32", "#b26a00"


class StandardizeTab(QWidget):
    def __init__(self, viewer=None, device_getter: Callable = lambda: None):
        super().__init__()
        self.viewer, self.device_getter = viewer, device_getter
        self.session = StandardizeSession(); self.results_folder = ""
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        self.add_files_btn = QPushButton("Add images..."); self.add_folder_btn = QPushButton("Add folder...")
        self.add_files_btn.clicked.connect(self._pick_files); self.add_folder_btn.clicked.connect(self._pick_folder)
        row.addWidget(self.add_files_btn); row.addWidget(self.add_folder_btn); lay.addLayout(row)
        self.summary = QLabel("No images yet."); self.summary.setWordWrap(True); lay.addWidget(self.summary)
        self.images = QListWidget(); self.images.setMaximumHeight(140); lay.addWidget(self.images)
        row = QHBoxLayout(); self.reason_cb = QComboBox(); self.reason_cb.setEditable(True); self.reason_cb.addItems(["folded", "torn", "other damage"])
        self.exclude_btn = QPushButton("Leave out of comparisons"); self.include_btn = QPushButton("Include again")
        self.exclude_btn.setToolTip("The selected image is still measured, but not used in the group comparisons (for folded or torn tissue)")
        self.exclude_btn.clicked.connect(lambda: self.exclude_selected()); self.include_btn.clicked.connect(lambda: self.include_selected())
        row.addWidget(self.reason_cb, 1); row.addWidget(self.exclude_btn); row.addWidget(self.include_btn); lay.addLayout(row)

        g = QGroupBox("Confirm once for this dataset"); gl = QVBoxLayout(g)
        self.stain_table = QTableWidget(0, 3); self.stain_table.setHorizontalHeaderLabels(["Images", "Channel", "Stain"])
        self.stain_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch); self.stain_table.verticalHeader().hide()
        gl.addWidget(QLabel("Stain in each channel, for each set of images with the same stains in the file name\n"
                            "(guessed from the filter and file name):")); gl.addWidget(self.stain_table)

        self.mouse_table = QTableWidget(0, 3); self.mouse_table.setHorizontalHeaderLabels(["Mouse", "Images", "Group"])
        self.mouse_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch); self.mouse_table.verticalHeader().hide()
        self.mouse_table.itemChanged.connect(self._group_edited)
        gl.addWidget(QLabel("Group of each mouse (mouse read from the start of the file name; leave the group empty to\n"
                            "measure a mouse without comparing it):")); gl.addWidget(self.mouse_table)
        row = QHBoxLayout(); row.addWidget(QLabel("Control group:")); self.control_cb = QComboBox()
        self.control_cb.currentTextChanged.connect(self._control_edited); row.addWidget(self.control_cb, 1); gl.addLayout(row)

        self.scale_box = QGroupBox("Scale (the files do not agree or do not say)"); fl = QFormLayout(self.scale_box); self.spin = []
        for name in ("Plane spacing (um)", "Pixel height (um)", "Pixel width (um)"):
            s = QDoubleSpinBox(); s.setDecimals(4); s.setRange(0.0, 50.0); s.setSingleStep(0.001); fl.addRow(name, s); self.spin.append(s)
        gl.addWidget(self.scale_box)

        gl.addWidget(QLabel("Bundle direction (bundles stand above the bright cuticular plates):"))
        row = QHBoxLayout(); self.side_btn = QPushButton("Show side view"); self.side_btn.clicked.connect(lambda: self.show_side_view())
        self.side_lower = QRadioButton("Toward lower plane numbers"); self.side_higher = QRadioButton("Toward higher plane numbers")
        self.side_group = QButtonGroup(self); self.side_group.addButton(self.side_lower); self.side_group.addButton(self.side_higher)
        self.side_lower.setChecked(True); row.addWidget(self.side_btn); row.addWidget(self.side_lower); row.addWidget(self.side_higher)
        gl.addLayout(row)

        row = QHBoxLayout(); self.confirm_btn = QPushButton("Confirm"); self.confirm_btn.clicked.connect(lambda: self.confirm())
        self.status = QLabel(""); self.status.setWordWrap(True); row.addWidget(self.confirm_btn); row.addWidget(self.status, 1)
        gl.addLayout(row); lay.addWidget(g); lay.addStretch(1)
        self._refresh()

    # images
    def _pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add images", "", "Images (*.nd2 *.tif *.tiff)")
        if paths: self.add_paths(paths)

    def _pick_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Add folder of images")
        if p: self.add_paths([p])

    def add_paths(self, paths):
        self.session.add_paths(list(paths))
        if not self.results_folder and self.session.infos:
            first = next(iter(self.session.infos.values())).path
            self.results_folder = os.path.join(os.path.dirname(first), "cochlea_phenotyping_results")
        self._refresh()

    def _selected_fields(self):
        return [it.data(Qt.UserRole) for it in self.images.selectedItems() if it.data(Qt.UserRole)]

    def exclude_selected(self, fields=None, reason: Optional[str] = None):
        """Leave the selected images out of the group comparisons (they are still measured)."""
        fields = fields or self._selected_fields()
        if not fields:
            self.status.setText(f"<span style='color:{WARN}'>Select an image in the list first</span>"); return []
        for f in fields: self.session.exclude(f, reason if reason is not None else self.reason_cb.currentText())
        self._refresh(); return fields

    def include_selected(self, fields=None):
        fields = fields or self._selected_fields()
        for f in fields: self.session.include(f)
        self._refresh(); return fields

    def _refresh(self):
        s = self.session; n = len(s.infos)
        self.images.clear()
        for f, i in sorted(s.infos.items()):
            notes = list(i.warnings) + ([f"left out of comparisons: {s.excluded[f]}"] if f in s.excluded else [])
            self.images.addItem(f + (f"   ({'; '.join(notes)})" if notes else ""))
            self.images.item(self.images.count() - 1).setData(Qt.UserRole, f)
        off = s.fields_off_scale()
        if n:
            vz, vy, vx = s.voxel_um or (None, None, None)
            text = f"{n} images" + (f", pixels {vx:.4f} um, planes {vz:.3f} um apart" if vx else "")
            if off: text += f". <span style='color:{WARN}'>{len(off)} differ in scale: {', '.join(off[:3])}{'...' if len(off) > 3 else ''}</span>"
            if s.errors: text += f". <span style='color:{WARN}'>{len(s.errors)} could not be read: " + "; ".join(f"{os.path.basename(p)} ({e})" for p, e in s.errors.items()) + "</span>"
            self.summary.setText(text)
        self.scale_box.setVisible(bool(n) and (bool(off) or s.voxel_um is None))
        for sp, v in zip(self.spin, s.voxel_um or (0, 0, 0)):
            sp.setValue(float(v))
        self.stain_table.setRowCount(0); panels = s.panels()
        for (p, ch), st in sorted(s.stains.items()):
            r = self.stain_table.rowCount(); self.stain_table.insertRow(r)
            files = panels.get(p, []); first = r == 0 or self.stain_table.item(r - 1, 0).data(Qt.UserRole) != p
            pit = QTableWidgetItem(f"{p} ({len(files)})" if first else ""); pit.setData(Qt.UserRole, p)
            pit.setFlags(pit.flags() & ~Qt.ItemIsEditable); pit.setToolTip("\n".join(files))
            it = QTableWidgetItem(ch); it.setFlags(it.flags() & ~Qt.ItemIsEditable); it.setToolTip(s.stain_reasons.get((p, ch), ""))
            self.stain_table.setItem(r, 0, pit); self.stain_table.setItem(r, 1, it)
            cb = QComboBox(); cb.addItems(KNOWN_STAINS if st in KNOWN_STAINS else KNOWN_STAINS + [st]); cb.setCurrentText(st)
            cb.setToolTip(s.stain_reasons.get((p, ch), ""))
            cb.currentTextChanged.connect(lambda text, p=p, ch=ch: self._edited(p, ch, text))
            self.stain_table.setCellWidget(r, 2, cb)
        self._fill_mouse_table()
        self._show_status()

    def _fill_mouse_table(self):
        s = self.session; self.mouse_table.blockSignals(True); self.mouse_table.setRowCount(0)
        by_mouse = {}
        for f, m in sorted(s.mouse_of.items()):
            by_mouse.setdefault(m, []).append(f)
        for m, files in sorted(by_mouse.items()):
            r = self.mouse_table.rowCount(); self.mouse_table.insertRow(r)
            a = QTableWidgetItem(m); a.setFlags(a.flags() & ~Qt.ItemIsEditable); a.setToolTip("\n".join(files))
            b = QTableWidgetItem(str(len(files))); b.setFlags(b.flags() & ~Qt.ItemIsEditable)
            c = QTableWidgetItem(s.group_of_mouse.get(m, ""))
            self.mouse_table.setItem(r, 0, a); self.mouse_table.setItem(r, 1, b); self.mouse_table.setItem(r, 2, c)
        self.mouse_table.blockSignals(False); self._fill_control()

    def _fill_control(self):
        s = self.session; self.control_cb.blockSignals(True); self.control_cb.clear()
        self.control_cb.addItems([""] + s.groups()); self.control_cb.setCurrentText(s.control_group if s.control_group in s.groups() else "")
        self.control_cb.blockSignals(False)

    def _group_edited(self, item):
        if item.column() != 2: return
        m = self.mouse_table.item(item.row(), 0).text(); self.session.set_group(m, item.text())
        self.session.stains_confirmed = False; self._fill_control(); self._show_status()

    def _control_edited(self, text):
        self.session.control_group = text; self.session.stains_confirmed = False; self._show_status()

    def _edited(self, panel, ch, text):
        self.session.set_stain(ch, text, panel=panel); self.session.bundle_side_confirmed = False; self._show_status()

    def _show_status(self, extra: str = ""):
        s = self.session
        if s.stains_confirmed and s.bundle_side_confirmed:
            self.status.setText(f"<span style='color:{OK}'>Confirmed{extra}</span>")
        elif s.infos:
            self.status.setText(f"<span style='color:{WARN}'>Check the stains, groups and bundle direction, then confirm{extra}</span>")
        else:
            self.status.setText("")

    def bundle_side(self) -> int:
        """-1: bundles toward lower plane numbers; +1: toward higher. (Read from the buttons directly: a Qt button
        id of -1 means 'assign one automatically'.)"""
        return -1 if self.side_lower.isChecked() else 1

    def confirm(self, results_folder: Optional[str] = None) -> Optional[str]:
        """Confirm stains, bundle direction and scale, and save dataset_settings.json in the results folder."""
        s = self.session
        if self.scale_box.isVisible():
            s.voxel_um = tuple(round(sp.value(), 4) for sp in self.spin)
        try:
            s.confirm_stains(); s.set_bundle_side(self.bundle_side())
            d = self.device_getter(); folder = results_folder or self.results_folder
            p = s.save(folder, confirmed_by=os.environ.get("USERNAME", os.environ.get("USER", "")),
                       device=d.as_dict() if d is not None else {})[0]
        except (ValueError, OSError) as e:
            s.stains_confirmed = s.bundle_side_confirmed = False
            self.status.setText(f"<span style='color:{WARN}'>{str(e)[:1].upper() + str(e)[1:]}</span>"); return None
        extra = f", saved in {folder}"
        nophal = s.panels_without_phalloidin()
        if nophal:
            n = sum(len(s.panels()[q]) for q in nophal)
            extra += f". <span style='color:{WARN}'>{n} images have no phalloidin channel and will not be measured ({'; '.join(nophal)})</span>"
        if not s.control_group and len(s.groups()) > 1:
            extra += f". <span style='color:{WARN}'>No control group chosen: groups will be described but not compared</span>"
        self._show_status(extra); return p

    def selected_field(self) -> Optional[str]:
        rows = [i.row() for i in self.images.selectedIndexes()]; names = sorted(self.session.infos)
        return names[rows[0]] if rows else (names[0] if names else None)

    def show_side_view(self, field: Optional[str] = None):
        from ._io import read_stack
        f = field or self.selected_field()
        if f is None:
            return None
        info = self.session.infos[f]; st = self.session.stains_of(f)
        if "Phalloidin" not in st:
            self.status.setText(f"<span style='color:{WARN}'>{f} has no phalloidin channel; pick another image</span>"); return None
        ph = read_stack(info.path)[:, st.index("Phalloidin")]; sv = side_view(ph); dz, dy, dx = info.voxel_um
        if self.viewer is not None:
            self.viewer.add_image(sv, name=f"{f} side view (plane number increases downward)", scale=(dz, dx), colormap="gray")
        return sv
