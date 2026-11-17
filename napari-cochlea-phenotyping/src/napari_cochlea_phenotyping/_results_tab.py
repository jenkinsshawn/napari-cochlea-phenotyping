"""Results tab: the validation cards, the group comparisons with their warnings, the result tables, and the measured
cells of the selected image as layers. It opens the folder of the run that just finished, or any results folder.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices
from qtpy.QtCore import QProcess, QProcessEnvironment
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QPushButton,
                            QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ._results import LEVEL_COLORS, LEVELS, TABLES, Results, open_results

OK, WARN, GRAY = "#2e7d32", "#b26a00", "#757575"
ROW_COLORS = {"OHC1": "#e53935", "OHC2": "#43a047", "OHC3": "#1e88e5", "OHC4": "#8e24aa"}
SHOWN_BY_DEFAULT = ("validated", "validated, more variable", "partly checked")


def _fmt(v, nd=2):
    try:
        if v != v: return ""
        return f"{v:.{nd}f}"
    except TypeError:
        return str(v)


def _p(v):
    if not (v == v): return ""
    return f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"


class ResultsTab(QWidget):
    def __init__(self, viewer=None, session_getter: Callable = lambda: None, selected_getter: Callable = lambda: None,
                 results_getter: Callable = lambda: ""):
        super().__init__()
        self.viewer, self.session_getter, self.selected_getter, self.results_getter = viewer, session_getter, selected_getter, results_getter
        self.results: Optional[Results] = None; self._comparisons = []
        lay = QVBoxLayout(self)
        row = QHBoxLayout(); self.open_btn = QPushButton("Open results folder..."); self.reveal_btn = QPushButton("Show folder")
        self.open_btn.clicked.connect(lambda: self._pick()); self.reveal_btn.clicked.connect(lambda: self._reveal())
        row.addWidget(self.open_btn); row.addWidget(self.reveal_btn); lay.addLayout(row)
        self.status = QLabel("Results appear here when a run finishes, or open a results folder."); self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText); lay.addWidget(self.status)

        lay.addWidget(QLabel("<b>Agreement with the hand references</b>"))
        self.cards = QTableWidget(0, 5); self.cards.setHorizontalHeaderLabels(["measure", "agreement", "95% interval", "documented", "reference"])
        self._setup(self.cards); lay.addWidget(self.cards)

        lay.addWidget(QLabel("<b>Group comparisons</b> (mouse-level P; hover a row for its notes)"))
        f = QHBoxLayout(); self.group_box = QComboBox(); self.group_box.currentIndexChanged.connect(lambda _: self._fill_comparisons())
        self.all_box = QCheckBox("include quality checks and measures that are not validated"); self.all_box.toggled.connect(lambda _: self._fill_comparisons())
        f.addWidget(self.group_box); f.addWidget(self.all_box, 1); lay.addLayout(f)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["measure", "group", "status", "control median", "group median", "difference", "P", "adjusted P"])
        self._setup(self.table); lay.addWidget(self.table, 1)
        self.warn = QLabel(""); self.warn.setWordWrap(True); self.warn.setTextFormat(Qt.RichText); lay.addWidget(self.warn)

        t = QHBoxLayout(); self.table_box = QComboBox(); self.table_box.addItems(list(TABLES)); self.table_btn = QPushButton("Open table")
        self.table_btn.clicked.connect(lambda: self.open_table()); t.addWidget(self.table_box, 1); t.addWidget(self.table_btn); lay.addLayout(t)
        self.layers_btn = QPushButton("Show measured cells on the selected image"); self.layers_btn.clicked.connect(lambda: self.show_layers())
        lay.addWidget(self.layers_btn)
        self.redo_btn = QPushButton("Redo statistics with a genotype table...")
        self.redo_btn.setToolTip("Recompute every group test of this run from a table with columns mouse, group, without measuring again")
        self.redo_btn.clicked.connect(lambda: self.redo_statistics()); lay.addWidget(self.redo_btn)
        self._proc = None

    @staticmethod
    def _setup(t):
        t.setEditTriggers(QTableWidget.NoEditTriggers); t.setSelectionBehavior(QTableWidget.SelectRows); t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents); t.horizontalHeader().setStretchLastSection(True)

    # loading --------------------------------------------------------------------------------------------------------
    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder of a run", self.results_getter() or "")
        if d: self.load(d)

    def _reveal(self):
        if self.results is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.results.folder))

    def load(self, folder: str) -> bool:
        try:
            self.results = open_results(folder)
        except FileNotFoundError as e:
            self.status.setText(f"<span style='color:{WARN}'>{e}</span>"); return False
        r = self.results; info = r.run_info
        txt = f"Results in {r.folder}"
        if info: txt += f"<br><span style='color:{GRAY}'>run started {info.get('started', '?')}, plugin {info.get('plugin_version', '?')}, graphics card: {info.get('graphics_card', '?')}</span>"
        miss = r.missing()
        if miss: txt += f"<br><span style='color:{WARN}'>not in this folder: {', '.join(miss)}</span>"
        self.status.setText(txt)
        self._fill_cards(); self._comparisons = r.comparisons()
        groups = sorted({c.group for c in self._comparisons})
        self.group_box.blockSignals(True); self.group_box.clear(); self.group_box.addItems([f"all groups against {r.control}"] + [f"{g} against {r.control}" for g in groups])
        self.group_box.blockSignals(False); self._fill_comparisons(); return True

    def _fill_cards(self):
        V = self.results.read("validation cards"); self.cards.setRowCount(0)
        if V is None or V.empty:
            self.cards.setRowCount(1); self.cards.setItem(0, 0, QTableWidgetItem("not assessed: this run has no hand references")); return
        self.cards.setRowCount(len(V))
        for i, (_, c) in enumerate(V.iterrows()):
            vals = [c.get("measure", ""), f"{_fmt(c.get('agreement_pct'), 1)}% of {int(c.get('cells', 0))} cells" if c.get("agreement_pct") == c.get("agreement_pct") else "not assessed",
                    f"{_fmt(c.get('interval_low_pct'), 1)}-{_fmt(c.get('interval_high_pct'), 1)}%" if c.get("interval_low_pct") == c.get("interval_low_pct") else "",
                    f"{_fmt(c.get('documented_pct'), 1)}%" if c.get("documented_pct") == c.get("documented_pct") else "", str(c.get("assessed_on", ""))]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v)); it.setToolTip(f"{c.get('agreement_rule', '')}\n{c.get('interval_method', '')}\n{c.get('reference', '')}")
                self.cards.setItem(i, j, it)

    def shown_comparisons(self):
        k = self.group_box.currentIndex(); groups = sorted({c.group for c in self._comparisons})
        keep = [c for c in self._comparisons if k <= 0 or c.group == groups[k - 1]]
        if not self.all_box.isChecked(): keep = [c for c in keep if c.level in SHOWN_BY_DEFAULT]
        return keep

    def _fill_comparisons(self):
        rows = self.shown_comparisons(); self.table.setRowCount(len(rows)); n_warn = 0
        for i, c in enumerate(rows):
            vals = [c.label, c.group, c.level, _fmt(c.control_median), _fmt(c.group_median), _fmt(c.difference),
                    _p(c.p) + (f" ({c.mice_control} v {c.mice_group} mice)" if c.mice_group else ""), _p(c.p_adjusted)]
            tip = c.note + ("\n\n" + "\n".join(c.warnings) if c.warnings else "")
            for j, v in enumerate(vals):
                it = QTableWidgetItem(("! " if (j == 0 and c.warnings) else "") + str(v)); it.setToolTip(tip)
                if j == 2: it.setForeground(QColor(LEVEL_COLORS.get(c.level, GRAY)))
                self.table.setItem(i, j, it)
            n_warn += bool(c.warnings)
        hidden = len([c for c in self._comparisons if c.level not in SHOWN_BY_DEFAULT])
        msg = []
        if n_warn: msg.append(f"<span style='color:{WARN}'>{n_warn} comparisons carry a note (marked !); hover for details</span>")
        if hidden and not self.all_box.isChecked(): msg.append(f"<span style='color:{GRAY}'>{hidden} comparisons of quality checks and measures that are not validated are hidden</span>")
        self.warn.setText("<br>".join(msg))

    # tables and layers ------------------------------------------------------------------------------------------------
    def open_table(self, name: Optional[str] = None) -> Optional[str]:
        if self.results is None: return None
        p = self.results.tables.get(name or self.table_box.currentText())
        if p and os.path.exists(p):
            QDesktopServices.openUrl(QUrl.fromLocalFile(p)); return p
        self.status.setText(f"<span style='color:{WARN}'>{os.path.basename(p or '')} is not in this results folder</span>"); return None

    def _pixel_um(self, field):
        s = self.session_getter()
        if s is not None and field in getattr(s, "infos", {}):
            return float(s.infos[field].voxel_um[2])
        ft = os.path.join(self.results.folder, "field_table.csv")
        if os.path.exists(ft):
            import pandas as pd
            d = pd.read_csv(ft); d = d[d.field == field]
            if len(d) and "pixel_um" in d: return float(d.pixel_um.iloc[0])
        return None

    def layer_data(self, field):
        """Points (in um) for the measured cells of field: inner hair cells with their bundle height, outer hair cells by row."""
        px = self._pixel_um(field)
        if px is None: raise ValueError(f"no pixel size known for {field}")
        out = []
        ih = self.results.read("inner hair cells")
        if ih is not None:
            d = ih[ih.field == field]
            if len(d): out.append(("inner hair cells", d[["y", "x"]].values * px, {"height_um": d.height_um.values, "category": d.category.astype(str).values}, "#ffeb3b"))
        oh = self.results.read("outer hair cells and rows")
        if oh is not None:
            d = oh[(oh.field == field) & oh.row.astype(str).str.startswith("OHC")]
            for rname, g in d.groupby("row"):
                out.append((f"outer hair cells {rname}", g[["y", "x"]].values * px, {}, ROW_COLORS.get(rname, "#ffffff")))
        return out

    def show_layers(self, field: Optional[str] = None):
        field = field or self.selected_getter()
        if self.results is None or not field:
            self.status.setText(f"<span style='color:{WARN}'>Open results and select an image in the Standardize tab first</span>"); return []
        try:
            data = self.layer_data(field)
        except ValueError as e:
            self.status.setText(f"<span style='color:{WARN}'>{e}</span>"); return []
        if not data:
            self.status.setText(f"<span style='color:{WARN}'>{field} has no measured cells in these results</span>"); return []
        if self.viewer is not None:
            for name, pts, props, color in data:
                self.viewer.add_points(pts, name=f"{field} {name}", size=2.0, face_color="transparent", border_color=color,
                                       properties=props or None)
        return data

    # redo statistics ---------------------------------------------------------------------------------------------
    def redo_statistics(self, groups_csv: str = "", control: str = ""):
        """Run phenotype_statistics.py on this results folder with another genotype table; reload when it finishes."""
        from ._run import groups_in_table, redo_statistics_command
        if self.results is None:
            self.status.setText(f"<span style='color:{WARN}'>Open a results folder first.</span>"); return None
        if not groups_csv:
            groups_csv, _ = QFileDialog.getOpenFileName(self, "Genotype table (columns mouse, group)", self.results.folder, "Tables (*.csv)")
            if not groups_csv: return None
        groups = groups_in_table(groups_csv)
        if not control:
            if len(groups) < 2:
                self.status.setText(f"<span style='color:{WARN}'>The table needs at least two groups; found {len(groups)}.</span>"); return None
            start = groups.index("control") if "control" in groups else 0
            control, ok = QInputDialog.getItem(self, "Control group", "Control group:", groups, start, False)
            if not ok: return None
        try:
            argv, env = redo_statistics_command(self.results.folder, groups_csv, control)
        except FileNotFoundError as e:
            self.status.setText(f"<span style='color:{WARN}'>{e}</span>"); return None
        pe = QProcessEnvironment()
        for k, v in env.items(): pe.insert(k, v)
        self._proc = QProcess(self); self._proc.setProcessEnvironment(pe); self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.setWorkingDirectory(os.path.dirname(argv[1]))
        folder = self.results.folder
        self._proc.finished.connect(lambda code, _status: self._redo_done(code, folder))
        self.redo_btn.setEnabled(False)
        self.status.setText(f"Redoing the statistics with {os.path.basename(groups_csv)} (control: {control}); this takes a few minutes...")
        self._proc.start(argv[0], argv[1:])
        return argv

    def _redo_done(self, code: int, folder: str):
        out = bytes(self._proc.readAll()).decode("utf-8", "replace") if self._proc is not None else ""
        with open(os.path.join(folder, "redo_statistics_log.txt"), "w", encoding="utf-8") as fh: fh.write(out)
        self.redo_btn.setEnabled(True)
        if code == 0:
            self.load(folder); self.status.setText(self.status.text() + "<br>Statistics redone; tables updated (log: redo_statistics_log.txt).")
        else:
            self.status.setText(f"<span style='color:{WARN}'>The statistics stopped with an error; see redo_statistics_log.txt in the results folder.</span>")
